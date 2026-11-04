#!/usr/bin/env python3
"""CPU-only preview of domain soft balance, task rotation and reply preference."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from preview_dedup import PROJECT, SEED, require, timestamp
from preview_r11_cap import (load_frozen, read_json, selection_hash, planned_exposure,
                             load_semantics, diagnostics)
from jev_train.common import digest, rows, write_json
from jev_train.domains import LABELS, label_acceptance
from jev_train.reply_origin import classify_origin
from jev_train.task_selection import select
from analyze_training_mechanism import (TurnView, source_units, classify_capture, BASE_FIELDS,
                                       termination_evidence, policy_flags, classify_respond, previous_error)


def restore_origins(kept, feature_path, build_config):
    targets = {sid: row for sid,row in kept.items() if row['gold_names'] == ['RESPOND']}
    features = {}
    for row in rows(feature_path):
        sid = row['id']
        if sid in targets:
            require(sid not in features, f'duplicate feature ID: {sid}')
            require(row['dataset'] == targets[sid]['dataset'] and row['split'] == 'train',
                    f'{sid}: saved feature identity differs')
            features[sid] = row
    require(features.keys() == targets.keys(), 'missing RESPOND state features')
    result = {}
    for unit in source_units(targets, build_config):
        view = TurnView(unit)
        descriptors = [{key: target[key] for key in BASE_FIELDS} | {'populations': [], 'source': target['source']}
                       for target in unit['targets']]
        terminal = termination_evidence(dict(unit, targets=descriptors))
        by_index = {target['turn_index']: target for target in unit['targets']}
        for row in classify_capture(unit['capture'], descriptors, unit['family'],
                                    unit['raw_records'], unit['prefix'], target_fields=BASE_FIELDS):
            sid, index = row['id'], row['turn_index']
            require(sid not in result, f'duplicate restored reply: {sid}')
            target, feature = by_index[index], features[sid]
            require(target['original_gold'] == target['gold_names'] == ['RESPOND'], 'C1 reply identity differs')
            row.update(termination_evidence=terminal[index], **policy_flags(row, terminal[index]))
            require(row['category'] == target['next_event'] == feature['next_event'], f'{sid}: successor differs')
            require(row['d2'] == feature['stop_reason_abnormal'] is False, f'{sid}: C1 D2 differs')
            prior = view.prior(index)
            require(prior['previous_turn_tool_calls'] == feature['turn_calls_before'], f'{sid}: prior calls differ')
            origin = classify_origin(view, index, prior, row['termination_status'])
            semantic, reason = classify_respond(row)
            result[sid] = dict(origin, dataset=target['dataset'], model=target['model'],
                task_group=target['task_group'], trajectory_id=target['trajectory_id'],
                source=row['source'], turn_index=index, next_event=row['category'],
                prior_calls=prior['previous_turn_tool_calls'], call_count_status=prior['call_count_status'],
                termination_status=row['termination_status'], termination_value=row['termination_value'],
                semantic=semantic, semantic_unknown_reason=reason, prior_error=previous_error(view,index,prior))
    require(result.keys() == targets.keys(), 'reply origin reconstruction coverage differs')
    return dict(sorted(result.items()))


def bounded_population(kept, labels):
    """A few complete retained trajectories per source/domain; never a formal training subset."""
    by_source, by_domain = defaultdict(list), defaultdict(list)
    for row in kept.values():
        by_domain[labels[row['task_group']]].append(row)
        if row['gold_names'] == ['RESPOND']:
            by_source[row['dataset']].append(row)
    selected = []
    for group in [*by_source.values(), *by_domain.values()]:
        selected.extend(sorted(group,key=lambda r:(digest(SEED,r['id']),r['id']))[:2])
    # Previously read, real positive examples exercise both input events and forced summaries.
    examples = ('963c7ee7e48b590f04a9405c0a333bd9df4d3441c4008b40a2de68f6bf53f4e5',
                '00454620316047cd2125ec71f8fa14d133f892e23fce4ab51e345b3b69ad98ec',
                '0068dbb8186661c6b9021a28eee8ef0040c80126f511d944d5825312a93c8d01',
                '001d9b76f4a534d47009fed4fb19003048da414eb1cf678b9fb8496ee21b54eb')
    selected.extend(kept[sid] for sid in examples if sid in kept)
    trajectories = {(r['dataset'],r['trajectory_id']) for r in selected}
    return {sid:r for sid,r in kept.items() if (r['dataset'],r['trajectory_id']) in trajectories}


def position_bins(kept):
    trajectories = defaultdict(list)
    for row in kept.values():
        trajectories[(row['dataset'],row['trajectory_id'])].append(row)
    result = {}
    for group in trajectories.values():
        for i,row in enumerate(sorted(group,key=lambda r:(r['turn_index'],r['id']))):
            result[row['id']] = min(9,(2*i+1)*10//(2*len(group)))
    return result


def describe(records, origins):
    replies = [r for r in records if r['gold_names'] == ['RESPOND']]
    tasks = Counter(r['task_group'] for r in records)
    trajectories = Counter((r['dataset'],r['trajectory_id']) for r in records)
    return dict(count=len(records), respond=len(replies), respond_fraction=len(replies)/len(records),
        reply_classes=dict(sorted(Counter(origins[r['id']]['category'] for r in replies).items())),
        reply_rules=dict(sorted(Counter(origins[r['id']]['rule'] for r in replies).items())),
        respond_next_event=dict(sorted(Counter(r['next_event'] for r in replies).items())),
        short_turn=sum(r['short_turn'] for r in records), direct=sum(r['direct'] for r in records),
        old_priority=sum(r['short_turn'] or r['direct'] for r in records),
        turn_calls_unknown=sum(r['turn_calls_unknown'] for r in records),
        tool_set=sum(len(r['gold_names'])>1 for r in records),
        strict_loop=dict(Counter(str(r['strict_loop']) for r in records)),
        task_groups=len(tasks), trajectories=len(trajectories),
        maximum_states_per_task=max(tasks.values()), maximum_states_per_trajectory=max(trajectories.values()),
        states_per_task_histogram=dict(sorted(Counter(tasks.values()).items())),
        states_per_trajectory_histogram=dict(sorted(Counter(trajectories.values()).items())),
        n_prompt_tokens=sum(r['n_prompt_tokens'] for r in records))


def summarize_selection(ids, kept, origins, labels, positions, r11_ids):
    records = [kept[sid] for sid in ids]
    result = describe(records,origins)
    sources, teachers, domains = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        sources[row['dataset']].append(row)
        teachers[(row['dataset'],row['model'])].append(row)
        domains[labels[row['task_group']]].append(row)
    result.update(by_source={key:describe(group,origins) for key,group in sorted(sources.items())},
        by_teacher=[dict(dataset=ds,model=model,**describe(group,origins)) for (ds,model),group in sorted(teachers.items())],
        by_domain={key:describe(group,origins) for key,group in sorted(domains.items())},
        relative_position_bins=[sum(positions[sid]==i for sid in ids) for i in range(10)],
        source_message_index_zero=sum(r['turn_index']==0 for r in records),
        overlap_with_R11_8k=len(set(ids)&set(r11_ids)),
        new_task_groups_vs_R11=len({r['task_group'] for r in records}-{kept[sid]['task_group'] for sid in r11_ids if sid in kept}))
    lengths = sorted(r['n_prompt_tokens'] for r in records)
    result['prompt_length'] = dict(method='nearest_rank', minimum=lengths[0], maximum=lengths[-1],
        mean=sum(lengths)/len(lengths), **{f'p{q}':lengths[math.ceil(q/100*len(lengths))-1] for q in (10,25,50,75,90,95,99)})
    # Match load_train_ids, not the order of the hierarchy traversal.
    training = sorted(records,key=lambda r:(digest(SEED,r['id']),r['id']))
    coefficients, schedule = planned_exposure(training)
    result['schedule'] = schedule
    buckets = defaultdict(float)
    for row in training:
        category = origins[row['id']]['category'] if row['gold_names']==['RESPOND'] else 'tool'
        buckets[category] += coefficients[row['id']]/schedule['updates']
    result['planned_linear_loss_shares'] = dict(sorted(buckets.items()))
    replies = [origins[sid] for sid in ids if sid in origins]
    known = [r['id'] for r in records if r['id'] in origins and origins[r['id']]['semantic']=='short_turn']
    unknown = [r['id'] for r in records if r['id'] in origins
               and r['next_event']=='user_turn_end' and origins[r['id']]['semantic']=='unknown']
    result['strict_short_stop'] = dict(known_positive=len(known), unknown=len(unknown), unknown_ids=unknown,
        loss_coefficient_fraction_lower_bound=sum(coefficients[sid] for sid in known)/schedule['updates'],
        loss_coefficient_fraction_upper_bound=sum(coefficients[sid] for sid in known+unknown)/schedule['updates'])
    result['respond_prior_error_status'] = dict(sorted(Counter(r['prior_error']['status'] for r in replies).items()))
    result['respond_gate_passed'] = .08 <= result['respond_fraction'] <= .20
    return result


def build_preview(config, smoke=False):
    require(config['counts'] == [8000,38000,76000] and config['seed']==SEED, 'task-reply protocol differs')
    kept, labels, reference_ids, r11_ids, regression, inputs, label_state = load_frozen(config)
    reference = Counter(labels[kept[sid]['task_group']] for sid in reference_ids)
    require(sum(reference.values())==8000 and set(reference)==set(LABELS), 'frozen R00 domain counts differ')
    parent = read_json(Path(inputs['dedup_preview'])/'summary.json')
    feature_path = Path(parent['inputs']['features'])/'state_features.jsonl'
    build_path = Path(parent['inputs']['build_config'])
    require(read_json(feature_path.parent/'metadata/run_status.json')['status']=='completed', 'features incomplete')
    cache = (PROJECT/config['mechanism']).resolve()
    old_semantics = load_semantics(cache,kept)
    old_order = sorted(r11_ids,key=lambda sid:(digest(SEED,sid),sid))
    baseline = dict(selected=regression['selected'], diagnostics=diagnostics(old_order,kept,r11_ids,old_semantics))
    old_cap = read_json(PROJECT/config['r11_cap_preview']/'summary.json')
    population = bounded_population(kept,labels) if smoke else kept
    print(f"restoring {sum(r['gold_names']==['RESPOND'] for r in population.values())} replies from {len(population)} states",flush=True)
    origins = restore_origins(population,feature_path,read_json(build_path))
    positions = position_bins(population)
    classes = {sid:row['category'] for sid,row in origins.items()}
    source_population = defaultdict(list)
    for row in population.values():
        source_population[row['dataset']].append(row)
    result = dict(recipe='task_reply_v1',seed=SEED,smoke=smoke,inputs=dict(inputs,
        state_features=str(feature_path),build_config=str(build_path),mechanism=str(cache),
        r11_cap_preview=str((PROJECT/config['r11_cap_preview']).resolve())),
        ordered_R11_8k_regression=True, label_acceptance=label_acceptance(label_state,Path(inputs['labels'])),
        reference_domain_counts=dict(reference), candidate_population=describe(list(population.values()),origins),
        candidate_sources={source:describe(group,origins) for source,group in sorted(source_population.items())},
        reply_classification=origins, targets={}, comparisons=dict(R11_8k=baseline,
            R11_cap={count:dict(status=arm['status'],selected=arm['selected'],diagnostics=arm['diagnostics'])
                     for count,arm in old_cap['targets'].items()}),
        definitions=dict(domain='0.5 * frozen R00 domain fraction + 0.5/7; cap unique states; redistribute by original weights.',
            candidate_population='All frozen C1-cleaned deduplicated states, including zero-R00 source/domain cells. No new raw states.',
            task='Within each domain: task_group round robin shared across sources; source -> teacher -> retained trajectory.',
            state='Trajectory: P(R)=2r/(2r+o), then independent phase-shifted spread streams; without replacement.',
            R='Normalized RESPOND with verifiable boundary and exact independent task-field/native user-event evidence; no confirmed framework finalization.',
            O='Tools, verified framework finalization and unknown reply origins. Unknown is not a natural-reply negative label.',
            no_C3='No old 25% short_turn/direct quota. No short-stop or context-length quota. No C4 filter.',
            gate='Global RESPOND 8%-20% review check; no automatic rule changes. Smoke records it but is not a formal quota test.',
            relative_position='Ten equal bins of (ordinal+.5)/retained_eligible_trajectory_states, ordered by (turn_index,id); not original full trajectory length.',
            loss='No added R/O loss weights; original microbatch mean loss unchanged. Predicted single-GPU accum=2 schedule, IDs sorted as load_train_ids; not measured gradients.',
            limitation='Origin evidence does not prove task completion or universal absence of framework influence; missing/null finish metadata is not inferred success.',
            frozen='Historical prompts/candidates remain unchanged; Codex candidate completeness/visibility remains uncertain.'),
        overlaps_between_sizes={})
    selections = {}
    for count in ([min(200,len(population))] if smoke else config['counts']):
        ids, allocation = select(population,labels,reference,classes,count,SEED)
        selected = summarize_selection(ids,population,origins,labels,positions,r11_ids)
        passed = selected['respond_gate_passed']
        result['targets'][str(count)] = dict(status='smoke_completed' if smoke else 'completed' if passed else 'failed',
            failure_stage=None if smoke or passed else 'RESPOND_gate', selection_sha256=selection_hash(ids),
            allocation=allocation, selected=selected)
        for previous, old_ids in selections.items():
            result['overlaps_between_sizes'][f'{previous}:{count}'] = dict(overlap=len(set(old_ids)&set(ids)),
                smaller_is_subset=set(old_ids)<=set(ids), note='Observed at these sizes, not a general nesting guarantee.')
        selections[count] = ids
        print(f"selected {count}: RESPOND={selected['respond']}, gate={passed}",flush=True)
    return result


def report_markdown(result):
    lines = ['# 领域软均衡、任务轮转与回复优先：CPU预览','',
        '仅生成预览；未写训练样本或ID文件，未训练、评测或调用模型。旧C1/去重/候选和领域标签冻结。',
        'R要求独立任务字段或Codex双事件的正向来源证据；明确框架收尾和证据未知均放O。此分类不证明任务完成。','',
        '| 状态数 | RESPOND | R | 框架收尾 | 未知回复 | 任务 | 轨迹 | token | 预计更新 | 门槛 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for count,arm in result['targets'].items():
        s=arm['selected'];c=s['reply_classes']
        lines.append(f"| {count} | {s['respond']} ({s['respond_fraction']:.2%}) | {c.get('verified_reply',0)} | {c.get('framework_forced',0)} | {c.get('unknown',0)} | {s['task_groups']} | {s['trajectories']} | {s['n_prompt_tokens']:,} | {s['schedule']['updates']} | {s['respond_gate_passed']} |")
    for count,arm in result['targets'].items():
        s=arm['selected']
        lines += ['',f'## {count}', '', '| 领域 | R00权重 | 软权重 | 容量 | 实取 |', '|---|---:|---:|---:|---:|']
        for d,p in arm['allocation'].items():
            lines.append(f"| {d} | {p['reference_count']} | {p['weight']:.4%} | {p['capacity']} | {p['quota']} |")
        lines += ['', '| 来源 | 实取 | RESPOND | R | 框架 | 未知 |', '|---|---:|---:|---:|---:|---:|']
        for ds,v in s['by_source'].items():
            c=v['reply_classes']
            lines.append(f"| {ds} | {v['count']} | {v['respond']} | {c.get('verified_reply',0)} | {c.get('framework_forced',0)} | {c.get('unknown',0)} |")
        lines += ['',f"位置十箱：{s['relative_position_bins']}；分母是去重后可用状态，不是完整原始轨迹。",
            f"单任务/轨迹最大状态数：{s['maximum_states_per_task']}/{s['maximum_states_per_trajectory']}。",
            f"与原R11-8k重合{s['overlap_with_R11_8k']}；旧优先类{s['old_priority']}仅作诊断。",
            f"严格短停已知/未知：{s['strict_short_stop']['known_positive']}/{s['strict_short_stop']['unknown']}。",
            f"R/O权重只影响选取，不改变损失；预计线性系数份额：{s['planned_linear_loss_shares']}。"]
    lines += ['', '逐ID来源证据、未知原因、来源×教师统计、长度分位数、原R11/R11-cap对照及规模重合见summary。',
        '只恢复回复的原始证据；previous_error按既有结构化标志判断，no_explicit_error不等于成功。',
        '当前轮界和严格短停仍是结构代理；未修复历史Codex候选、未新增领域标注。',
        '8%–20%只作审阅门槛；失败不补样、不调权重，停在用户审阅点。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/selection/task_reply.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.smoke and not out.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under project tmp')
    out.mkdir(parents=True,exist_ok=False);(out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    state=dict(status='running',started_at=timestamp(),smoke=args.smoke)
    write_json(out/'metadata/run_status.json',state)
    try:
        result=build_preview(read_json(args.config),args.smoke)
        result['config']=str(args.config.resolve())
        write_json(out/'summary.json',result);(out/'report.md').write_text(report_markdown(result))
        failed=[n for n,a in result['targets'].items() if a['status']=='failed']
        state.update(status='failed' if failed else 'smoke_completed' if args.smoke else 'completed',
                     ended_at=timestamp(),failed_targets=failed,
                     reply_states=len(result['reply_classification']),ordered_R11_8k_regression=True)
        write_json(out/'metadata/run_status.json',state)
    except BaseException as exc:
        state.update(status='failed',ended_at=timestamp(),error=str(exc))
        write_json(out/'metadata/run_status.json',state)
        raise
    print(json.dumps(dict(output=str(out),status=state['status']),ensure_ascii=False),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    sys.exit(main())
