#!/usr/bin/env python3
"""CPU-only v2 preview: source quotas, shared task counts, verifiable replies."""
import argparse
from collections import Counter, defaultdict, deque
import json
import os
from pathlib import Path
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from preview_dedup import PROJECT, SEED, require, timestamp
from preview_r11_cap import load_frozen, read_json, selection_hash
from preview_task_reply import bounded_population, describe, summarize_selection
from jev_train.common import digest, write_json
from jev_train.domains import label_acceptance
from jev_train.reply_origin import reply_priority, zclaw_framework_message, hermes_budget_pressure
from jev_train.task_selection import select, select_with_sources
from analyze_training_mechanism import source_units, TurnView, classify_capture, BASE_FIELDS
from dataset_tool_inventory.adapters import review_formats as formats
from dataset_tool_inventory.adapters.review_common import json_value


REVIEW_SOURCES = ('DeepSeek-v4-Pro-Agent', 'hermes-agent-reasoning-traces', 'GLM-5.2-Agent',
                 'claude-fable-5-claude-code-traces', 'mimo-claude-code-traces-1k', 'ZClawBench')


def check_restored_source(target, origin):
    """The two exact locator representations produced by classify_capture.

    Native raw._source stays unchanged. Record adapters add the normalized
    message index to context.source, whereas canonical source names the record.
    """
    expected = target['source']
    if '#normalized_messages[' in origin['source']:
        expected += f"#normalized_messages[{target['turn_index']}]"
    require(origin['source'] == expected, f"{target['id']}: reply source locator differs")


def review_ids(origins, r11_ids):
    """20 newly eligible unknown-origin replies/source plus old unknown short stops."""
    reasons = defaultdict(list)
    for source in REVIEW_SOURCES:
        strata = defaultdict(list)
        for sid, row in origins.items():
            if (row['dataset'] == source and row['category'] == 'unknown'
                    and row['call_count_status'] == 'known'):
                strata[row['prior_calls'] == 0, row['next_event']].append(sid)
        streams = deque(deque(sorted(ids, key=lambda sid: (digest(SEED, 'boundary_review_v2', sid), sid)))
                        for _, ids in sorted(strata.items()))
        chosen = []
        while streams and len(chosen) < 20:
            stream = streams.popleft()
            chosen.append(stream.popleft())
            if stream:
                streams.append(stream)
        require(len(chosen) == 20, f'{source}: insufficient bounded review supply')
        for sid in chosen:
            reasons[sid].append('source_unknown_20')
    strict = [sid for sid in r11_ids if sid in origins and origins[sid]['category'] == 'unknown'
              and origins[sid]['semantic'] == 'short_turn']
    require(len(strict) == 63, 'frozen R11 unknown strict-stop count differs')
    for sid in strict:
        reasons[sid].append('R11_unknown_strict_short_stop')
    return dict(sorted(reasons.items()))


def load_origins(directory, kept, inputs):
    """Use the finished v1 reconstruction, including its documented gate-only failure."""
    old, status = read_json(directory/'summary.json'), read_json(directory/'metadata/run_status.json')
    require(old['recipe'] == 'task_reply_v1' and old['seed'] == SEED and not old['smoke']
            and old['ordered_R11_8k_regression'] and status['ordered_R11_8k_regression']
            and not status['smoke'] and 'error' not in status and status['ended_at'],
            'reply evidence requires the fully reconstructed v1 preview')
    require(set(old['targets']) == {'8000', '38000', '76000'}, 'v1 target coverage differs')
    failed = []
    for count, arm in old['targets'].items():
        s = arm['selected']
        passed = .08 <= s['respond_fraction'] <= .20
        require(s['count'] == int(count) and s['respond_gate_passed'] == passed
                and arm['status'] == ('completed' if passed else 'failed')
                and arm['failure_stage'] == (None if passed else 'RESPOND_gate'),
                'v1 failure is not the recorded completed-selection RESPOND gate')
        if not passed:
            failed.append(count)
    require(status['failed_targets'] == failed
            and status['status'] == ('failed' if failed else 'completed'), 'v1 run status differs')
    for key, value in inputs.items():
        require(old['inputs'][key] == value, f'frozen reply input differs: {key}')
    origins = old['reply_classification']
    targets = {sid: r for sid, r in kept.items() if r['gold_names'] == ['RESPOND']}
    require(origins.keys() == targets.keys() and status['reply_states'] == len(origins),
            'reply evidence must cover every frozen normalized RESPOND')
    for sid, origin in origins.items():
        for field in ('dataset', 'model', 'task_group', 'trajectory_id', 'turn_index', 'next_event'):
            require(origin[field] == targets[sid][field], f'{sid}: reply evidence identity differs: {field}')
        check_restored_source(targets[sid], origin)
        reply_priority(origin)
    return origins, old


def load_review(path, origins, r11_ids):
    # Both the temporary review input and a retained preview use this same field.
    review = read_json(path)['reply_boundary_review']
    require(review['pending_policy_questions'] == [],
            'bounded source review has unresolved scientific policy questions; stop before formal preview')
    require(review['policy_decisions'] == dict(
        hermes_nonexhausted_final_now='retain_priority_with_diagnostic',
        zclaw_framework_boundaries='exclude_explicit_notification_boundaries'), 'approved boundary policies differ')
    expected = review_ids(origins, r11_ids)
    cases = review['cases']
    require(review['protocol'] == 'task_reply_v2_boundary_review' and set(cases) == set(expected),
            'bounded review coverage differs')
    for sid, case in cases.items():
        origin = origins[sid]
        require(case['id'] == sid and case['source'] == origin['source']
                and case['user_boundary_source'] == origin['user_boundary_source'],
                f'{sid}: review source locator differs')
        require(case['review_reasons'] == expected[sid], f'{sid}: review stratum differs')
        require(case['origin_category'] == origin['category'] == 'unknown', 'review must preserve unknown origin')
        require(case['boundary_consistent'] is True and case['termination_consistent'] is True,
                f'{sid}: unresolved boundary/termination review')
        require(case['framework_forced_evidence'] is False,
                f'{sid}: new framework evidence requires an explicit source rule before preview')
        require(bool(case['evidence']), f'{sid}: missing review evidence')
        require((origin['next_event'] == 'trajectory_end' and case['next_source'] is None)
                or (origin['next_event'] == 'user_turn_end' and isinstance(case['next_source'], str)
                    and bool(case['next_source'])), f'{sid}: review successor locator differs')
    return review


def restore_boundary_evidence(kept, origins, build_config):
    """Recheck only the two affected sources; never rewrite frozen C1 or origins."""
    sources = {'ZClawBench', 'hermes-agent-reasoning-traces'}
    targets = {sid: kept[sid] for sid, row in origins.items() if row['dataset'] in sources}
    checked, boundary_matches, pressure, diagnostic_unknown = set(), {}, {}, {}
    for unit in source_units(targets, build_config):
        view = TurnView(unit)
        source = unit['capture']['dataset']
        expected_family = 'anthropic' if source == 'ZClawBench' else 'hermes'
        require(unit['family'] == expected_family, 'boundary source adapter differs')
        locators = {view.locator(i): i for i in range(len(view.raw))}
        if source == 'ZClawBench':
            raw_map = [(i, message) for i, raw in enumerate(json_value(unit['record']['trajectory']))
                       for message in formats.anthropic_messages([raw])]
            require([m for _, m in raw_map] == view.raw, 'ZClaw original/normalized mapping differs')
        else:
            require(len(unit['record']['conversations']) == len(view.raw), 'Hermes message mapping differs')
        descriptors = [{k: r[k] for k in BASE_FIELDS} | {'source': r['source'], 'populations': []}
                       for r in unit['targets']]
        for row in classify_capture(unit['capture'], descriptors, unit['family'],
                                    unit['raw_records'], unit['prefix'], target_fields=BASE_FIELDS):
            sid, index = row['id'], row['turn_index']
            require(sid not in checked, f'duplicate boundary audit ID: {sid}')
            checked.add(sid)
            old = origins[sid]
            for field in ('source', 'user_boundary_source', 'call_count_status'):
                require(row[field] == old[field], f'{sid}: raw boundary evidence differs: {field}')
            require(row['category'] == old['next_event'] and row['previous_turn_tool_calls'] == old['prior_calls'],
                    f'{sid}: frozen successor/prior calls differ')
            boundary = locators[row['user_boundary_source']] if row['call_count_status'] == 'known' else None
            if source == 'ZClawBench':
                inspect = ([] if boundary is None else [('current_boundary', boundary)])
                inspect += [('successor', locators[location]) for location in row['next_sources']]
                evidence = []
                for relation, position in inspect:
                    message = view.raw[position]
                    match = zclaw_framework_message(message['content']) if message['role'] == 'user' else None
                    if match is not None:
                        evidence.append(dict(relation=relation, source=view.locator(position),
                            raw_source=unit['capture']['context']['source']+f'#trajectory[{raw_map[position][0]}]',
                            **match))
                if evidence:
                    boundary_matches[sid] = dict(source=old['source'], evidence=evidence,
                        otherwise_priority=reply_priority(old)['eligible'])
            elif boundary is not None:
                evidence, issues = [], []
                for position in range(boundary+1, index):
                    if view.raw[position]['role'] != 'tool':
                        continue
                    raw = unit['record']['conversations'][position]
                    require(raw['from'] == 'tool' and raw['value'] == view.raw[position]['content'],
                            'Hermes original tool content differs')
                    hits, unknown = hermes_budget_pressure(raw['value'])
                    location = dict(source=view.locator(position),
                        raw_source=unit['capture']['context']['source']+f'#conversations[{position}]')
                    evidence.extend(dict(location, **hit) for hit in hits)
                    issues.extend(dict(location, reason=reason) for reason in unknown)
                if evidence:
                    pressure[sid] = dict(source=old['source'], evidence=evidence)
                if issues:
                    diagnostic_unknown[sid] = issues
    require(checked == targets.keys(), 'affected-source boundary audit coverage differs')
    return dict(checked_ids=sorted(checked),
        by_source=dict(sorted(Counter(origins[sid]['dataset'] for sid in checked).items())),
        unverifiable_current_boundary_ids=sorted(sid for sid in checked if origins[sid]['call_count_status'] != 'known'),
        summary=dict(zclaw_matched_states=len(boundary_matches),
            zclaw_priority_removed=sum(row['otherwise_priority'] for row in boundary_matches.values()),
            zclaw_by_relation=dict(sorted(Counter(relation for row in boundary_matches.values()
                for relation in {e['relation'] for e in row['evidence']}).items())),
            zclaw_by_rule=dict(sorted(Counter(rule for row in boundary_matches.values()
                for rule in {e['rule'] for e in row['evidence']}).items())),
            hermes_pressure_states=len(pressure),
            hermes_by_rule=dict(sorted(Counter(rule for row in pressure.values()
                for rule in {e['rule'] for e in row['evidence']}).items())),
            hermes_diagnostic_unknown_states=len(diagnostic_unknown)),
        zclaw_framework_boundaries=dict(sorted(boundary_matches.items())),
        hermes_nonexhausted_pressure=dict(sorted(pressure.items())),
        hermes_pressure_diagnostic_unknown=dict(sorted(diagnostic_unknown.items())))


def preference_summary(ids, origins, framework_ids):
    replies = [reply_priority(origins[sid], framework_boundary=sid in framework_ids) for sid in ids if sid in origins]
    rules = Counter(row['rule'] for row in replies)
    return dict(priority_replies=sum(row['eligible'] for row in replies),
                priority_rules=dict(sorted(rules.items())))


def unit_positions(records):
    """V2 positions use retained teacher/trajectory units, not physical files."""
    units = defaultdict(list)
    for row in records.values():
        units[row['dataset'], row['model'], row['trajectory_id']].append(row)
    result = {}
    for group in units.values():
        for index, row in enumerate(sorted(group, key=lambda r: (r['turn_index'], r['id']))):
            result[row['id']] = (2*index+1)*10//(2*len(group))
    return result


def add_v2_diagnostics(selected, ids, origins, kept, audit):
    framework_ids = set(audit['zclaw_framework_boundaries'])
    selected.update(preference_summary(ids, origins, framework_ids))
    selected['zclaw_framework_boundary_ids'] = sorted(set(ids) & framework_ids)
    selected['hermes_nonexhausted_pressure_ids'] = sorted(set(ids) & audit['hermes_nonexhausted_pressure'].keys())
    units = Counter((kept[sid]['dataset'], kept[sid]['model'], kept[sid]['trajectory_id']) for sid in ids)
    selected.update(retained_units=len(units), maximum_states_per_retained_unit=max(units.values()),
                    states_per_retained_unit_histogram=dict(sorted(Counter(units.values()).items())))
    sources = Counter(kept[sid]['dataset'] for sid in ids)
    largest = sorted(sources, key=lambda s: (-sources[s], s))[:3]
    selected['top3_sources'] = {s: sources[s] for s in largest}
    selected['top3_source_fraction'] = sum(sources[s] for s in largest)/len(ids)
    for source, row in selected['by_source'].items():
        row.update(preference_summary([sid for sid in ids if kept[sid]['dataset'] == source], origins, framework_ids))


def build_preview(config, smoke=False, review_path=None):
    require(config['counts'] in ([8000, 38000], [8000, 38000, 76000])
            and config['seed'] == SEED, 'v2 preview protocol differs')
    kept, labels, reference_ids, r11_ids, regression, inputs, label_state = load_frozen(config)
    directory = (PROJECT/config['reply_origin_preview']).resolve()
    origins, old = load_origins(directory, kept, inputs)
    review = load_review(review_path, origins, r11_ids) if review_path is not None else None
    audit = restore_boundary_evidence(kept, origins, read_json(Path(old['inputs']['build_config'])))
    framework_ids = set(audit['zclaw_framework_boundaries'])
    if review is not None:
        for sid, case in review['cases'].items():
            if origins[sid]['dataset'] == 'ZClawBench':
                expected = case['framework_boundary_evidence'] or case['successor_framework_evidence']
                require((sid in framework_ids) == expected, f'{sid}: ZClaw rule disagrees with original-source review')
            if origins[sid]['dataset'] == 'hermes-agent-reasoning-traces':
                require((sid in audit['hermes_nonexhausted_pressure']) == case['framework_pressure_evidence'],
                        f'{sid}: Hermes pressure diagnostic differs from review')
    reference = Counter(labels[kept[sid]['task_group']] for sid in reference_ids)
    cells = {d: {s: c['reference_count'] for s, c in sources.items()}
             for d, sources in regression['cells'].items()}
    require(sum(sum(c.values()) for c in cells.values()) == 8000, 'R00 reference total differs')
    population = bounded_population(kept, labels) if smoke else kept
    if smoke and review is not None:
        example_ids = [sid for sid, case in review['cases'].items()
                       if sid in framework_ids or (origins[sid]['dataset'] == 'hermes-agent-reasoning-traces'
                                                   and case['explicit_final_now_pressure'])]
        units = {(kept[sid]['dataset'], kept[sid]['trajectory_id']) for sid in example_ids}
        population.update({sid: r for sid, r in kept.items() if (r['dataset'], r['trajectory_id']) in units})
    active_origins = {sid: row for sid, row in origins.items() if sid in population}
    positions = unit_positions(population)
    baseline = summarize_selection(r11_ids, kept, origins, labels, unit_positions(kept), r11_ids)
    add_v2_diagnostics(baseline, r11_ids, origins, kept, audit)
    result = dict(recipe='task_reply_v2', seed=SEED, smoke=smoke,
        inputs=dict(inputs, reply_origin_preview=str(directory)), ordered_R11_8k_regression=True,
        label_acceptance=label_acceptance(label_state, Path(inputs['labels'])),
        reference_domain_counts=dict(reference), reference_cells=cells,
        candidate_population=describe(list(population.values()), active_origins),
        supported_candidates=sum(cells[labels[r['task_group']]].get(r['dataset'], 0) > 0 for r in population.values()),
        reply_priority={sid: dict(origin_category=row['category'], **reply_priority(row, framework_boundary=sid in framework_ids))
                        for sid, row in active_origins.items()}, reply_boundary_review=review, boundary_audit=audit,
        targets={}, comparisons=dict(R11_8k=baseline), overlaps_between_sizes={},
        definitions=dict(domain='0.5*R00_domain_fraction+0.5/7; cap supported capacity; fixed-weight redistribution.',
            source='Within each domain, sqrt(frozen R00 source/domain STATE counts); zero-reference cells excluded.',
            task='Digest-ordered source round robin to fixed quotas; each source selects its least-used available canonical task; counters shared across sources.',
            unit='Within source/task, round robin retained (model, trajectory_id) units; dedup leaves one trajectory per teacher.',
            R='C1 RESPOND with verifiable prior boundary and successor, no terminal conflict, explicit framework-forced finalization or matched ZClaw completion notification boundary. Unknown origin alone does not exclude R.',
            boundary_rules='ZClaw: complete background completion or internal runtime completion templates at current/next boundary -> O. Hermes: non-exhausted budget pressure is diagnostic only. No sidechain/subagent-context-only exclusion.',
            state='P(R)=2r/(2r+o) among remaining states of a trajectory; independent spread phases, no replacement.',
            loss='Selection preference only. Original loss unchanged; no C3, C4, short-turn quota, copying or RESPOND target quota.',
            gate='RESPOND 8%-20% is a review check, not an optimizer target. Failure stops without adjustment.',
            limitations='Boundary validity does not prove natural input or task completion. Missing/null terminal metadata follows established semantics. Frozen Codex candidate limitations remain.',
            relative_position='(ordinal+.5)/retained eligible (dataset,model,trajectory_id) unit states; not full raw trajectory length. Historical v1 comparisons keep their physical-trajectory bins.',
            trajectories='trajectories is the physical (dataset,trajectory_id) count for historical comparison; retained_units keeps distinct model identities.',
            schedule='Predicted original single-GPU accum=2, 32768 tokens, 1 epoch; not a measured training run.'))
    selections = {}
    for count in ([200] if smoke else config['counts']):
        ids, allocation = select_with_sources(population, labels, cells, active_origins, count, SEED,
                                             framework_boundary_ids=framework_ids & active_origins.keys())
        selected = summarize_selection(ids, population, active_origins, labels, positions, r11_ids)
        add_v2_diagnostics(selected, ids, active_origins, population, audit)
        if not smoke:
            old_ids, _ = select(kept, labels, reference, {sid: o['category'] for sid, o in origins.items()}, count, SEED)
            require(selection_hash(old_ids) == old['targets'][str(count)]['selection_sha256'],
                    f'v1 ordered ID regression failed at {count}')
            selected['overlap_with_v1_same_size'] = len(set(ids) & set(old_ids))
            result['comparisons'][f'v1_{count}'] = old['targets'][str(count)]['selected']
        passed = selected['respond_gate_passed']
        result['targets'][str(count)] = dict(status='smoke_completed' if smoke else 'completed' if passed else 'failed',
            failure_stage=None if smoke or passed else 'RESPOND_gate', selection_sha256=selection_hash(ids),
            allocation=allocation, selected=selected)
        for previous, previous_ids in selections.items():
            result['overlaps_between_sizes'][f'{previous}:{count}'] = dict(overlap=len(set(ids) & set(previous_ids)),
                smaller_is_subset=set(previous_ids) <= set(ids), note='Observed only; no general nesting guarantee.')
        selections[count] = ids
        print(f"selected {count}: RESPOND={selected['respond']}, priority={selected['priority_replies']}, gate={passed}", flush=True)
    return result


def report_markdown(result):
    lines = ['# 来源配额与回复优先 v2：CPU预览', '',
        '冻结旧C1/去重/领域标签；不写训练样本或ID文件，不训练或评测模型。来源未知与轮界未知分别记录。', '',
        '| 方案 | 状态 | RESPOND | 回复优先R | 严格短停 | 任务 | 前三来源占比 | 预计更新 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    comparisons = [('R11-8k', result['comparisons']['R11_8k'])]
    comparisons += [(f'v2-{n}', a['selected']) for n, a in result['targets'].items()]
    for name, s in comparisons:
        lines.append(f"| {name} | {s['count']} | {s['respond']} ({s['respond_fraction']:.2%}) | {s['priority_replies']} | {s['strict_short_stop']['known_positive']} | {s['task_groups']} | {s['top3_source_fraction']:.2%} | {s['schedule']['updates']} |")
    for count, arm in result['targets'].items():
        lines += ['', f'## {count} 的配额与实取', '', '| 领域 | 来源 | R00状态数 | 可用状态 | 配额 |', '|---|---|---:|---:|---:|']
        for domain, plan in arm['allocation'].items():
            for source, cell in plan['sources'].items():
                lines.append(f"| {domain} | {source} | {cell['reference_count']} | {cell['capacity']} | {cell['quota']} |")
        lines += ['', '| 来源 | 实取 | RESPOND | R |', '|---|---:|---:|---:|']
        for source, s in arm['selected']['by_source'].items():
            lines.append(f"| {source} | {s['count']} | {s['respond']} | {s['priority_replies']} |")
        s = arm['selected']
        lines += ['', f"门槛：{arm['status']}；严格短停已知/未知 {s['strict_short_stop']['known_positive']}/{s['strict_short_stop']['unknown']}。",
                  f"位置十箱：{s['relative_position_bins']}（保留状态分母）；原始消息index=0：{s['source_message_index_zero']}。",
                  f"任务/保留教师轨迹单元最大状态数：{s['maximum_states_per_task']}/{s['maximum_states_per_retained_unit']}；保留单元 {s['retained_units']}；token {s['n_prompt_tokens']:,}。",
                  f"来源类别：{s['reply_classes']}；回复优先规则：{s['priority_rules']}。",
                  f"选入ZClaw框架边界回复 {len(s['zclaw_framework_boundary_ids'])} 条，作为O；Hermes非耗尽催促 {len(s['hermes_nonexhausted_pressure_ids'])} 条，仅作诊断。"]
    if result['reply_boundary_review'] is not None:
        lines += ['', f"有界原文审查：{len(result['reply_boundary_review']['cases'])}个去重状态；逐ID定位及摘录见summary的reply_boundary_review。"]
    lines += ['', f"受影响来源的原始证据重核：{result['boundary_audit']['by_source']}；汇总：{result['boundary_audit']['summary']}。",
              'ZClaw只匹配完整后台完成/内部完成通知，不扩到单独子代理上下文或sidechain标志；不修改C1和旧来源类别。',
              'Hermes非耗尽的final NOW与consolidation催促分别计数，均不改变R资格；原预算耗尽双证据规则保持。']
    lines += ['', '来源配额优先，任务计数跨来源共享；不保证任务全局等额。教师与保留轨迹合并为一个轮转层。',
              '未知来源仍为unknown；轮界可核实不等于自然请求或任务完成。有界审查不是全量来源正确性证明。',
              '缺失/null终止字段不推断成功，D2仍只认既定error/length。未对aborted/max_tokens另立过滤。',
              '取消C3与短轮配额，不复制状态，损失不加权。RESPOND 8%–20%仅审阅门槛，不据结果调参。',
              '旧R11和v1同规模ID回归、各类统计/来源/教师/长度/预期损失份额、规模交集见summary。',
              '结果仅说明组成变化，不能推断在线收益。下一步等待用户审阅，不自动生成样本或开始训练。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/selection/task_reply_v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, help='JSON containing reply_boundary_review; an existing v2 summary can be reused')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke and args.review is None:
        parser.error('formal preview requires the bounded original-source review')
    out = args.output.resolve()
    if args.smoke and not out.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under project tmp')
    out.mkdir(parents=True, exist_ok=False)
    (out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable, *sys.argv])}\n')
    state = dict(status='running', started_at=timestamp(), smoke=args.smoke)
    write_json(out/'metadata/run_status.json', state)
    try:
        result = build_preview(read_json(args.config), args.smoke, args.review)
        result['config'] = str(args.config.resolve())
        write_json(out/'summary.json', result)
        (out/'report.md').write_text(report_markdown(result))
        failed = [n for n, a in result['targets'].items() if a['status'] == 'failed']
        state.update(status='failed' if failed else 'smoke_completed' if args.smoke else 'completed',
                     ended_at=timestamp(), failed_targets=failed, reply_states=len(result['reply_priority']),
                     ordered_R11_8k_regression=True, ordered_v1_regression=not args.smoke)
        write_json(out/'metadata/run_status.json', state)
    except BaseException as exc:
        state.update(status='failed', ended_at=timestamp(), error=str(exc))
        write_json(out/'metadata/run_status.json', state)
        raise
    print(json.dumps(dict(output=str(out), status=state['status']), ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
