#!/usr/bin/env python3
"""Stage 1: recover canonical inputs, or reuse them for the approved reallocated preview."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(REPO/'third_party/decider'), str(PROJECT.parent/'dataset_tool_inventory/src')]
from jev_train.common import digest, rows, write_json
from jev_train.data import SampleIndex, load_selected, tool_name, validate
from jev_train.curation import c1_plan
from jev_train.dedup import preview, redistributed_preview
from jev_train.domains import clean_request, annotation_requests
from jev_train.build import canonical_g3, setup_entry, swe_repos, codex_call_types, light, initial_request
from dataset_tool_inventory.adapters.review_formats import scan
from dataset_tool_inventory.review_policy import eligibility
from dataset_tool_inventory.schema import DecisionEvent

SEED = 20261015


def timestamp():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_candidates(args):
    status = json.loads((args.features/'metadata/run_status.json').read_text())
    require(status['status']=='completed', 'state_features not completed')
    features = {}
    for row in rows(args.features/'state_features.jsonl'):
        require(row['id'] not in features, 'duplicate feature ID')
        features[row['id']] = row
    index = SampleIndex(args.samples/'train.jsonl')
    quotas = dict(Counter(e['dataset'] for e in load_selected(index, 8000, SEED)))
    reference = json.loads((args.reference/'metadata/curation.json').read_text())
    require(reference['rule_version']=='B2-C1-C2-C3-no-C4', 'reference is not approved B2')
    require(quotas==reference['source_quotas'], 'reference source quotas changed')
    for key in ('samples','pool','features','build_config'):
        require(Path(reference['inputs'][key]).resolve()==getattr(args,key).resolve(), f'reference input differs: {key}')
    reference_ids = (args.reference/'train_ids.txt').read_text().splitlines()
    require(len(reference_ids)==len(set(reference_ids))==8000, 'invalid B2 reference IDs')
    plans, counts = {}, {}
    for split in ('train','val','test_id'):
        path = (args.samples if split=='train' else args.pool)/f'{split}.jsonl'
        seen, drops = set(), Counter()
        for raw in rows(path):
            sid = raw['id']
            require(sid not in seen and sid not in plans, f'duplicate input ID: {sid}')
            seen.add(sid)
            plan = c1_plan(raw, features[sid])
            require(plan['split']==split, 'feature split mismatch')
            if plan['drops']:
                drops.update(plan['drops'])
                continue
            plan.update(model=raw['model'], task_group=raw['task_group'], n_prompt_tokens=raw['n_prompt_tokens'],
                        original_options=[tool_name(v) for v in raw['options']], candidate_level=raw['candidate_level'])
            for key in ('trajectory_id','turn_index'):
                if key in raw:
                    plan[key] = raw[key]
            plans[sid] = plan
        require(seen=={sid for sid,f in features.items() if f['split']==split}, f'{split}: feature ID coverage mismatch')
        counts[split] = dict(input=len(seen), c1_kept=sum(r['split']==split for r in plans.values()), drop_reasons=dict(drops))
        if split != 'train':
            normalized = set()
            for raw in rows(args.normalized/f'{split}.jsonl'):
                validate(raw)
                sid = raw['id']
                require(sid not in normalized and sid in plans, f'{split}: normalized ID mismatch')
                normalized.add(sid)
                p = plans[sid]
                for key in ('dataset','model','task_group','n_prompt_tokens','trajectory_id','turn_index','gold','short_turn','direct'):
                    require(raw[key]==p[key], f'{sid}: normalized {key} differs')
            require(normalized=={sid for sid,p in plans.items() if p['split']==split}, f'{split}: normalized coverage mismatch')
        print(f'planned {split}: {counts[split]["c1_kept"]}', flush=True)
    return plans, quotas, reference_ids, counts


def recover_source(job):
    entry, config, wanted, directory, smoke = job
    dataset = entry['name']
    run = REPO/config['review_run']
    root = REPO/config['data_root']/entry['path']
    groups = json.loads((REPO/config['task_groups']).read_text())
    repo_map = swe_repos(root, entry) if dataset==groups['swe_repo_dataset'] else {}
    duplicates = json.loads((run/'duplicates.json').read_text())
    duplicate_ids = {t for g in duplicates['exact'] for d,t in g['duplicates'] if d==dataset}
    policy = json.loads((REPO/config['review_policy']).read_text())
    entry = setup_entry(entry, run)
    recovered, requests = {}, {}
    for event in scan(dataset, root, entry):
        if not isinstance(event, DecisionEvent):
            continue
        # Forward construction from canonical fields, never inverse inference from a saved ID.
        sid = digest(dataset, event.trajectory_id, event.turn_index)
        if sid not in wanted:
            continue
        require(sid not in recovered, f'duplicate source event: {sid}')
        require(event.trajectory_id not in duplicate_ids, f'{sid}: source now marked exact duplicate')
        require(eligibility(event,policy)['eligibility'] in ('confirmed','conditional'), f'{sid}: source no longer eligible')
        calls = codex_call_types(event.source.rsplit(':',1)[0]) if entry['review_adapter']=='native_codex' else None
        record = light(event, groups, repo_map, calls)
        for key in ('trajectory_id','model'):
            require(isinstance(record[key],str) and bool(record[key]), f'{sid}: missing/invalid canonical {key}')
        plan = wanted[sid]
        for key in ('id','dataset','model','outcome','candidate_level'):
            require(record[key]==plan[key], f'{sid}: canonical {key} differs')
        for key in ('trajectory_id','turn_index'):
            if key in plan:
                require(record[key]==plan[key], f'{sid}: saved pool {key} differs')
        require(record['candidate_names']+['RESPOND']==plan['original_options'], f'{sid}: source options differ')
        gold = ['RESPOND'] if record['action_kind']=='RESPOND' else record['action_names']
        require(set(gold)==set(plan['original_gold']), f'{sid}: original source gold differs')
        text, _ = initial_request(event.metadata['_history'])
        require(hashlib.sha256(text.encode()).hexdigest()==record['request_hash'], f'{sid}: request_hash mismatch')
        req = dict(dataset=dataset, trajectory_id=record['trajectory_id'], task_group=plan['task_group'],
                   split=plan['split'], request_hash=record['request_hash'], request=clean_request(text,record['request_hash']))
        key = record['trajectory_id']
        if key in requests:
            require(requests[key]==req, f'{dataset}/{key}: request/split/task_group changed within trajectory')
        requests[key] = req
        recovered[sid] = dict(plan, trajectory_id=record['trajectory_id'], turn_index=record['turn_index'],
                              request_hash=record['request_hash'], source=record['source'])
        if smoke and len(recovered)>=2:
            break
        if len(recovered)==len(wanted):
            break
    if not smoke:
        require(recovered.keys()==wanted.keys(), f'{dataset}: canonical recovery missing {len(wanted.keys()-recovered.keys())} IDs: {sorted(wanted.keys()-recovered.keys())[:5]}')
    else:
        require(bool(recovered), f'{dataset}: smoke did not recover a real state')
    with (directory/(dataset+'.states.jsonl')).open('w') as handle:
        for sid in sorted(recovered):
            handle.write(json.dumps(recovered[sid], ensure_ascii=False)+'\n')
    with (directory/(dataset+'.requests.jsonl')).open('w') as handle:
        for key in sorted(requests):
            handle.write(json.dumps(requests[key], ensure_ascii=False)+'\n')
    return dict(dataset=dataset, states=len(recovered), trajectories=len(requests))


def run(args):
    plans, quotas, reference_ids, counts = load_candidates(args)
    config = json.loads(args.build_config.read_text())
    canonical_g3(config)
    registry = {e['name']:e for e in json.loads((REPO/config['inventory_config']).read_text())['datasets']}
    per_source = defaultdict(dict)
    for sid, row in plans.items():
        per_source[row['dataset']][sid] = row
    work = Path(tempfile.mkdtemp(prefix='dedup_recovery_', dir=PROJECT/'tmp'))
    args.work = work
    print(f'canonical recovery work: {work}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = [executor.submit(recover_source, (registry[d],config,p,work,args.smoke)) for d,p in sorted(per_source.items())]
        for future in as_completed(jobs):
            print(json.dumps(future.result()), flush=True)
    restored, requests = {}, {}
    for dataset in sorted(per_source):
        for row in rows(work/(dataset+'.states.jsonl')):
            require(row['id'] not in restored, 'duplicate recovered state')
            restored[row['id']] = row
        for row in rows(work/(dataset+'.requests.jsonl')):
            key = row['dataset'],row['trajectory_id']
            require(key not in requests, 'duplicate recovered trajectory request')
            requests[key] = row
    train = {sid:r for sid,r in restored.items() if r['split']=='train'}
    if args.smoke:
        per = defaultdict(list)
        for r in train.values():
            per[r['dataset']].append(r)
        quotas = {d:len(v) for d,v in per.items()}
    _, ids, report = preview(train, quotas, SEED, reference_ids)
    label_inputs = annotation_requests(restored.values(), requests, SEED)
    report.update(seed=SEED, smoke=args.smoke, c1=counts,
        inputs={k:str(getattr(args,k).resolve()) for k in ('samples','pool','features','build_config','normalized','reference')},
        identity_recovery=dict(states=len(restored), trajectories=len(requests),
            definition='Forward source adapter -> build.light; group IDs reused from canonical samples. Historical hashes were not persisted; hashes are reconstructed, not independently compared to a saved historical hash.'),
        recovery_inputs=dict(canonical_states=str(args.output/'metadata/canonical_states.jsonl'),
                             annotation_requests=str(args.output/'metadata/annotation_requests.jsonl')),
        domain_inputs={s:dict(task_groups=sum(r['split']==s for r in label_inputs),
            fixed=sum(r['split']==s and r['fixed_label'] is not None for r in label_inputs),
            api=sum(r['split']==s and r['fixed_label'] is None for r in label_inputs)) for s in ('train','val','test_id')},
        definitions=dict(before='C1 surviving capped candidates, before trajectory deduplication.',
            dedup='One minimum digest(seed,trajectory_id) per (dataset,task_group,model); retain all its C1-surviving capped states.',
            priority='Unchanged B2: at most round(.25*k) priority; fill only same-source ordinary; C4 off.',
            scope='No canonical samples or train_ids written. Domain members are those present in the current C1 candidate/normalized input scope.',
            review='Stop for stage-1 review. Stage-2 labels require separate 100-case review; no stage-3 preview or training yet.'))
    write_json(args.output/'summary.json', report)
    with (args.output/'metadata/canonical_states.jsonl').open('w') as handle:
        for sid in sorted(restored):
            row = dict(restored[sid]); del row['original_options']
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (args.output/'metadata/annotation_requests.jsonl').open('w') as handle:
        for row in label_inputs:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    shutil.rmtree(work)
    if not args.smoke:
        require(not report['shortfalls'], f'fixed source quotas infeasible: {report["shortfalls"]}; stop, no redistribution')
        require(report['respond_gate_passed'], 'RESPOND outside [0.08,0.20]; stop, no samples written')
    return report


def run_recovered(args):
    previous = json.loads((args.recovery/'summary.json').read_text())
    require(not previous['smoke'] and previous['seed']==SEED, 'requires full recovery with the fixed seed')
    # The previous preview stopped at capacity checking, after all canonical inputs were saved.
    paths = {name:(args.recovery/f'metadata/{name}.jsonl').resolve()
             for name in ('canonical_states','annotation_requests')}
    train, seen, split_counts = {}, set(), Counter()
    for row in rows(paths['canonical_states']):
        sid = row['id']
        require(sid not in seen, f'duplicate recovered ID: {sid}')
        seen.add(sid);split_counts[row['split']]+=1
        require(not row['drops'], f'recovery contains a C1-discarded row: {sid}')
        if row['split']=='train':
            train[sid] = row
    require(len(seen)==previous['identity_recovery']['states'], 'recovery state coverage differs')
    require(dict(split_counts)=={s:r['c1_kept'] for s,r in previous['c1'].items()}, 'recovery split coverage differs')
    reference_path = Path(previous['inputs']['reference'])
    reference = json.loads((reference_path/'metadata/curation.json').read_text())
    require(reference['rule_version']=='B2-C1-C2-C3-no-C4', 'reference is not approved B2')
    original_quotas = reference['source_quotas']
    require(original_quotas=={d:r['original_quota'] for d,r in previous['sources'].items()}, 'original quotas differ')
    reference_ids = (reference_path/'train_ids.txt').read_text().splitlines()
    require(len(reference_ids)==len(set(reference_ids))==sum(original_quotas.values())==8000, 'invalid reference size/IDs')
    _, selected, report = redistributed_preview(train, original_quotas, SEED, reference_ids)
    # Compare via JSON because histogram keys become strings when saved.
    serialized = json.loads(json.dumps(report))
    for key in ('before','after'):
        require(serialized[key]==previous[key], f'reconstructed dedup population changed: {key}')
    label_counts = {s:dict(task_groups=0,fixed=0,api=0) for s in ('train','val','test_id')}
    label_groups = set()
    for request in rows(paths['annotation_requests']):
        require(request['task_group'] not in label_groups, 'duplicate annotation group')
        label_groups.add(request['task_group'])
        item = label_counts[request['split']];item['task_groups']+=1
        item['fixed' if request['fixed_label'] is not None else 'api']+=1
    require(label_counts==previous['domain_inputs'], 'annotation input coverage differs')
    reference_rows = [train[sid] for sid in reference_ids]
    report['reference'].update(respond=sum(r['gold_names']==['RESPOND'] for r in reference_rows),
        priority=sum(r['short_turn'] or r['direct'] for r in reference_rows),
        n_prompt_tokens=sum(r['n_prompt_tokens'] for r in reference_rows))
    report.update(seed=SEED,smoke=args.smoke,c1=previous['c1'],inputs=dict(previous['inputs'],recovery=str(args.recovery.resolve())),
        identity_recovery=previous['identity_recovery'],recovery_inputs={k:str(v) for k,v in paths.items()},
        domain_inputs=label_counts,definitions=dict(previous['definitions'],
            redistribution='0813 capped at its B2 capacity; release deficit once to other sources in original quota proportions, integer largest remainder, dataset-name ties; fail if any new quota exceeds capacity.',
            priority='round(.25*new_source_quota), ordinary-only backfill within each source; C4 off.',
            review='Stop after this preview. No API labels until explicit preview approval.'))
    require(len(selected)==8000 and len(set(selected))==8000, 'selection total/uniqueness failed')
    write_json(args.output/'summary.json',report)
    write_report(args.output/'report.md',report)
    require(report['respond_gate_passed'], 'RESPOND outside [0.08,0.20]; stop, no samples written')
    return report


def write_report(path, report):
    chosen, old = report['selected'], report['reference']
    lines = ['# B2-dedup-8k：一次重分配后的 CPU 预览','',
        '仅预览，等待用户审阅；未写训练样本/ID，未调用 API、启动训练或评测。','',
        f"- 总量 {chosen['count']}；独立任务组 {old['task_groups']} → {chosen['task_groups']}；与 B2-8k 重合 {chosen['overlap_with_reference']} 个状态。",
        f"- RESPOND {chosen['respond']}（{chosen['respond_fraction']:.4%}）；优先类 {chosen['priority']}（{chosen['priority_fraction']:.4%}）；prompt token {chosen['n_prompt_tokens']:,}。",
        '- 0813保留实际容量；缺口只按其余来源原配额用最大余数法分一次，余数并列按来源名。优先上限按新k重算；重分配后容量不足即报错。','',
        '| 来源 | 原k | 新k | 增减 | 任务组：B2→新组 | 重合状态 | 优先上限/实取 | RESPOND | B2容量 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,r in report['sources'].items():
        s = r['selected']
        lines.append(f"| {dataset} | {r['original_quota']} | {r['final_quota']} | {r['final_quota']-r['original_quota']:+d} | "
            f"{r['reference']['task_groups']} → {s['task_groups']} | {s['overlap_with_reference']} | {r['priority_quota']} / {s['priority']} | {s['respond']} | {r['b2_capacity']} |")
    lines += ['', '独立任务组总数对task_group做全局去重，不能把逐来源的数相加当总数。去重前后每任务状态数直方图、来源×任务/教师数、RESPOND next_event和缺口流向详见summary.json。','',
        '恢复清单未重建、未移动或复制，仍在旧预览的metadata/下，属于下游标注/选样输入；新summary.recovery_inputs记录准确路径，新metadata只包含command.txt和run_status.json。','',
        '请求hash只能验证现有构造口径重建的自洽性，历史hash未保存，不能声称独立历史比对。','',
        '解释边界：同一教师对同一任务的轨迹可能包含有用的探索差异；按键去重不等于内容近似重复。如果模型变差，应区分轨迹多样性损失与其他因素。此外，0813占比下降是本组实际变化之一，结果不能单独归因于去重。','',
        '领域均衡组的领域内、跨领域缺口规则保持原计划；本轮未计算该组配额。API待本预览审阅通过才启动。']
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','pool','features','build-config','normalized','reference'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recovery',type=Path,help='Reuse the first completed canonical recovery; apply approved one-pass reallocation.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    inputs = [getattr(args,k) for k in ('samples','pool','features','build_config','normalized','reference')]
    if args.recovery:
        if any(v is not None for v in inputs):
            parser.error('--recovery takes its original inputs from summary; do not also supply raw inputs')
    elif any(v is None for v in inputs):
        parser.error('provide --recovery or all six raw input paths')
    if args.workers<1:
        parser.error('workers must be positive')
    args.output = args.output.resolve()
    if args.smoke and not args.output.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under jev_train/tmp')
    (PROJECT/'tmp').mkdir(exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'metadata').mkdir()
    (args.output/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status = dict(status='running', started_at=timestamp(), smoke=args.smoke, stage=1)
    write_json(args.output/'metadata/run_status.json', status)
    try:
        result = run_recovered(args) if args.recovery else run(args)
    except BaseException as exc:
        status.update(status='failed', ended_at=timestamp(), error=str(exc))
        if hasattr(args,'work') and args.work.exists():
            status['diagnostics'] = str(args.work)
        write_json(args.output/'metadata/run_status.json', status)
        raise
    status.update(status='smoke_completed' if args.smoke else 'completed', ended_at=timestamp(), awaiting_review=not args.smoke)
    write_json(args.output/'metadata/run_status.json', status)
    print(json.dumps(dict(output=str(args.output), selected=result['selected'], shortfalls=result['shortfalls'], domain_inputs=result['domain_inputs']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
