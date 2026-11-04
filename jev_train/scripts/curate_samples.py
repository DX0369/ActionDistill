#!/usr/bin/env python
"""C1-normalized B1/evaluation files and B2 preview only; CPU, immutable inputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys

from analyze_data_curation import (PROJECT, WORKSPACE, read_json, write_json, require,
    SampleIndex, load_selected, sample_metadata, source_units, locate_missing, history_loop,
    codex_call_types, digest, DecisionEvent, emit_trajectory)
from jev_train.build import history_features
from jev_train.curation import c1_plan, c4_drop, materialize, select_versions
from jev_train.data import make_item, validate

SEED = 20261015
SPLITS = ('train', 'val', 'test_id', 'test_ood')
DEFINITIONS = dict(
    C1='D1改为源后继工具集合；缺候选、narration_successor、D2、unjudgeable丢弃。原因可重叠，唯一丢弃取并集。',
    C2='整轮调用数1–3且C1后轮尾为RESPOND，整轮内状态均为short_turn；null不属于short_turn。',
    C3='整轮0调用且C1后gold为RESPOND，标记direct。',
    C4='仅B2训练：C1保留且strict_loop=true，修正后gold恰为循环工具一个名字才丢弃；无报错豁免。',
    selection='B0逐来源配额固定；B1按digest(seed,id)前k；B2优先至多round(.25k)，余下仅从非short_turn/direct补齐。不得跨来源补足或放宽。',
    rounding='Python round，.5 ties-to-even；不是始终向上舍入。',
    unknown='整轮调用数null正常进入普通池，strict_loop=null不命中C4；按split/source/ID记录。',
    inputs='context/question/options/token数/截断字段保持原值；仅gold、依赖gold的已有元数据及short_turn/direct标记变化。',
    gate='本入口只写B1和C1规范化评测集；B2只在内存选样并报告预览，不写B2样本或ID文件。',
)


def load(args):
    status = read_json(args.features/'metadata/run_status.json')
    approved = read_json(args.features/'summary.json')
    require(status['status'] == 'completed' and not approved['validation']['smoke'], 'requires completed full Stage 1')
    require(approved['validation']['all_reference_fields_equal'] and approved['validation']['reference_states_compared'] == 21231,
            'approved Stage 1 regression missing')
    features = {}
    with (args.features/'state_features.jsonl').open() as handle:
        for line in handle:
            f = json.loads(line)
            require(f['id'] not in features, 'duplicate feature ID')
            features[f['id']] = f
    indices = {s:SampleIndex((args.samples if s == 'train' else args.pool)/f'{s}.jsonl') for s in SPLITS}
    for split, index in indices.items():
        require({e['id'] for e in index.entries} == {sid for sid,f in features.items() if f['split'] == split},
                f'feature/sample ID coverage differs: {split}')
    b0 = load_selected(indices['train'], 8000, SEED)
    quotas = dict(Counter(e['dataset'] for e in b0))
    flags = Counter()
    for entry in b0:
        f = features[entry['id']]
        d1 = f['next_event'] == 'intermediate_tool_continuation'
        d2 = f['stop_reason_abnormal'] is True
        flags.update(respond=int(f['gold_kind']=='RESPOND'),d1=int(d1),d2=int(d2),both=int(d1 and d2))
    require(dict(flags) == dict(respond=1039,d1=469,d2=37,both=3), 'B0 regression differs')
    require(len(indices['train'].entries)==261663, 'full train count differs')
    require(sum(f['split']=='train' and f['next_event']=='intermediate_tool_continuation' for f in features.values())==15834,
            'recovered train D1 count differs')
    selected = set(features)
    if args.smoke:
        slots = defaultdict(list)
        for sid,f in features.items():
            key = (f['split'],f['dataset'],f['gold_kind'],f['next_event'],f['stop_reason_abnormal'],
                   f['narration_successor'],f['strict_loop'],f['turn_total_calls'] in (0,1,2,3))
            slots[key].append(sid)
        selected = {sid for ids in slots.values() for sid in sorted(ids,key=lambda sid:digest(SEED,sid))[:8]}
    plans, offsets, targets = {}, {}, {}
    for split,index in indices.items():
        with index.path.open('rb') as handle:
            for entry in index.entries:
                sid = entry['id']
                if sid not in selected:
                    continue
                handle.seek(entry['offset'])
                raw = json.loads(handle.readline())
                plan = c1_plan(raw,features[sid])
                require(raw['id']==sid, 'index read returned wrong ID')
                plans[sid],offsets[sid] = plan,entry['offset']
                if (split == 'train' and plan['strict_loop'] is True) or (not plan['drops'] and plan['d1'] and 'loop_state' in raw):
                    targets[sid] = sample_metadata(raw)
                    for key in ('prev_action_names','transition','loop_state'):
                        if key in raw:
                            targets[sid][key] = raw[key]
        print(f'planned {split}: {sum(p["split"]==split for p in plans.values())}',flush=True)
    if args.smoke:
        counts = Counter(p['dataset'] for p in plans.values() if p['split']=='train' and not p['drops'])
        quotas = {ds:min(4,n) for ds,n in counts.items()}
    else:
        require(sum('C1_D1_missing_candidate' in p['drops'] for p in plans.values() if p['split']=='train')==483,
                'expected 483 train D1 missing-candidate discards')
    return features,indices,plans,offsets,targets,quotas,dict(flags),approved['validation']


def recover(targets, plans, config):
    """Restore only C4 identities and the existing gold-dependent heldout metadata."""
    locate_missing(targets,config)
    restored, loops, seen = {}, {}, set()
    for unit in source_units(targets,config):
        cap = unit['capture']
        wanted = {r['id']:r for r in unit['targets']}
        call_types = codex_call_types(Path(cap['context']['source'])) if unit['family']=='native_codex' else None
        for event in emit_trajectory(cap['dataset'],cap['trajectory_id'],cap['messages'],cap['tools'],cap['context']):
            if not isinstance(event,DecisionEvent):
                continue
            sid = digest(event.dataset,event.trajectory_id,event.turn_index)
            if sid not in wanted:
                continue
            require(sid not in seen, f'duplicate source state: {sid}')
            seen.add(sid)
            target,plan,meta = wanted[sid],plans[sid],event.metadata
            gold = ['RESPOND'] if event.action_kind=='RESPOND' else list(event.action_names)
            require(target['names']==list(event.candidate_tools)+['RESPOND'] and set(target['gold_names'])==set(gold),
                    f'original source gold/options differ: {sid}')
            require(target['model']==event.model and all(target[k]==meta[k] for k in ('outcome','candidate_level')),
                    f'original source metadata differ: {sid}')
            history = meta['_history']
            if 'loop_state' in target:
                original = history_features(history,gold,call_types)
                require(all(original[k]==target[k] for k in ('prev_action_names','transition','loop_state')), f'old features differ: {sid}')
                restored[sid] = history_features(history,plan['gold_names'],call_types)
            if plan['split']=='train' and plan['strict_loop'] is True:
                loop = history_loop(history,call_types)
                require(loop['loop_history'] is True, f'strict loop no longer reproducible: {sid}')
                # Retain bounded excerpts of source messages for the requested manual review.
                calls = []
                for i,message in enumerate(history):
                    if message['role']=='user':
                        calls.clear()
                    elif message['role']=='assistant':
                        calls.extend((i,c) for c in message.get('tool_calls',[]))
                evidence = []
                for i,call in calls[-3:]:
                    following = []
                    for message in history[i+1:]:
                        if message['role'] in ('assistant','user'):
                            break
                        if message['role']=='tool' and message.get('tool_call_id')==call.get('id'):
                            following.append(str(message.get('content'))[:1000])
                    evidence.append(dict(call=call,results=following))
                loops[sid] = dict(tool=loop['loop_tool'],source=event.source,trajectory_id=event.trajectory_id,
                                  turn_index=event.turn_index,last_three=evidence)
        if len(seen)%250==0:
            print(f'source states restored: {len(seen)}/{len(targets)}',flush=True)
    require(seen==targets.keys(), 'source recovery coverage differs')
    for sid,plan in plans.items():
        plan['c4_drop'] = c4_drop(plan,loops[sid]['tool'] if sid in loops else None) if plan['split']=='train' else False
    return restored,loops


def summarize(plans, ids):
    rows = [plans[sid] for sid in ids]
    return dict(count=len(rows),respond=sum(r['gold_names']==['RESPOND'] for r in rows),
                respond_fraction=sum(r['gold_names']==['RESPOND'] for r in rows)/len(rows) if rows else None,
                short_turn=sum(r['short_turn'] for r in rows),direct=sum(r['direct'] for r in rows),
                turn_calls_unknown=sum(r['turn_calls_unknown'] for r in rows),
                respond_next_event=dict(Counter(r['next_event'] for r in rows if r['gold_names']==['RESPOND'])),
                by_dataset={d:dict(count=sum(r['dataset']==d for r in rows),
                                  respond=sum(r['dataset']==d and r['gold_names']==['RESPOND'] for r in rows),
                                  respond_next_event=dict(Counter(r['next_event'] for r in rows if r['dataset']==d and r['gold_names']==['RESPOND'])))
                            for d in sorted({r['dataset'] for r in rows})})


def ledger(plans):
    result = {split:{} for split in SPLITS}
    for sid,p in plans.items():
        bucket = result[p['split']].setdefault(p['dataset'],{})
        reasons = list(p['drops'])
        if p['drops']:
            reasons.append('C1_unique_discard')
        if p['d1']:
            reasons.append('C1_D1_before_discard')
            if not p['drops']:
                reasons.append('C1_D1_retained_relabel')
        if p['turn_calls_unknown']:
            reasons.append('turn_calls_unknown_input')
            if not p['drops']:
                reasons.append('turn_calls_unknown_C1_retained')
        if p['strict_loop'] is None:
            reasons.append('strict_loop_unknown')
        if p['split']=='train':
            if p['strict_loop'] is True:
                reasons.append('strict_loop_before_C1')
            if p['c4_drop']:
                reasons.append('C4')
            if not p['drops'] and not p['c4_drop']:
                if p['short_turn']:
                    reasons.append('C2')
                if p['direct']:
                    reasons.append('C3')
        for reason in reasons:
            bucket.setdefault(reason,[]).append(sid)
    return result


def audit_cases(counts, plans, indices, offsets, loops):
    cases = []
    rules = ('C1_D1_retained_relabel','C1_D1_missing_candidate','C1_narration_successor','C1_D2','C1_unjudgeable','C2','C3','C4')
    for rule in rules:
        sources = sorted(counts['train'],key=lambda d:(-len(counts['train'][d].get(rule,[])),d))
        sources = [d for d in sources if counts['train'][d].get(rule)][:3]
        if rule=='C4':
            sources = list(dict.fromkeys(sources+['mle-trajectories','ZClawBench','Toolathlon-Trajectories']))
        for ds in sources:
            ids = sorted(counts['train'][ds].get(rule,[]),key=lambda sid:digest(SEED,sid))[:20]
            for sid in ids:
                raw = indices['train'].read(offsets[sid])
                p = plans[sid]
                row = dict(rule=rule,dataset=ds,id=sid,before=p['original_gold'],
                           after='DISCARD' if p['drops'] or rule=='C4' else p['gold_names'],
                           context_tail=raw['context'][-900:])
                if rule=='C4':
                    row['source_evidence']=loops[sid]
                cases.append(row)
    return cases


def write_samples(path, ids, split, indices, offsets, plans, restored):
    index = indices[split]
    seen = set()
    with index.path.open('rb') as source, path.open('w') as dest:
        for sid in sorted(ids,key=lambda sid:offsets[sid]):
            source.seek(offsets[sid])
            raw = json.loads(source.readline())
            require(raw['id']==sid and sid not in seen,'write ID mismatch')
            seen.add(sid)
            result = materialize(raw,plans[sid],restored.get(sid))
            allowed = {'gold','short_turn','direct','action_kind','transition','loop_state'}
            require({k:v for k,v in raw.items() if k not in allowed}=={k:v for k,v in result.items() if k not in allowed},
                    f'protected input fields changed: {sid}')
            dest.write(json.dumps(result,ensure_ascii=False)+'\n')
    reread = set()
    with path.open() as handle:
        for line in handle:
            row = validate(json.loads(line))
            require(row['id'] not in reread and row['gold']==plans[row['id']]['gold'],'written duplicate/label differs')
            reread.add(row['id'])
    require(seen==reread==set(ids),'written coverage differs')
    return len(seen)


def token_check(indices, offsets, plans, restored, tokenizer_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    groups = defaultdict(list)
    for sid,p in plans.items():
        if not p['drops']:
            groups[(p['split'],p['d1'])].append(sid)
    checked = []
    for ids in groups.values():
        for sid in sorted(ids,key=lambda sid:digest(SEED,sid))[:3]:
            raw = indices[plans[sid]['split']].read(offsets[sid])
            changed = materialize(raw,plans[sid],restored.get(sid))
            old_item,new_item = make_item(raw,tokenizer,seed=SEED),make_item(changed,tokenizer,seed=SEED)
            require(old_item['ids']==new_item['ids'] and old_item['perms']==new_item['perms'],f'encoded prompt changed: {sid}')
            require(abs(float(new_item['targets'].sum())-1)<1e-6,'invalid soft target mass')
            checked.append(sid)
    return checked


def report(summary):
    lines = ['# 第二阶段：B1与B2配额预览\n',summary['created_at'],
             '\nB1和规范化评测集已写出；B2未写样本或ID文件，等待用户确认。未训练、评测或操作GPU。\n']
    lines.extend(f'- {key}：{value}' for key,value in DEFINITIONS.items())
    lines.extend(['\n## 配额预览\n','| 来源 | k | round(.25k) | 可用优先类 | 实取优先类 | B1 RESPOND | B2 RESPOND |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for d,r in summary['preview'].items():
        lines.append(f'| {d} | {r["k"]} | {r["priority_quota"]} | {r["priority_available"]} | {r["priority_selected"]} | {r["b1_respond"]} | {r["b2_respond"]} |')
    lines.extend(['\n## 总体构成\n','```json',json.dumps(summary['composition'],ensure_ascii=False,indent=2),'```',
                  '\n## 规则计数\n','| split | 来源 | 规则 | n |','|---|---|---|---:|'])
    for split,sources in summary['rule_ids'].items():
        for ds,rules in sources.items():
            if not rules:
                lines.append(f'| {split} | {ds} | 无规则命中 | 0 |')
            for rule,ids in rules.items():
                lines.append(f'| {split} | {ds} | {rule} | {len(ids)} |')
    if 'manual_review' in summary:
        lines.extend(['\n## 人工复核结论\n','```json',json.dumps(summary['manual_review'],ensure_ascii=False,indent=2),'```'])
    lines.append('\n## 人工审核样例\n每规则影响最大三个来源，各按digest取20条（不足全取）；C4额外保证MLE/ZClawBench/Toolathlon。C2/C3展示B2候选池标记，并非保证入选。上下文为保存渲染文本末900字符，源调用证据未参与模型输入。\n')
    for c in summary['manual_cases']:
        lines.extend([f'### {c["rule"]} / {c["dataset"]} / {c["id"]}\n',f'gold: {c["before"]} → {c["after"]}',
                      '\n```text',c['context_tail'].replace('```','｀｀｀'),'```'])
        if 'source_evidence' in c:
            lines.extend(['\n```json',json.dumps(c['source_evidence'],ensure_ascii=False,indent=2),'```'])
        if 'c1_gold' in c:
            lines.append(f'\nC1保留：{c["c1_retained"]}；C1修正后gold：{c["c1_gold"]}。')
        if 'manual_review' in c:
            lines.extend(['\n人工复核：','```json',json.dumps(c['manual_review'],ensure_ascii=False,indent=2),'```'])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','pool','features','build-config','output','b1-output','eval-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    started = datetime.now().astimezone().isoformat(timespec='seconds')
    paths = (args.output,args.b1_output,args.eval_output)
    require(len({p.resolve() for p in paths})==3, 'output directories must be distinct')
    require(all(not p.exists() for p in paths), 'output directory exists; use new paths')
    created = []
    try:
        for path in paths:
            path.mkdir(parents=True,exist_ok=False)
            (path/'metadata').mkdir()
            created.append(path)
            (path/'metadata/command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
            write_json(path/'metadata/run_status.json',dict(status='running',started_at=started))
        features,indices,plans,offsets,targets,quotas,b0,approved = load(args)
        restored,loops = recover(targets,plans,read_json(args.build_config))
        # This entry reproduces the original, now-reviewed C4 preview.
        # Current v2 selections are written with write_task_reply_samples.py.
        b1,b2,preview = select_versions(plans,quotas,SEED,apply_c4=True)
        counts = ledger(plans)
        composition = dict(B1=summarize(plans,b1),B2_preview=summarize(plans,b2))
        composition['B2_priority_fraction'] = sum(r['priority_selected'] for r in preview.values())/len(b2)
        composition['B2_respond_outside_8_20_percent'] = not .08<=composition['B2_preview']['respond_fraction']<=.20
        checked = token_check(indices,offsets,plans,restored,read_json(args.build_config)['tokenizer'])
        print(f'token identity passed: {len(checked)} rows',flush=True)
        written = {'B1':write_samples(args.b1_output/'train.jsonl',b1,'train',indices,offsets,plans,restored)}
        (args.b1_output/'train_ids.txt').write_text(''.join(sid+'\n' for sid in b1))
        for split in SPLITS[1:]:
            ids = [sid for sid,p in plans.items() if p['split']==split and not p['drops']]
            written[split] = write_samples(args.eval_output/f'{split}.jsonl',ids,split,indices,offsets,plans,restored)
        summary = dict(created_at=started,smoke=args.smoke,definitions=DEFINITIONS,
                       inputs={k:str(getattr(args,k).resolve()) for k in ('samples','pool','features','build_config')},
                       outputs=dict(B1=str(args.b1_output.resolve()),normalized_eval=str(args.eval_output.resolve()),B2=None),
                       validation=dict(b0=b0,approved_stage1=approved,source_recovered=len(targets),loop_identities=len(loops),
                                       protected_fields_equal=True,token_identity_ids=checked,written_counts=written),
                       preview=preview,composition=composition,rule_ids=counts,
                       manual_cases=audit_cases(counts,plans,indices,offsets,loops))
        write_json(args.output/'summary.json',summary)
        (args.output/'report.md').write_text(report(summary))
        for path in (args.b1_output,args.eval_output):
            write_json(path/'metadata/curation.json',dict(rule_version='C1-C4-20261025-preview',definitions=DEFINITIONS,
                        quotas=quotas,composition=composition['B1'] if path==args.b1_output else written,
                        rule_ledger=str((args.output/'summary.json').resolve())))
    except Exception as exc:
        for path in created:
            write_json(path/'metadata/run_status.json',dict(status='failed',started_at=started,error=repr(exc)))
        raise
    for path in (args.output,args.b1_output,args.eval_output):
        write_json(path/'metadata/run_status.json',dict(status='smoke_completed' if args.smoke else 'completed',
                    stage='B2_preview_awaiting_user_confirmation',started_at=started,
                    completed_at=datetime.now().astimezone().isoformat(timespec='seconds')))
    print(json.dumps(dict(composition=composition,written=written),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
