#!/usr/bin/env python
"""CPU-only teacher loop histories and saved Jev predictions; no model calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from analyze_respond import (PROJECT, WORKSPACE, SampleIndex, load_selected, tool_name,
                             validate, validate_part, completed_parts, read_json, write_json, require)
from jev_train.build import history_features, codex_call_types
from jev_train.common import digest
from dataset_tool_inventory.common import iter_jsonl
from dataset_tool_inventory.schema import DecisionEvent
from dataset_tool_inventory.adapters.review_common import emit_trajectory, visible

from jev_train.reply_capture import source_units, RawTree


def history_loop(history, call_types=None):
    """Only past calls determine membership, including repeats within a batch."""
    calls = []
    for index, message in enumerate(history):
        if message['role'] == 'user':
            calls.clear()
        elif message['role'] == 'assistant':
            calls.extend((index, call) for call in message.get('tool_calls', []))
    result = dict(loop_history=False, current_turn_call_count=len(calls), loop_tool=None,
                  last_three_batches=None, argument_forms=[])
    if len(calls) < 3:
        return result
    last = calls[-3:]
    # Reuse the builder's approved JSON/custom/empty/unparsed comparison rules.
    # The supplied action comes from the history, never from the current gold.
    synthetic = [dict(role='assistant', content=None, tool_calls=[call]) for _, call in last]
    feature = history_features(synthetic, [last[-1][1]['function']['name']], call_types)
    result.update(loop_history=feature['loop_state'] == 'yes', argument_forms=feature['loop_argument_forms'],
                  last_three_batches=len({i for i, _ in last}))
    if result['loop_history']:
        result['loop_tool'] = last[-1][1]['function']['name']
    return result


def gold_relation(gold, tool):
    require(gold and ('RESPOND' not in gold or gold == ['RESPOND']), 'invalid gold actions')
    if gold == ['RESPOND']:
        return 'RESPOND'
    return 'contains_loop_tool' if tool in gold else 'different_tool'


def history_status(index, raw, messages, tree=None):
    """Validate the current canonical user turn, not RESPOND semantic boundaries."""
    abandoned = raw[index].get('_abandoned', frozenset())
    history = [i for i in range(index) if not (raw[i].get('_uuids') and raw[i]['_uuids'] <= abandoned)]
    boundary = next((i for i in reversed(history) if messages[i]['role'] == 'user'), None)
    if boundary is None:
        return 'no_user_boundary_in_record'
    if tree:
        ancestors, issue = tree.ancestry_to_user(index, boundary)
        if issue:
            return issue
        for i in history:
            if i <= boundary or not messages[i].get('tool_calls'):
                continue
            if not tree.message_nodes[i] & ancestors:
                return 'tool_message_outside_target_ancestry'
            if max(tree.nodes[n][0] for n in tree.message_nodes[i]) >= min(tree.nodes[n][0] for n in tree.message_nodes[index]):
                return 'tool_fragment_after_target'
    return 'known'


def slim(raw, sample_source):
    validate(raw)
    row = {k: raw[k] for k in ('id', 'dataset', 'model', 'outcome', 'candidate_level', 'messages_dropped', 'truncation_counts')}
    row.update(names=[tool_name(o) for o in raw['options']], gold_indices=raw['gold'], sample_source=sample_source)
    require(len(set(row['names'])) == len(row['names']) and row['names'][-1] == 'RESPOND', 'invalid option identities')
    row['gold_names'] = [row['names'][i] for i in row['gold_indices']]
    for key in ('trajectory_id', 'turn_index', 'harness', 'action_kind', 'prev_action_names', 'transition', 'loop_state'):
        if key in raw:
            row[key] = raw[key]
    return row


def cached_index(path):
    cache = path.parent.parent/'index_cache'/path.parent.name/(path.stem+'.json')
    require(cache.is_file(), f'read-only analysis requires existing index: {cache}')
    return SampleIndex(path)


def load_inputs(args):
    saved = read_json(args.training_analysis)
    selection = saved['training_selection']
    tokens = shlex.split(Path(selection['command']).read_text().splitlines()[1])
    for flag, value in (('--samples', selection['samples']), ('--train_limit', '8000'), ('--seed', '20261015')):
        require(tokens.count(flag) == 1 and tokens[tokens.index(flag)+1] == value, f'training command differs: {flag}')
    require('--train_ids' not in tokens and selection['train_limit'] == 8000 and selection['seed'] == 20261015,
            'expected actual A 8k selection')
    targets, populations = {}, defaultdict(set)
    smoke_seen = defaultdict(set)
    def add(row, population):
        sid = row['id']
        if args.smoke:
            # A bounded real-source check per dataset/action type, plus saved-loop positives.
            kind = 'RESPOND' if row['gold_names'] == ['RESPOND'] else 'TOOL' if len(row['gold_names']) == 1 else 'TOOL_SET'
            slot = (row['dataset'], 'saved_loop' if row.get('loop_state') == 'yes' else kind)
            if slot in smoke_seen[population]:
                return
            smoke_seen[population].add(slot)
        if sid in targets:
            old = targets[sid]
            for key in ('dataset', 'model', 'outcome', 'candidate_level', 'names', 'gold_indices', 'messages_dropped', 'truncation_counts'):
                require(old[key] == row[key], f'overlapping sample differs: {sid}/{key}')
            row = {**old, **row}
        targets[sid] = row
        require(sid not in populations[population], f'duplicate {population} ID: {sid}')
        populations[population].add(sid)
    index = cached_index(Path(selection['samples'])/'train.jsonl')
    selected = load_selected(index, 8000, 20261015)
    require(len(selected) == 8000, 'training selection incomplete')
    print('reading exact 8000 training rows', flush=True)
    for entry in selected:
        add(slim(index.read(entry['offset']), f'{index.path}#byte={entry["offset"]}'), 'A_train')
    pool_count = 0
    print('streaming pool training rows', flush=True)
    for number, raw in iter_jsonl(args.pool/'train.jsonl'):
        add(slim(raw, f'{args.pool}/train.jsonl:{number}'), 'pool_train')
        pool_count += 1
        if pool_count % 20000 == 0:
            print(f'pool rows read: {pool_count}', flush=True)
    require(pool_count == read_json(args.pool/'build_report.json')['lengths']['train']['count'], 'pool coverage differs')
    state = read_json(args.evaluation/'metadata/run_status.json')
    expected_model = Path(selection['command']).parents[1]/'model'
    require(state['status'] == 'completed' and Path(state['configuration']['model']).resolve() == expected_model.resolve(),
            'saved evaluation is not the selected 2B 8k model')
    require(Path(state['configuration']['samples']).resolve() == args.pool.resolve(), 'eval/pool differ')
    predictions = {}
    for split in ('val', 'test_id'):
        data, seed = completed_parts(args.evaluation, split)
        require(seed == 20261015, 'evaluation seed differs')
        validate_part(data)
        require(len(data['id']) == 1000, f'expected fixed 1000 {split} IDs')
        index = cached_index(args.pool/f'{split}.jsonl')
        offsets = {e['id']: e['offset'] for e in index.entries}
        for i, sid in enumerate(data['id']):
            row = slim(index.read(offsets[sid]), f'{index.path}#byte={offsets[sid]}')
            names = data['tool_names'][i]
            gold = [name for j, name in enumerate(names) if data['targets'][i, j] > 0]
            require(len(names) == len(row['names']) and set(names) == set(row['names']) and set(gold) == set(row['gold_names']),
                    f'saved logits labels/options differ: {sid}')
            require(all(row[k] == data[k][i] for k in ('dataset', 'outcome', 'candidate_level', 'transition')), 'part metadata differ')
            add(row, split)
            if sid in populations[split]:
                position = int(data['logits'][i].argmax())
                predictions[sid] = dict(prediction=names[position], hit=bool(data['targets'][i, position] > 0))
    for population, ids in populations.items():
        for sid in ids:
            targets[sid].setdefault('populations', []).append(population)
    print('input coverage '+str({k: len(v) for k, v in populations.items()}), flush=True)
    return targets, populations, predictions, selection


def locate_missing(targets, config):
    missing = {sid: r for sid, r in targets.items() if 'trajectory_id' not in r}
    if not missing:
        return
    entries = {e['name']: e for e in read_json(WORKSPACE/config['inventory_config'])['datasets']}
    datasets = {r['dataset'] for r in missing.values()}
    codex = {d for d in datasets if entries[d]['review_adapter'] == 'native_codex'}
    maxima, found = {}, set()
    for number, row in iter_jsonl(WORKSPACE/config['review_run']/'states.jsonl'):
        ds = row['dataset']
        if ds not in datasets:
            continue
        key = ds, row['trajectory_id']
        if ds in codex:
            maxima[key] = max(maxima.get(key, -1), row['turn_index'])
        else:
            sid = digest(*key, row['turn_index'])
            if sid in missing:
                missing[sid].update(trajectory_id=key[1], turn_index=row['turn_index'])
                found.add(sid)
    # The old Codex review precedes merged calls: its indices are only an upper
    # bound for locating IDs. All current labels/history are revalidated below.
    for key, maximum in maxima.items():
        for turn in range(maximum+1):
            sid = digest(*key, turn)
            if sid in missing:
                missing[sid].update(trajectory_id=key[1], turn_index=turn)
                found.add(sid)
    require(found == missing.keys(), f'unlocated training IDs: {sorted(missing.keys()-found)[:5]}')
    print(f'located {len(missing)} old training IDs outside pool', flush=True)


def rebuild(unit):
    cap, family = unit['capture'], unit['family']
    raw, context = cap['messages'], cap['context']
    messages = [visible(m, context.get('strip_think', False)) for m in raw]
    tree = RawTree(family, unit['raw_records'], raw) if family in ('native_pi', 'native_claude') else None
    call_types = codex_call_types(Path(context['source'])) if family == 'native_codex' else None
    wanted = {r['id']: r for r in unit['targets']}
    found = set()
    # Explicitly replay all saved supervised prefixes from the longest equal
    # prefix. source_units already verified prefix equality and supervision.
    for event in emit_trajectory(cap['dataset'], cap['trajectory_id'], raw, cap['tools'], context):
        if not isinstance(event, DecisionEvent):
            continue
        sid = digest(event.dataset, event.trajectory_id, event.turn_index)
        if sid not in wanted:
            continue
        require(sid not in found, f'duplicate reconstructed ID: {sid}')
        found.add(sid)
        target = wanted[sid]
        meta = event.metadata
        gold = ['RESPOND'] if event.action_kind == 'RESPOND' else list(event.action_names)
        require(target['names'] == list(event.candidate_tools)+['RESPOND'] and set(target['gold_names']) == set(gold), f'source labels/options differ: {sid}')
        require(target['model'] == event.model and all(target[k] == meta[k] for k in ('outcome', 'candidate_level')), f'source metadata differs: {sid}')
        for key in ('harness', 'action_kind'):
            if key in target:
                require(target[key] == getattr(event, key), f'source {key} differs: {sid}')
        old_features = history_features(meta['_history'], gold, call_types)
        for key in ('prev_action_names', 'transition', 'loop_state'):
            if key in target:
                require(target[key] == old_features[key], f'saved feature differs: {sid}/{key}')
        index = meta['target_message_index']
        reason = history_status(index, raw, messages, tree)
        loop = history_loop(meta['_history'], call_types) if reason == 'known' else dict(loop_history=None, loop_tool=None)
        result = {k: target[k] for k in ('id', 'dataset', 'model', 'populations', 'messages_dropped', 'truncation_counts')}
        result.update(trajectory_id=event.trajectory_id, turn_index=event.turn_index, source=event.source,
                      outcome=meta['outcome'], harness=event.harness, family=family, gold_names=gold,
                      saved_loop_state=old_features['loop_state'], history_status=reason, **loop)
        if loop['loop_history']:
            result.update(gold_relation=gold_relation(gold, loop['loop_tool']), gold_multi_tool=len(gold)>1)
        yield result
    require(found == wanted.keys(), f'raw source coverage incomplete: {sorted(wanted.keys()-found)[:5]}')


def aggregate(rows):
    known = [r for r in rows if r['loop_history'] is not None]
    loops = [r for r in known if r['loop_history']]
    pred = [r for r in loops if 'prediction' in r]
    return dict(count=len(rows), known=len(known), unknown=len(rows)-len(known), loops=len(loops),
        loop_rate=len(loops)/len(known) if known else None,
        unknown_reasons=dict(Counter(r['history_status'] for r in rows if r['loop_history'] is None)),
        gold_relation=dict(Counter(r['gold_relation'] for r in loops)),
        contains_loop_tool_with_other_gold=sum(r['gold_multi_tool'] and r['gold_relation']=='contains_loop_tool' for r in loops),
        by_outcome=dict(Counter(r['outcome'] for r in loops)),
        last_three_batches=dict(Counter(str(r['last_three_batches']) for r in loops)),
        argument_forms_all_known=dict(Counter(form for r in known for form in r['argument_forms'])),
        argument_forms=dict(Counter(form for r in loops for form in r['argument_forms'])),
        with_deleted_history=sum(r['messages_dropped']>0 for r in loops),
        with_truncated_arguments=sum(r['truncation_counts'].get('tool_call_arguments_truncated', 0)>0 for r in loops),
        prediction=dict(count=len(pred), hit=sum(r['hit'] for r in pred),
            repeats_loop_tool=sum(r['prediction']==r['loop_tool'] for r in pred),
            incorrect_repeats=sum(r['prediction']==r['loop_tool'] and not r['hit'] for r in pred),
            by_gold_relation={kind: dict(count=len(group), repeats_loop_tool=sum(r['prediction']==r['loop_tool'] for r in group),
                                        hit=sum(r['hit'] for r in group))
                for kind in ('RESPOND','contains_loop_tool','different_tool')
                if (group := [r for r in pred if r['gold_relation']==kind])}),
        examples=[{k: r[k] for k in ('id','source','loop_tool','gold_names')} for r in loops[:3]],
        unknown_examples=[{k: r[k] for k in ('id','source','history_status')} for r in rows if r['loop_history'] is None][:3])


DEFINITIONS = {
    'loop_history': 'Only current user turn: reset at each canonical user message; flatten assistant.tool_calls in source order; the last three calls have identical names and normalized arguments. No dependence on current gold. Same-batch repetitions count and number of distinct batches is reported.',
    'arguments': 'Reuse history_features: function JSON canonicalized; empty string is named empty_arguments -> {}; invalid nonempty JSON compared by exact text and counted; Codex custom input compared exactly by original source type. Missing required fields or source mismatches fail.',
    'unknown': 'Missing parent, ambiguous tool branches/fragments or absent current user boundary are unknown, not negatives. Every canonical user resets the turn, including framework/summary user messages. A new user inside a continued part establishes a known local turn; no earlier part is needed. Missing source counted separately; rates use the known denominator.',
    'gold_relation': 'RESPOND; contains_loop_tool (with multi-tool gold separately counted); different_tool (loop tool absent). This compares current tool names only, not current arguments or action quality.',
    'prediction': 'Saved val/test_id argmax, hit iff targets[argmax]>0. No inference or temperature fitting; positive scalar calibration does not change argmax. No OOD used.',
    'visibility': 'Rebuilt full source-visible history, before rendering. messages_dropped and truncated tool arguments are reported separately; a loop in source history is not proof all three calls survived in the model input.',
    'populations': 'Exact A 8k and entire pool train reported separately; overlapping IDs are shared observations, not independent samples. No new training selection or labels.'
}


def report(summary):
    lines = ['# 教师循环历史与 Jev 重复推荐', '', '时间：'+summary['created_at'], '',
             '只读 CPU 分析；不运行模型、不改数据、不生成训练选样。循环由历史单独决定，当前 gold 仅用于事后分类。', '',
             '| 人群 | 状态 | 不可判 | 循环历史 | gold 含同工具 | gold 换工具 | gold RESPOND |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, pop in summary['populations'].items():
        a=pop['overall'];g=a['gold_relation']
        lines.append(f'| {name} | {a["count"]} | {a["unknown"]} | {a["loops"]} | {g.get("contains_loop_tool",0)} | {g.get("different_tool",0)} | {g.get("RESPOND",0)} |')
    lines += ['', '| 评测循环子集 | 数量 | Jev 推荐重复工具 | 重复且不在 gold | top-1 命中 |', '|---|---:|---:|---:|---:|']
    for name in ('val','test_id'):
        a=summary['populations'][name]['overall']['prediction']
        lines.append(f'| {name} | {a["count"]} | {a["repeats_loop_tool"]} | {a["incorrect_repeats"]} | {a["hit"]} |')
    lines += ['', '| 训练来源 | A 循环/已知 | A 同工具/换工具/RESPOND | pool 循环/已知 | pool 同工具/换工具/RESPOND |', '|---|---:|---|---:|---|']
    for ds in sorted(summary['populations']['pool_train']['by_dataset']):
        cells=[]
        for population in ('A_train','pool_train'):
            a=summary['populations'][population]['by_dataset'].get(ds)
            cells.extend(['0/0','0/0/0'] if a is None else [f'{a["loops"]}/{a["known"]}', '/'.join(str(a['gold_relation'].get(k,0)) for k in ('contains_loop_tool','different_tool','RESPOND'))])
        lines.append('| '+ds+' | '+' | '.join(cells)+' |')
    lines += ['', '完整按来源/教师、同批情况、未知原因、参数形式、截断及少量源定位见 summary.json；正例逐状态见 loop_states.jsonl。', '',
              '这些是教师标签和已保存预测的描述性统计。重复同名工具不等于错误或相同参数；不能直接推出过滤、补样会提升在线成功率。少量离线循环状态不能代表全部 BFCL 循环。', '',
              '口径：', *['- '+k+'：'+v for k,v in DEFINITIONS.items()]]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-analysis',type=Path,required=True)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    require(args.output.resolve().parent == (PROJECT/('tmp' if args.smoke else 'outputs')).resolve(), 'wrong independent output root')
    args.output.mkdir(parents=False,exist_ok=False)
    (args.output/'metadata').mkdir()
    (args.output/'metadata/command.txt').write_text(shlex.join([sys.executable,*sys.argv])+'\n')
    state=dict(status='running',started_at=datetime.now().astimezone().isoformat())
    write_json(args.output/'metadata/run_status.json',state)
    try:
        targets,populations,predictions,selection=load_inputs(args)
        config=read_json(PROJECT/'configs/build.json')
        locate_missing(targets,config)
        results={};missing_sources={};units=0
        for unit in source_units(targets,config,missing_sources):
            for row in rebuild(unit):
                require(row['id'] not in results, 'duplicate source state')
                row.update(predictions.get(row['id'],{}))
                results[row['id']]=row
            units+=1
            if units%1000==0:print(f'rebuilt {units} trajectories / {len(results)}/{len(targets)} states',flush=True)
        for sid,reason in missing_sources.items():
            target=targets[sid]
            results[sid]={**target,'source':reason,'history_status':'missing_source','loop_history':None}
        require(results.keys()==targets.keys(),'final ID coverage differs')
        summary=dict(created_at=datetime.now().astimezone().isoformat(),smoke=args.smoke,definitions=DEFINITIONS,
            training_selection=selection,evaluation=str(args.evaluation),pool=str(args.pool),
            coverage=dict(unique_states=len(results),source_units=units,source_missing=len(missing_sources),
                exact_source_labels_validated=len(results)-len(missing_sources),
                A_pool_overlap=len(populations['A_train']&populations['pool_train'])),populations={})
        for name,ids in populations.items():
            rows=[results[sid] for sid in sorted(ids)]
            datasets=defaultdict(list);teachers=defaultdict(list)
            for row in rows:
                datasets[row['dataset']].append(row)
                teachers[(row['dataset'],row['model'])].append(row)
            summary['populations'][name]=dict(overall=aggregate(rows),
                by_dataset={d:aggregate(group) for d,group in sorted(datasets.items())},
                by_dataset_model=[dict(dataset=d,model=m,**aggregate(group)) for (d,m),group in sorted(teachers.items())])
        write_json(args.output/'summary.json',summary)
        with (args.output/'loop_states.jsonl').open('w') as handle:
            for sid,row in sorted(results.items()):
                if row['loop_history']:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        (args.output/'report.md').write_text(report(summary))
        state.update(status='completed',coverage=summary['coverage'])
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        state['ended_at']=datetime.now().astimezone().isoformat()
        write_json(args.output/'metadata/run_status.json',state)


if __name__=='__main__':main()
