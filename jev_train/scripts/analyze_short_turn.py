#!/usr/bin/env python
"""CPU-only RESPOND semantic slices of saved evaluation parts, with exact ID joins."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(WORKSPACE/'third_party/decider'),
               str(PROJECT.parent/'dataset_tool_inventory/src'), str(PROJECT/'scripts')]
import torch
from jev_train.common import digest
from jev_train.data import SampleIndex, load_selected, tool_name, validate
from jev_train.evaluation import read_parts, paired_difference, sign_test
from analyze_respond import read_json, write_json, require, validate_part, gold_respond, quantiles
from jev_train.reply_capture import termination_status as source_termination_status

SPLITS = ('val', 'test_id', 'test_ood')
GROUPS = ('short_turn', 'short_end', 'long_turn_or_end', 'no_prior_call', 'mid_commentary', 'unknown')
PAIRS = (('B1', 'B0'), ('B2', 'B0'), ('B2', 'B1'))
IDENTITY = ('id', 'dataset', 'model', 'trajectory_id', 'turn_index')
FEATURE_FIELDS = (*IDENTITY, 'category', 'previous_turn_tool_calls', 'call_count_status',
                  'termination_status', 'termination_value', 'termination_evidence', 'populations')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def readonly_index(samples, split):
    cache = samples.parent/'index_cache'/samples.name/f'{split}.json'
    require(cache.exists(), f'read-only analysis requires existing index cache: {cache}')
    return SampleIndex(samples/f'{split}.jsonl')


def read_evaluation(directory, samples):
    state = read_json(directory/'metadata/run_status.json')
    require(state['status'] == 'completed', f'{directory}: evaluation not completed')
    c = state['configuration']
    require(Path(c['samples']).resolve() == samples.resolve(), 'evaluation sample directory differs')
    require(c['seed'] == 20261015 and type(c['seed']) is int, 'evaluation seed differs')
    require(all(c['limits'][s] == 1000 for s in SPLITS), 'expected three 1000-state evaluation splits')
    temperature = read_json(directory/'temperature.json')['temperature']
    require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
            'temperature must be finite and positive')
    return c, temperature


def classify(feature):
    if feature is None:
        return 'unknown', 'feature_unmatched'
    category, calls = feature['category'], feature['previous_turn_tool_calls']
    require(category in {'user_turn_end', 'trajectory_end', 'intermediate_tool_continuation', 'unjudgeable'}, 'unknown successor category')
    require(calls is None or type(calls) is int and calls >= 0, 'invalid previous_turn_tool_calls')
    if category == 'unjudgeable':
        return 'unknown', 'successor_unjudgeable'
    if calls is None or feature['call_count_status'] != 'known':
        return 'unknown', 'prior_call_count_unknown'
    require(type(calls) is int and calls >= 0, 'invalid previous_turn_tool_calls')
    if category == 'intermediate_tool_continuation':
        return 'mid_commentary', None
    if calls == 0:
        return 'no_prior_call', None
    if calls >= 4:
        return 'long_turn_or_end', None
    if category == 'trajectory_end':
        return 'short_end', None
    status, value = feature['termination_status'], feature['termination_value']
    require(status in {'known', 'missing', 'null', 'conflict'}, 'unknown termination status')
    if status == 'conflict':
        return 'unknown', 'termination_conflict'
    if status == 'known' and value in {'error', 'length'}:
        return 'unknown', 'abnormal_short_user_turn'
    return 'short_turn', None


def load_features(semantics, policy):
    summary = read_json(policy/'summary.json')
    require(read_json(policy/'metadata/run_status.json')['status'] == 'completed', 'policy reconstruction incomplete')
    features = {}
    for row in summary['states']:
        require(row['id'] not in features, f'duplicate policy state: {row["id"]}')
        features[row['id']] = row
    checked = 0
    with semantics.open() as stream:
        seen = set()
        for line in stream:
            row = json.loads(line); sid = row['id']
            require(sid not in seen, f'duplicate semantics state: {sid}'); seen.add(sid)
            if sid in features:
                fields = (*IDENTITY, 'category', 'previous_turn_tool_calls', 'call_count_status')
                require(all(row[k] == features[sid][k] for k in fields), f'semantics/policy conflict: {sid}')
                checked += 1
    return features, summary['sources'], checked


def join_one(row, split, feature, original):
    if original is None:
        return None, 'original_sample_id_missing'
    if any(row[k] != original[k] for k in IDENTITY) or row['options'] != original['options']:
        return None, 'original_identity_or_candidates_differ'
    if not gold_respond(original):
        return None, 'original_gold_not_respond'
    if digest(row['dataset'], row['trajectory_id'], row['turn_index']) != row['id']:
        return None, 'canonical_id_digest_differs'
    if feature is None:
        return None, 'feature_id_missing'
    missing = [k for k in FEATURE_FIELDS if k not in feature]
    if missing:
        return None, 'feature_fields_missing:'+','.join(missing)
    if any(row[k] != feature[k] for k in IDENTITY):
        return None, 'feature_identity_differs'
    if split not in feature['populations']:
        return None, 'feature_split_differs'
    try:
        status, value = source_termination_status(feature['termination_evidence'])
    except (KeyError, ValueError, TypeError):
        return None, 'termination_evidence_invalid'
    if (status, value) != (feature['termination_status'], feature['termination_value']):
        return None, 'termination_evidence_conflict_with_saved_fields'
    return feature, None


def audit_split(index, original_index, features, split, selected_ids, smoke_limit=None):
    original = {e['id']: e for e in original_index.entries}
    selected = set(selected_ids)
    sample_rows, joined, misses = {}, {}, {}
    counts = Counter(); selected_counts = Counter(); termination = Counter(); groups = Counter(); reasons = Counter()
    entries = index.entries if smoke_limit is None else [e for e in index.entries if e['id'] in selected]
    for e in entries:
        row = validate(index.read(e['offset'])); sid = row['id']; is_respond = gold_respond(row)
        require(sid == e['id'], 'sample ID differs from cached entry')
        counts['states'] += 1
        old = original_index.read(original[sid]['offset']) if sid in original else None
        if old is None:
            counts['original_id_missing'] += 1
        else:
            require(all(row[k] == old[k] for k in IDENTITY), f'{sid}: normalized identity changed')
            require(row['options'] == old['options'], f'{sid}: normalized candidates changed')
            counts['original_id_matched'] += 1
            counts['gold_changed'] += row['gold'] != old['gold']
            counts['new_respond'] += is_respond and not gold_respond(old)
        if sid in selected:
            sample_rows[sid] = {k: row[k] for k in (*IDENTITY, 'options', 'gold')}
        if not is_respond:
            continue
        counts['gold_respond'] += 1
        feature, reason = join_one(row, split, features.get(sid), old)
        if reason:
            misses[sid] = reason; counts['unmatched'] += 1
        else:
            joined[sid] = feature; counts['matched'] += 1
            termination[feature['termination_status']] += 1
        group, why = classify(feature)
        groups[group] += 1
        if why: reasons[why] += 1
        if sid in selected:
            selected_counts['gold_respond'] += 1
            selected_counts['unmatched' if reason else 'matched'] += 1
    require(sample_rows.keys() == selected, f'{split}: selected IDs missing from sample audit')
    def complete(c):
        result = {k: c[k] for k in ('gold_respond', 'matched', 'unmatched')}
        result['unmatched_fraction'] = c['unmatched']/c['gold_respond'] if c['gold_respond'] else 0.
        return result
    report = dict(scope='all_normalized_split' if smoke_limit is None else 'bounded_old_label_smoke',
                  states=counts['states'], original_id_matched=counts['original_id_matched'],
                  original_id_missing=counts['original_id_missing'], gold_changed=counts['gold_changed'],
                  new_respond=counts['new_respond'], population=complete(counts), selected=complete(selected_counts),
                  unmatched_reasons=dict(Counter(misses.values())), unmatched=[dict(id=sid, reason=r) for sid, r in misses.items()],
                  termination_status=dict(termination), population_groups={g: groups[g] for g in GROUPS},
                  unknown_reasons=dict(reasons), newly_reconstructed=0)
    report['stop'] = any(report[k]['unmatched_fraction'] > .05 for k in ('population', 'selected'))
    return report, sample_rows, joined


def prediction_rows(dump, temperature, samples, features):
    positions = validate_part(dump)
    logits = dump['logits'].float()/temperature
    probabilities = torch.softmax(logits, dim=1)
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    pred = logits.argmax(1)
    result = {}
    for i, sid in enumerate(dump['id']):
        row = samples[sid]; n = dump['nopts'][i]; positive = torch.nonzero(dump['targets'][i] > 0).flatten().tolist()
        require(Counter(dump['tool_names'][i][j] for j in positive) == Counter(tool_name(row['options'][j]) for j in row['gold']),
                f'{sid}: sample and saved target names differ')
        require(n == len(row['options']) and dump['dataset'][i] == row['dataset'], f'{sid}: sample and saved metadata differ')
        require(Counter(dump['tool_names'][i]) == Counter(tool_name(option) for option in row['options']),
                f'{sid}: sample and saved candidate names differ')
        pos = int(positions[i]); gold = bool(dump['targets'][i, pos] > 0)
        require(gold == gold_respond(row), f'{sid}: RESPOND label differs')
        group, reason = classify(features.get(sid)) if gold else (None, None)
        result[sid] = dict(gold_respond=gold, predicted_respond=int(pred[i]) == pos,
            correct=bool(dump['targets'][i, pred[i]] > 0), p_respond=float(probabilities[i, pos]),
            respond_rank=int(torch.nonzero(order[i] == pos)[0, 0])+1, group=group, unknown_reason=reason)
    return result


def summarize(rows):
    rows = list(rows); positives = [r for r in rows if r['gold_respond']]; negatives = [r for r in rows if not r['gold_respond']]
    def ratio(n, d): return n/d if d else None
    groups = {}
    for group in GROUPS:
        selected = [r for r in positives if r['group'] == group]; n = len(selected)
        groups[group] = dict(count=n, respond_hits=sum(r['predicted_respond'] for r in selected),
            respond_recall=ratio(sum(r['predicted_respond'] for r in selected), n),
            p_respond_median=quantiles([r['p_respond'] for r in selected])['p50'],
            respond_rank2_count=sum(r['respond_rank'] == 2 for r in selected),
            respond_rank2_rate=ratio(sum(r['respond_rank'] == 2 for r in selected), n),
            caution='样本太少，只作参考' if n < 30 else None)
    require(sum(g['count'] for g in groups.values()) == len(positives), 'nonexclusive RESPOND groups')
    return dict(count=len(rows), overall_acc=ratio(sum(r['correct'] for r in rows), len(rows)),
        gold_respond=len(positives), nonrespond_count=len(negatives),
        premature_respond_count=sum(r['predicted_respond'] for r in negatives),
        premature_respond_rate=ratio(sum(r['predicted_respond'] for r in negatives), len(negatives)), groups=groups)


def mcnemar(a, b, group):
    require(a.keys() == b.keys(), 'paired state IDs differ')
    ids = [sid for sid in a if a[sid]['gold_respond'] and a[sid]['group'] == group]
    require(all(a[sid]['group'] == b[sid]['group'] and a[sid]['gold_respond'] == b[sid]['gold_respond'] for sid in a), 'paired state groups differ')
    improved = sum(a[s]['predicted_respond'] and not b[s]['predicted_respond'] for s in ids)
    regressed = sum(b[s]['predicted_respond'] and not a[s]['predicted_respond'] for s in ids)
    return dict(count=len(ids), improved=improved, regressed=regressed, discordant=improved+regressed,
                recall_difference=(improved-regressed)/len(ids) if ids else None,
                p_value=sign_test(improved, regressed) if improved+regressed else (1. if ids else None),
                caution='样本太少，只作参考' if len(ids) < 30 else None)


def render(summary):
    def f(v): return 'NA' if v is None else f'{v:.4g}'
    lines = ['# RESPOND 短轮专项分析', '', f"生成时间：{summary['created_at']}", '',
             '只读已保存的logits；本轮工具未进行模型推理。trajectory_end不是任务成功证明。', '',
             '| split | 全量gold RESPOND | 特征匹配/未匹配 | ID匹配/缺失 | gold变化 | 评测子集匹配/未匹配 |', '|---|---:|---|---|---:|---|']
    for split, a in summary['join'].items():
        p, s = a['population'], a['selected']
        lines.append(f"| {split} | {p['gold_respond']} | {p['matched']}/{p['unmatched']} | {a['original_id_matched']}/{a['original_id_missing']} | {a['gold_changed']} | {s['matched']}/{s['unmatched']} |")
    if summary['status'] != 'completed':
        lines += ['', '未生成正式模型比较：'+summary['status']]
    if summary['smoke_limit'] is not None:
        lines += ['', '**旧标签有界smoke，只用于验证流程，不作为科研结论。**']
    for split, models in summary.get('metrics', {}).items():
        lines += ['', f'## {split}', '', '| 模型 | 总数 | acc | 非RESPOND误停/分母 | 误停率 |', '|---|---:|---:|---|---:|']
        for name, m in models.items():
            lines.append(f"| {name} | {m['count']} | {f(m['overall_acc'])} | {m['premature_respond_count']}/{m['nonrespond_count']} | {f(m['premature_respond_rate'])} |")
        lines += ['', '| 模型/组 | 数量 | RESPOND召回 | P(RESPOND)中位数 | 排第2比例 | 限制 |', '|---|---:|---:|---:|---:|---|']
        for name, m in models.items():
            for group, v in m['groups'].items():
                lines.append(f"| {name}/{group} | {v['count']} | {f(v['respond_recall'])} | {f(v['p_respond_median'])} | {f(v['respond_rank2_rate'])} | {v['caution'] or ''} |")
        lines += ['', '| 配对/组 | 数量 | 改善/退步 | 不一致 | p |', '|---|---:|---|---:|---:|']
        for pair, groups in summary.get('pairs', {}).get(split, {}).items():
            for group, v in groups.items():
                lines.append(f"| {pair}/{group} | {v['count']} | {v['improved']}/{v['regressed']} | {v['discordant']} | {f(v['p_value'])} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', action='append', default=[], metavar='LABEL=DIR')
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--semantics', type=Path, required=True)
    parser.add_argument('--policy-impact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--smoke-limit', type=int)
    args = parser.parse_args()
    specs = {}
    for value in args.eval:
        label, sep, directory = value.partition('=')
        require(sep and directory and label not in specs and label in {'B0', 'B1', 'B2'}, 'invalid/duplicate evaluation label')
        specs[label] = Path(directory).resolve()
    require(args.audit_only or set(specs) == {'B0', 'B1', 'B2'} or (args.smoke_limit is not None and set(specs) == {'B0'}), 'formal analysis requires B0/B1/B2; smoke permits only B0')
    require(len(set(specs.values())) == len(specs), 'evaluation directories must be distinct')
    require(args.smoke_limit is None or args.smoke_limit > 0, 'smoke limit must be positive')
    samples = args.samples.resolve(); output = args.output.resolve()
    require(not any(output.is_relative_to(p) for p in [samples, args.policy_impact.resolve(), args.semantics.resolve().parent, *specs.values()]), 'output must be separate from inputs')
    output.mkdir(parents=True, exist_ok=False); (output/'metadata').mkdir()
    (output/'metadata/command.txt').write_text(shlex.join([sys.executable, *sys.argv])+'\n')
    status = dict(status='running', started_at=now()); write_json(output/'metadata/run_status.json', status)
    summary = dict(created_at=now(), status='running', smoke_limit=args.smoke_limit, join={}, metrics={}, pairs={}, evaluations={},
                   definitions=dict(short_turn='gold RESPOND; next=user_turn_end, prior calls 1..3, no explicit error/length',
                       priority='unknown successor/count; mid_commentary; 0 calls; >=4; short_end; short_turn or unknown abnormal termination',
                       missing='exact ID only; unmatched stays unknown; >5% unmatched in either population or selected split stops',
                       termination='missing/null retained and counted separately; conflict cannot establish normal short_turn',
                       p_respond='softmax(float32 logits / per-model saved temperature); stable descending rank',
                       mcnemar='exact two-sided binomial on discordance; p=1 with nonempty identical predictions, null for empty group',
                       feature_source='saved policy reconstruction, checked against semantics; newly_reconstructed=0'))
    try:
        configurations = {label: read_evaluation(p, samples) for label, p in specs.items()}
        for label, (c, t) in configurations.items():
            summary['evaluations'][label] = dict(directory=str(specs[label]), model=c['model'], samples=c['samples'],
                                                temperature_file=str(specs[label]/'temperature.json'), temperature=t, seed=c['seed'])
        features, sources, checked = load_features(args.semantics, args.policy_impact)
        summary['feature_sources'] = dict(policy=str(args.policy_impact.resolve()), semantics=str(args.semantics.resolve()), crosschecked=checked)
        for split in SPLITS:
            index = readonly_index(samples, split)
            original = readonly_index(Path(sources['samples']), split)
            entries = load_selected(index, 1000, 20261015)
            dumps = {}
            for label, (c, t) in configurations.items():
                dump = read_parts(entries, shards=c['shards'], directory=specs[label]/'shards'/split)
                validate_part(dump)
                dumps[label] = dump
            wanted = sorted(e['id'] for e in entries)
            if args.smoke_limit is not None: wanted = wanted[:args.smoke_limit]
            audit, sample_rows, joined = audit_split(index, original, features, split, wanted, args.smoke_limit)
            summary['join'][split] = audit
            print(json.dumps(dict(split=split, population=audit['population'], selected=audit['selected'], stop=audit['stop']), ensure_ascii=False), flush=True)
            if audit['stop']:
                continue
            if args.audit_only:
                continue
            predictions = {}
            for label, dump in dumps.items():
                if args.smoke_limit is not None:
                    positions = [i for i, sid in enumerate(dump['id']) if sid in set(wanted)]
                    dump = {k: v[positions] if torch.is_tensor(v) else [v[i] for i in positions] for k, v in dump.items()}
                    dumps[label] = dump
                predictions[label] = prediction_rows(dump, configurations[label][1], sample_rows, joined)
            summary['metrics'][split] = {label: summarize(r.values()) for label, r in predictions.items()}
            summary['pairs'][split] = {}
            for a, b in PAIRS:
                if a not in dumps or b not in dumps: continue
                paired_difference(dumps[a], dumps[b], seed=configurations[a][0]['seed'], reference_seed=configurations[b][0]['seed'])
                other = dict(zip(dumps[b]['id'], dumps[b]['tool_names']))
                require(all(names == other[sid] for sid, names in zip(dumps[a]['id'], dumps[a]['tool_names'])),
                        'paired candidate name order differs')
                summary['pairs'][split][f'{a}:{b}'] = {g: mcnemar(predictions[a], predictions[b], g) for g in GROUPS}
        stopped = any(r['stop'] for r in summary['join'].values())
        summary['status'] = 'stopped_join_threshold' if stopped else 'audit_only' if args.audit_only else 'completed'
        if stopped:
            summary['metrics'] = {}; summary['pairs'] = {}
        write_json(output/'summary.json', summary); (output/'report.md').write_text(render(summary))
        require(not stopped, 'unmatched RESPOND features exceed 5%; see summary.json; no model comparison published')
        status.update(status='completed', ended_at=now(), audit_only=args.audit_only)
    except BaseException as exc:
        status.update(status='failed', ended_at=now(), error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(output/'metadata/run_status.json', status)


if __name__ == '__main__':
    main()
