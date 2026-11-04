#!/usr/bin/env python3
"""Write the reviewed v2 38k selection on CPU, preserving canonical inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys

from preview_dedup import PROJECT, SEED, SampleIndex, require, timestamp
from preview_r11_cap import load_frozen, read_json, selection_hash
from preview_task_reply import summarize_selection
from preview_task_reply_v2 import load_origins, load_review, unit_positions, add_v2_diagnostics
from curate_samples import write_samples, token_check
from jev_train.common import write_json
from jev_train.curation import kind, materialize
from jev_train.data import make_item, tool_name, validate
from jev_train.reply_origin import reply_priority
from jev_train.task_selection import select, select_with_sources

COUNT = 38000


def replay(preview):
    approved = read_json(preview/'summary.json')
    status = read_json(preview/'metadata/run_status.json')
    require(status['status'] == 'completed' and not status['smoke'] and not status['failed_targets']
            and not approved['smoke'] and approved['recipe'] == 'task_reply_v2'
            and approved['seed'] == SEED, 'requires completed full v2 preview')
    target = approved['targets'][str(COUNT)]
    require(target['status'] == 'completed' and target['failure_stage'] is None
            and target['selected']['count'] == COUNT and target['selected']['respond_gate_passed']
            and .08 <= target['selected']['respond_fraction'] <= .20, '38k target/gate differs')
    config = read_json(approved['config'])
    kept, labels, reference_ids, r11_ids, regression, inputs, _ = load_frozen(config)
    directory = (PROJECT/config['reply_origin_preview']).resolve()
    require(dict(inputs, reply_origin_preview=str(directory)) == approved['inputs'], 'preview inputs differ')
    origins, old = load_origins(directory, kept, inputs)
    load_review(preview/'summary.json', origins, r11_ids)
    # The approved original-source audit is frozen evidence, not a new review.
    audit = approved['boundary_audit']
    framework_ids = set(audit['zclaw_framework_boundaries'])
    priority = {sid: dict(origin_category=row['category'],
                         **reply_priority(row, framework_boundary=sid in framework_ids))
                for sid, row in origins.items()}
    require(priority == approved['reply_priority'], 'reply priority differs from approved preview')
    reference = dict(Counter(labels[kept[sid]['task_group']] for sid in reference_ids))
    cells = {d: {s: c['reference_count'] for s, c in sources.items()}
             for d, sources in regression['cells'].items()}
    require(reference == approved['reference_domain_counts'] and cells == approved['reference_cells'],
            'frozen reference counts differ')
    ids, allocation = select_with_sources(kept, labels, cells, origins, COUNT, SEED,
                                          framework_boundary_ids=framework_ids)
    require(selection_hash(ids) == target['selection_sha256'], 'ordered 38k IDs differ from preview')
    require(allocation == target['allocation'], '38k source/domain allocation differs')
    selected = summarize_selection(ids, kept, origins, labels, unit_positions(kept), r11_ids)
    add_v2_diagnostics(selected, ids, origins, kept, audit)
    old_ids, _ = select(kept, labels, reference, {sid: o['category'] for sid, o in origins.items()}, COUNT, SEED)
    require(selection_hash(old_ids) == old['targets'][str(COUNT)]['selection_sha256'],
            'v1 reference ordered IDs differ')
    selected['overlap_with_v1_same_size'] = len(set(ids) & set(old_ids))
    require(json.loads(json.dumps(selected)) == target['selected'], '38k full statistics differ')
    print(f'replayed {len(ids)} ordered IDs; allocation, priority and full statistics match', flush=True)
    return approved, kept, labels, ids


def smoke_ids(ids, plans, labels, approved):
    """Represent the approved selection, without running a smaller selection."""
    buckets = {}
    for sid in ids:
        plan = plans[sid]
        priority = sid in approved['reply_priority'] and approved['reply_priority'][sid]['eligible']
        key = (plan['dataset'], labels[plan['task_group']], plan['d1'], kind(plan['gold_names']), priority)
        buckets.setdefault(key, sid)
    chosen = set(buckets.values())
    chosen.update((min(ids, key=lambda sid: plans[sid]['n_prompt_tokens']),
                   max(ids, key=lambda sid: plans[sid]['n_prompt_tokens'])))
    # Cover explicit framework replies and both approved source-boundary rules.
    for group in (
        [sid for sid in ids if sid in approved['reply_priority']
         and approved['reply_priority'][sid]['origin_category'] == 'framework_forced'],
        approved['targets'][str(COUNT)]['selected']['zclaw_framework_boundary_ids'],
        approved['targets'][str(COUNT)]['selected']['hermes_nonexhausted_pressure_ids'],
    ):
        if group:
            chosen.add(group[0])
    return [sid for sid in ids if sid in chosen]


def check_encoding(index, offsets, plans, tokenizer_path):
    checked = token_check({'train': index}, offsets, plans, {}, tokenizer_path)
    # Exercise longest/shortest inputs and each C1/gold-kind path as well.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    examples = {}
    for sid, plan in plans.items():
        examples.setdefault((plan['d1'], kind(plan['gold_names'])), sid)
    examples['longest'] = max(plans, key=lambda sid: plans[sid]['n_prompt_tokens'])
    examples['shortest'] = min(plans, key=lambda sid: plans[sid]['n_prompt_tokens'])
    for sid in dict.fromkeys(examples.values()):
        raw = index.read(offsets[sid])
        result = materialize(raw, plans[sid])
        before, after = make_item(raw, tokenizer, seed=SEED), make_item(result, tokenizer, seed=SEED)
        require(before['ids'] == after['ids'] and before['perms'] == after['perms'], f'encoded input differs: {sid}')
        perm = after['perms'][0]
        expected = [1/len(result['gold']) if i in result['gold'] else 0 for i in perm]
        require(all(abs(float(after['targets'][j])-value) < 1e-6 for j, value in enumerate(expected))
                and not after['targets'][len(perm):].any(), f'permuted C1 soft targets differ: {sid}')
        if sid not in checked:
            checked.append(sid)
    return checked


def readback(path, ids, index, offsets, plans, labels, priority):
    """Compare every persisted row with the original input and frozen C1 plan."""
    sources, domains = Counter(), Counter()
    tasks, respond, replies, tokens, count = set(), 0, 0, 0, 0
    with index.path.open('rb') as source, path.open() as written:
        for sid in sorted(ids, key=lambda sid: offsets[sid]):
            raw_line = written.readline()
            require(bool(raw_line), f'written sample missing: {sid}')
            row = validate(json.loads(raw_line))
            require(row['id'] == sid, f'written sample order/ID differs: {sid}')
            source.seek(offsets[sid])
            raw = validate(json.loads(source.readline()))
            plan = plans[sid]
            require(raw['id'] == sid and all(raw[k] == plan[k]
                    for k in ('dataset', 'model', 'task_group', 'outcome', 'n_prompt_tokens')),
                    f'canonical source/plan identity differs: {sid}')
            names = [tool_name(option) for option in raw['options']]
            require([names[i] for i in raw['gold']] == plan['original_gold'], f'original gold differs: {sid}')
            require([names[i] for i in row['gold']] == plan['gold_names'], f'C1 gold names differ: {sid}')
            require(row == materialize(raw, plan), f'persisted input/label differs: {sid}')
            count += 1
            sources[row['dataset']] += 1
            domains[labels[row['task_group']]] += 1
            tasks.add(row['task_group'])
            respond += plan['gold_names'] == ['RESPOND']
            replies += sid in priority and priority[sid]['eligible']
            tokens += row['n_prompt_tokens']
        require(written.readline() == '', 'unexpected extra written sample')
    return dict(count=count, respond=respond, priority_replies=replies, task_groups=len(tasks),
                n_prompt_tokens=tokens, source_counts=dict(sources), domain_counts=dict(domains))


def run(args):
    approved, plans, labels, ids = replay(args.preview)
    target = approved['targets'][str(COUNT)]
    written_ids = smoke_ids(ids, plans, labels, approved) if args.smoke else ids
    index = SampleIndex(Path(approved['inputs']['source_samples'])/'train.jsonl')
    wanted = set(written_ids)
    offsets = {r['id']: r['offset'] for r in index.entries if r['id'] in wanted}
    require(offsets.keys() == wanted, 'selected IDs missing from source')
    selected_plans = {sid: plans[sid] for sid in written_ids}
    checks = check_encoding(index, offsets, selected_plans, args.tokenizer)
    count = write_samples(args.output/'train.jsonl', written_ids, 'train', {'train': index}, offsets, plans, {})
    id_path = args.output/'train_ids.txt'
    id_path.write_text('\n'.join(written_ids)+'\n')
    require(id_path.read_text().splitlines() == written_ids, 'persisted ordered IDs differ')
    actual = readback(args.output/'train.jsonl', written_ids, index, offsets, plans, labels, approved['reply_priority'])
    require(actual['count'] == count == len(written_ids), 'written count differs')
    if not args.smoke:
        expected = target['selected']
        for key in ('count', 'respond', 'priority_replies', 'task_groups', 'n_prompt_tokens'):
            require(actual[key] == expected[key], f'written {key} differs from preview')
        require(actual['source_counts'] == {s: r['count'] for s, r in expected['by_source'].items()}
                and actual['domain_counts'] == {d: r['count'] for d, r in expected['by_domain'].items()},
                'written source/domain counts differ')
    summary = dict(created_at=timestamp(), recipe='task_reply_v2', seed=SEED, count=count, smoke=args.smoke,
                   preview=str(args.preview), config=approved['config'], source=str(index.path),
                   approved_count=COUNT, approved_selection_sha256=target['selection_sha256'],
                   written_selection_sha256=selection_hash(written_ids), composition=actual,
                   validation=dict(ordered_selection_equal=True, full_preview_statistics_equal=True,
                                   reply_priority_equal=True, all_rows_readback_equal=True,
                                   token_permutation_checked_ids=checks),
                   ordering=dict(jsonl='original source offset', train_ids='approved selection traversal',
                                 training='use --train_ids with --train_limit 0; load_train_ids sorts by digest(seed,id)'),
                   scope='C1 materialization only; no new rendering, labels, filtering, sampling or training.')
    write_json(args.output/'metadata/curation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', default='/data/wangsq/model/Mapika-decider-2b')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.preview, args.output = args.preview.resolve(), args.output.resolve()
    if args.smoke and not args.output.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under tmp')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.output/'metadata'
    metadata.mkdir()
    (metadata/'command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable, *sys.argv])}\n')
    status = dict(status='running', started_at=timestamp(), smoke=args.smoke)
    write_json(metadata/'run_status.json', status)
    try:
        result = run(args)
    except BaseException as exc:
        status.update(status='failed', ended_at=timestamp(), error=str(exc))
        write_json(metadata/'run_status.json', status)
        raise
    status.update(status='smoke_completed' if args.smoke else 'completed', ended_at=timestamp(), count=result['count'])
    write_json(metadata/'run_status.json', status)
    print(json.dumps(dict(output=str(args.output), **status), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
