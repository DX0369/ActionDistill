#!/usr/bin/env python3
"""Export frozen C1/dedup results; full export is a separate user-started CPU run."""
import argparse
from datetime import datetime
from itertools import islice
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'scripts'), str(WORKSPACE/'third_party/decider'),
               str(PROJECT.parent/'dataset_tool_inventory/src')]

from jev_train.common import rows, write_json
from jev_train.curation import materialize
from jev_train.data import SampleIndex, validate
from jev_train.portable_pool import (DEFAULTS, DEDUP_SEED, FORMAT, file_identity, load_pool,
                                     select_pool, selection_hash)
from jev_train.reply_origin import reply_priority
from jev_train.task_selection import select
from preview_task_reply import summarize_selection
from preview_task_reply_v2 import add_v2_diagnostics, unit_positions
from write_task_reply_samples import replay, smoke_ids


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_eval_source(directory, approved):
    parent = read_json(Path(approved['inputs']['dedup_preview'])/'summary.json')
    require(directory.resolve() == Path(parent['inputs']['normalized']).resolve(),
            'eval directory differs from the frozen normalized input')
    require(read_json(directory/'metadata/run_status.json')['status'] == 'completed',
            'normalized eval input is incomplete')
    curation = read_json(directory/'metadata/curation.json')
    require(curation['rule_version'] == 'C1-C4-20261025-preview', 'normalized eval C1 rule differs')
    for split in ('val', 'test_id', 'test_ood'):
        require((directory/f'{split}.jsonl').is_file(), f'missing normalized eval split: {split}')
    return curation


def load_export_inputs(args):
    approved = read_json(args.preview/'summary.json')
    state = read_json(args.preview/'metadata/run_status.json')
    require(state['status'] == 'completed' and not state['smoke'] and not state['failed_targets']
            and approved['recipe'] == 'task_reply_v2' and not approved['smoke']
            and approved['seed'] == DEDUP_SEED, 'requires an approved completed v2 preview')
    ids38 = (args.samples_38k/'train_ids.txt').read_text().splitlines()
    require(len(ids38) == len(set(ids38)) == 38000
            and selection_hash(ids38) == approved['targets']['38000']['selection_sha256'],
            'delivered 38k IDs differ from the approved preview')
    curation = read_json(args.samples_38k/'metadata/curation.json')
    require(curation['recipe'] == 'task_reply_v2' and not curation['smoke']
            and curation['approved_selection_sha256'] == selection_hash(ids38), '38k delivery identity differs')
    inputs = approved['inputs']
    validate_eval_source(args.eval_samples, approved)
    label_records = {row['task_group']: row for row in rows(Path(inputs['labels'])/'labels.jsonl')}
    labels = {task: row['label'] for task, row in label_records.items()}
    if args.smoke:
        # Inspect saved plans only; no full selection/dedup replay and no raw parsing.
        wanted = set(ids38)
        plans = {}
        for row in rows(Path(inputs['canonical_states'])):
            if row['id'] in wanted:
                require(row['id'] not in plans and not row['drops'], 'duplicate/discarded 38k plan')
                plans[row['id']] = row
        require(plans.keys() == wanted, '38k canonical plan coverage differs')
        ids = smoke_ids(ids38, plans, labels, approved)
        # A bounded, direct equality check against the delivered JSONL.
        delivered = list(islice(rows(args.samples_38k/'train.jsonl'), 16))
        ids = list(dict.fromkeys([*ids, *(row['id'] for row in delivered)]))
        plans = {sid: plans[sid] for sid in ids}
    else:
        _, plans, restored_labels, replayed = replay(args.preview)
        require(replayed == ids38 and all(labels[r['task_group']] == restored_labels[r['task_group']]
                                         for r in plans.values()), 'replayed delivery/labels differ')
        delivered = None
        require(len(plans) == 150234, 'full frozen dedup population differs')
    old = read_json(Path(inputs['reply_origin_preview'])/'summary.json')
    origins = {sid: old['reply_classification'][sid] for sid, row in plans.items()
               if row['gold_names'] == ['RESPOND']}
    r00 = (Path(inputs['baseline_samples'])/'train_ids.txt').read_text().splitlines()
    r11 = (Path(inputs['r11_samples'])/'train_ids.txt').read_text().splitlines()
    return approved, plans, labels, label_records, origins, ids38, r00, r11, delivered


def annotation(plan, label_records, origins, approved):
    sid = plan['id']
    origin = origins[sid] if plan['gold_names'] == ['RESPOND'] else None
    audit = approved['boundary_audit']
    boundary = None
    priority = None
    if origin is not None:
        boundary = dict(checked=sid in audit['checked_ids'],
                        unverifiable_current_boundary=sid in audit['unverifiable_current_boundary_ids'],
                        zclaw_framework_boundary=audit['zclaw_framework_boundaries'].get(sid),
                        hermes_nonexhausted_pressure=audit['hermes_nonexhausted_pressure'].get(sid),
                        hermes_diagnostic_unknown=audit['hermes_pressure_diagnostic_unknown'].get(sid))
        priority = dict(origin_category=origin['category'], **reply_priority(
            origin, framework_boundary=boundary['zclaw_framework_boundary'] is not None))
        require(priority == approved['reply_priority'][sid], f'{sid}: priority differs from review')
    label = label_records[plan['task_group']]
    return dict(plan, domain=label['label'], domain_label=label, reply_origin=origin,
                reply_priority=priority, boundary_evidence=boundary)


def reference(approved, r00, r11, eval_curation):
    return dict(dedup_seed=DEDUP_SEED, selection_defaults=DEFAULTS,
                reference_cells=approved['reference_cells'],
                reference_domain_counts=approved['reference_domain_counts'],
                label_acceptance=approved['label_acceptance'],
                reply_boundary_review=approved['reply_boundary_review'],
                boundary_audit_summary={k: approved['boundary_audit'][k] for k in ('by_source', 'summary')},
                golden_targets=approved['targets'], comparisons=approved['comparisons'],
                expected_population=approved['candidate_population'],
                supported_population=approved['supported_candidates'], r00_ids=r00, r11_ids=r11,
                definitions=approved['definitions'],
                source_provenance=dict(inputs=approved['inputs'], config=approved['config'],
                                       eval_curation=eval_curation))


def write_jsonl_row(handle, row):
    handle.write((json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n').encode())


def write_pool(root, index, plans, label_records, origins, approved):
    offsets = {row['id']: row['offset'] for row in index.entries if row['id'] in plans}
    require(offsets.keys() == plans.keys(), 'source body coverage differs from annotations')
    new_offsets = {}
    with index.path.open('rb') as source, (root/'pool/train.jsonl').open('wb') as body, \
            (root/'pool/annotations.jsonl').open('wb') as notes:
        for sid in sorted(plans, key=offsets.__getitem__):
            source.seek(offsets[sid])
            raw = validate(json.loads(source.readline()))
            plan = plans[sid]
            require(raw['id'] == sid and all(raw[k] == plan[k] for k in
                    ('dataset', 'model', 'task_group', 'outcome', 'n_prompt_tokens')),
                    f'{sid}: source/plan identity differs')
            require([index_name(raw, i) for i in raw['gold']] == plan['original_gold'],
                    f'{sid}: original gold names differ')
            require([index_name(raw, i) for i in plan['gold']] == plan['gold_names'],
                    f'{sid}: C1 gold names differ')
            new_offsets[sid] = body.tell()
            write_jsonl_row(body, materialize(raw, plan))
            write_jsonl_row(notes, annotation(plan, label_records, origins, approved))
    # Full body readback, using an existing source handle and bounded row memory.
    with index.path.open('rb') as source:
        seen = set()
        for row in rows(root/'pool/train.jsonl'):
            sid = row['id']
            require(sid not in seen and sid in offsets, 'written ID duplicate/outside pool')
            source.seek(offsets[sid])
            require(row == materialize(json.loads(source.readline()), plans[sid]), f'{sid}: body readback differs')
            seen.add(sid)
    require(seen == plans.keys(), 'body readback coverage differs')
    return new_offsets


def index_name(row, index):
    from jev_train.data import tool_name
    return tool_name(row['options'][index])


def compare_delivery(root, offsets, args, delivered):
    expected = delivered if args.smoke else rows(args.samples_38k/'train.jsonl')
    seen = set()
    with (root/'pool/train.jsonl').open('rb') as handle:
        for row in expected:
            sid = row['id']
            require(sid not in seen and sid in offsets, 'delivered ID missing/duplicate')
            handle.seek(offsets[sid])
            require(json.loads(handle.readline()) == row, f'{sid}: full row differs from delivered 38k')
            seen.add(sid)
    if not args.smoke:
        require(seen == set((args.samples_38k/'train_ids.txt').read_text().splitlines()), '38k body/ID coverage differs')
    return len(seen)


def replay_goldens(root, approved, plans, labels, origins, r11):
    positions = unit_positions(plans)
    for count in (8000, 38000, 76000):
        ids, result = select_pool(root, dict(DEFAULTS, count=count))
        target = approved['targets'][str(count)]
        require(result['selection_sha256'] == target['selection_sha256']
                and result['allocation'] == target['allocation'], f'{count}: portable IDs/allocation differ')
        selected = summarize_selection(ids, plans, origins, labels, positions, r11)
        add_v2_diagnostics(selected, ids, origins, plans, approved['boundary_audit'])
        old_ids, _ = select(plans, labels, approved['reference_domain_counts'],
                            {sid: origin['category'] for sid, origin in origins.items()}, count, DEDUP_SEED)
        selected['overlap_with_v1_same_size'] = len(set(ids) & set(old_ids))
        require(json.loads(json.dumps(selected)) == target['selected'], f'{count}: complete golden statistics differ')
        print(f'{count}: portable ordered IDs, allocation and full statistics match', flush=True)


def run(args):
    dirty = bool(subprocess.check_output(
        ['git', '-C', str(PROJECT.parent), 'status', '--porcelain', '--',
         'jev_train/src', 'jev_train/scripts', 'jev_train/configs/selection'], text=True).strip())
    require(args.smoke or not dirty, 'commit the construction/selection code before the full export')
    approved, plans, labels, label_records, origins, ids38, r00, r11, delivered = load_export_inputs(args)
    source_path = Path(approved['inputs']['source_samples'])/'train.jsonl'
    cache = source_path.parent.parent/'index_cache'/source_path.parent.name/'train.json'
    require(not args.smoke or cache.is_file(), 'bounded smoke requires the existing source index; no full scan')
    index = SampleIndex(source_path)
    for name in ('pool', 'eval', 'selections', 'recipes'):
        (args.output/name).mkdir()
    offsets = write_pool(args.output, index, plans, label_records, origins, approved)
    compared = compare_delivery(args.output, offsets, args, delivered)
    for split in ('val', 'test_id', 'test_ood'):
        with (args.eval_samples/f'{split}.jsonl').open('rb') as source, \
                (args.output/'eval'/f'{split}.jsonl').open('wb') as target:
            if args.smoke:
                target.writelines(islice(source, 2))
            else:
                for block in iter(lambda: source.read(1024*1024), b''):
                    target.write(block)
    write_json(args.output/'pool/selection_reference.json', reference(
        approved, r00, r11, read_json(args.eval_samples/'metadata/curation.json')))
    selected_ids = [sid for sid in ids38 if sid in plans]
    (args.output/'selections/task_reply_v2_38k_ids.txt').write_text('\n'.join(selected_ids)+'\n')
    write_json(args.output/'recipes/task_reply_v2_38k.json', DEFAULTS)
    (args.output/'README.md').write_text(
        '# Frozen Jev selection pool\n\n'
        'C1-normalized, trajectory-deduplicated text samples. Annotations are selection evidence, not model input.\n'
        'dedup_seed=20261015; selection and training seeds are separate. Raw data is not needed for pool selection.\n'
        + ('\nThis is a partial engineering smoke package; never use it as formal training data.\n' if args.smoke else ''))
    files = {str(path.relative_to(args.output)): file_identity(path) for path in sorted(args.output.rglob('*'))
             if path.is_file() and 'metadata' not in path.relative_to(args.output).parts}
    manifest = dict(format=FORMAT, smoke=args.smoke, dedup_seed=DEDUP_SEED, count=len(plans),
                    respond_count=len(origins), files=files,
                    construction_code=subprocess.check_output(['git', '-C', str(PROJECT.parent), 'rev-parse', 'HEAD'], text=True).strip(),
                    construction_dirty=dirty,
                    tokenizer=dict(path=str(args.tokenizer.resolve()),
                                   files={name: file_identity(args.tokenizer/name) for name in
                                          ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}),
                    validation=dict(all_rows_readback_equal=True, delivered_rows_equal=compared,
                                    golden_replays=[]))
    write_json(args.output/'manifest.json', manifest)
    # Exercise the new portable boundary even for a partial smoke package.
    load_pool(args.output)
    if not args.smoke:
        replay_goldens(args.output, approved, plans, labels, origins, r11)
        manifest['validation']['golden_replays'] = [8000, 38000, 76000]
        write_json(args.output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--samples-38k', type=Path, required=True)
    parser.add_argument('--eval-samples', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.smoke and not args.output.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under jev_train/tmp')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.output/'metadata'
    metadata.mkdir()
    (metadata/'command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable, *sys.argv])}\n')
    status = dict(status='running', started_at=datetime.now().astimezone().isoformat(), smoke=args.smoke)
    write_json(metadata/'run_status.json', status)
    try:
        result = run(args)
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      ended_at=datetime.now().astimezone().isoformat())
        write_json(metadata/'run_status.json', status)
        raise
    status.update(status='completed', ended_at=datetime.now().astimezone().isoformat(), count=result['count'])
    write_json(metadata/'run_status.json', status)
    print(args.output)


if __name__ == '__main__':
    main()
