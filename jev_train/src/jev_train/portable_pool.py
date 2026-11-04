"""Portable, already-C1-normalized trajectory-deduplicated selection inputs."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

FORMAT = 'jev_selection_pool_v1'
DEDUP_SEED = 20261015
DEFAULTS = dict(recipe='task_reply_v2', count=38000, seed=20261015,
                domain_uniform_mix=0.5, source_weight_power=0.5, reply_priority_weight=2)
ANNOTATION_FIELDS = {'domain', 'domain_label', 'reply_origin', 'reply_priority', 'boundary_evidence'}
BOUNDARY_FIELDS = {'checked', 'unverifiable_current_boundary', 'zclaw_framework_boundary',
                   'hermes_nonexhausted_pressure', 'hermes_diagnostic_unknown'}


def selection_hash(ids):
    return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()


def file_identity(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            checksum.update(block)
    return dict(size_bytes=Path(path).stat().st_size, sha256=checksum.hexdigest())


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != set(DEFAULTS):
        raise ValueError(f'recipe requires exactly {sorted(DEFAULTS)}')
    if recipe['recipe'] != 'task_reply_v2':
        raise ValueError('unsupported selection recipe')
    for key in ('count', 'reply_priority_weight'):
        if type(recipe[key]) is not int or recipe[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(recipe['seed']) is not int:
        raise ValueError('selection seed must be an integer')
    for key in ('domain_uniform_mix', 'source_weight_power'):
        value = recipe[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
    if recipe['domain_uniform_mix'] > 1:
        raise ValueError('domain_uniform_mix must be in [0,1]')
    return recipe


def package_file(root, manifest, name):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('package file escapes the data root')
    identity = manifest['files'][name]
    if not path.is_file() or path.stat().st_size != identity['size_bytes']:
        raise ValueError(f'package file missing or size differs: {name}')
    return path


def verify_files(root):
    """Full transfer verification, separate from ordinary metadata-only selection."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_identity(package_file(root, manifest, name)) != expected:
            raise ValueError(f'package checksum differs: {name}')


def load_pool(root):
    """Read frozen annotations only; never recover raw histories or deduplicate."""
    from .domains import LABELS
    from .reply_origin import reply_priority
    from .task_selection import validate_inputs

    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest['format'] != FORMAT or manifest['dedup_seed'] != DEDUP_SEED
            or type(manifest['smoke']) is not bool):
        raise ValueError('unsupported pool format/dedup seed/smoke status')
    package_file(root, manifest, 'pool/train.jsonl')
    reference_path = package_file(root, manifest, 'pool/selection_reference.json')
    if file_identity(reference_path) != manifest['files']['pool/selection_reference.json']:
        raise ValueError('selection reference checksum differs')
    reference = json.loads(reference_path.read_text())
    if reference['dedup_seed'] != DEDUP_SEED or reference['selection_defaults'] != DEFAULTS:
        raise ValueError('frozen reference protocol differs')
    acceptance = reference['label_acceptance']
    if (acceptance['authority'] != 'user' or acceptance['decision'] != 'accept_existing_labels'
            or not isinstance(acceptance['reason'], str) or not acceptance['reason'].strip()
            or datetime.fromisoformat(acceptance['accepted_at']).utcoffset() is None):
        raise ValueError('frozen domain-label acceptance differs')
    review = reference['reply_boundary_review']
    if (review['protocol'] != 'task_reply_v2_boundary_review' or review['pending_policy_questions'] != []
            or review['policy_decisions'] != dict(
                hermes_nonexhausted_final_now='retain_priority_with_diagnostic',
                zclaw_framework_boundaries='exclude_explicit_notification_boundaries')):
        raise ValueError('frozen reply-boundary decisions differ')
    cells = reference['reference_cells']
    if set(cells) != set(LABELS):
        raise ValueError('reference must cover the seven frozen domains')
    if (reference['reference_domain_counts'] != {d: sum(v.values()) for d, v in cells.items()}
            or any(type(n) is not int or n < 0 for v in cells.values() for n in v.values())):
        raise ValueError('invalid frozen source/domain counts')
    annotation_path = package_file(root, manifest, 'pool/annotations.jsonl')
    records, labels, origins, framework = {}, {}, {}, set()
    checksum = hashlib.sha256()
    with annotation_path.open('rb') as handle:
        for line in handle:
            checksum.update(line)
            annotation = json.loads(line)
            if not ANNOTATION_FIELDS <= annotation.keys():
                raise ValueError('annotation lacks required frozen evidence fields')
            plan = {k: v for k, v in annotation.items() if k not in ANNOTATION_FIELDS}
            sid, task = plan['id'], plan['task_group']
            if sid in records:
                raise ValueError(f'duplicate annotated ID: {sid}')
            label = annotation['domain_label']
            if (label['task_group'] != task or label['split'] != 'train'
                    or label['label'] != annotation['domain']
                    or not isinstance(label['request_sha256'], str) or not label['request_sha256']):
                raise ValueError(f'{sid}: domain label identity differs')
            if task in labels and labels[task] != label['label']:
                raise ValueError(f'{sid}: conflicting task domain')
            labels[task] = label['label']
            origin, priority, boundary = (annotation[k] for k in
                                         ('reply_origin', 'reply_priority', 'boundary_evidence'))
            if plan['gold_names'] == ['RESPOND']:
                if (not isinstance(origin, dict) or not isinstance(priority, dict)
                        or not isinstance(boundary, dict) or set(boundary) != BOUNDARY_FIELDS):
                    raise ValueError(f'{sid}: reply state lacks frozen evidence')
                for key in ('dataset', 'model', 'task_group', 'trajectory_id', 'turn_index', 'next_event'):
                    if origin[key] != plan[key]:
                        raise ValueError(f'{sid}: reply identity differs: {key}')
                for key in ('checked', 'unverifiable_current_boundary'):
                    if type(boundary[key]) is not bool:
                        raise ValueError(f'{sid}: boundary {key} must be boolean')
                is_framework = boundary['zclaw_framework_boundary'] is not None
                if is_framework and not isinstance(boundary['zclaw_framework_boundary'], dict):
                    raise ValueError(f'{sid}: invalid framework boundary evidence')
                expected = dict(origin_category=origin['category'],
                                **reply_priority(origin, framework_boundary=is_framework))
                if priority != expected:
                    raise ValueError(f'{sid}: frozen reply priority differs')
                if is_framework:
                    framework.add(sid)
                origins[sid] = origin
            elif origin is not None or priority is not None or boundary is not None:
                raise ValueError(f'{sid}: tool state must not carry reply evidence')
            records[sid] = plan
    if checksum.hexdigest() != manifest['files']['pool/annotations.jsonl']['sha256']:
        raise ValueError('annotation checksum differs')
    validate_inputs(records, labels, {sid: o['category'] for sid, o in origins.items()})
    if (type(manifest['count']) is not int or type(manifest['respond_count']) is not int
            or len(records) != manifest['count']
            or len(origins) != manifest['respond_count']):
        raise ValueError('pool population differs from manifest')
    return manifest, reference, records, labels, origins, framework


def select_pool(root, recipe):
    from .task_selection import select_with_sources

    validate_recipe(recipe)
    manifest, reference, records, labels, origins, framework = load_pool(root)
    ids, allocation = select_with_sources(
        records, labels, reference['reference_cells'], origins, recipe['count'], recipe['seed'],
        framework_boundary_ids=framework, domain_uniform_mix=recipe['domain_uniform_mix'],
        source_weight_power=recipe['source_weight_power'], reply_priority_weight=recipe['reply_priority_weight'])
    selected = [records[sid] for sid in ids]
    summary = dict(recipe=recipe, smoke=manifest['smoke'], dedup_seed=manifest['dedup_seed'],
                   selection_sha256=selection_hash(ids), allocation=allocation,
                   count=len(ids), sources=dict(Counter(r['dataset'] for r in selected)),
                   domains=dict(Counter(labels[r['task_group']] for r in selected)),
                   respond=sum(r['gold_names'] == ['RESPOND'] for r in selected),
                   task_groups=len({r['task_group'] for r in selected}),
                   n_prompt_tokens=sum(r['n_prompt_tokens'] for r in selected))
    summary['respond_fraction'] = summary['respond']/len(ids)
    summary['respond_gate_passed'] = .08 <= summary['respond_fraction'] <= .20
    return ids, summary
