from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
from jev_train.common import write_json
from jev_train.curation import c1_plan, materialize
from jev_train.data import SampleIndex
from jev_train.domains import LABELS
from jev_train.portable_pool import (DEFAULTS, FORMAT, DEDUP_SEED, file_identity,
                                    load_pool, select_pool, validate_recipe, verify_files)
from jev_train.task_selection import select_with_sources
from curation_fixtures import sample, feature
from test_task_reply_v2 import origin
import export_selection_pool as exporter


def fixture(root):
    source = root/'source'
    source.mkdir()
    plans, labels, label_records, origins = {}, {}, {}, {}
    raw_rows = []
    for domain in LABELS:
        for ds in ('a', 'b'):
            task = f'{domain}/{ds}'
            labels[task] = domain
            label_records[task] = dict(task_group=task, split='train', label=domain, request_sha256='f'*64)
            for i in range(6):
                sid = f'{task}/{i}'
                raw = dict(sample(sid, gold=[2] if i in (0, 1) else [0]), dataset=ds, task_group=task)
                evidence = feature(raw, **(dict(next_event='intermediate_tool_continuation',
                                                next_tool_names=['run', 'other']) if i == 1 else {}))
                plan = dict(c1_plan(raw, evidence), model=raw['model'], task_group=task,
                            trajectory_id=task, turn_index=i, outcome=raw['outcome'],
                            n_prompt_tokens=raw['n_prompt_tokens'], split='train')
                plans[sid] = plan
                if i == 0:
                    origins[sid] = dict(origin(), id=sid, **{k: plan[k] for k in
                        ('dataset', 'model', 'task_group', 'trajectory_id', 'turn_index', 'next_event')})
                raw_rows.append(raw)
    (source/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw_rows))
    audit = dict(checked_ids=list(origins), unverifiable_current_boundary_ids=[],
                 zclaw_framework_boundaries={}, hermes_nonexhausted_pressure={},
                 hermes_pressure_diagnostic_unknown={})
    approved = dict(boundary_audit=audit, reply_priority={sid: dict(origin_category='unknown',
                    eligible=True, rule='verifiable_reply_boundary') for sid in origins})
    pool = root/'package'
    (pool/'pool').mkdir(parents=True)
    index = SampleIndex(source/'train.jsonl')
    offsets = exporter.write_pool(pool, index, plans, label_records, origins, approved)
    reference = dict(dedup_seed=DEDUP_SEED, selection_defaults=DEFAULTS,
                     reference_cells={d: {'a': 4, 'b': 1} for d in LABELS},
                     reference_domain_counts=dict.fromkeys(LABELS, 5),
                     label_acceptance=dict(authority='user', decision='accept_existing_labels',
                                           reason='synthetic test fixture', accepted_at='2026-11-03T12:00:00+08:00'),
                     reply_boundary_review=dict(protocol='task_reply_v2_boundary_review', pending_policy_questions=[],
                         policy_decisions=dict(hermes_nonexhausted_final_now='retain_priority_with_diagnostic',
                                               zclaw_framework_boundaries='exclude_explicit_notification_boundaries')),
                     source_provenance={'old_path': '/does/not/exist'})
    write_json(pool/'pool/selection_reference.json', reference)
    (pool/'metadata').mkdir()
    write_json(pool/'metadata/run_status.json', dict(status='completed', smoke=True))
    manifest = dict(format=FORMAT, dedup_seed=DEDUP_SEED, count=len(plans), respond_count=len(origins),
                    smoke=True, files={str(p.relative_to(pool)): file_identity(p) for p in (pool/'pool').iterdir()})
    write_json(pool/'manifest.json', manifest)
    return pool, source, plans, labels, origins, reference, offsets, raw_rows


class PortablePoolTests(unittest.TestCase):
    def test_relocated_pool_defaults_weights_and_no_raw_dependency(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            pool, source, plans, labels, origins, ref, _, _ = fixture(root)
            expected, allocation = select_with_sources(plans, labels, ref['reference_cells'], origins, 42, 20261015)
            shutil.rmtree(source)
            moved = root/'another_root'/'data'
            moved.parent.mkdir()
            pool.rename(moved)
            verify_files(moved)
            ids, report = select_pool(moved, dict(DEFAULTS, count=42))
            self.assertEqual(ids, expected)
            self.assertEqual(report['allocation'], allocation)
            self.assertEqual(report['selection_sha256'],
                             '3285241c57dd8ce70b6ff16c547a467cb6580a3256ac8bb8236b5d2e7930b5ac')
            # The supported sources stay fixed even when power is zero.
            changed, report = select_pool(moved, dict(DEFAULTS, count=42, source_weight_power=0,
                                                    domain_uniform_mix=1, reply_priority_weight=1))
            self.assertEqual(report['sources'], {'a': 21, 'b': 21})
            self.assertNotEqual(changed, ids)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(load_pool(moved)[0]['dedup_seed'], 20261015)

    def test_required_recipe_evidence_identity_and_checksum_fail_explicitly(self):
        for key in DEFAULTS:
            recipe = dict(DEFAULTS)
            del recipe[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_recipe(recipe)
        for key, value in [('count', 0), ('seed', True), ('domain_uniform_mix', 1.1),
                           ('source_weight_power', float('nan')), ('source_weight_power', -1),
                           ('reply_priority_weight', 2.5)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_recipe(dict(DEFAULTS, **{key: value}))
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            pool, *_ = fixture(Path(directory))
            path = pool/'pool/annotations.jsonl'
            valid = path.read_text()
            manifest = json.loads((pool/'manifest.json').read_text())
            for mutation in ('required', 'label', 'priority', 'duplicate'):
                records = [json.loads(line) for line in valid.splitlines()]
                if mutation == 'required': del records[0]['boundary_evidence']
                if mutation == 'label': records[0]['domain_label']['task_group'] = 'wrong'
                if mutation == 'priority': records[0]['reply_priority']['eligible'] = False
                if mutation == 'duplicate': records.append(deepcopy(records[0]))
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                manifest['files']['pool/annotations.jsonl'] = file_identity(path)
                write_json(pool/'manifest.json', manifest)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_pool(pool)
            path.write_text(valid)
            manifest['files']['pool/annotations.jsonl'] = file_identity(path)
            write_json(pool/'manifest.json', manifest)
            path.write_text(valid.replace('repo_swe', 'repo_xyz'))
            with self.assertRaises(ValueError):
                verify_files(pool)

    def test_body_is_existing_materialize_and_delivery_matches_all_fields(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            pool, _, plans, _, _, _, offsets, raw = fixture(Path(directory))
            written = [json.loads(line) for line in (pool/'pool/train.jsonl').read_text().splitlines()]
            self.assertEqual(written, [materialize(row, plans[row['id']]) for row in raw])
            self.assertEqual(written[1]['gold'], [0, 1])
            self.assertTrue(all('domain' not in row and 'reply_origin' not in row for row in written))
            delivery = Path(directory)/'delivered'
            delivery.mkdir()
            (delivery/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in written))
            (delivery/'train_ids.txt').write_text('\n'.join(r['id'] for r in written)+'\n')
            args = SimpleNamespace(samples_38k=delivery, smoke=False)
            self.assertEqual(exporter.compare_delivery(pool, offsets, args, None), len(written))
            written[0]['context'] += ' changed'
            (delivery/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in written))
            with self.assertRaisesRegex(ValueError, 'full row differs'):
                exporter.compare_delivery(pool, offsets, args, None)

    def test_cli_records_partial_status_and_print_only_migration_plan(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            pool, *_ = fixture(root)
            config = root/'recipe.json'
            write_json(config, dict(DEFAULTS, count=42))
            output = root/'selection'
            subprocess.run([sys.executable, '-B', str(P/'scripts/select_pool.py'), '--pool', str(pool),
                            '--config', str(config), '--output', str(output), '--verify-files'],
                           check=True, capture_output=True, text=True)
            status = json.loads((output/'metadata/run_status.json').read_text())
            self.assertEqual((status['status'], status['smoke'], status['count']), ('completed', True, 42))
            manifest = json.loads((pool/'manifest.json').read_text())
            manifest['smoke'] = False
            write_json(pool/'manifest.json', manifest)
            write_json(pool/'metadata/run_status.json', dict(status='completed', smoke=False))
            # A one-state selection is necessarily 0% or 100% RESPOND.
            write_json(config, dict(DEFAULTS, count=1))
            failed = root/'failed_selection'
            response = subprocess.run([sys.executable, '-B', str(P/'scripts/select_pool.py'), '--pool', str(pool),
                                       '--config', str(config), '--output', str(failed)], capture_output=True, text=True)
            self.assertNotEqual(response.returncode, 0)
            self.assertFalse((failed/'train_ids.txt').exists())
            self.assertFalse(json.loads((failed/'summary.json').read_text())['respond_gate_passed'])
            self.assertEqual(json.loads((failed/'metadata/run_status.json').read_text())['status'], 'failed')
            script = P.parent/'migration/prepare.py'
            spec = spec_from_file_location('migration_prepare', script)
            migration = module_from_spec(spec)
            spec.loader.exec_module(migration)
            assets = json.loads(script.with_name('assets.json').read_text())
            migration.validate_assets(assets)
            for field, value in [('revision', ''), ('path', '../escape'), ('source', None)]:
                invalid = deepcopy(assets)
                invalid['models'][0][field] = value
                with self.assertRaises(ValueError):
                    migration.validate_assets(invalid)
            target = root/'target with spaces'
            lines = migration.plan(assets, target, root/'models', root/'package', 'sources')
            self.assertFalse(target.exists())
            text = '\n'.join(lines)
            self.assertIn('tau2-bench-verified.bundle', text)
            self.assertIn('checkout --detach 3d82db8', text)
            self.assertIn("'"+str(target/'third_party')+"'", text)

    def test_export_requires_the_frozen_completed_normalized_eval(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            dedup = root/'dedup'
            dedup.mkdir()
            evaluation = root/'normalized'
            (evaluation/'metadata').mkdir(parents=True)
            write_json(dedup/'summary.json', {'inputs': {'normalized': str(evaluation)}})
            write_json(evaluation/'metadata/run_status.json', {'status': 'completed'})
            write_json(evaluation/'metadata/curation.json', {'rule_version': 'C1-C4-20261025-preview'})
            for split in ('val', 'test_id', 'test_ood'):
                (evaluation/f'{split}.jsonl').write_text('{}\n')
            approved = {'inputs': {'dedup_preview': str(dedup)}}
            exporter.validate_eval_source(evaluation, approved)
            with self.assertRaisesRegex(ValueError, 'frozen normalized'):
                exporter.validate_eval_source(root/'wrong', approved)
            write_json(evaluation/'metadata/run_status.json', {'status': 'failed'})
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                exporter.validate_eval_source(evaluation, approved)
            write_json(evaluation/'metadata/run_status.json', {'status': 'completed'})
            write_json(evaluation/'metadata/curation.json', {'rule_version': 'unapproved'})
            with self.assertRaisesRegex(ValueError, 'C1 rule'):
                exporter.validate_eval_source(evaluation, approved)


if __name__ == '__main__':
    unittest.main()
