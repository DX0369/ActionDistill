from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
import write_task_reply_samples as writer
from jev_train.common import digest
from jev_train.curation import c1_plan
from jev_train.selection import load_train_ids
from curation_fixtures import sample, feature


class ReplyWriterTests(unittest.TestCase):
    def test_reviewed_respond_gate_keeps_inclusive_boundaries(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            (root/'metadata').mkdir()
            (root/'config.json').write_text('{}')
            (root/'metadata/run_status.json').write_text(json.dumps(dict(
                status='completed', smoke=False, failed_targets=[])))
            for fraction in (.08, .20, .07999, .20001):
                approved = dict(smoke=False, recipe='task_reply_v2', seed=writer.SEED,
                    config=str(root/'config.json'), targets={'38000': dict(
                        status='completed', failure_stage=None, selected=dict(
                            count=38000, respond_gate_passed=True, respond_fraction=fraction))})
                (root/'summary.json').write_text(json.dumps(approved))
                with self.subTest(fraction=fraction), patch.object(
                    writer, 'load_frozen', side_effect=RuntimeError('source load reached')
                ) as load:
                    if fraction in (.08, .20):
                        with self.assertRaisesRegex(RuntimeError, 'source load reached'):
                            writer.replay(root)
                        load.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError, '38k target/gate differs'):
                            writer.replay(root)
                        load.assert_not_called()

    def test_rejects_unapproved_or_missing_required_preview_fields_before_source_load(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            (root/'metadata').mkdir()
            good_status = dict(status='completed', smoke=False, failed_targets=[])
            good = dict(smoke=False, recipe='task_reply_v2', seed=writer.SEED,
                        targets={'38000': dict(status='completed', failure_stage=None,
                                              selected=dict(count=38000, respond_gate_passed=True,
                                                            respond_fraction=.11))})
            for change in ('failed', 'smoke', 'recipe', 'gate', 'count', 'missing_seed', 'missing_gate'):
                status, approved = deepcopy(good_status), deepcopy(good)
                if change == 'failed': status['status'] = 'failed'
                if change == 'smoke': status['smoke'] = True
                if change == 'recipe': approved['recipe'] = 'task_reply_v1'
                if change == 'gate': approved['targets']['38000']['selected']['respond_fraction'] = .07
                if change == 'count': approved['targets']['38000']['selected']['count'] = 76000
                if change == 'missing_seed': del approved['seed']
                if change == 'missing_gate': del approved['targets']['38000']['selected']['respond_gate_passed']
                (root/'summary.json').write_text(json.dumps(approved))
                (root/'metadata/run_status.json').write_text(json.dumps(status))
                with self.subTest(change=change), patch.object(writer, 'load_frozen') as load:
                    with self.assertRaises((ValueError, KeyError)):
                        writer.replay(root)
                    load.assert_not_called()

    def test_replay_refuses_ordered_ids_allocation_or_reply_priority_drift(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            (root/'metadata').mkdir()
            config = dict(reply_origin_preview='origins')
            (root/'config.json').write_text(json.dumps(config))
            (root/'metadata/run_status.json').write_text(json.dumps(dict(status='completed', smoke=False, failed_targets=[])))
            ids = ['a', 'b']
            target = dict(status='completed', failure_stage=None, selection_sha256=writer.selection_hash(ids),
                          allocation={}, selected=dict(count=38000, respond_gate_passed=True, respond_fraction=.11))
            base = dict(smoke=False, recipe='task_reply_v2', seed=writer.SEED, targets={'38000': target},
                        config=str(root/'config.json'), inputs={'reply_origin_preview': str(writer.PROJECT/'origins')},
                        boundary_audit={'zclaw_framework_boundaries': {}}, reply_priority={},
                        reference_domain_counts={}, reference_cells={})
            for change in ('order', 'allocation', 'priority'):
                approved = deepcopy(base)
                chosen = ids
                if change == 'order': chosen = list(reversed(ids))
                if change == 'allocation': approved['targets']['38000']['allocation'] = {'changed': 1}
                if change == 'priority': approved['reply_priority'] = {'a': {'eligible': True}}
                (root/'summary.json').write_text(json.dumps(approved))
                with self.subTest(change=change), \
                     patch.object(writer, 'load_frozen', return_value=({}, {}, [], [], {'cells': {}}, {}, {})), \
                     patch.object(writer, 'load_origins', return_value=({}, {})), \
                     patch.object(writer, 'load_review'), \
                     patch.object(writer, 'select_with_sources', return_value=(chosen, {})), \
                     patch.object(writer, 'summarize_selection') as summarize:
                    with self.assertRaises(ValueError):
                        writer.replay(root)
                    summarize.assert_not_called()

    def fixture(self, root):
        source = root/'samples'
        source.mkdir()
        original = [sample('a'), sample('z')]
        original[0]['options'] = ['run: 查询 (params: x)', 'other: 写入', 'RESPOND']
        original[0]['context'] = '中文原始历史\n工具结果：{ "x": 1 }'
        with (source/'train.jsonl').open('w') as handle:
            for row in original:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        plans = {}
        for row in original:
            f = feature(row, **(dict(next_event='intermediate_tool_continuation', next_tool_names=['run', 'other'])
                               if row['id'] == 'a' else {}))
            plans[row['id']] = dict(c1_plan(row, f), model=row['model'], task_group=row['task_group'],
                                    n_prompt_tokens=row['n_prompt_tokens'])
        ids = ['z', 'a']
        selected = dict(count=2, respond=1, priority_replies=1, task_groups=1, n_prompt_tokens=24,
                        by_source={'d': {'count': 2}}, by_domain={'repo_swe': {'count': 2}},
                        zclaw_framework_boundary_ids=[], hermes_nonexhausted_pressure_ids=[])
        approved = dict(config='frozen.json', inputs={'source_samples': str(source)},
                        reply_priority={'z': dict(eligible=True, origin_category='unknown')},
                        targets={'2': dict(selected=selected, selection_sha256=writer.selection_hash(ids))})
        output = root/'written'
        (output/'metadata').mkdir(parents=True)
        args = SimpleNamespace(preview=root/'preview', output=output, tokenizer='unused', smoke=False)
        return original, approved, plans, {'g': 'repo_swe'}, ids, args

    def test_write_preserves_inputs_c1_labels_and_distinct_file_and_training_orders(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            original, approved, plans, labels, ids, args = self.fixture(root)
            with patch.object(writer, 'COUNT', 2), \
                 patch.object(writer, 'replay', return_value=(approved, plans, labels, ids)), \
                 patch.object(writer, 'check_encoding', return_value=[]):
                result = writer.run(args)
            written = [json.loads(line) for line in (args.output/'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in written], ['a', 'z'])
            self.assertEqual((args.output/'train_ids.txt').read_text().splitlines(), ids)
            self.assertEqual(written[0]['gold'], [0, 1])
            self.assertEqual(written[1]['gold'], [2])
            for raw, row in zip(original, written):
                self.assertEqual({k: v for k, v in raw.items() if k != 'gold'},
                                 {k: v for k, v in row.items() if k not in {'gold', 'short_turn', 'direct'}})
                self.assertNotIn('reply_priority', row)
                self.assertNotIn('domain', row)
            index = writer.SampleIndex(args.output/'train.jsonl')
            actual = load_train_ids(index, args.output/'train_ids.txt', writer.SEED)
            self.assertEqual([r['id'] for r in actual], sorted(ids, key=lambda sid: (digest(writer.SEED, sid), sid)))
            self.assertEqual(result['composition']['respond'], 1)
            self.assertEqual(result['written_selection_sha256'], approved['targets']['2']['selection_sha256'])

    def test_readback_rejects_input_label_coverage_and_required_field_corruption(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory)
            _, approved, plans, labels, ids, args = self.fixture(root)
            index = writer.SampleIndex(Path(approved['inputs']['source_samples'])/'train.jsonl')
            offsets = {r['id']: r['offset'] for r in index.entries}
            path = args.output/'train.jsonl'
            writer.write_samples(path, ids, 'train', {'train': index}, offsets, plans, {})
            valid = [json.loads(line) for line in path.read_text().splitlines()]
            for change in ('context', 'options', 'gold', 'missing', 'extra', 'required'):
                rows = deepcopy(valid)
                if change == 'context': rows[0]['context'] += 'modified'
                if change == 'options': rows[0]['options'][0] = 'wrong'
                if change == 'gold': rows[0]['gold'] = [2]
                if change == 'missing': rows.pop()
                if change == 'extra': rows.append(deepcopy(rows[0]))
                if change == 'required': del rows[0]['question']
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    writer.readback(path, ids, index, offsets, plans, labels, approved['reply_priority'])


if __name__ == '__main__':
    unittest.main()
