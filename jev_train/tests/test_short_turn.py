"""Exact joins and semantic boundaries for read-only short-turn analysis."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
from analyze_short_turn import (classify, join_one, audit_split, read_evaluation, digest,
                                prediction_rows, summarize, mcnemar)
import torch


def sample(n=0):
    return dict(id=digest('dataset', 'trajectory', n), dataset='dataset', model='teacher', trajectory_id='trajectory',
                turn_index=n, options=['tool: description', 'RESPOND'], gold=[1], task_group='group',
                outcome='success', candidate_level='S', context='', question='Next?', n_prompt_tokens=10,
                messages_dropped=0, truncation_counts={})


def feature(row, category='user_turn_end', calls=3, termination='stop'):
    return dict(**{k: row[k] for k in ('id', 'dataset', 'model', 'trajectory_id', 'turn_index')},
                category=category, previous_turn_tool_calls=calls, call_count_status='known' if calls is not None else 'unknown',
                termination_status='known' if termination is not None else 'missing', termination_value=termination,
                termination_evidence=[dict(present=termination is not None, value=termination)], populations=['val'])


class Index:
    def __init__(self, rows):
        self.rows = rows
        self.entries = [dict(id=r['id'], offset=i) for i, r in enumerate(rows)]
    def read(self, offset):
        return self.rows[offset]


def dump(row):
    logits = torch.full((1, 255), -float('inf')); logits[0, :2] = torch.tensor([2., 1.])
    targets = torch.zeros((1, 255)); targets[0, 1] = 1.
    return dict(logits=logits, targets=targets, id=[row['id']], dataset=['dataset'], outcome=['success'],
                candidate_level=['S'], nopts=[2], tool_names=[['tool', 'RESPOND']], transition=['change'], copy_prev_hit=[False])


class ShortTurnTests(unittest.TestCase):
    def test_exclusive_boundaries_and_priority(self):
        row = sample()
        self.assertEqual(classify(feature(row, calls=3))[0], 'short_turn')
        self.assertEqual(classify(feature(row, calls=4))[0], 'long_turn_or_end')
        self.assertEqual(classify(feature(row, category='trajectory_end'))[0], 'short_end')
        self.assertEqual(classify(feature(row, calls=0))[0], 'no_prior_call')
        self.assertEqual(classify(feature(row, category='intermediate_tool_continuation', calls=0))[0], 'mid_commentary')
        self.assertEqual(classify(feature(row, category='unjudgeable', calls=0))[0], 'unknown')
        self.assertEqual(classify(feature(row, category='intermediate_tool_continuation', calls=None))[0], 'unknown')
        for terminal in ('error', 'length'):
            self.assertEqual(classify(feature(row, termination=terminal)), ('unknown', 'abnormal_short_user_turn'))
        self.assertEqual(classify(feature(row, termination=None))[0], 'short_turn')
        conflict = feature(row); conflict.update(termination_status='conflict', termination_value=None)
        self.assertEqual(classify(conflict), ('unknown', 'termination_conflict'))
        self.assertEqual(classify({**conflict, 'category': 'trajectory_end'})[0], 'short_end')
        with self.assertRaises(ValueError): classify(feature(row, calls=True))

    def test_unmatched_is_reported_not_dropped(self):
        rows = [sample(0), sample(1)]
        features = {rows[0]['id']: feature(rows[0])}
        report, selected, joined = audit_split(Index(rows), Index(rows), features, 'val', [r['id'] for r in rows])
        self.assertEqual(report['population'], dict(gold_respond=2, matched=1, unmatched=1, unmatched_fraction=.5))
        self.assertTrue(report['stop'])
        self.assertEqual(report['unmatched'], [dict(id=rows[1]['id'], reason='feature_id_missing')])
        self.assertEqual(len(selected), 2)
        self.assertEqual(report['population_groups']['unknown'], 1)
        self.assertNotIn(rows[1]['id'], joined)

    def test_join_conflicts_and_required_fields(self):
        row = sample(); f = feature(row)
        self.assertEqual(join_one(row, 'val', f, row), (f, None))
        original = {**row, 'gold': [0]}
        self.assertEqual(join_one(row, 'val', f, original)[1], 'original_gold_not_respond')
        bad = deepcopy(f); del bad['termination_status']
        self.assertIn('feature_fields_missing', join_one(row, 'val', bad, row)[1])
        bad = {**f, 'termination_value': None}
        self.assertEqual(join_one(row, 'val', bad, row)[1], 'termination_evidence_conflict_with_saved_fields')
        self.assertEqual(join_one(row, 'test_id', f, row)[1], 'feature_split_differs')

    def test_incomplete_eval_and_missing_temperature_fail(self):
        with tempfile.TemporaryDirectory(prefix='short_turn_test_', dir=PROJECT/'tmp') as d:
            directory = Path(d); (directory/'metadata').mkdir()
            status = dict(status='running', configuration=dict(samples=str(directory), seed=20261015,
                limits=dict(val=1000, test_id=1000, test_ood=1000)))
            path = directory/'metadata/run_status.json'; path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, 'not completed'): read_evaluation(directory, directory)
            status['status'] = 'completed'; path.write_text(json.dumps(status))
            with self.assertRaises(FileNotFoundError): read_evaluation(directory, directory)
            for t in (0, True, float('nan')):
                (directory/'temperature.json').write_text(json.dumps(dict(temperature=t)))
                with self.assertRaises(ValueError): read_evaluation(directory, directory)

    def test_saved_probabilities_and_candidate_validation(self):
        row = sample(); data = dump(row)
        result = prediction_rows(data, 2., {row['id']: row}, {row['id']: feature(row)})
        r = result[row['id']]
        self.assertEqual(r['respond_rank'], 2)
        self.assertAlmostEqual(r['p_respond'], 1/(1+__import__('math').exp(.5)), places=6)
        self.assertEqual(summarize(result.values())['groups']['short_turn']['count'], 1)
        data['tool_names'][0][0] = 'substituted'
        with self.assertRaisesRegex(ValueError, 'candidate names'): prediction_rows(data, 2., {row['id']: row}, {})

    def test_mcnemar_direction_no_discordance_and_empty(self):
        a = {str(i): dict(gold_respond=True, group='short_turn', predicted_respond=True) for i in range(3)}
        b = {s: {**r, 'predicted_respond': False} for s, r in a.items()}
        result = mcnemar(a, b, 'short_turn')
        self.assertEqual((result['improved'], result['regressed'], result['discordant'], result['p_value']), (3, 0, 3, .25))
        self.assertEqual(mcnemar(b, a, 'short_turn')['recall_difference'], -1.)
        self.assertEqual(mcnemar(a, a, 'short_turn')['p_value'], 1.)
        self.assertIsNone(mcnemar(a, b, 'short_end')['p_value'])


if __name__ == '__main__':
    unittest.main()
