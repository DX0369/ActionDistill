import json
from pathlib import Path
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
from jev_train.domains import (LABELS, balance_selection, capped_equal_allocation,
                               BalanceCapacityError)
from preview_r11_cap import diagnostics, load_semantics, planned_exposure, report_markdown
from test_balance_selection import fixture
from test_dedup import plan


def cap_select(rows, labels, reference, count):
    return balance_selection(rows, labels, reference, count, 7,
                             domain_alpha=0, source_beta=.5, cap_domains=True)


class R11CapTests(unittest.TestCase):
    def test_waterfill_restarts_rounding_after_caps(self):
        self.assertEqual(capped_equal_allocation(8, {'a':99, 'b':2, 'c':99}),
                         {'a':3, 'b':2, 'c':3})
        capacities = {'a':1, 'b':2, 'c':20, 'd':20}
        self.assertEqual(capped_equal_allocation(14, capacities), {'a':1, 'b':2, 'c':6, 'd':5})
        self.assertEqual(capped_equal_allocation(43, capacities), capacities)
        self.assertEqual(capped_equal_allocation(14, dict(reversed(list(capacities.items())))),
                         capped_equal_allocation(14, capacities))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            capped_equal_allocation(44, capacities)
        for bad in (-1, None, True):
            with self.assertRaisesRegex(ValueError, 'inputs'):
                capped_equal_allocation(1, {'a': bad})

    def test_uncapped_ordered_R11_and_report_regression(self):
        rows, labels, reference = fixture()
        old_ids, old_report = balance_selection(rows, labels, reference, 210, 7, domain_alpha=0, source_beta=.5)
        ids, report = cap_select(rows, labels, reference, 210)
        self.assertEqual(ids, old_ids)
        report.pop('cap_allocation')
        self.assertEqual(report, old_report)
        ids, report = cap_select(rows, labels, reference, 280)
        self.assertEqual(len(ids), 280)
        self.assertEqual(report['cap_allocation']['reference_count'], 210)
        self.assertEqual({r['quota'] for r in report['domains'].values()}, {40})
        self.assertEqual((ids, report), cap_select(dict(reversed(list(rows.items()))), labels, reference, 280))

    def test_unsupported_cells_do_not_increase_domain_capacity(self):
        rows, labels, reference = {}, {}, []
        for domain in LABELS:
            for i in range(30):
                sid = f'{domain}/{i}'; labels[sid] = domain
                rows[sid] = plan(sid, sid, task_group=sid)
                if i < 10:
                    reference.append(sid)
        rows = {sid:r for sid,r in rows.items() if labels[sid] != 'other' or sid in reference}
        for i in range(200):
            sid=f'extra/{i}'; labels[sid]='other'
            rows[sid]=plan(sid, sid, dataset='unsupported', task_group=sid)
        ids, report = cap_select(rows, labels, reference, 140)
        self.assertEqual(report['cap_allocation']['domains']['other']['supported_capacity'], 10)
        self.assertEqual(report['domains']['other']['quota'], 10)
        self.assertEqual(report['cells']['other']['unsupported']['final_quota'], 0)
        self.assertFalse(any(sid.startswith('extra/') for sid in ids))

    def test_raw_capacity_passes_but_C3_stops_with_fixed_quota_evidence(self):
        rows, labels, reference = fixture()
        for row in rows.values():
            row['short_turn'] = True
        with self.assertRaises(BalanceCapacityError) as caught:
            cap_select(rows, labels, reference, 280)
        report = caught.exception.report
        self.assertEqual(sum(r['quota'] for r in report['domains'].values()), 280)
        self.assertTrue(report['c3_failures'])
        self.assertNotIn('selected', report)
        for source, row in report['c3_failures'].items():
            self.assertEqual(row['priority_quota'], round(.25*row['quota']))
            self.assertEqual(row['mandatory_priority'], row['quota'])
            self.assertEqual(row['priority_shortfall'], row['quota']-row['priority_quota'])
        for cells in report['cells'].values():
            self.assertTrue(all(r['priority_selected'] is None for r in cells.values()))
        text = report_markdown(dict(targets={'280': dict(status='failed', error=str(caught.exception),
            allocation=report, selected=None, diagnostics=None)}))
        self.assertIn('尚无合法选中集合', text)

    def test_missing_required_feature_fails(self):
        rows, labels, reference = fixture()
        del rows[reference[0]]['short_turn']
        with self.assertRaisesRegex(ValueError, 'short_turn'):
            cap_select(rows, labels, reference, 280)

    def test_predicted_schedule_handles_odd_epoch_and_batch_weights(self):
        records = [plan(str(i), n_prompt_tokens=32704) for i in range(3)]
        weights, schedule = planned_exposure(records)
        self.assertEqual(schedule['updates'], 2)
        self.assertEqual(schedule['occurrences'], 4)
        self.assertEqual(schedule['effective_prompt_tokens'], 4*32704)
        self.assertEqual(sum(weights.values()), 2)
        self.assertNotIn('all_step_token_counts_matched', schedule)
        weights, schedule = planned_exposure([plan('a'), plan('b')])
        self.assertEqual(dict(weights), {'a': .5, 'b': .5})

    def test_uncached_RESPOND_is_unknown_not_a_negative(self):
        kept = {sid: plan(sid, sid, task_group=sid, gold_names=['RESPOND'],
                          next_event='user_turn_end') for sid in ('known', 'uncached', 'conflict')}
        kept['end'] = plan('end', gold_names=['RESPOND'], next_event='trajectory_end')
        semantics = {'known': dict(semantic='short_turn'), 'conflict': dict(semantic='unknown')}
        r = diagnostics(list(kept), kept, ['known'], semantics)
        strict = r['strict_short_stop']
        self.assertEqual(strict['known_positive'], 1)
        self.assertEqual(strict['unknown'], 2)
        self.assertFalse(strict['complete'])
        self.assertEqual(strict['count_upper_bound'], 3)
        self.assertEqual(strict['loss_coefficient_fraction_lower_bound'], .25)
        self.assertEqual(strict['loss_coefficient_fraction_upper_bound'], .75)
        self.assertEqual(r['overlap_with_R11_8k'], 1)

    def test_invalid_cached_semantic_fails_at_input_boundary(self):
        kept = {'s': plan('s', turn_index=0, gold_names=['RESPOND'], next_event='user_turn_end')}
        with tempfile.TemporaryDirectory(dir=P/'tmp', prefix='r11_cap_test_') as directory:
            path = Path(directory); (path/'metadata').mkdir()
            (path/'metadata/run_status.json').write_text(json.dumps(dict(status='completed', smoke=False, states=1)))
            row = dict(kept['s'], normalized_gold=['RESPOND'])
            for bad in (None, 'unrecognized', False, 'missing'):
                if bad == 'missing':
                    row.pop('semantic')
                else:
                    row['semantic'] = bad
                (path/'states.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaises((ValueError, KeyError)):
                    load_semantics(path, kept)


if __name__ == '__main__':
    unittest.main()
