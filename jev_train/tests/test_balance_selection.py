from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
from jev_train.common import digest
from jev_train.curation import select_versions
from jev_train.domains import LABELS, balance_selection, _select_balance_source, capped_allocation, allocation_flows
from test_dedup import plan


def fixture():
    rows, labels = {}, {}
    for d, domain in enumerate(sorted(LABELS)):
        for source, n in [('a', 30+d), ('b', 20+2*d)]:
            for i in range(n):
                sid = f'{domain}/{source}/{i}'
                labels[sid] = domain
                rows[sid] = plan(sid, sid, dataset=source, task_group=sid, short_turn=i%6 == 0,
                                 gold_names=['RESPOND'] if i%6 == 0 else ['run'])
    _, ids, _ = select_versions(rows, {'a': 140, 'b': 70}, 7, apply_c4=False)
    return rows, labels, ids


class BalanceTests(unittest.TestCase):
    def test_R00_ordered_regression_and_four_group_invariants(self):
        rows, labels, baseline = fixture()
        for alpha, beta in [(1,1), (0,1), (1,.5), (0,.5)]:
            ids, report = balance_selection(rows, labels, baseline, 210, 7, domain_alpha=alpha, source_beta=beta)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 210)
            if alpha == beta == 1:
                self.assertEqual(ids, baseline)
            if alpha == 0:
                self.assertEqual(set(r['quota'] for r in report['domains'].values()), {30})
            for r in report['by_source'].values():
                self.assertLessEqual(r['priority'], round(.25*r['count']))
            self.assertEqual((ids, report), balance_selection(dict(reversed(list(rows.items()))), dict(reversed(list(labels.items()))),
                baseline, 210, 7, domain_alpha=alpha, source_beta=beta))

    def test_budget_shared_across_domains_not_per_cell(self):
        cells = {'a': {str(i): plan(str(i), short_turn=True) for i in range(4)},
                 'b': {str(i): plan(str(i)) for i in range(4, 16)}}
        selected, report = _select_balance_source(cells, {'a': 4, 'b': 12}, 7)
        self.assertEqual({d: len(v) for d, v in selected.items()}, {'a':4, 'b':12})
        self.assertEqual(report['priority_selected'], 4)

    def test_mandatory_priority_reserved_before_optional(self):
        order = sorted((str(i) for i in range(20)), key=lambda sid: (digest(7,sid),sid))
        cells = {'a': {sid: plan(sid, short_turn=True) for sid in order[-4:]},
                 'b': {sid: plan(sid, short_turn=i<4) for i,sid in enumerate(order[:-4])}}
        selected, report = _select_balance_source(cells, {'a':4, 'b':12}, 7)
        self.assertEqual(report['priority_by_domain'], {'a':4, 'b':0})
        self.assertEqual(set(selected['a']), set(order[-4:]))

    def test_insufficient_ordinary_fails_no_constraint_relaxation(self):
        cells = {'a': {str(i): plan(str(i), direct=True) for i in range(5)},
                 'b': {str(i): plan(str(i)) for i in range(5, 16)}}
        with self.assertRaisesRegex(ValueError, 'mandatory priority 5'):
            _select_balance_source(cells, {'a':5, 'b':11}, 7)

    def test_priority_rounds_ties_to_even(self):
        cells = {'a': {str(i): plan(str(i), short_turn=i<8) for i in range(20)}}
        for n, expected in [(2,0), (6,2), (10,2), (14,4)]:
            _, report = _select_balance_source(cells, {'a':n}, 7)
            self.assertEqual(report['priority_quota'], expected)

    def test_fixed_weight_local_redistribution(self):
        result = capped_allocation({'a':8, 'b':2}, 10, {'a':5, 'b':20})
        self.assertEqual(result, {'a':5, 'b':5})
        self.assertEqual(allocation_flows({'a':8,'b':2}, result), [dict(source='a',destination='b',count=3)])

    def test_zero_baseline_cell_cannot_fill_short_domain(self):
        rows, labels, baseline = fixture()
        domain = labels[rows[baseline[0]]['task_group']]
        # Make this domain's reference count one; unsupported source z has capacity.
        base = [sid for sid in baseline if labels[rows[sid]['task_group']] != domain] + [baseline[0]]
        trimmed = {sid:r for sid,r in rows.items() if labels[r['task_group']] != domain or sid == baseline[0]}
        for i in range(200):
            sid=f'z/{i}'; labels[sid]=domain
            trimmed[sid]=plan(sid,sid,dataset='z',task_group=sid)
        with self.assertRaisesRegex(ValueError, 'no cross-domain backfill'):
            balance_selection(trimmed, labels, base, len(base), 7, domain_alpha=0, source_beta=1)

    def test_missing_invalid_labels_and_required_booleans(self):
        rows, labels, baseline = fixture()
        first = next(iter(rows))
        for alpha,beta in ((True,1),(0,True),('0',1)):
            with self.assertRaisesRegex(ValueError,'exponents'):
                balance_selection(rows,labels,baseline,210,7,domain_alpha=alpha,source_beta=beta)
        for field in ('short_turn', 'direct', 'turn_calls_unknown'):
            for value in (None, 0, 'false', 'missing'):
                bad=deepcopy(rows)
                if value == 'missing': del bad[first][field]
                else: bad[first][field]=value
                with self.assertRaisesRegex(ValueError, field):
                    balance_selection(bad, labels, baseline, 210, 7, domain_alpha=1, source_beta=1)
        for value in (None, 'unknown'):
            bad=dict(labels); bad[first]=value
            with self.assertRaisesRegex(ValueError, 'task domain'):
                balance_selection(rows, bad, baseline, 210, 7, domain_alpha=1, source_beta=1)


if __name__ == '__main__':
    unittest.main()
