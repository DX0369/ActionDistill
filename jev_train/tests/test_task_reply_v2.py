from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
from jev_train.domains import LABELS
from jev_train.reply_origin import (reply_priority, zclaw_framework_message, hermes_budget_pressure,
    ZCLAW_BACKGROUND_DIRECTIVE, ZCLAW_INTERNAL_HEADER, ZCLAW_INTERNAL_DIRECTIVE)
from jev_train.task_selection import source_allocation, source_quota_order, select_with_sources
from test_task_selection import state


def origin(**changes):
    return dict(dict(category='unknown', next_event='user_turn_end', termination_status='missing',
                     termination_value=None, call_count_status='known', user_boundary_source='raw:1',
                     prior_calls=2), **changes)


class ReplyV2Tests(unittest.TestCase):
    def test_zclaw_requires_complete_notification_not_quoted_or_partial_markers(self):
        background = 'A background task "task" just completed successfully.\n\nFindings:\ndone\n\n'+ZCLAW_BACKGROUND_DIRECTIVE
        internal = ZCLAW_INTERNAL_HEADER+'\n\n[Internal task completion event]\nresult\n\nAction:\n'+ZCLAW_INTERNAL_DIRECTIVE
        for text, rule in [(background, 'zclaw_background_task_completion'), (internal, 'zclaw_internal_task_completion')]:
            for content in (text, '[Sun 2026-03-15 19:15 UTC] '+text, [dict(type='text', text=text)]):
                self.assertEqual(zclaw_framework_message(content)['rule'], rule)
            for text in ('Analyze this quoted log:\n'+text, text[:-10], text+'\nActually this is a new user task.'):
                self.assertIsNone(zclaw_framework_message(text))
        self.assertIsNone(zclaw_framework_message('[Subagent Context] You are running as a subagent\n[Subagent Task]: hello'))
        self.assertIsNone(zclaw_framework_message('A background task "my own job" is in progress.'))
        with self.assertRaises(KeyError):
            zclaw_framework_message([dict(type='text')])

    def test_hermes_nonexhausted_pressure_is_diagnostic_only(self):
        phrase = 'Provide your final response NOW. No more tool calls unless absolutely critical.'
        def content(warning):
            return '<tool_response>'+json.dumps({'content': {'_budget_warning': warning}})+'</tool_response>'
        row = origin(); original = deepcopy(row)
        hits, issues = hermes_budget_pressure(content('Iteration 24/25. Only 1 iteration(s) left. '+phrase))
        self.assertEqual([h['rule'] for h in hits], ['hermes_nonexhausted_final_now'])
        self.assertEqual(issues, [])
        self.assertTrue(reply_priority(row)['eligible'])
        self.assertEqual(row, original)
        hits, _ = hermes_budget_pressure(content('Iteration 11/15. 4 iterations left. Start consolidating your work.'))
        self.assertEqual(hits[0]['rule'], 'hermes_nonexhausted_consolidation')
        self.assertEqual(hermes_budget_pressure(content('Iteration 25/25. Only 0 iteration(s) left. '+phrase)), ([], []))
        self.assertEqual(hermes_budget_pressure('<tool_response>broken</tool_response>'), ([], ['unparseable_tool_response']))

    def test_framework_boundary_removes_preference_without_dropping_or_relabeling(self):
        records = {f'{t}/{i}': state(f'{t}/{i}', source='ZClawBench', task=str(t), trajectory=str(t), step=i, reply=i == 0)
                   for t in range(1000) for i in range(4)}
        origins = {f'{t}/0': origin() for t in range(1000)}
        labels = {str(t): 'repo_swe' for t in range(1000)}
        cells = {domain: {'ZClawBench': 1} for domain in LABELS}
        before = deepcopy((records, origins))
        ids, _ = select_with_sources(records, labels, cells, origins, 1000, 7, framework_boundary_ids=set(origins))
        hits = sum(records[sid]['gold_names'] == ['RESPOND'] for sid in ids)
        self.assertTrue(220 <= hits <= 280, hits)  # One reply among four ordinary states.
        all_ids, _ = select_with_sources(records, labels, cells, origins, 4000, 7, framework_boundary_ids=set(origins))
        self.assertEqual(set(all_ids), set(records))
        self.assertEqual((records, origins), before)
        self.assertFalse(reply_priority(origin(), framework_boundary=True)['eligible'])
        with self.assertRaisesRegex(ValueError, 'normalized RESPOND IDs'):
            select_with_sources(records, labels, cells, origins, 1, 7, framework_boundary_ids={'missing'})

    def test_formal_preview_rejects_unresolved_source_review(self):
        from preview_task_reply_v2 import load_review
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            path = Path(directory)/'review.json'
            path.write_text(json.dumps({'reply_boundary_review': {
                'pending_policy_questions': ['zclaw_framework_boundaries']}}))
            with self.assertRaisesRegex(ValueError, 'unresolved scientific policy'):
                load_review(path, {}, [])

    def test_report_positions_keep_teachers_of_same_physical_trajectory_distinct(self):
        from preview_task_reply_v2 import unit_positions
        records = {f'{model}/{i}': state(f'{model}/{i}', teacher=model, step=i)
                   for model, count in [('x', 10), ('y', 2)] for i in range(count)}
        positions = unit_positions(records)
        self.assertEqual([positions[f'x/{i}'] for i in range(10)], list(range(10)))
        self.assertEqual([positions[f'y/{i}'] for i in range(2)], [2, 7])

    def test_cache_uses_exact_native_or_normalized_source_locator(self):
        from preview_task_reply_v2 import check_restored_source
        target = dict(id='a', source='raw.jsonl:20', turn_index=98)
        for source in ('raw.jsonl:20', 'raw.jsonl:20#normalized_messages[98]'):
            check_restored_source(target, dict(source=source))
        for source in ('raw.jsonl:21', 'raw.jsonl:20#normalized_messages[99]',
                       'other.jsonl:20#normalized_messages[98]'):
            with self.assertRaisesRegex(ValueError, 'locator differs'):
                check_restored_source(target, dict(source=source))

    def test_priority_uses_boundary_not_positive_origin_and_keeps_terminal_semantics(self):
        for category in ('unknown', 'verified_reply'):
            for status, value in [('missing', None), ('null', None), ('known', 'stop'),
                                  ('known', 'aborted'), ('known', 'max_tokens')]:
                row = origin(category=category, termination_status=status, termination_value=value)
                self.assertTrue(reply_priority(row)['eligible'])
                self.assertEqual(row['category'], category)
        for changed in (dict(category='framework_forced'), dict(termination_status='conflict'),
                        dict(next_event='unjudgeable'),
                        dict(call_count_status='parent_chain_missing_before_user', user_boundary_source=None, prior_calls=None)):
            self.assertFalse(reply_priority(origin(**changed))['eligible'])
        for value in ('length', 'error'):
            with self.assertRaisesRegex(ValueError, 'D2'):
                reply_priority(origin(termination_status='known', termination_value=value))
        for field in origin():
            missing = origin(); del missing[field]
            with self.assertRaises(KeyError):
                reply_priority(missing)
        for changed in (dict(category='natural'), dict(termination_status='inferred'),
                        dict(call_count_status='inferred'), dict(next_event='tool_call'),
                        dict(prior_calls=None), dict(user_boundary_source=None)):
            with self.assertRaises(ValueError):
                reply_priority(origin(**changed))

    def test_reference_source_quotas_zero_cells_caps_and_exact_selection(self):
        records, labels, cells = {}, {}, {}
        for domain in LABELS:
            labels[domain] = domain
            cells[domain] = {'a': 9, 'b': 1, 'zero': 0}
            for source in cells[domain]:
                # More distinct zero-reference tasks must not grant them quota.
                for i in range(40):
                    sid = f'{domain}/{source}/{i}'
                    task = f'{domain}/{source}/{i}'
                    labels[task] = domain
                    records[sid] = state(sid, task=task, source=source, trajectory=task)
        allocation = source_allocation(records, labels, cells, 140)
        for plan in allocation.values():
            self.assertEqual(plan['quota'], 20)
            self.assertEqual(plan['capacity'], 80)
            self.assertEqual({s: c['quota'] for s, c in plan['sources'].items()}, {'a': 15, 'b': 5, 'zero': 0})
        original = deepcopy((records, labels, cells))
        ids, actual = select_with_sources(records, labels, cells, {}, 140, 20261015)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(actual, allocation)
        self.assertNotIn('zero', {records[sid]['dataset'] for sid in ids})
        self.assertEqual((records, labels, cells), original)
        self.assertEqual((ids, actual), select_with_sources(dict(reversed(list(records.items()))), labels, cells, {}, 140, 20261015))
        scarce = next(iter(LABELS))
        fewer = {sid: r for sid, r in records.items()
                 if not (labels[r['task_group']] == scarce and r['dataset'] == 'b' and not sid.endswith('/0'))}
        capped = source_allocation(fewer, labels, cells, 140)
        self.assertEqual(capped[scarce]['sources']['b']['quota'], 1)
        self.assertEqual(capped[scarce]['sources']['a']['quota'], 19)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            select_with_sources(records, labels, cells, {}, 600, 1)  # Zero cells cannot fill missing support.

    def test_shared_task_counters_and_flat_teacher_units(self):
        records = [state(f'{source}/{task}/{model}/{i}', source=source, task=task, teacher=model,
                         trajectory=task, step=i)
                   for source in ('a', 'b') for task in ('shared', f'{source}_only')
                   for model in ('x', 'y') for i in range(4)]
        order = list(source_quota_order(records, {'a': 8, 'b': 8}, set(), 7, 'repo_swe'))
        remaining = Counter((r['dataset'], r['task_group']) for r in records)
        selected = Counter()
        units = {}
        for row in order:
            source, task = row['dataset'], row['task_group']
            possible = [t for (s, t), n in remaining.items() if s == source and n]
            self.assertEqual(selected[task], min(selected[t] for t in possible))
            selected[task] += 1
            remaining[source, task] -= 1
            units.setdefault((source, task), []).append(row['model'])
        self.assertEqual(Counter(r['dataset'] for r in order), {'a': 8, 'b': 8})
        self.assertTrue(all(set(models[:2]) == {'x', 'y'} for models in units.values()))
        self.assertEqual(order, list(source_quota_order(list(reversed(records)), {'b': 8, 'a': 8}, set(), 7, 'repo_swe')))
        invalid = records + [state('extra', source='a', task='shared', teacher='x', trajectory='second')]
        with self.assertRaisesRegex(ValueError, 'one retained trajectory'):
            list(source_quota_order(invalid, {'a': 1}, set(), 7, 'repo_swe'))

    def test_unknown_origin_reply_is_weighted_without_copying(self):
        records = {f'{t}/{i}': state(f'{t}/{i}', task=str(t), trajectory=str(t), step=i, reply=i == 0)
                   for t in range(1000) for i in range(4)}
        origins = {f'{t}/0': origin() for t in range(1000)}
        labels = {str(t): 'repo_swe' for t in range(1000)}
        cells = {domain: {'a': 1} for domain in LABELS}
        ids, _ = select_with_sources(records, labels, cells, origins, 1000, 7)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len({records[sid]['task_group'] for sid in ids}), 1000)
        hits = sum(records[sid]['gold_names'] == ['RESPOND'] for sid in ids)
        self.assertTrue(370 <= hits <= 430, hits)


if __name__ == '__main__':
    unittest.main()
