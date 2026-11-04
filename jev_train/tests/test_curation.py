import copy
from pathlib import Path
import sys
import unittest

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(R/'third_party/decider'),
               str(R/'experiments/dataset_tool_inventory/src')]
from jev_train.curation import c1_plan, c4_drop, materialize, select_versions
from jev_train.common import digest
from test_data_curation import unit, call, tool
from analyze_data_curation import rebuild
from curation_fixtures import sample, feature




class CurationTests(unittest.TestCase):
    def test_d1_and_narration_source_chain(self):
        source = unit([dict(role='user', content='work'), dict(role='assistant', content='I will inspect'),
                       tool(call(), call('other')), dict(role='tool', content='ok', tool_call_id='c'),
                       dict(role='assistant', content='done')])
        restored = [f for f, _, _ in rebuild(source, {})]
        for f in restored:
            row = dict(sample(f['id'], [0, 1] if f['gold_kind'] == 'TOOL_SET' else [2]),
                       dataset=f['dataset'])
            plan = c1_plan(row, f)
            if f['next_event'] == 'intermediate_tool_continuation':
                self.assertEqual(plan['gold'], [0, 1])
                self.assertEqual(plan['drops'], [])
                changed = materialize(row, plan)
                self.assertEqual({k:v for k,v in changed.items() if k not in ('gold','short_turn','direct')},
                                 {k:v for k,v in row.items() if k != 'gold'})
            elif f['narration_successor']:
                self.assertEqual(plan['drops'], ['C1_narration_successor'])

    def test_combined_message_unchanged(self):
        source = unit([dict(role='user',content='work'), dict(tool(call()),content='I will inspect')])
        f = next(rebuild(source, {}))[0]
        row = dict(sample(f['id'],[0]),dataset=f['dataset'])
        plan = c1_plan(row, f)
        self.assertEqual(plan['drops'], [])
        self.assertEqual(materialize(row,plan)['gold'], row['gold'])

    def test_missing_candidate_and_overlapping_drops(self):
        row = sample()
        f = feature(row, next_event='intermediate_tool_continuation', next_tool_names=['absent'],
                    stop_reason_abnormal=True, narration_successor=True)
        plan = c1_plan(row, f)
        self.assertEqual(plan['missing_candidates'], ['absent'])
        self.assertEqual(plan['drops'], ['C1_D1_missing_candidate','C1_narration_successor','C1_D2'])
        with self.assertRaises(ValueError):
            materialize(row,plan)
        self.assertEqual(c1_plan(row, feature(row,next_event='unjudgeable'))['drops'], ['C1_unjudgeable'])

    def test_c4_exact_single_gold_after_c1(self):
        for gold, expected in (([0],True),([1],False),([0,1],False),([2],False)):
            row = sample(gold=gold)
            plan = c1_plan(row, feature(row,strict_loop=True))
            self.assertEqual(c4_drop(plan,'run'),expected)
        row = sample()
        plan = c1_plan(row,feature(row,strict_loop=True,next_event='intermediate_tool_continuation',next_tool_names=['run']))
        self.assertTrue(c4_drop(plan,'run'))
        with self.assertRaisesRegex(ValueError,'loop tool'):
            c4_drop(plan,None)

    def test_c4_no_error_retry_exemption_and_different_arguments(self):
        for arguments, expected in (('{}',True), ('{"changed":1}',False)):
            messages = [dict(role='user',content='work')]
            for i in range(3):
                messages.extend([tool(call(arguments=arguments if i==2 else '{}')),
                                 dict(role='tool',content='Error: retry',tool_call_id='c')])
            messages.append(tool(call()))
            f = list(rebuild(unit(messages),{}))[-1][0]
            row = dict(sample(f['id'],[0]),dataset=f['dataset'])
            self.assertEqual(c4_drop(c1_plan(row,f),'run'),expected)

    def test_short_direct_unknown_boundaries(self):
        for calls, end, short, direct in ((3,'RESPOND',True,False),(4,'RESPOND',False,False),
                                         (0,'RESPOND',False,True),(None,None,False,False),
                                         (3,'incomplete',False,False)):
            row = sample()
            plan = c1_plan(row,feature(row,turn_total_calls=calls,turn_end_kind=end))
            self.assertEqual((plan['short_turn'],plan['direct']),(short,direct))
            self.assertEqual(plan['turn_calls_unknown'],calls is None)

    def test_missing_or_invalid_rule_inputs(self):
        row = sample()
        for key in feature(row):
            f = feature(row)
            del f[key]
            with self.subTest(key=key), self.assertRaises((KeyError,ValueError)):
                c1_plan(row,f)
        for changes in (dict(turn_total_calls=True),dict(turn_total_calls=-1),dict(strict_loop=1),
                        dict(next_event=None),dict(next_tool_names=[]),dict(stop_reason_abnormal=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c1_plan(row,feature(row,**changes))

    def test_relabel_updates_existing_features_only(self):
        row = dict(sample(), action_kind='RESPOND',prev_action_names=['run'],transition='change',loop_state='no')
        plan = c1_plan(row,feature(row,next_event='intermediate_tool_continuation',next_tool_names=['run']))
        with self.assertRaisesRegex(ValueError,'full source'):
            materialize(row,plan)
        changed = materialize(row,plan,dict(prev_action_names=['run'],transition='repeat',loop_state='yes'))
        self.assertEqual((changed['action_kind'],changed['transition'],changed['loop_state']),('TOOL','repeat','yes'))
        self.assertEqual(row['gold'],[2])
        self.assertNotIn('loop_state',materialize(sample(),plan))

    def test_quota_determinism_and_exclusive_remainder(self):
        plans = {}
        for i in range(40):
            row = sample(str(i),gold=[2] if i<20 else [0])
            plan = c1_plan(row,feature(row,turn_total_calls=0 if i<20 else None))
            plans[row['id']] = dict(plan,c4_drop=False)
        original = copy.deepcopy(plans)
        b1,b2,preview = select_versions(plans,{'d':10},7,apply_c4=True)
        self.assertEqual(b1,sorted(plans,key=lambda x:(digest(7,x),x))[:10])
        self.assertEqual(select_versions(dict(reversed(list(plans.items()))),{'d':10},7,apply_c4=True),(b1,b2,preview))
        self.assertEqual(preview['d']['priority_selected'],2)  # round(2.5) ties-to-even
        self.assertEqual(sum(plans[i]['direct'] for i in b2),2)
        self.assertEqual(preview['d']['b2_unknown_turns'],8)
        self.assertEqual(plans,original)
        for i in range(20,40):
            plans[str(i)]['c4_drop']=True
            plans[str(i)]['strict_loop']=True
        with self.assertRaisesRegex(ValueError,'ordinary pool'):
            select_versions(plans,{'d':10},7,apply_c4=True)
        # With C4 withdrawn all ordinary loop states re-enter selection.
        restored_b1,restored_b2,restored_preview=select_versions(plans,{'d':10},7,apply_c4=False)
        self.assertEqual(restored_b1,b1)
        self.assertEqual(restored_b2,b2)
        self.assertEqual(restored_preview,preview)

    def test_priority_shortage_uses_only_ordinary_remainder(self):
        plans = {}
        for i in range(20):
            row = sample(str(i))
            plan = c1_plan(row,feature(row,turn_total_calls=0 if i==0 else 4))
            plans[row['id']] = dict(plan,c4_drop=False)
        _, b2, preview = select_versions(plans,{'d':12},7,apply_c4=False)
        self.assertIn('0',b2)
        self.assertEqual(preview['d']['priority_quota'],3)
        self.assertEqual(preview['d']['priority_selected'],1)
        self.assertEqual(preview['d']['ordinary_selected'],11)


if __name__ == '__main__':
    unittest.main()
