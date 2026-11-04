from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
import analyze_data_curation as a


def call(name='run', arguments='{}', cid='c'):
    return dict(id=cid, type='function', function=dict(name=name, arguments=arguments))


def tool(*calls):
    return dict(role='assistant', content=None, tool_calls=list(calls))


def unit(messages, indices=None):
    names = ['run', 'other', 'RESPOND']
    cap = dict(dataset='test', trajectory_id='t', messages=messages,
               tools=[dict(name=n, parameters={}) for n in names[:-1]],
               context=dict(source='source.json', strip_think=False, model='teacher', outcome='success'))
    targets = []
    for i, m in enumerate(messages):
        if m['role'] != 'assistant' or (indices is not None and i not in indices):
            continue
        gold = list(dict.fromkeys(c['function']['name'] for c in m.get('tool_calls', []))) or ['RESPOND']
        targets.append(dict(id=a.digest('test','t',i), dataset='test', trajectory_id='t', turn_index=i,
                            model='teacher', outcome='success', candidate_level='S', names=names,
                            gold_names=gold, gold_kind='RESPOND' if gold == ['RESPOND'] else 'TOOL' if len(gold)==1 else 'TOOL_SET',
                            split='train', populations=['full_train']))
    return dict(capture=cap, targets=targets, family='openai', record=dict(messages=messages), raw_records=(), prefix=None)


class DataCurationTests(unittest.TestCase):
    def test_narration_successor_and_parallel_call_count(self):
        messages = [dict(role='user',content='do work'), dict(role='assistant',content='I will inspect'),
                    tool(call(),call('other'),call()), dict(role='tool',content='ok',tool_call_id='c'),
                    dict(role='assistant',content='done'), dict(role='user',content='next')]
        rows = list(a.rebuild(unit(messages), {}))
        narration, successor, ending = [r[0] for r in rows]
        self.assertEqual(narration['next_event'], 'intermediate_tool_continuation')
        self.assertEqual(narration['next_tool_names'], ['run','other'])
        self.assertTrue(successor['narration_successor'])
        self.assertEqual([r['turn_total_calls'] for r in (narration,successor,ending)], [3,3,3])
        self.assertEqual(ending['turn_calls_before'], 3)
        self.assertEqual(ending['turn_end_kind'], 'RESPOND')
        self.assertEqual(tuple(narration), a.FEATURE_FIELDS)

    def test_combined_text_and_call_is_one_tool_decision(self):
        messages = [dict(role='user',content='work'), dict(tool(call()),content='I will inspect'),
                    dict(role='tool',content='ok',tool_call_id='c'), dict(role='assistant',content='done')]
        rows = list(a.rebuild(unit(messages), {}))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0][0]['gold_kind'],'TOOL')
        self.assertIsNone(rows[0][0]['next_event'])
        self.assertFalse(rows[0][0]['narration_successor'])

    def test_d2_only_respond_and_c1_end(self):
        messages = [dict(role='user',content='work'), dict(tool(call()),finish_reason='length'),
                    dict(role='tool',content='error',tool_call_id='c'),
                    dict(role='assistant',content='failed',finish_reason='error')]
        rows = list(a.rebuild(unit(messages), {}))
        self.assertIsNone(rows[0][0]['stop_reason_abnormal'])
        self.assertTrue(rows[1][0]['stop_reason_abnormal'])
        self.assertEqual(rows[0][0]['turn_end_kind'], 'incomplete')

    def test_full_turn_not_just_selected_states_and_user_boundary(self):
        for calls in (0,3,4):
            with self.subTest(calls=calls):
                messages = [dict(role='user',content='first')]
                for i in range(calls):
                    messages.extend([tool(call(arguments=f'{{"i":{i}}}')),dict(role='tool',content='ok',tool_call_id='c')])
                end = len(messages)
                messages.extend([dict(role='assistant',content='done'),dict(role='user',content='new'),dict(role='assistant',content='reply')])
                rows = list(a.rebuild(unit(messages, [end, end+2]), {}))
                self.assertEqual(rows[0][0]['turn_total_calls'],calls)
                self.assertEqual(rows[1][0]['turn_total_calls'],0)
                self.assertEqual(rows[1][0]['turn_calls_before'],0)

    def test_structural_loop_keeps_error_retry_as_positive(self):
        messages = [dict(role='user',content='work')]
        for _ in range(3):
            messages.extend([tool(call()),dict(role='tool',content='Error: retry',tool_call_id='c')])
        messages.append(dict(role='assistant',content='done'))
        self.assertTrue(list(a.rebuild(unit(messages), {}))[-1][0]['strict_loop'])
        messages[-3] = tool(call(arguments='{"different":1}'))
        self.assertFalse(list(a.rebuild(unit(messages), {}))[-1][0]['strict_loop'])

    def test_incomplete_prefix_and_missing_user_are_unknown(self):
        source = unit([dict(role='assistant',content='reply')])
        row, _, diag = next(a.rebuild(source, {}))
        self.assertIsNone(row['turn_calls_before'])
        self.assertIsNone(row['strict_loop'])
        self.assertIn('turn_calls_before',diag['reasons'])
        source = unit([dict(role='user',content='q'),dict(role='assistant',content='reply')])
        source['prefix'] = dict(source_fully_captured=False, known_continuation=True, continuation=False, trace_part=1)
        row, _, diag = next(a.rebuild(source, {}))
        self.assertEqual(row['next_event'],'unjudgeable')
        self.assertIsNone(row['turn_total_calls'])
        self.assertIsNone(row['turn_end_kind'])

    def test_raw_parent_branch_is_unknown(self):
        source = unit([dict(role='user',content='q'),tool(call()),dict(role='assistant',content='left'),dict(role='assistant',content='right')])
        source['family']='native_pi'
        records=[]
        for i,m in enumerate(source['capture']['messages']):
            m['_source']=f'raw.jsonl:{i+1}'
            records.append((i+1, dict(id=str(i),parentId=None if i==0 else '0' if i==1 else '1',message=m)))
        source['raw_records']=records
        view=a.TurnView(source)
        self.assertEqual(view.suffix(1)[2], 'multiple_tree_successors')
        source['raw_records'][1][1]['parentId']='missing'
        view=a.TurnView(source)
        self.assertIsNone(view.prior(1)['previous_turn_tool_calls'])

    def test_default_classification_projection_unchanged(self):
        source=unit([dict(role='user',content='q'),dict(role='assistant',content='reply')])
        target=dict(source['targets'][0],transition='first',trajectory_position=0,trajectory_decisions=1)
        full=list(a.classify_capture(source['capture'],[target],'openai'))[0]
        reduced=list(a.classify_capture(source['capture'],[target],'openai',target_fields=a.BASE_FIELDS))[0]
        self.assertEqual({k:v for k,v in full.items() if k not in set(a.TARGET_FIELDS)-set(a.BASE_FIELDS)},reduced)

    def test_conflicting_claude_terminal_evidence_is_unknown(self):
        source=unit([dict(role='user',content='q',_uuids={'u'}),
                     dict(role='assistant',content='reply',_uuids={'r1','r2'})])
        source['family']='native_claude'
        source['raw_records']=[(1,dict(uuid='u',parentUuid=None,message=dict(role='user',content='q'))),
                               (2,dict(uuid='r1',parentUuid='u',message=dict(stop_reason='error'))),
                               (3,dict(uuid='r2',parentUuid='r1',message=dict(stop_reason='end_turn')))]
        feature,classified,diagnostic=next(a.rebuild(source,{}))
        self.assertEqual(classified['termination_status'],'conflict')
        self.assertIsNone(feature['turn_end_kind'])
        self.assertEqual(diagnostic['reasons']['turn_end_kind'],'conflicting_terminal_evidence')


if __name__ == '__main__':
    unittest.main()
