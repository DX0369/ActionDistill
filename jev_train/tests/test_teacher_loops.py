import json
from pathlib import Path
import sys
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
import analyze_teacher_loops as analysis


def call(arguments='{}',name='run',cid='c'):
    return dict(id=cid,type='function',function=dict(name=name,arguments=arguments))


def message(*calls):
    return dict(role='assistant',content=None,tool_calls=list(calls))


class TeacherLoopTests(unittest.TestCase):
    def test_history_only_json_equivalence_and_batches(self):
        history=[dict(role='user',content='work'),message(call('{"a":1,"b":2}')),
                 message(call('{"b":2, "a":1}')),dict(role='assistant',content='progress'),
                 message(call({'a':1,'b':2}))]
        result=analysis.history_loop(history)
        self.assertTrue(result['loop_history'])
        self.assertEqual(result['last_three_batches'],3)
        for gold,expected in ((['RESPOND'],'RESPOND'),(['run'],'contains_loop_tool'),
                              (['other'],'different_tool'),(['run','other'],'contains_loop_tool')):
            self.assertEqual(analysis.gold_relation(gold,result['loop_tool']),expected)
        self.assertTrue(analysis.history_loop([message(call(),call(),call())])['loop_history'])
        self.assertEqual(analysis.history_loop([message(call(),call(),call())])['last_three_batches'],1)

    def test_new_user_resets_and_names_parameters_must_match(self):
        repeated=[message(call()),message(call()),message(call())]
        self.assertFalse(analysis.history_loop(repeated+[dict(role='user',content='next')])['loop_history'])
        self.assertFalse(analysis.history_loop(repeated[:2]+[dict(role='user',content='next'),repeated[-1]])['loop_history'])
        self.assertFalse(analysis.history_loop(repeated[:2]+[message(call('{"a":1}'))])['loop_history'])
        self.assertFalse(analysis.history_loop(repeated[:2]+[message(call(name='other'))])['loop_history'])
        self.assertFalse(analysis.history_loop(repeated[:2])['loop_history'])

    def test_approved_argument_forms_and_source_custom_type(self):
        result=analysis.history_loop([message(call('')),message(call('{}')),message(call({}))])
        self.assertTrue(result['loop_history']);self.assertEqual(result['argument_forms'],['empty_arguments'])
        result=analysis.history_loop([message(call('{broken')) for _ in range(3)])
        self.assertTrue(result['loop_history']);self.assertEqual(result['argument_forms'],['unparsed_arguments'])
        text='*** Begin Patch\n*** End Patch'
        calls=[call(text,cid=str(i)) for i in range(3)]
        types={(c['id'],'run',json.dumps(text,ensure_ascii=False,sort_keys=True)):'custom_tool_call' for c in calls}
        result=analysis.history_loop([message(c) for c in calls],types)
        self.assertTrue(result['loop_history']);self.assertEqual(result['argument_forms'],[])
        with self.assertRaises(KeyError):
            analysis.history_loop([message(c) for c in calls],{})
        invalid=call();del invalid['function']['arguments']
        with self.assertRaises(KeyError):
            analysis.history_loop([message(invalid) for _ in range(3)])

    def test_source_reconstruction_gold_independence_and_local_prefix_boundary(self):
        raw=[dict(role='user',content='work'),message(call(),call(),call()),dict(role='assistant',content='finished')]
        context=dict(source='/source.json:1',model='teacher',outcome='success',strip_think=False,harness='test')
        cap=dict(dataset='dataset',trajectory_id='record:1',messages=raw,
                 tools=[dict(name=n,parameters={'type':'object'}) for n in ('run','other')],context=context)
        sid=analysis.digest('dataset','record:1',2)
        target=dict(id=sid,dataset='dataset',model='teacher',trajectory_id='record:1',turn_index=2,
                    names=['run','other','RESPOND'],gold_names=['RESPOND'],outcome='success',candidate_level='S',
                    messages_dropped=0,truncation_counts={},populations=['A_train'],loop_state='no')
        unit=dict(capture=cap,targets=[target],family='openai',raw_records=(),prefix=None)
        row=list(analysis.rebuild(unit))[0]
        self.assertTrue(row['loop_history']);self.assertEqual(row['gold_relation'],'RESPOND')
        self.assertEqual(row['saved_loop_state'],'no')
        summary=analysis.aggregate([row])
        self.assertEqual((summary['known'],summary['loops'],summary['gold_relation']),(1,1,{'RESPOND':1}))
        unit['prefix']=dict(continuation=True,trace_part=2)
        row=list(analysis.rebuild(unit))[0]
        self.assertTrue(row['loop_history'])
        self.assertEqual(analysis.aggregate([row])['known'],1)
        self.assertEqual(analysis.history_status(2,raw,[dict(m,role='system') if i==0 else m for i,m in enumerate(raw)]),
                         'no_user_boundary_in_record')
        raw[0]['isMeta']=True
        self.assertEqual(analysis.history_status(2,raw,raw),'known')
        raw[1]['tool_calls'][-1]['function']['arguments']='{broken'
        row=list(analysis.rebuild(unit))[0]
        self.assertFalse(row['loop_history'])
        summary=analysis.aggregate([row])
        self.assertEqual(summary['argument_forms_all_known'],{'unparsed_arguments':1})
        self.assertEqual(summary['argument_forms'],{})
        target['names']=['wrong','other','RESPOND']
        with self.assertRaisesRegex(ValueError,'source labels/options differ'):
            list(analysis.rebuild(unit))


if __name__=='__main__':unittest.main()
