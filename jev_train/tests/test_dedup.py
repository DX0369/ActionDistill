import hashlib
from pathlib import Path
import sys
import unittest

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(R/'third_party/decider'),
               str(P.parent/'dataset_tool_inventory/src')]
from jev_train.common import digest
from jev_train.dedup import deduplicate, b2_capacity, preview, reallocate_shortfall, redistributed_preview, SHORT_SOURCE
from jev_train.domains import clean_request, fixed_label, parse_label, annotation_requests
from jev_train.build import initial_request, light
from test_build import event


def plan(sid, trajectory='t', **changes):
    row = dict(id=sid, dataset='a', task_group='g', model='teacher', trajectory_id=trajectory,
               split='train', drops=[], gold_names=['run'], short_turn=False, direct=False,
               turn_calls_unknown=False, next_event=None, n_prompt_tokens=100)
    row.update(changes)
    return row


class DedupTests(unittest.TestCase):
    def test_missing_canonical_fields_never_inferred(self):
        for key in ('trajectory_id','model','task_group'):
            for value in ('missing',None,''):
                row = plan('s')
                if value=='missing':
                    del row[key]
                else:
                    row[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    deduplicate({'s':row}, 7)

    def test_choose_only_eligible_trajectory_and_keep_all_states(self):
        names = sorted(['t1','t2','t3'], key=lambda t: (digest(7,t),t))
        rows = {'bad':plan('bad',names[0],drops=['C1_D2']),
                'one':plan('one',names[1]), 'two':plan('two',names[1]),
                'last':plan('last',names[2]),
                'teacher2':plan('teacher2',names[0],dataset='b',model='other')}
        kept = deduplicate(rows,7)
        self.assertEqual(set(kept), {'one','two','teacher2'})
        self.assertEqual(kept,deduplicate(dict(reversed(list(rows.items()))),7))
        self.assertEqual({r['trajectory_id'] for r in kept.values() if r['dataset']=='a'}, {names[1]})

    def test_different_sources_and_teachers_do_not_merge(self):
        rows = {r['id']:r for r in [plan('a','x'), plan('b','y',model='second'), plan('c','x',dataset='b')]}
        self.assertEqual(deduplicate(rows,7), rows)
        rows['same_trajectory_other_teacher'] = plan('same_trajectory_other_teacher','x',model='second')
        with self.assertRaisesRegex(ValueError,'partially retained'):
            deduplicate(rows,7)
        del rows['b']
        self.assertIn('same_trajectory_other_teacher',deduplicate(rows,7))
        rows['bad'] = plan('bad','x',task_group='different')
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            deduplicate(rows,7)

    def test_capacity_includes_rounding_and_ordinary_only_constraint(self):
        for priority in range(20):
            for ordinary in range(20):
                feasible = [k for k in range(priority+ordinary+1) if k-min(priority,round(.25*k))<=ordinary]
                self.assertEqual(b2_capacity(priority,ordinary),max(feasible))
        self.assertEqual(b2_capacity(100,0),0)

    def test_preview_preserves_quotas_and_all_8000_distinct_ids(self):
        rows = {str(i):plan(str(i),str(i),task_group=str(i),dataset='a' if i<6000 else 'b',
                            short_turn=i%10==0, gold_names=['RESPOND'] if i%10==0 else ['run'],
                            next_event='trajectory_end' if i%10==0 else None) for i in range(10000)}
        _, ids, report = preview(rows,{'a':5000,'b':3000},7,set(rows))
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),8000)
        self.assertEqual(report['selected']['overlap_with_reference'],8000)
        self.assertEqual(report['selected']['n_prompt_tokens'],800000)
        self.assertTrue(report['respond_gate_passed'])
        self.assertEqual(report['selected']['respond'],1000)
        self.assertEqual(report['redistribution'],[])

    def test_shortfall_does_not_redistribute_or_write_partial_selection(self):
        rows = {'a':plan('a',short_turn=True), 'b':plan('b',dataset='b')}
        _, ids, report = preview(rows,{'a':1,'b':1},7,[])
        self.assertEqual(ids,[])
        self.assertIsNone(report['selected'])
        self.assertEqual(report['shortfalls'],{'a':1})
        self.assertEqual(report['sources']['a']['final_quota'],1)

    def test_largest_remainder_is_exact_and_ties_match_source_name(self):
        quotas={SHORT_SOURCE:2,'b':3,'a':3}
        new,flows=reallocate_shortfall(quotas,{SHORT_SOURCE:1,'a':20,'b':20})
        self.assertEqual(new,{SHORT_SOURCE:1,'b':3,'a':4})
        self.assertEqual(flows,[dict(source=SHORT_SOURCE,destination='a',count=1,original_weight=3)])
        self.assertEqual(sum(new.values()),sum(quotas.values()))
        self.assertEqual(reallocate_shortfall(dict(reversed(list(quotas.items()))),{SHORT_SOURCE:1,'a':20,'b':20}),(new,flows))

    def test_reallocation_does_not_repeat_or_relax_capacity(self):
        with self.assertRaisesRegex(ValueError,'after single reallocation'):
            reallocate_shortfall({SHORT_SOURCE:6,'a':3,'b':3},{SHORT_SOURCE:2,'a':4,'b':100})
        with self.assertRaisesRegex(ValueError,'only'):
            reallocate_shortfall({SHORT_SOURCE:6,'a':3,'b':3},{SHORT_SOURCE:2,'a':2,'b':100})

    def test_redistributed_8000_rechecks_new_priority_and_reports_groups(self):
        rows={}
        for g in range(7):
            for t in range(20):
                for i in range(4):
                    sid=f'ds/{g}/{t}/{i}'
                    rows[sid]=plan(sid,f'{g}/{t}',dataset=SHORT_SOURCE,task_group=f'ds/{g}')
        ds_ids=list(rows)[:552]
        for dataset,n in (('a',4500),('b',4300)):
            for i in range(n):
                sid=f'{dataset}/{i}'
                rows[sid]=plan(sid,sid,dataset=dataset,task_group=sid,short_turn=i%10==0,
                    gold_names=['RESPOND'] if i%10==0 else ['run'])
        reference=ds_ids+[f'a/{i}' for i in range(4000)]+[f'b/{i}' for i in range(3448)]
        kept,ids,report=redistributed_preview(rows,{SHORT_SOURCE:552,'a':4000,'b':3448},7,reference)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),8000)
        self.assertEqual({d:r['final_quota'] for d,r in report['sources'].items()}, {SHORT_SOURCE:28,'a':4281,'b':3691})
        self.assertEqual(sum(r['count'] for r in report['redistribution']),524)
        self.assertEqual(report['sources']['a']['priority_quota'],round(.25*4281))
        self.assertEqual(report['reference']['task_groups'],7455)
        self.assertEqual(report['selected']['task_groups'],7979)
        self.assertEqual(report['selected']['overlap_with_reference'],len(set(ids)&set(reference)))
        for d,r in report['sources'].items():
            chosen=[kept[sid] for sid in ids if kept[sid]['dataset']==d]
            self.assertEqual(len(chosen),r['final_quota'])
            self.assertLessEqual(sum(p['short_turn'] or p['direct'] for p in chosen),r['priority_quota'])

    def test_task_groups_are_distinct_globally_not_sum_of_sources(self):
        rows={'a':plan('a'), 'b':plan('b',dataset='b')}
        _,_,report=preview(rows,{'a':1,'b':1},7,['a','b'])
        self.assertEqual(report['selected']['task_groups'],1)
        self.assertEqual(sum(r['selected']['task_groups'] for r in report['sources'].values()),2)

    def test_canonical_request_helper_preserves_builder_hash(self):
        history = [dict(role='system',content='DO NOT INCLUDE'),
                   dict(role='user',content='<environment_context>secret env</environment_context>\n work '),
                   dict(role='user',content=' second '), dict(role='assistant',content='done')]
        text, skipped = initial_request(history)
        self.assertEqual(text,'work\nsecond')
        e = event(history,kind='RESPOND',actions=())
        e.dataset='d';e.trajectory_id='t';e.turn_index=2;e.model='m';e.source='source'
        e.metadata.update(task_id=None,outcome='success',candidate_level='S')
        record = light(e,dict(source_task_ids={},swe_datasets=[],swe_repo_dataset='swe'),{})
        self.assertEqual(record['request_hash'],hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(record['skipped_framework_greeting'],skipped)

    def test_request_hash_checked_before_cleaning_and_truncation(self):
        request='[Sun 2026-10-18 10:00 UTC] <file path="x">abc</file>'+('z'*4000)
        sha=hashlib.sha256(request.encode()).hexdigest()
        self.assertEqual(clean_request(request,sha),'abc'+('z'*2997))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            clean_request(request,'0'*64)

    def test_fixed_conflict_and_invalid_api_label_fail(self):
        self.assertEqual(fixed_label(['Open-SWE-Traces','unknown']), 'repo_swe')
        with self.assertRaisesRegex(ValueError,'conflicting'):
            fixed_label(['Open-SWE-Traces','mle-trajectories'])
        for label in (None, '', 'repo_swe.', '"repo_swe"', 'Here is repo_swe'):
            with self.assertRaises(ValueError):
                parse_label(label)
        self.assertEqual(parse_label(' repo_swe\n'),'repo_swe')

    def test_group_representative_and_cross_split_error(self):
        records=[plan('a','t1'),plan('b','t2',dataset='Open-SWE-Traces')]
        requests={(r['dataset'],r['trajectory_id']):dict(task_group='g',split='train',request_hash='h',request=r['id']) for r in records}
        result=annotation_requests(records,requests,7)
        expected=min(records,key=lambda r:(digest(7,r['trajectory_id']),r['trajectory_id'],r['dataset']))
        self.assertEqual(result[0]['request'],expected['id'])
        self.assertEqual(result[0]['fixed_label'],'repo_swe')
        records[1]=dict(records[1],split='val')
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            annotation_requests(records,requests,7)


if __name__=='__main__':
    unittest.main()
