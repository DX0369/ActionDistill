from collections import Counter
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(P/'src'),str(P/'scripts'),str(P.parents[1]/'third_party/decider'),str(P.parent/'dataset_tool_inventory/src')]
from jev_train.domains import apportion,capped_allocation,allocation_flows,balanced_selection
from test_dedup import plan


class DomainSelectionTests(unittest.TestCase):
    def test_capacity_redistribution_and_conservation(self):
        self.assertEqual(apportion({'a':1,'b':1,'c':1},8),{'a':3,'b':3,'c':2})
        initial=apportion({'a':4,'b':2,'c':1},70)
        final=capped_allocation({'a':4,'b':2,'c':1},70,{'a':8,'b':21,'c':60})
        self.assertEqual(final,{'a':8,'b':21,'c':41})
        flows=allocation_flows(initial,final)
        rebuilt=dict(initial)
        for f in flows:
            rebuilt[f['source']]-=f['count'];rebuilt[f['destination']]+=f['count']
        self.assertEqual(rebuilt,final)
        with self.assertRaisesRegex(ValueError,'cannot fill'):
            capped_allocation({'a':1,'b':1},3,{'a':1,'b':1})

    def test_global_redistribution_uses_b2_capacity(self):
        rows={};labels={}
        for label,n,priority in [('repo_swe',10,False),('data_file',10,False),('other',20,True)]:
            for i in range(n):
                sid=f'{label}/{i}';labels[sid]=label
                rows[sid]=plan(sid,sid,task_group=sid,short_turn=priority)
        ids,report=balanced_selection(rows,labels,{'a':10},10,7)
        self.assertEqual(Counter(labels[rows[s]['task_group']] for s in ids),{'repo_swe':5,'data_file':5})
        self.assertEqual(report['domains']['other']['final_quota'],0)
        self.assertTrue(report['redistribution']['across_domains'])
        self.assertEqual(report['cells']['other']['a']['b2_capacity'],0)

    def test_8000_with_cell_shortfall_reproducibility_and_budget(self):
        rows={};labels={}
        for ds,domain,n in [('a','repo_swe',4500),('b','repo_swe',3000),('a','data_file',500),('b','data_file',4000)]:
            for i in range(n):
                sid=f'{ds}/{domain}/{i}';labels[sid]=domain
                rows[sid]=plan(sid,sid,dataset=ds,task_group=sid,short_turn=i%10==0,
                    gold_names=['RESPOND'] if i%10==0 else ['run'])
        ids,report=balanced_selection(rows,labels,{'a':5000,'b':3000},8000,7)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),8000)
        self.assertEqual(report['cells']['data_file']['a']['final_quota'],500)
        self.assertTrue(report['redistribution']['within_domains']['data_file'])
        self.assertTrue(report['respond_gate_passed'])
        for sources in report['cells'].values():
            for r in sources.values():
                self.assertLessEqual(r['final_quota'],r['b2_capacity'])
                self.assertLessEqual(r['priority_selected'],round(.25*r['final_quota']))
        self.assertEqual((ids,report),balanced_selection(dict(reversed(list(rows.items()))),labels,{'b':3000,'a':5000},8000,7))

    def test_missing_or_invalid_labels_fail(self):
        rows={'s':plan('s')}
        with self.assertRaises(KeyError):balanced_selection(rows,{}, {'a':1},1,7)
        with self.assertRaisesRegex(ValueError,'invalid domain'):
            balanced_selection(rows,{'g':'unknown'}, {'a':1},1,7)

    def test_cross_source_task_counts_once_in_domain_weight(self):
        rows={'a':plan('a'),'b':plan('b',dataset='b')}
        ids,report=balanced_selection(rows,{'g':'repo_swe'},{'a':1,'b':1},2,7)
        self.assertEqual(report['domains']['repo_swe']['G_D'],1)
        self.assertEqual(report['selected']['task_groups'],1)

    def test_cross_domain_additions_never_revoke_existing_cell_seats(self):
        rows={};labels={'g1':'data_file','g2':'repo_swe'}
        for ds,n in [('a',2),('b',1),('c',1)]:
            for i in range(n):
                sid=f'data/{ds}/{i}'
                rows[sid]=plan(sid,f'data/{ds}',dataset=ds,task_group='g1')
        for ds in ('a','b','c'):
            for i in range(4):
                sid=f'repo/{ds}/{i}'
                rows[sid]=plan(sid,f'repo/{ds}',dataset=ds,task_group='g2')
        _,report=balanced_selection(rows,labels,{'a':5,'b':3,'c':1},9,7)
        cells=report['cells']['repo_swe']
        self.assertEqual([cells[d]['initial_quota'] for d in ('a','b','c')],[2,1,1])
        self.assertEqual([cells[d]['final_quota'] for d in ('a','b','c')],[3,1,1])


if __name__=='__main__':unittest.main()
