import json,sys,tempfile,unittest,math
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path[:0]=[str(P/'src'),str(R/'third_party/decider'),str(R/'experiments/dataset_tool_inventory/src')]
from jev_train.selection import select,filter_pool,load_train_ids,load_pool,spread_order,pool_sample,proportional_quotas,take_quota,selection_summary,position_summary
from unittest.mock import patch


def record(d,t,i,**kw):
    return dict({'id':f'{d}/{t}/{i}','dataset':d,'trajectory_id':str(t),'turn_index':i,'task_group':f'{d}/g{t//2}',
                 'harness':'fixture','action_kind':'TOOL','gold_names':['abc'[i%3]],'outcome':'success',
                 'transition':'change','loop_state':'no','trajectory_decisions':30,'trajectory_position':i,'n_prompt_tokens':100+i},**kw)


class SelectionTests(unittest.TestCase):
    def config(self,name='C_transition'):
        return json.loads((P/f'configs/selection/{name}.json').read_text())

    def test_budgets_caps_reproducibility_and_spread(self):
        rows=[record(str(d),t,i,outcome='failure' if t%7==0 else 'success',transition='repeat' if i%5==0 else 'change') for d in range(8) for t in range(12) for i in range(30)]
        chosen,summary=select(rows,self.config(),160,7)
        self.assertEqual(len(chosen),160);self.assertEqual(select(list(reversed(rows)),self.config(),160,7),(chosen,summary))
        self.assertTrue(all(n<=8 for n in Counter((r['dataset'],r['trajectory_id']) for r in chosen).values()))
        for stats in summary['by_dataset'].values():
            self.assertLessEqual(stats['failure_fraction'],.3);self.assertLessEqual(stats['repeat_fraction'],.3);self.assertLessEqual(stats['dominant_tool_fraction'],.4)
        quotas,upper=proportional_quotas({'a':1.,'b':3.,'c':2.},{'a':100,'b':12,'c':100},40,5,30,7)
        self.assertEqual(quotas,{'a':9,'b':12,'c':19})
        self.assertTrue(all(5<=quotas[d]<=upper[d] for d in quotas))

    def test_reference_quotas_are_exact_and_never_reassigned(self):
        rows=[record(str(d),t,i,transition='repeat' if i%5==0 else 'change') for d in range(3) for t in range(12) for i in range(30)]
        quotas={'0':3,'1':0,'2':57}
        chosen,summary=select(rows,self.config(),60,7,quotas)
        self.assertEqual(Counter(r['dataset'] for r in chosen),Counter({'0':3,'2':57}))
        self.assertEqual((summary['final_quotas'],summary['quota_source']),(quotas,'reference'))
        self.assertEqual(select(rows,self.config(),60,7,quotas),(chosen,summary))
        for bad in ({'0':3,'2':57},{'0':3,'1':0,'2':56},{'0':3,'1':-1,'2':58}):
            with self.assertRaises(ValueError):select(rows,self.config(),60,7,bad)
        # Budgets bind: all-repeat dataset with a 30% repeat budget cannot fill a positive quota.
        stuck=[record('r',t,i,transition='repeat') for t in range(4) for i in range(10)]
        with self.assertRaisesRegex(ValueError,'infeasible'):select(stuck,self.config(),5,7,{'r':5})

    def test_discrete_quota_holes_and_no_relaxation(self):
        # Every state consumes exactly one of the three budgets: feasible q includes 10,20 but not 15.
        rows=[]
        for d in ('a','b'):
            for t in range(10):
                for i in range(10):
                    k=i%3
                    rows.append(record(d,t,i,action_kind='TOOL' if k<2 else 'RESPOND',gold_names=[('a','b','RESPOND')[k]],
                                       outcome='failure' if k==1 else 'success',transition='repeat' if k==2 else 'change'))
        config=dict(self.config(),min_fraction=.1,max_fraction=.8)
        _,summary=select(rows,config,30,7)
        self.assertEqual(sorted(summary['initial_quotas'].values()),[15,15]);self.assertEqual(sorted(summary['final_quotas'].values()),[10,20])
        with self.assertRaisesRegex(ValueError,'cannot fill'):select(rows,config,31,7)
        with self.assertRaisesRegex(ValueError,'minimum'):select([record('a',0,0)],config,30,7)
        flags=[(1,0,0),(1,1,1),(0,1,0),(0,0,1),(0,0,0)]
        nonmonotone=[record('a',i,i,action_kind='TOOL' if t else 'RESPOND',gold_names=['a'] if t else ['RESPOND'],
                            outcome='failure' if f else 'success',transition='repeat' if r else 'change') for i,(t,f,r) in enumerate(flags)]
        self.assertEqual(len(take_quota(nonmonotone,5,config,'a')),3)
        self.assertEqual(len(take_quota(nonmonotone,4,config,'a')),4)

    def test_spread_distribution_and_small_pool_cap(self):
        seed=20261015
        trajectories=[[record('d',t,i,task_group=str(t),trajectory_decisions=50,action_kind='TOOL_SET',gold_names=['a','b']) for i in range(50)] for t in range(1000)]
        def uniform(selected):
            counts=Counter(r['turn_index']//5 for r in selected)
            self.assertEqual(len(counts),10)
            self.assertTrue(all(abs(n/len(selected)-.1)<=.03 for n in counts.values()),counts)
        uniform([next(spread_order(t,seed,'d',t[0]['trajectory_id'])) for t in trajectories])
        rows=[r for t in trajectories for r in t];pool=pool_sample(rows,seed,16,3000)
        self.assertEqual(len(pool),3000);self.assertEqual(Counter(r['trajectory_id'] for r in pool),dict.fromkeys(map(str,range(1000)),3))
        uniform(pool)
        self.assertEqual(pool,pool_sample(list(reversed(rows)),seed,16,3000))
        # Independent second-stage phase also avoids correlation after only 3 pool states survive.
        chosen,_=select(pool,dict(self.config(),max_fraction=1.),1000,seed)
        self.assertEqual(len({r['trajectory_id'] for r in chosen}),1000);uniform(chosen)

    def test_spread_prefix_gaps_and_digest_edges(self):
        for n in (1,3,7,16,50,97):
            rows=[record('d',0,i,trajectory_decisions=n) for i in range(n)]
            order=list(spread_order(rows,20261015,'d','0'))
            self.assertEqual({r['id'] for r in order},{r['id'] for r in rows})
            for k in range(1,n+1):
                positions=sorted(r['turn_index'] for r in order[:k])
                gaps=[b-a for a,b in zip(positions,positions[1:])]+[n+positions[0]-positions[-1]]
                self.assertLessEqual(max(gaps),math.ceil(2*n/k))
        self.assertEqual(list(spread_order([],7,'d','0')),[])
        with patch('jev_train.selection.digest',return_value='f'*64):self.assertEqual(next(spread_order(rows,7,'d','0'))['turn_index'],96)
        with patch('jev_train.selection.digest',return_value='0'*64):self.assertEqual(next(spread_order(rows,7,'d','0'))['turn_index'],0)
        report=position_summary([record('d',0,0,trajectory_decisions=1),record('d',1,99,trajectory_position=5,trajectory_decisions=10)])
        self.assertEqual(report['bin_counts'],[1,0,0,0,0,1,0,0,0,0]);self.assertEqual(report['single_state_trajectory_count'],1)

    def test_concise_within_dataset_groups_and_loop_filter(self):
        rows=[record('a',0,0,task_group='shared',trajectory_decisions=50),
              record('a',1,0,task_group='shared',trajectory_decisions=10),
              record('b',0,0,task_group='shared',trajectory_decisions=20),
              record('b',1,0,task_group='shared',outcome='failure',trajectory_decisions=1),
              record('c',0,0,task_group='shared',outcome='unknown',trajectory_decisions=2),
              record('a',2,0,loop_state='yes'),record('a',3,0)]
        selected,report=filter_pool(rows,self.config('D_concise'),7)
        self.assertEqual({r['id'] for r in selected},{'a/1/0','b/0/0','c/0/0','a/3/0'})
        without_a,_=filter_pool([r for r in rows if r['dataset']!='a'],self.config('D_concise'),7)
        self.assertEqual(without_a,[r for r in selected if r['dataset']!='a'])
        self.assertEqual(report,{'input':7,'after_loop_filter':6,'after_concise_filter':4})

    def test_ids_and_required_pool_fields(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            path=Path(directory)/'ids.txt';index=SimpleNamespace(entries=[{'id':'a'},{'id':'b'}])
            path.write_text('b\na\n');selected=load_train_ids(index,path,7)
            path.write_text('a\nb\n');self.assertEqual(selected,load_train_ids(index,path,7))
            path.write_text('a\na\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):load_train_ids(index,path,7)
            path.write_text('missing\n')
            with self.assertRaisesRegex(ValueError,'missing'):load_train_ids(index,path,7)
            row=dict(record('d',0,0),harness=None,model='m',candidate_level='S',context='x',question='q',options=['a','RESPOND'],gold=[0],messages_dropped=0,truncation_counts={})
            path.write_text(json.dumps(row)+'\n');self.assertEqual(load_pool(path)[0]['gold_names'],['a'])
            self.assertEqual(set(selection_summary([load_pool(path)[0],record('e',0,0)],{'d':'a','e':'a'})['by_harness']),{None,'fixture'})
            for invalid in (-1,30,True,None):
                path.write_text(json.dumps(dict(row,trajectory_position=invalid))+'\n')
                with self.assertRaisesRegex(ValueError,'trajectory_position'):load_pool(path)
            missing=dict(row);del missing['trajectory_position'];path.write_text(json.dumps(missing)+'\n')
            with self.assertRaisesRegex(ValueError,'trajectory_position'):load_pool(path)
            del row['trajectory_id'];path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'trajectory_id'):load_pool(path)

if __name__=='__main__':unittest.main()
