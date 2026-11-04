import sys,unittest,json,tempfile,random,math
from pathlib import Path
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path[:0]=[str(P/'src'),str(R/'third_party/decider'),str(R/'experiments/dataset_tool_inventory/src')]
import decider
from jev_train.loss import soft_loss,metrics,fit_temperature
from jev_train.data import SampleIndex,make_item,load_selected,tool_name
from jev_train.render import prompt_item
from jev_train.evaluation import evaluate,measured,training_majority_tools,run_parts,read_parts,plan_parts,load_majority,paired_difference,transition_metadata
from decider.train import batches_by_tokens
from test_build import Tokenizer

class TrainTests(unittest.TestCase):
    def test_global_microbatch_cursor(self):
        sys.path.insert(0,str(P/'scripts'))
        import train_jev as training
        items=[{'ids':range(n)} for n in (12,19,25,31,38)]
        for smoke in (False,True):
            epoch=cursor=0;old_epoch=old_cursor=0
            for step in range(9):
                expected=[]
                while len(expected)<2:
                    batches=batches_by_tokens(items,40,random.Random(7+old_epoch))
                    if smoke:
                        j=next(j for j,b in enumerate(batches) if 4 in b);batches.insert(0,batches.pop(j))
                    if old_cursor>=len(batches):old_epoch+=1;old_cursor=0;continue
                    expected.append(batches[old_cursor]);old_cursor+=1
                actual=training.next_step_batches(items,40,7,epoch,cursor,2,smoke,4)
                self.assertEqual(actual,(expected,old_epoch,old_cursor))
                epoch,cursor=actual[1:]
                if step==2:resume=(epoch,cursor)
                if step==3:resumed_expected=(expected,old_epoch,old_cursor)
            self.assertGreater(epoch,0)
            self.assertNotEqual(resume[1],0)
            # A fresh invocation has no hidden iterator state at resume.
            self.assertEqual(training.next_step_batches(items,40,7,*resume,2,smoke,4),resumed_expected)

    def test_training_steps_and_distributed_boundaries(self):
        sys.path.insert(0,str(P/'scripts'))
        import train_jev as training
        for count in (1,5,12):
            for accum in (1,2,3):
                for epochs in (1.,2.,1.5):
                    old=max(1,int(math.ceil(count/accum)*epochs))
                    self.assertEqual(training.total_steps(count,1,accum,epochs,0),old)
                    self.assertEqual(training.total_steps(count,1,2*accum,epochs,0),
                                     training.total_steps(count,2,accum,epochs,0))
        self.assertEqual(training.total_steps(12,2,1,2.,5),5)
        self.assertEqual(training.distributed_environment({}),(1,0,0))
        self.assertEqual(training.distributed_environment({'WORLD_SIZE':'2','RANK':'1','LOCAL_RANK':'1'}),(2,1,1))
        for invalid in ({'WORLD_SIZE':'2'},{'WORLD_SIZE':'2','RANK':'2','LOCAL_RANK':'0'}):
            with self.assertRaises(ValueError):training.distributed_environment(invalid)
        training.check_device_count(2,2)
        with self.assertRaisesRegex(ValueError,'device_count=1, world_size=2'):training.check_device_count(1,2)
        training.check_resume_configuration({'world_size':2},{'world_size':2})
        with self.assertRaisesRegex(ValueError,'checkpoint world_size=2, current=1'):
            training.check_resume_configuration({'world_size':2},{'world_size':1})
        with self.assertRaisesRegex(ValueError,'old commit'):training.check_resume_configuration({}, {'world_size':1})

    def test_tool_name_boundaries(self):
        for display in ('write_file (params: path, content)', 'write_file: Write a file',
                        'write_file: Write a file (params: path)', 'write_file',
                        'write_file: Description contains (params: example)'):
            with self.subTest(display=display):self.assertEqual(tool_name(display),'write_file')

    def test_positional_paired_difference(self):
        current={'id':['x','y','z'],'nopts':[3,3,3],
                 'targets':torch.tensor([[1.,0.,0.],[.5,.5,0.],[0.,0.,1.]]),
                 'logits':torch.tensor([[2.,1.,0.],[0.,2.,1.],[2.,1.,0.]]),
                 'tool_names':[['new','names','RESPOND']]*3}
        reference={'id':['z','y','x'],'nopts':[3,3,3],
                   'targets':current['targets'].flip(0),
                   'logits':torch.tensor([[2.,1.,0.],[0.,0.,2.],[2.,1.,0.]]),
                   'tool_names':[['old (params','old (params','RESPOND']]*3}
        difference=paired_difference(current,reference,seed=7,reference_seed=7)
        acc=lambda dump:float((dump['targets'].gather(1,dump['logits'].argmax(1)[:,None])>0).float().mean())
        self.assertAlmostEqual(difference['acc_difference'],acc(current)-acc(reference))
        self.assertEqual((difference['improved'],difference['regressed'],difference['discordant']),(1,0,1))
        self.assertEqual(difference['sign_test_p'],1.0)
        from jev_train.evaluation import sign_test
        self.assertEqual((sign_test(0,0),sign_test(5,0),sign_test(3,3)),(None,0.0625,1.0))
        with self.assertRaisesRegex(ValueError,'seeds'):paired_difference(current,reference,seed=7,reference_seed=8)
        changed=dict(reference,targets=reference['targets'].clone());changed['targets'][0]=torch.tensor([1.,0.,0.])
        for invalid in (changed,dict(reference,nopts=[2,3,3])):
            with self.assertRaisesRegex(ValueError,'targets/nopts'):paired_difference(current,invalid,seed=7,reference_seed=7)

    def test_prediction_diagnostics_and_stale_majority(self):
        dump={'dataset':['d']*5+['e'],'tool_names':[['a','b','RESPOND']]*6,
              'outcome':['unknown']*6,'candidate_level':['S']*6,
              'transition':['first','repeat','change','repeat','change','first'],'copy_prev_hit':[False,True,False,True,False,False],
              'targets':torch.tensor([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,1.],[.5,.5,0.]]),
              'logits':torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.],[2.,0.,0.],[0.,0.,2.],[0.,0.,2.]])}
        absent=measured(dump,train_majority={'d':'a (params'})['datasets']['d']
        self.assertEqual((absent['majority_tool_candidate_count'],absent['train_majority_tool_acc']),(0,0.0))
        result=measured(dump,train_majority={'d':'a','e':'b'})
        d=result['datasets']['d']
        self.assertEqual(d['respond'],{'gold_count':2,'predicted_count':2,'true_positive_count':1,'recall':.5,'precision':.5})
        self.assertEqual(d['tool_prediction']['a'],{'gold_count':2,'predicted_count':2,'true_positive_count':1,'recall':.5,'precision':.5})
        self.assertEqual(d['tool_prediction']['b']['precision'],1.)
        self.assertEqual(d['tool_prediction']['b']['recall'],.5)
        self.assertNotIn('RESPOND',d['tool_prediction'])
        self.assertEqual(result['datasets']['e']['top_confusions'],[{'gold':'a','predicted':'RESPOND','count':1}])
        self.assertIsNone(result['datasets']['e']['respond']['recall'])
        self.assertEqual(result['overall']['respond'],{'precision':1/3,'recall':.5})
        self.assertIn('diagnostic_definition',result)
        self.assertAlmostEqual(result['overall']['copy_prev_acc'],1/3)
        self.assertEqual(d['copy_prev_acc'],.4)
        self.assertEqual(result['overall']['by_transition']['repeat'],{'acc':.5,'count':2})
        self.assertEqual(result['datasets']['e']['by_transition']['change'],{'acc':None,'count':0})
        self.assertEqual(transition_metadata({'prev_action_names':['a'],'transition':'change','options':['a','b'],'gold':[0,1]}),('change',True))
        for previous,gold,expected in [([], [0], False),(['a'],[0],True),(['b'],[0],False),
                                        (['a','b'],[0,1],True),(['a','b'],[0],False),(['a','b'],[1,2],False),(['a','b'],[0,1,2],False),
                                        (['RESPOND'],[3],True),(['RESPOND'],[0],False)]:
            self.assertEqual(transition_metadata({'prev_action_names':previous,'transition':'change','options':['a','b','c','RESPOND'],'gold':gold})[1],expected)
        with self.assertRaises(KeyError):transition_metadata({'transition':'first'})

    def test_evaluation_model_source_arguments(self):
        import io
        from contextlib import redirect_stderr
        sys.path.insert(0,str(P/'scripts'));import eval_jev
        args=['eval_jev.py','--samples',str(P/'tmp'),'--out',str(P/'tmp/eval'),
              '--majority','unused.json','--limit_val','1','--limit_test_id','1','--limit_test_ood','1','--seed','7','--gpus','7']
        for source in ([],['--run',str(P),'--model',str(P)],['--model',str(P/'tmp')]):
            with self.subTest(source=source),patch.object(sys,'argv',args+source),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:eval_jev.main()
                self.assertEqual(exc.exception.code,2)

    def test_lazy_evaluation_parts_and_resume(self):
        tok=Tokenizer();tok.pad_token_id=0
        records=[]
        for i in range(7):
            row={'id':str(i),'dataset':'d','model':'teacher','task_group':'g','outcome':'success','candidate_level':'S',
                 'prev_action_names':['a'],'transition':'repeat',
                 'context':'x'*(20-i),'question':'q','options':['a: Read.','b: Write!','RESPOND: reply'],'gold':[0],
                 'n_prompt_tokens':0,'messages_dropped':0,'truncation_counts':{}}
            row['n_prompt_tokens']=len(prompt_item(row['context'],row['question'],row['options'],row['gold'],tok)['ids'])
            records.append(row)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.tok=tok
            def forward(self,batch):
                logits=torch.full((len(batch['nopts']),255),float('-inf'))
                logits[:,:3]=0;logits[:,0]=batch['input_ids'].sum(1).float()
                return logits
        model=Model()
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            folder=Path(directory);path=folder/'samples/train.jsonl';path.parent.mkdir()
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            index=SampleIndex(path);entries=load_selected(index)
            self.assertEqual([index.read(e['offset']) for e in entries],records)
            actual=[make_item(r,tok,seed=6) for r in records]
            self.assertEqual(batches_by_tokens(actual,32768,random.Random(6)),batches_by_tokens([{'ids':range(e['n_prompt_tokens'])} for e in entries],32768,random.Random(6)))
            full=evaluate(model,index,entries,seed=6)
            for i,value in enumerate(full['id']):self.assertEqual(full['logits'][i,0],sum(actual[int(value)]['ids']))
            majority=training_majority_tools(records)
            self.assertEqual(majority,{'d':'a'})
            self.assertEqual(measured(full,train_majority=majority)['overall']['train_majority_tool_acc'],1.)
            assigned=[e['id'] for k in range(2) for group in plan_parts(entries,k,2,2) for e in group]
            self.assertEqual(sorted(assigned),sorted(e['id'] for e in entries))
            import jev_train.evaluation as evaluation
            original=evaluation.evaluate;calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt()
                return original(*args,**kwargs)
            with patch.object(evaluation,'evaluate',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):run_parts(model,index,entries,seed=6,shard=0,shards=1,directory=folder/'parts/0',part_size=2)
            with self.assertRaises(ValueError):read_parts(entries,shards=1,directory=folder/'parts',part_size=2)
            first=folder/'parts/0/part_000000.pt';stamp=first.stat().st_mtime_ns
            run_parts(model,index,entries,seed=6,shard=0,shards=1,directory=folder/'parts/0',part_size=2)
            self.assertEqual(first.stat().st_mtime_ns,stamp)
            resumed=read_parts(entries,shards=1,directory=folder/'parts',part_size=2)
            self.assertEqual(resumed['id'],full['id']);self.assertTrue(torch.equal(resumed['logits'],full['logits']))
            self.assertEqual(paired_difference(resumed,full,seed=6,reference_seed=6)['discordant'],0)
            with self.assertRaises(ValueError):paired_difference(resumed,dict(full,id=['changed',*full['id'][1:]]),seed=6,reference_seed=6)
            with self.assertRaises(FileNotFoundError):load_majority(folder/'missing.json')

    def test_one_hot_ce_and_invalid_inf_no_nan(self):
        logits=torch.full((2,255),float('-inf'));logits[:,:3]=torch.tensor([[2.,1.,0.],[0.,1.,2.]])
        targets=torch.zeros_like(logits);targets[0,0]=1;targets[1,2]=1
        loss,ce,brier=soft_loss(logits,targets,torch.tensor([3,3]))
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(brier))
        self.assertTrue(torch.allclose(ce,F.cross_entropy(logits,torch.tensor([0,2]))))
        with self.assertRaises(ValueError):soft_loss(logits,targets*0,torch.tensor([3,3]))
    def test_soft_target_gradient(self):
        logits=torch.zeros((1,255),requires_grad=True);targets=torch.zeros_like(logits);targets[0,:2]=.5
        loss,_,_=soft_loss(logits,targets,torch.tensor([255]));loss.backward()
        self.assertAlmostEqual(float(logits.grad[0,0]),float(logits.grad[0,1]))
        self.assertLess(float(logits.grad[0,0]),float(logits.grad[0,2]))
    def test_permutation_targets_and_no_silent_truncation(self):
        row={'id':'sample','dataset':'d','model':'teacher','task_group':'g','outcome':'success','candidate_level':'S',
             'context':'context','question':'q','options':['one','two','RESPOND: reply'],'gold':[0,1],
             'n_prompt_tokens':0,'messages_dropped':0,'truncation_counts':{}}
        row['n_prompt_tokens']=len(prompt_item(row['context'],row['question'],row['options'],row['gold'],Tokenizer())['ids'])
        it=make_item(row,Tokenizer(),32768,6)
        with self.assertRaisesRegex(ValueError,'length invariant'):make_item(dict(row,n_prompt_tokens=row['n_prompt_tokens']+1),Tokenizer(),32768,6)
        self.assertEqual(float(it['targets'].sum()),1.)
        for j,index in enumerate(it['perms'][0]):self.assertEqual(float(it['targets'][j]),.5 if index in (0,1) else 0.)
        with self.assertRaises(ValueError):make_item(row,Tokenizer(),max_len=1)
    def test_set_hit_metrics_and_calibration(self):
        logits=torch.tensor([[2.,1.,float('-inf')],[0.,2.,1.]])
        targets=torch.tensor([[.5,.5,0.],[0.,1.,0.]])
        result=metrics(logits,targets,['a','b'],[True,True])
        self.assertEqual(result['overall']['acc'],1.)
        self.assertAlmostEqual(result['overall']['uniform_random_acc'],(1+1/3)/2,places=5)
        self.assertGreater(fit_temperature(logits,targets)['temperature'],0)
    def test_single_answer_ece_and_action_slices(self):
        logits=torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]])
        targets=torch.tensor([[1.,0.,0.],[0.,1.,0.],[.5,0.,.5]])
        result=metrics(logits,targets,['a','a','b'],slices={'action_kind':['RESPOND','TOOL','TOOL_SET']},respond_hits=[True,False,False])
        self.assertEqual(result['overall']['ece_count'],2)
        self.assertAlmostEqual(result['overall']['ece'],1-float(torch.softmax(logits[0],0).max()),places=6)
        self.assertIsNone(result['datasets']['b']['ece'])
        self.assertEqual(result['overall']['by_action_kind'],{k:{'count':1,'acc':1.} for k in ('RESPOND','TOOL','TOOL_SET')})
        self.assertAlmostEqual(result['overall']['always_respond_acc'],1/3)
        self.assertEqual(result['overall']['macro_by_dataset'],1.)
        logits[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):metrics(logits,targets,['a','a','b'])

    def test_first_learning_rate_and_boolean_config(self):
        sys.path.insert(0,str(P/'scripts'))
        import train_jev as training
        self.assertAlmostEqual(training.learning_rate(0,1000,1e-5,150),1e-5/150)
        with self.assertRaises(ValueError):training.reject_boolean_config({'warmup':False})
        boundary=fit_temperature(torch.tensor([[2.,0.]]),torch.tensor([[1.,0.]]))
        self.assertTrue(boundary['at_grid_boundary'])

    def test_selection_and_monitor_coverage(self):
        row={'id':'sample','dataset':'d','model':'teacher','task_group':'g','outcome':'success','candidate_level':'S',
             'context':'context','question':'q','options':['one','RESPOND: reply'],'gold':[0],
             'n_prompt_tokens':120,'messages_dropped':0,'truncation_counts':{}}
        (P/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            path=Path(directory)/'samples/train.jsonl';path.parent.mkdir()
            records=[dict(row,id=str(i)) for i in range(10)]+[dict(row,id='rare',dataset='rare')]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            index=SampleIndex(path)
            # Golden IDs/order from the pre-index load_selected on this fixture.
            self.assertEqual([r['id'] for r in load_selected(index,2,0)],['5','2'])
            chosen=load_selected(index,2,min_per_dataset=True)
            self.assertEqual({r['dataset'] for r in chosen},{'d','rare'})
            with self.assertRaises(ValueError):load_selected(index,1,min_per_dataset=True)
            for i,bad in enumerate((0,-1,True)):
                invalid=path.with_name(f'invalid_{i}.jsonl')
                invalid.write_text(json.dumps(dict(row,n_prompt_tokens=bad))+'\n')
                with self.assertRaisesRegex(ValueError,'n_prompt_tokens'):SampleIndex(invalid)

    def test_decider_source(self):
        self.assertTrue(Path(decider.__file__).resolve().is_relative_to(R/'third_party/decider'))

if __name__=='__main__':unittest.main()
