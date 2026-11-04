import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'scripts'))
from compare_evals import clustered_pair,subset,domain_accuracy,read_domain_labels,summary
import compare_evals
from jev_train.evaluation import measured


def dump(hits):
    n=len(hits);logits=torch.full((n,255),-1e9);targets=torch.zeros((n,255));targets[:,0]=1
    for i,hit in enumerate(hits):logits[i,:2]=torch.tensor([2.,0.] if hit else [0.,2.])
    return dict(id=[str(i) for i in range(n)],logits=logits,targets=targets,nopts=[2]*n,
        dataset=['a']*n,outcome=['success']*n,candidate_level=['trajectory']*n,
        tool_names=[['run','RESPOND'] for _ in hits],transition=['first']*n,copy_prev_hit=[False]*n)


class CompareTests(unittest.TestCase):
    def test_val_only_comparison_is_explicit_and_default_still_requires_tests(self):
        values=dump([1,0,1]);majority={'a':'run'}
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);paths=[root/'a',root/'b']
            for path in paths:
                (path/'metadata').mkdir(parents=True)
                (path/'metadata/run_status.json').write_text(json.dumps(dict(configuration=dict(samples=str(root/'samples'),majority=majority))))
                (path/'eval.json').write_text(json.dumps(dict(val=measured(values,2,train_majority=majority),temperature={'temperature':2})))
            def saved(path,split):
                if split!='val':raise ValueError('requested split absent')
                return values,7
            for partial in (True,False):
                output=root/('val_only' if partial else 'full')
                argv=['compare_evals.py','--eval',f'a={paths[0]}','--eval',f'b={paths[1]}','--reference','a','--out',str(output)]
                if partial:argv+=['--splits','val']
                with patch.object(sys,'argv',argv),patch.object(compare_evals,'completed_parts',side_effect=saved),\
                     patch.object(compare_evals,'canonical_groups',return_value={'0':'big','1':'big','2':'small'}),redirect_stdout(io.StringIO()):
                    if partial:compare_evals.main()
                    else:
                        with self.assertRaisesRegex(ValueError,'absent'):compare_evals.main()
                if partial:
                    result=json.loads((output/'summary.json').read_text())
                    self.assertEqual(set(result['splits']),{'val'})
                    self.assertEqual(result['splits']['val']['a']['task_equal_acc'],.75)
                    self.assertEqual(result['val_ranking'],['a','b'])

    def test_cluster_whole_tasks_task_equal_and_permutation(self):
        a,b=dump([1,1,1,0]),dump([0,0,0,1]);groups={'0':'big','1':'big','2':'big','3':'small'}
        result=clustered_pair(a,b,groups,7,7)
        self.assertEqual(result['acc_difference'],.5)
        self.assertEqual(result['task_equal_difference'],0)
        self.assertEqual(result['task_sign_test_p'],1)
        self.assertEqual(result['cluster_bootstrap_ci95'],[-1,1])
        self.assertEqual(result,clustered_pair(subset(a,[3,1,0,2]),b,groups,7,7))
        same=clustered_pair(a,a,groups,7,7)
        self.assertEqual(same['task_sign_test_p'],1)
        with self.assertRaises(KeyError):clustered_pair(a,b,{},7,7)
        with self.assertRaises(ValueError):clustered_pair(a,b,groups,7,8)
        changed=dump([0,0,0,1]);changed['targets'][0]=0
        with self.assertRaises(ValueError):clustered_pair(a,changed,groups,7,7)

    def test_domain_count_boundary_and_missing_label(self):
        values=dump([1]*59);groups={str(i):str(i) for i in range(59)}
        labels={str(i):dict(label='repo_swe' if i<29 else 'data_file',split='test_id') for i in range(59)}
        result=domain_accuracy(values,groups,labels)
        self.assertTrue(result['repo_swe']['descriptive_only']);self.assertFalse(result['data_file']['descriptive_only'])
        del labels['0']
        with self.assertRaises(KeyError):domain_accuracy(values,groups,labels)
        labels['0']=dict(label='unknown',split='test_id')
        with self.assertRaises(ValueError):domain_accuracy(values,groups,labels)

    def test_subset_uses_temperature_and_preserves_respond_and_ece_semantics(self):
        values=subset(dump([1,1,0]),[0,1]);result=summary(measured(values,2,train_majority={'a':'run'}))
        self.assertIsNone(result['respond']['precision'])
        self.assertEqual(result['ece_count'],2)
        self.assertNotEqual(result['nll'],summary(measured(values,1,train_majority={'a':'run'}))['nll'])

    def test_label_duplicate_or_missing_required_field_fails(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);(root/'metadata').mkdir()
            preview=root/'preview';preview.mkdir();requests=root/'requests.jsonl'
            request=dict(task_group='g',split='test_id',member_datasets=['a'],request_sha256='a'*64,request='fix bug',fixed_label=None)
            requests.write_text(json.dumps(request)+'\n')
            (preview/'summary.json').write_text(json.dumps(dict(recovery_inputs=dict(annotation_requests=str(requests)))))
            status=dict(status='completed',failure_count=0,review=dict(count=100,agreement=.95),item_count=1,completed_count=1,preview=str(preview))
            (root/'metadata/run_status.json').write_text(json.dumps(status))
            row=dict(task_group='g',label='repo_swe',split='test_id',label_source='api',member_datasets=['a'],request_sha256='a'*64)
            path=root/'labels.jsonl';path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            with self.assertRaises(ValueError):read_domain_labels(root)
            del row['request_sha256'];path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):read_domain_labels(root)
            row['request_sha256']='b'*64;path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'input differs'):read_domain_labels(root)

    def test_full_cli_uses_canonical_majority_mapping_and_saved_temperature(self):
        values=dump([1,0,1]);values['dataset']=['a','a',compare_evals.EXCLUDED]
        majority={'a':'run',compare_evals.EXCLUDED:'run'}
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);a=root/'a';b=root/'b';output=root/'comparison'
            for path in (a,b):
                (path/'metadata').mkdir(parents=True)
                (path/'metadata/run_status.json').write_text(json.dumps(dict(configuration=dict(samples=str(root/'samples'),majority=majority))))
                report={split:measured(values,2,train_majority=majority) for split in compare_evals.SPLITS}
                report['temperature']={'temperature':2};(path/'eval.json').write_text(json.dumps(report))
            argv=['compare_evals.py','--eval',f'a={a}','--eval',f'b={b}','--reference','a','--out',str(output)]
            with patch.object(sys,'argv',argv),patch.object(compare_evals,'completed_parts',return_value=(values,7)),\
                 patch.object(compare_evals,'canonical_groups',return_value={str(i):str(i) for i in range(3)}),redirect_stdout(io.StringIO()):
                compare_evals.main()
            result=json.loads((output/'summary.json').read_text())
            excluded=result['splits']['val']['b']['without_deepseek_0813']
            self.assertEqual(excluded['temperature'],2);self.assertEqual(excluded['excluded_count'],1)
            self.assertEqual(excluded['clustered_vs_reference']['count'],2)


if __name__=='__main__':unittest.main()
