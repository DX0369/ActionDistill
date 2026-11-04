import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(R/'third_party/decider'),
               str(R/'experiments/dataset_tool_inventory/src')]
import train_jev as training
from jev_train.training_history import read_training_history, resume_boundary


def row(step, total=5):
    return dict(event='train', step=step, total=total, loss=float(step), total_loss=float(step), lr=.01, grad_norm=1.)


class ResumeTests(unittest.TestCase):
    def setup_run(self, root):
        (root/'metadata').mkdir(); cp=root/'checkpoints/step_0000002';cp.mkdir(parents=True)
        cfg={'world_size':1,'lr':1e-5}
        status=dict(status='failed', ended_at='old', error_type='OSError', error='disk full',
                    swanlab_run_id='original-id',configuration=cfg)
        (root/'metadata/run_status.json').write_text(json.dumps(status))
        events=[dict(event='data',train_count=8000)]+[row(n) for n in range(1,4)]
        (root/'train.log').write_text(''.join(json.dumps(e)+'\n' for e in events))
        state=dict(step=2,configuration=cfg,rank_rng=[{}],loss_history=[1.,2.])
        torch.save(state,cp/'state.pt')
        return cp,cfg,status,state

    def test_resume_keeps_failed_tail_and_reconstructs_effective_history(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
            root=Path(folder);cp,cfg,prior,state=self.setup_run(root)
            old=(root/'train.log').read_bytes()
            loaded,previous,boundary=training.prepare_resume(root,cp,cfg,1,False)
            self.assertEqual(loaded['step'],2);self.assertEqual(previous,prior)
            self.assertEqual(boundary['metric_prefix'],'resume_1/')
            with (root/'train.log').open('a') as f:
                for e in [boundary,dict(event='data',train_count=8000),row(3),row(4),row(5)]:
                    f.write(json.dumps(e)+'\n')
            view=read_training_history(root/'train.log')
            self.assertTrue((root/'train.log').read_bytes().startswith(old))
            self.assertEqual([r['step'] for r in view['train']],[1,2,3,4,5])
            second=resume_boundary(view,cp,state,prior,'later')
            self.assertEqual(second['metric_prefix'],'resume_2/')
            self.assertEqual(view['resumes'][0]['previous_status']['error'],'disk full')

    def test_resume_preflight_errors_do_not_modify_status_or_log(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
            root=Path(folder);cp,cfg,prior,state=self.setup_run(root)
            saved={p:p.read_bytes() for p in (root/'train.log',root/'metadata/run_status.json')}
            for current in ({**cfg,'lr':3e-5},{**cfg,'world_size':2}):
                with self.assertRaises(ValueError):training.prepare_resume(root,cp,current,1,False)
            (root/'checkpoints/step_0000004').mkdir()
            with self.assertRaisesRegex(ValueError,'later checkpoint'):
                training.prepare_resume(root,cp,cfg,1,False)
            for p,b in saved.items():self.assertEqual(p.read_bytes(),b)

    def test_rejects_unmarked_duplicates_gaps_nonfinite_and_incomplete_lines(self):
        for extra in ([row(3)],[row(5)],[dict(row(4),loss=float('nan'))],[dict(event='data',train_count=8000)]):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
                root=Path(folder);self.setup_run(root)
                with (root/'train.log').open('a') as f:
                    for e in extra:f.write(json.dumps(e)+'\n')
                with self.assertRaises(ValueError):read_training_history(root/'train.log')
        with tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
            root=Path(folder);self.setup_run(root)
            with (root/'train.log').open('a') as f:f.write('{')
            with self.assertRaisesRegex(ValueError,'incomplete'):read_training_history(root/'train.log')

    def test_resume_rejects_existing_final_exports_without_mutation(self):
        for artifact in ('model','training_majority_tools.json'):
            with self.subTest(artifact=artifact),tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
                root=Path(folder);cp,cfg,prior,state=self.setup_run(root)
                saved={p:p.read_bytes() for p in (root/'train.log',root/'metadata/run_status.json')}
                (root/artifact).touch()
                with self.assertRaisesRegex(ValueError,'final exports'):
                    training.prepare_resume(root,cp,cfg,1,False)
                for p,b in saved.items():self.assertEqual(p.read_bytes(),b)

    def test_rejects_invalid_resume_boundary_and_checkpoint_history(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as folder:
            root=Path(folder);cp,cfg,prior,state=self.setup_run(root)
            view=read_training_history(root/'train.log')
            with self.assertRaisesRegex(ValueError,'loss history differs'):
                resume_boundary(view,cp,dict(state,loss_history=[1.,9.]),prior,'now')
            event=resume_boundary(view,cp,state,prior,'now');event['previous_last_step']=99
            with (root/'train.log').open('a') as f:f.write(json.dumps(event)+'\n')
            with self.assertRaisesRegex(ValueError,'invalid resume boundary'):read_training_history(root/'train.log')

    def test_rng_restore_uses_cpu_checkpoint_and_same_rank(self):
        s=dict(python_rng='p',numpy_rng='n',torch_rng=torch.tensor([1],dtype=torch.uint8),
               cuda_rng=torch.tensor([2],dtype=torch.uint8))
        with patch.object(training.random,'setstate') as p,patch.object(training.np.random,'set_state') as n,\
             patch.object(training.torch,'set_rng_state') as t,patch.object(training.torch.cuda,'set_rng_state') as c:
            training.restore_rng(s)
        p.assert_called_once_with('p');n.assert_called_once_with('n')
        self.assertEqual(t.call_args.args[0].device.type,'cpu')
        self.assertEqual(c.call_args.args[0].device.type,'cpu')


if __name__=='__main__':unittest.main()
