import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'scripts'))
import eval_jev
from jev_train.data import SampleIndex, load_selected
from jev_train.evaluation import (
    COPY_PREV_RULE,
    calibration_temperature,
    completed_parts,
    plan_parts,
    read_parts,
    requested_splits,
)
from test_compare_evals import dump


def fixture(root):
    samples=root/'samples';samples.mkdir()
    rows=[dict(id=str(i),dataset='a',model='teacher',task_group=f'g{i}',outcome='success',candidate_level='S',
               context='history',question='next?',options=['run','RESPOND'],gold=[0],n_prompt_tokens=20,
               messages_dropped=0,truncation_counts={}) for i in range(3)]
    (samples/'val.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output=root/'dev';(output/'metadata').mkdir(parents=True)
    config=dict(model=str(root/'model'),model_source='run',samples=str(samples),limits=dict(val=1000,test_id=1000,test_ood=1000),
                seed=20261015,shards=1,majority={'a':'run'},copy_prev_rule=COPY_PREV_RULE,output=str(output),splits=['val'])
    entries=load_selected(SampleIndex(samples/'val.jsonl'),1000,20261015)
    values=dump([1,0,1]);values['id']=[e['id'] for e in sorted(entries,key=lambda e:(e['n_prompt_tokens'],e['id']))]
    directory=output/'shards/val/0';directory.mkdir(parents=True)
    torch.save(values,directory/'part_000000.pt')
    return config,values


class EvalSplitTests(unittest.TestCase):
    def test_cli_resumes_val_only_and_rejects_scope_or_calibration_changes(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);config,_=fixture(root);output=Path(config['output'])
            run=root/'train';model=run/'model';model.mkdir(parents=True);config['model']=str(model)
            majority=root/'majority.json';majority.write_text(json.dumps(config['majority']))
            state=output/'metadata/run_status.json';state.write_text(json.dumps(dict(status='running',configuration=config)))
            argv=['eval_jev.py','--run',str(run),'--samples',config['samples'],'--out',str(output),'--majority',str(majority),
                  '--limit_val','1000','--limit_test_id','1000','--limit_test_ood','1000','--seed','20261015','--gpus','2','--finalize_only']
            with patch.object(sys,'argv',argv+['--splits','val']),patch.object(eval_jev,'fit_temperature',return_value={'temperature':1.5}):
                eval_jev.main()
            self.assertEqual(json.loads(state.read_text())['status'],'completed')
            with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'resume configuration differs'):eval_jev.main()
            for extra in (['--splits','test_id','test_ood'],['--splits','val','--calibration_eval',str(output)],
                          ['--splits','test_id','test_ood','--calibration_eval',str(output.parent)]):
                with patch.object(sys,'argv',argv+extra),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):eval_jev.main()

    def test_default_scope_and_invalid_partial_scopes(self):
        self.assertEqual(requested_splits({}),('val','test_id','test_ood'))
        for splits in ([],['val','val'],['test_id'],['test_ood','test_id']):
            with self.assertRaises(ValueError):requested_splits(dict(splits=splits))

    def test_val_only_has_no_test_files_and_completed_scope_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            config,_=fixture(Path(directory));output=Path(config['output'])
            parts = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.rglob('part_*.pt')}
            with patch.object(eval_jev,'fit_temperature',return_value={'temperature':1.5}) as fit:
                eval_jev.finalize(config,config['majority'])
                fit.assert_called_once()
            self.assertEqual(parts, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in parts})
            report=json.loads((output/'eval.json').read_text())
            self.assertEqual(set(report),{'val','temperature'})
            (output/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',configuration=config)))
            self.assertEqual(len(completed_parts(output,'val')[0]['id']),3)
            with self.assertRaisesRegex(ValueError,'absent'):completed_parts(output,'test_id')
            self.assertFalse((Path(config['samples'])/'test_id.jsonl').exists())
            legacy=dict(config);del legacy['splits']
            (output/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',configuration=legacy)))
            with self.assertRaisesRegex(ValueError,'legacy'):completed_parts(output,'val')

    def test_configuration_required_fields_and_identity(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root = Path(directory); config, _ = fixture(root)
            run = root/'train'; model = run/'model'; model.mkdir(parents=True)
            config['model'] = str(model)
            majority = root/'majority.json'; majority.write_text(json.dumps(config['majority']))
            args = argparse.Namespace(run=run, model=None, samples=Path(config['samples']),
                out=Path(config['output']), majority=majority, seed=20261015,
                limit_val=1000, limit_test_id=1000, limit_test_ood=1000,
                splits=['val'], calibration_eval=None, finalize_only=False)
            (args.out/'metadata/run_status.json').write_text(json.dumps(dict(status='running', configuration=config)))
            for field, value in (('limit_val', True), ('seed', False), ('model', model)):
                invalid = argparse.Namespace(**{**vars(args), field: value})
                with self.subTest(field=field), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    eval_jev.prepare_configuration(invalid, 1, argparse.ArgumentParser())
            self.assertEqual(eval_jev.prepare_configuration(args, 1, argparse.ArgumentParser())[2], config)
            with self.assertRaisesRegex(ValueError, 'resume configuration differs'):
                eval_jev.prepare_configuration(args, 2, argparse.ArgumentParser())

    def test_completed_parts_reject_missing_extra_and_corrupt_ids(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            config, _ = fixture(Path(directory)); output = Path(config['output'])
            with patch.object(eval_jev, 'fit_temperature', return_value={'temperature': 1.5}):
                eval_jev.finalize(config, config['majority'])
            (output/'metadata/run_status.json').write_text(json.dumps(dict(status='completed', configuration=config)))
            path = output/'shards/val/0/part_000000.pt'
            extra = path.with_name('part_000001.pt'); original = path.read_bytes()
            for corruption in ('missing', 'extra', 'wrong_id'):
                with self.subTest(corruption=corruption):
                    if corruption == 'missing': path.unlink()
                    elif corruption == 'extra': extra.write_bytes(b'invalid')
                    else:
                        values = torch.load(path, weights_only=True)
                        values['id'][0] = 'wrong'; torch.save(values, path)
                    try:
                        with self.assertRaises(ValueError): completed_parts(output, 'val')
                    finally:
                        path.write_bytes(original); extra.unlink(missing_ok=True)

    def test_four_logical_shards_preserve_exact_parts_coverage(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            config, _ = fixture(Path(directory)); output = Path(config['output'])
            entries = load_selected(SampleIndex(Path(config['samples'])/'val.jsonl'), 1000, config['seed'])
            (output/'shards/val/0/part_000000.pt').unlink()
            target = output/'shards/val'
            for shard in range(4):
                for number, group in enumerate(plan_parts(entries, shard, 4)):
                    path = target/str(shard)/f'part_{number:06d}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
                    values = dump([1]*len(group)); values['id'] = [entry['id'] for entry in group]
                    torch.save(values, path)
            expected = [entry['id'] for entry in entries]
            self.assertEqual(read_parts(entries, shards=4, directory=target)['id'], expected)
            self.assertFalse((target/'3').exists())
            extra = target/'4'; extra.mkdir()
            self.assertEqual(read_parts(entries, shards=4, directory=target)['id'], expected)
            torch.save(dump([1]), extra/'part_000000.pt')
            with self.assertRaisesRegex(ValueError, 'unexpected shard parts'):
                read_parts(entries, shards=4, directory=target)

    def test_test_only_never_fits_temperature_and_checks_saved_calibration(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);config,values=fixture(root);dev=Path(config['output'])
            with patch.object(eval_jev,'fit_temperature',return_value={'temperature':1.5}):eval_jev.finalize(config,config['majority'])
            (dev/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',configuration=config)))
            final=root/'final'
            test=dict(config,output=str(final),splits=['test_id','test_ood'],calibration_eval=str(dev),calibration_temperature={'temperature':1.5})
            for split in ('test_id','test_ood'):
                sample=Path(config['samples'])/(split+'.jsonl');sample.write_text((Path(config['samples'])/'val.jsonl').read_text())
                part=final/'shards'/split/'0';part.mkdir(parents=True);torch.save(values,part/'part_000000.pt')
            with patch.object(eval_jev,'fit_temperature',side_effect=AssertionError('test refit')) as fit:
                eval_jev.finalize(test,test['majority']);fit.assert_not_called()
            report=json.loads((final/'eval.json').read_text())
            self.assertEqual(set(report),{'test_id','test_ood','temperature'})
            (final/'metadata').mkdir()
            for missing in ('calibration_eval','calibration_temperature'):
                invalid=dict(test);del invalid[missing]
                (final/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',configuration=invalid)))
                with self.assertRaises(KeyError):completed_parts(final,'test_id')
            test['calibration_temperature']={'temperature':2}
            with self.assertRaisesRegex(ValueError,'changed'):eval_jev.finalize(test,test['majority'])

    def test_legacy_full_evaluation_without_splits_keeps_original_behavior(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            config,values=fixture(Path(directory));del config['splits'];output=Path(config['output'])
            for split in ('test_id','test_ood'):
                sample=Path(config['samples'])/(split+'.jsonl');sample.write_text((Path(config['samples'])/'val.jsonl').read_text())
                part=output/'shards'/split/'0';part.mkdir(parents=True);torch.save(values,part/'part_000000.pt')
            with patch.object(eval_jev,'fit_temperature',return_value={'temperature':1.5}) as fit:
                eval_jev.finalize(config,config['majority']);fit.assert_called_once()
            (output/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',configuration=config)))
            for split in ('val','test_id','test_ood'):self.assertEqual(len(completed_parts(output,split)[0]['id']),3)

    def test_invalid_calibration_identity_temperature_and_coverage_fail(self):
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            config,_=fixture(Path(directory));dev=Path(config['output'])
            with patch.object(eval_jev,'fit_temperature',return_value={'temperature':1.5}):eval_jev.finalize(config,config['majority'])
            status=dict(status='completed',configuration=config);path=dev/'metadata/run_status.json'
            path.write_text(json.dumps(status))
            self.assertEqual(calibration_temperature(dev,config),{'temperature':1.5})
            for key in ('model','model_source','samples','seed','majority','copy_prev_rule'):
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity'):
                    calibration_temperature(dev,dict(config,**{key:'different'}))
            with self.assertRaisesRegex(ValueError,'limit'):
                calibration_temperature(dev,dict(config,limits=dict(config['limits'],val=1)))
            for status_value in ('failed','running'):
                path.write_text(json.dumps(dict(status,status=status_value)))
                with self.assertRaisesRegex(ValueError,'completed'):calibration_temperature(dev,config)
            path.write_text(json.dumps(status))
            for temperature in ({},{'temperature':0},{'temperature':float('nan')},{'temperature':True},{'temperature':2}):
                (dev/'temperature.json').write_text(json.dumps(temperature))
                with self.assertRaises((ValueError,KeyError)):calibration_temperature(dev,config)
            (dev/'temperature.json').write_text(json.dumps({'temperature':1.5}))
            (dev/'shards/val/0/part_000000.pt').unlink()
            with self.assertRaisesRegex(ValueError,'parts'):calibration_temperature(dev,config)


if __name__=='__main__':unittest.main()
