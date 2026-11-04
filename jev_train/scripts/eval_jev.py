#!/usr/bin/env python3
"""Independent, resumable GPU-sharded evaluation; never modifies a training run."""
import argparse
import json
import multiprocessing as mp
import os
import shlex
import sys
import time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(REPO/'third_party/decider'),str(REPO/'experiments/dataset_tool_inventory/src')]
from jev_train.common import now, write_json
from jev_train.data import SampleIndex, load_selected
from jev_train.evaluation import (
    COPY_PREV_RULE,
    calibration_temperature,
    completed_parts,
    load_majority,
    measured,
    paired_difference,
    read_parts,
    requested_splits,
    run_parts,
)
from jev_train.loss import fit_temperature

SPLITS=('val','test_id','test_ood')


def worker(gpu,shard,configuration):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    from decider.model import DecisionModel
    torch.manual_seed(configuration['seed'])
    model=DecisionModel(configuration['model'],grad_ckpt=False).cuda()
    model.lm.config.use_cache=False
    for split in requested_splits(configuration):
        index=SampleIndex(Path(configuration['samples'])/(split+'.jsonl'))
        entries=load_selected(index,configuration['limits'][split],configuration['seed'])
        run_parts(model,index,entries,seed=configuration['seed'],shard=shard,shards=configuration['shards'],
                  directory=Path(configuration['output'])/'shards'/split/str(shard))


def finalize(configuration,majority,compare_to=None):
    output=Path(configuration['output']);results={};comparison={};temperature=None
    splits=requested_splits(configuration)
    if 'val' not in splits:
        temperature=calibration_temperature(configuration['calibration_eval'],configuration)
        if temperature!=configuration['calibration_temperature']:raise ValueError('calibration changed after test configuration was saved')
    for split in splits:
        index=SampleIndex(Path(configuration['samples'])/(split+'.jsonl'))
        entries=load_selected(index,configuration['limits'][split],configuration['seed'])
        dump=read_parts(entries,shards=configuration['shards'],directory=output/'shards'/split)
        if split=='val':
            temperature=fit_temperature(dump['logits'],dump['targets'])
        results[split]=measured(dump,temperature['temperature'],train_majority=majority,oracle=split=='test_ood')
        results[split]['selection']={'count':len(entries),'population':len(index.entries),'limit':configuration['limits'][split],
                                     'seed':configuration['seed'],'model':configuration['model']}
        if compare_to is not None:
            reference,reference_seed=completed_parts(compare_to,split)
            comparison[split]=paired_difference(dump,reference,seed=configuration['seed'],reference_seed=reference_seed)
    results['temperature']=temperature
    if compare_to is not None:results['comparison']={'reference':str(compare_to), 'splits':comparison}
    write_json(output/'temperature.json',temperature)
    write_json(output/'eval.json',results)


def prepare_configuration(args, shards, parser):
    """Validate CLI identity and saved evaluation configuration before execution."""
    if (args.run is None) == (args.model is None):parser.error('exactly one of run/model is required')
    if type(shards) is not int or shards < 1:parser.error('shards must be a positive integer')
    if type(args.seed) is not int:parser.error('seed must be an integer')
    limits={s:getattr(args,'limit_'+s) for s in SPLITS}
    if any(type(v) is not int or v<0 for v in limits.values()):parser.error('limits must be nonnegative integers (0 means all)')
    output=args.out.resolve();source_path=(args.run if args.run is not None else args.model).resolve();samples=args.samples.resolve()
    model=source_path/'model' if args.run is not None else source_path
    if output.is_relative_to(source_path) or output.is_relative_to(model.resolve()) or output.is_relative_to(samples):parser.error('evaluation output must be separate from run/model and samples')
    majority=load_majority(args.majority)
    configuration={'model':str(model),'model_source':'run' if args.run is not None else 'model','samples':str(samples),'limits':limits,'seed':args.seed,'shards':shards,'copy_prev_rule':COPY_PREV_RULE,
                   'majority':majority,'output':str(output)}
    if args.splits is not None:configuration['splits']=args.splits
    splits=requested_splits(configuration)
    if ('val' not in splits)!=bool(args.calibration_eval):parser.error('test-only requires --calibration_eval; val-containing runs must fit their own temperature')
    if args.calibration_eval:
        calibration=args.calibration_eval.resolve()
        if output.is_relative_to(calibration) or calibration.is_relative_to(output):parser.error('test-only output must be separate from completed calibration evaluation')
        configuration['calibration_eval']=str(args.calibration_eval.resolve())
        configuration['calibration_temperature']=calibration_temperature(args.calibration_eval,configuration)
    if not model.is_dir():raise ValueError('model directory does not exist')
    status_path=output/'metadata/run_status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        if previous['configuration']!=configuration:raise ValueError('resume configuration differs, including shard count; use a new output directory')
    elif args.finalize_only:raise ValueError('finalize_only requires an existing evaluation run')
    elif (output/'shards').exists():raise ValueError('existing shards have no matching run metadata')
    return output, majority, configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run',type=Path)
    source.add_argument('--model',type=Path)
    for name in ('samples','out','majority'):parser.add_argument('--'+name,type=Path,required=True)
    for split in SPLITS:parser.add_argument('--limit_'+split,type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gpus',required=True)
    parser.add_argument('--finalize_only',action='store_true')
    parser.add_argument('--compare_to',type=Path)
    parser.add_argument('--splits',nargs='+',choices=SPLITS,help='val for development, test_id test_ood for final; default all three')
    parser.add_argument('--calibration_eval',type=Path,help='Completed val evaluation for this same model; required for test-only')
    args=parser.parse_args()
    gpus=args.gpus.split(',')
    if not gpus or any(not v.isdigit() for v in gpus) or len(set(gpus))!=len(gpus):parser.error('gpus must be distinct numeric GPU indices')
    output,majority,configuration=prepare_configuration(args,len(gpus),parser)
    status_path=output/'metadata/run_status.json'
    (output/'metadata').mkdir(parents=True,exist_ok=True)
    with (output/'metadata/command.txt').open('a') as f:f.write(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status={'status':'running','started_at':now(),'configuration':configuration,'gpus':gpus,'finalize_only':args.finalize_only}
    write_json(status_path,status)
    processes=[]
    try:
        if not args.finalize_only:
            ctx=mp.get_context('spawn')
            for shard,gpu in enumerate(gpus):
                process=ctx.Process(target=worker,args=(gpu,shard,configuration));process.start();processes.append(process)
            while any(p.is_alive() for p in processes):
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('evaluation worker failed; completed parts retained')
                time.sleep(.5)
            if any(p.exitcode!=0 for p in processes):raise RuntimeError('evaluation worker failed; completed parts retained')
        finalize(configuration,majority,args.compare_to.resolve() if args.compare_to else None)
        status.update(status='completed',ended_at=now());write_json(status_path,status)
    except BaseException as exc:
        for process in processes:
            if process.is_alive():process.terminate()
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',ended_at=now(),error_type=type(exc).__name__,error=str(exc))
        write_json(status_path,status);raise
    finally:
        for process in processes:process.join()


if __name__=='__main__':main()
