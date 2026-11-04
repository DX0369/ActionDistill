#!/usr/bin/env python3
"""Select a fixed-size training subset from an immutable pool."""
import argparse,json,shlex,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];REPO=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(REPO/'third_party/decider'),str(REPO/'experiments/dataset_tool_inventory/src')]
from jev_train.common import now,write_json
from jev_train.selection import load_pool,select,validate_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','config','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--reference_quotas',type=Path,help='JSON dataset->integer quota; replaces sqrt(task-group) weights and min/max fractions')
    args=parser.parse_args();pool=args.pool.resolve();out=args.out.resolve()
    if args.count<=0:parser.error('count must be positive')
    if out.is_relative_to(pool):parser.error('output must be outside the input pool')
    config=json.loads(args.config.read_text());validate_config(config)
    reference=json.loads(args.reference_quotas.read_text()) if args.reference_quotas else None
    out.mkdir(parents=True,exist_ok=False);(out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status={'status':'running','started_at':now(),'pool':str(pool),'config':config,'count':args.count,'seed':args.seed,'reference_quotas':str(args.reference_quotas.resolve()) if reference is not None else None}
    write_json(out/'metadata/run_status.json',status)
    try:
        chosen,summary=select(load_pool(pool/'train.jsonl'),config,args.count,args.seed,reference)
        (out/'train_ids.txt').write_text(''.join(row['id']+'\n' for row in chosen))
        if reference is not None:summary['reference_quotas_path']=str(args.reference_quotas.resolve())
        write_json(out/'summary.json',summary)
        status.update(status='completed',ended_at=now(),selected=len(chosen))
    except BaseException as exc:
        status.update(status='failed',ended_at=now(),error_type=type(exc).__name__,error=str(exc));raise
    finally:write_json(out/'metadata/run_status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
