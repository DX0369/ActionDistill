#!/usr/bin/env python3
import argparse,json,shlex,sys,tempfile,shutil
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];REPO=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(REPO/'third_party/decider'),str(REPO/'experiments/dataset_tool_inventory/src')]
from jev_train.common import now,write_json
from jev_train.build import build_samples,canonical_g3,GroupingReviewError


def main():
    parser=argparse.ArgumentParser(description='Build Jev decision samples from approved source trajectories')
    parser.add_argument('--review-run',type=Path)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/build.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--limit-trajectories',type=int)
    parser.add_argument('--representative-states',action='store_true',help='smoke only: render first/middle/last selected state per trajectory')
    parser.add_argument('--group-only',action='store_true',help='review only: full first pass and grouping, no selection or rendering; tmp output required')
    args=parser.parse_args()
    if args.workers<=0 or (args.limit_trajectories is not None and args.limit_trajectories<=0):parser.error('workers/limit must be positive')
    if args.representative_states and args.limit_trajectories is None:parser.error('representative states require --limit-trajectories (smoke only)')
    if args.group_only and (args.limit_trajectories is not None or args.representative_states):parser.error('group-only requires full inputs without representative sampling')
    config=json.loads(args.config.read_text());run=(args.review_run or REPO/config['review_run']).resolve();out=args.output.resolve()
    if (args.limit_trajectories is not None or args.group_only) and not out.is_relative_to(PROJECT/'tmp'):parser.error('smoke/group review output must be under jev_train/tmp')
    canonical_g3(config)
    registry=json.loads((REPO/config['inventory_config']).read_text())['datasets'];entries=[e for e in registry if e['name'] in config['datasets']]
    if len(entries)!=len(config['datasets']) or any(e['review_adapter'] in {'excluded','unresolved'} for e in entries):raise ValueError('invalid dataset scope')
    out.mkdir(parents=True,exist_ok=False);(out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status={'status':'running','started_at':now(),'bounded':args.limit_trajectories is not None,'group_only':args.group_only,'representative_states':args.representative_states,'review_run':str(run),'workers':args.workers}
    write_json(out/'metadata/run_status.json',status)
    (PROJECT/'tmp').mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='build_work_',dir=PROJECT/'tmp'))
    try:
        report=build_samples(entries,config,run,tmp,out,args.workers,args.limit_trajectories,args.representative_states,args.group_only)
        status.update(status='partial' if args.limit_trajectories is not None else 'completed',ended_at=now())
        if not args.group_only:status['split_counts']={s:sum(v[s] for v in report['final_dataset_counts'].values()) for s in ('train','val','test_id','test_ood')}
    except BaseException as exc:
        status.update(status='failed',ended_at=now(),error_type=type(exc).__name__,error=str(exc))
        if isinstance(exc,GroupingReviewError) and args.group_only:
            shutil.rmtree(tmp)
        else:status['diagnostics']=str(tmp.relative_to(PROJECT))
        write_json(out/'metadata/run_status.json',status);raise
    else:
        shutil.rmtree(tmp);write_json(out/'metadata/run_status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
