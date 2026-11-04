#!/usr/bin/env python3
"""Read-only comparison of completed evaluation directories on identical fixed subsets.

Metrics come from each eval.json (the unique final source); paired differences against the
reference are recomputed from the saved parts with paired_difference, plus an exact two-sided
sign test on the discordant states. Nothing is re-evaluated and no input directory is modified.
"""
import argparse
import json
import random
import shlex
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(REPO/'third_party/decider'),str(REPO/'experiments/dataset_tool_inventory/src')]
from jev_train.common import write_json
from jev_train.evaluation import completed_parts,paired_difference,sign_test,LIST_FIELDS,measured
from jev_train.data import SampleIndex
from jev_train.domains import LABELS,label_acceptance

SPLITS=('val','test_id','test_ood')
FIELDS=('acc','macro_by_dataset','nll','ece','copy_prev_acc')
EXCLUDED='deepseek-v4-pro-0813-agentic'


def clustered_pair(dump,reference,groups,seed,reference_seed):
    paired_difference(dump,reference,seed=seed,reference_seed=reference_seed)
    def hits(values):
        return {sid:int(values['targets'][i,int(values['logits'][i].argmax())]>0) for i,sid in enumerate(values['id'])}
    a,b=hits(dump),hits(reference);members={}
    for sid in sorted(a):
        group=groups[sid]
        if not isinstance(group,str) or not group:raise ValueError('invalid canonical task_group')
        members.setdefault(group,[]).append(sid)
    names=list(members)
    diffs={g:sum(a[s]-b[s] for s in ids) for g,ids in members.items()}
    rng=random.Random(20261015);boot=[]
    for _ in range(10000):
        drawn=[rng.choice(names) for _ in names]
        boot.append(sum(diffs[g] for g in drawn)/sum(len(members[g]) for g in drawn))
    boot.sort();improved=sum(v>0 for v in diffs.values());regressed=sum(v<0 for v in diffs.values())
    sizes=sorted(map(len,members.values()),reverse=True)
    return dict(count=len(a),task_groups=len(names),acc_difference=sum(diffs.values())/len(a),
        cluster_bootstrap_ci95=[boot[249],boot[9749]],bootstrap_resamples=10000,bootstrap_seed=20261015,
        task_equal_difference=sum(diffs[g]/len(members[g]) for g in names)/len(names),
        task_improved=improved,task_regressed=regressed,task_tied=len(names)-improved-regressed,
        task_sign_test_p=sign_test(improved,regressed) if improved+regressed else 1.,
        largest_five_group_sizes=sizes[:5],largest_five_share=sum(sizes[:5])/len(a))


def canonical_groups(samples,split,dump):
    index=SampleIndex(samples/(split+'.jsonl'));wanted=set(dump['id']);result={};datasets=dict(zip(dump['id'],dump['dataset']))
    for entry in index.entries:
        if entry['id'] not in wanted:continue
        row=index.read(entry['offset']);sid=row['id']
        if row['dataset']!=datasets[sid]:raise ValueError('sample/part dataset differs')
        group=row['task_group']
        if not isinstance(group,str) or not group:raise ValueError('invalid canonical task_group')
        result[sid]=group
    if result.keys()!=wanted:raise ValueError('sample/part ID coverage differs')
    return result


def subset(dump,indices):
    return {**{k:dump[k][indices] for k in ('logits','targets')},**{k:[dump[k][i] for i in indices] for k in LIST_FIELDS}}


def read_domain_labels(directory):
    from label_domains import read_requests,existing_labels
    state=json.loads((directory/'metadata/run_status.json').read_text())
    label_acceptance(state,directory)
    preview=json.loads((Path(state['preview'])/'summary.json').read_text())
    requests=read_requests(Path(preview['recovery_inputs']['annotation_requests']))
    result=existing_labels(directory/'labels.jsonl',requests)
    if result.keys()!=requests.keys() or not len(result)==state['item_count']==state['completed_count']:
        raise ValueError('domain label coverage differs')
    return result


def domain_accuracy(dump,groups,labels):
    result={}
    for i,sid in enumerate(dump['id']):
        record=labels[groups[sid]]
        if record['split']!='test_id':raise ValueError('domain label split differs')
        label=record['label']
        if label not in LABELS:raise ValueError('invalid domain label')
        row=result.setdefault(label,dict(count=0,correct=0));row['count']+=1
        row['correct']+=int(dump['targets'][i,int(dump['logits'][i].argmax())]>0)
    return {d:dict(r,acc=r['correct']/r['count'],descriptive_only=r['count']<30) for d,r in sorted(result.items())}


def summary(report):
    overall=report['overall']
    return {**{k:overall.get(k) for k in FIELDS},'respond':overall['respond'],'ece_count':overall['ece_count'],
            'by_transition':{k:v['acc'] for k,v in overall['by_transition'].items()},
            'datasets':{d:v['acc'] for d,v in report['datasets'].items()}}


def task_equal_accuracy(dump, groups):
    counts={}
    for i,sid in enumerate(dump['id']):
        correct,total=counts.get(groups[sid],(0,0))
        counts[groups[sid]]=(correct+int(dump['targets'][i,int(dump['logits'][i].argmax())]>0),total+1)
    if not counts:raise ValueError('task-equal accuracy requires nonempty tasks')
    return sum(correct/total for correct,total in counts.values())/len(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval',action='append',required=True,metavar='LABEL=DIR')
    parser.add_argument('--reference',required=True,help='label of the paired-comparison reference')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--domain-labels',type=Path,help='Reviewed labels directory; adds test_id domain slices')
    parser.add_argument('--splits',nargs='+',choices=SPLITS,default=list(SPLITS))
    args=parser.parse_args()
    if len(set(args.splits))!=len(args.splits):parser.error('duplicate split')
    evals={}
    for value in args.eval:
        label,sep,path=value.partition('=')
        if not sep or not label or not path or label in evals:parser.error(f'invalid or duplicate --eval {value!r}')
        evals[label]=Path(path).resolve()
    if len(evals)<2 or args.reference not in evals:parser.error('need >=2 evaluations and a reference among them')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    reports={label:json.loads((path/'eval.json').read_text()) for label,path in evals.items()}
    configurations={label:json.loads((path/'metadata/run_status.json').read_text())['configuration'] for label,path in evals.items()}
    samples=Path(configurations[args.reference]['samples']).resolve()
    if any(Path(c['samples']).resolve()!=samples for c in configurations.values()):raise ValueError('comparison sample directories differ')
    domains=read_domain_labels(args.domain_labels) if args.domain_labels else None
    result={'evaluations':{k:str(v) for k,v in evals.items()},'reference':args.reference,'splits':{}}
    for split in args.splits:
        reference,reference_seed=completed_parts(evals[args.reference],split)
        groups=canonical_groups(samples,split,reference)
        rows={}
        for label,path in evals.items():
            dump,seed=(reference,reference_seed) if label==args.reference else completed_parts(path,split)
            rows[label]=summary(reports[label][split])
            rows[label]['task_equal_acc']=task_equal_accuracy(dump,groups)
            if label!=args.reference:
                paired=paired_difference(dump,reference,seed=seed,reference_seed=reference_seed)
                rows[label]['paired_vs_reference']=paired
                rows[label]['clustered_vs_reference']=clustered_pair(dump,reference,groups,seed,reference_seed)
            if split=='test_id' and domains is not None:rows[label]['by_domain']=domain_accuracy(dump,groups,domains)
            if split=='test_ood':
                rows[label]['ood_datasets']={d:{k:v[k] for k in ('count','acc','ece','ece_count','respond')} for d,v in reports[label][split]['datasets'].items()}
            if split=='val':
                kept=subset(dump,[i for i,d in enumerate(dump['dataset']) if d!=EXCLUDED])
                temperature=reports[label]['temperature']['temperature']
                excluded=summary(measured(kept,temperature,train_majority=configurations[label]['majority']))
                excluded.update(count=len(kept['id']),excluded_count=len(dump['id'])-len(kept['id']),excluded_dataset=EXCLUDED,temperature=temperature)
                if label!=args.reference:
                    ref_kept=subset(reference,[i for i,d in enumerate(reference['dataset']) if d!=EXCLUDED])
                    excluded['clustered_vs_reference']=clustered_pair(kept,ref_kept,groups,seed,reference_seed)
                rows[label]['without_deepseek_0813']=excluded
        result['splits'][split]=rows
    if 'val' in result['splits']:
        result['val_ranking']=sorted(result['splits']['val'],key=lambda name:(-result['splits']['val'][name]['task_equal_acc'],name))
    result['definition']=('Metrics copied from each eval.json. paired_vs_reference uses positional top-1 set hits on identical '
                          'ids/targets/seed; sign_test_p is the exact two-sided binomial test on discordant states (p=0.5). '
                          'Cluster bootstrap resamples whole task_groups 10000 times, seed 20261015, sorted ranks 249/9749; '
                          'task-equal delta weights tasks equally; task sign p=1 if no task changes. '
                          'Val excluding deepseek-0813 reuses saved temperature, never refits. Domains with n<30 are descriptive only.')
    result['domain_labels']=str(args.domain_labels.resolve()) if args.domain_labels else None
    result['domain_label_acceptance']=(label_acceptance(
        json.loads((args.domain_labels/'metadata/run_status.json').read_text()),args.domain_labels)
        if args.domain_labels else None)
    write_json(out/'summary.json',result)
    write_json(out/'metadata/run_status.json',dict(status='completed',ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')))
    for split in args.splits:
        print(f'\n## {split}\n| label | acc | macro | nll | ece | first | repeat | change | Δacc vs {args.reference} | +/- | p |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for label,row in result['splits'][split].items():
            t=row['by_transition'];p=row.get('paired_vs_reference')
            fmt=lambda v:'-' if v is None else f'{v:.3f}'
            delta=f"{p['acc_difference']:+.3f} | {p['improved']}/{p['regressed']} | {fmt(p['sign_test_p'])}" if p else '- | - | -'
            print(f"| {label} | {fmt(row['acc'])} | {fmt(row['macro_by_dataset'])} | {fmt(row['nll'])} | {fmt(row['ece'])} | "
                  f"{fmt(t.get('first'))} | {fmt(t.get('repeat'))} | {fmt(t.get('change'))} | {delta} |")


if __name__=='__main__':main()
