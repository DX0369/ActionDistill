#!/usr/bin/env python3
"""Domain-balanced B2 preview from reviewed labels and existing canonical recovery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys

from preview_dedup import PROJECT, SEED, timestamp, require
from jev_train.common import rows, write_json
from jev_train.dedup import redistributed_preview
from jev_train.domains import balanced_selection, label_acceptance


def load_prepared(preview_path):
    state=json.loads((preview_path/'metadata/run_status.json').read_text())
    require(state['status']=='completed' and not state['smoke'],'requires full completed dedup preview')
    summary=json.loads((preview_path/'summary.json').read_text())
    require(summary['seed']==SEED and summary['respond_gate_passed'],'dedup preview protocol differs')
    train={};seen=set()
    for row in rows(Path(summary['recovery_inputs']['canonical_states'])):
        require(row['id'] not in seen,'duplicate canonical ID');seen.add(row['id'])
        if row['split']=='train':train[row['id']]=row
    require(len(seen)==summary['identity_recovery']['states'] and len(train)==summary['c1']['train']['c1_kept'],'canonical coverage differs')
    reference=Path(summary['inputs']['reference'])
    reference_ids=(reference/'train_ids.txt').read_text().splitlines()
    weights={d:r['original_quota'] for d,r in summary['sources'].items()}
    _,dedup_ids,report=redistributed_preview(train,weights,SEED,reference_ids)
    require(json.loads(json.dumps(report['selected']))==summary['selected'],'dedup selection no longer reproduces')
    return summary,train,weights,reference_ids,dedup_ids


def load_labels(directory, preview_path):
    from label_domains import existing_labels, read_requests
    state=json.loads((directory/'metadata/run_status.json').read_text())
    label_acceptance(state,directory)
    require(state['train_other_fraction']<=.10,'training other fraction above threshold')
    require(Path(state['preview']).resolve()==preview_path.resolve(),'label preview identity differs')
    preview=json.loads((preview_path/'summary.json').read_text())
    requests=read_requests(Path(preview['recovery_inputs']['annotation_requests']))
    records=existing_labels(directory/'labels.jsonl',requests)
    require(records.keys()==requests.keys(),'label/request coverage differs')
    labels={g:r['label'] for g,r in records.items()}
    require(len(labels)==state['item_count']==state['completed_count'],'label coverage differs')
    return labels,state


def run(args):
    parent,train,weights,reference_ids,dedup_ids=load_prepared(args.dedup_preview)
    labels,state=load_labels(args.labels,args.dedup_preview)
    ids,result=balanced_selection(train,labels,weights,8000,SEED)
    result['comparison']={name:dict(states=len(values),domains=dict(Counter(labels[train[sid]['task_group']] for sid in values)),
        task_groups=len({train[sid]['task_group'] for sid in values}),overlap_with_domain_group=len(set(ids)&set(values)))
        for name,values in [('B2_8k',reference_ids),('B2_dedup_8k',dedup_ids)]}
    result.update(seed=SEED,smoke=args.smoke,recipe='B2-dedupdom-8k',
        inputs=dict(dedup_preview=str(args.dedup_preview.resolve()),labels=str(args.labels.resolve()),samples=parent['inputs']['samples']),
        recovery_inputs=parent['recovery_inputs'],label_review=state['review'],label_acceptance=label_acceptance(state,args.labels),
        domain_label_model=dict(requested=state['model'],actual=state['api_response_models'],thinking=state['thinking']))
    write_json(args.output/'summary.json',result)
    lines=['# B2-dedupdom-8k 领域均衡预览','',
        f"总数8000；RESPOND {result['selected']['respond']}（{result['selected']['respond_fraction']:.4%}）；优先类{result['selected']['priority']}（{result['selected']['priority_fraction']:.4%}）；prompt token {result['selected']['n_prompt_tokens']:,}。",'',
        '| 领域 | G_D | 初始K_D | 最终配额 | B2-8k | dedup-8k |',
        '|---|---:|---:|---:|---:|---:|']
    for domain,r in sorted(result['domains'].items()):
        counts=[]
        for name in ('B2_8k','B2_dedup_8k'):
            counts.append(result['comparison'][name]['domains'].get(domain,0))
        lines.append(f"| {domain} | {r['G_D']} | {r['K_D']} | {r['final_quota']} | {counts[0]} | {counts[1]} |")
    lines+=['','## 来源 × 领域','', '| 来源 | 领域 | 初始配额 | 最终配额 | 优先上限/实取 | 容量 |','|---|---|---:|---:|---:|---:|']
    for domain,sources in sorted(result['cells'].items()):
        for dataset,r in sorted(sources.items()):
            lines.append(f"| {dataset} | {domain} | {r['initial_quota']} | {r['final_quota']} | {r['priority_quota']} / {r['priority_selected']} | {r['b2_capacity']} |")
    lines+=['','优先预算按最终来源×领域单元计算；有序流量记账与全局/逐来源构成在summary.json。未复制恢复清单，metadata只含command.txt/run_status.json。',
        '去重可能减少有用的探索路径多样性；领域均衡仍未覆盖客服领域。评测保持规范化3×1000，不从OOD选配额或改定义。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    require(result['respond_gate_passed'],'domain selection RESPOND outside [0.08,0.20]; stop')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dedup-preview','labels','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.smoke and not args.output.is_relative_to(PROJECT/'tmp'):parser.error('smoke output must be under tmp')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'metadata').mkdir()
    (args.output/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status=dict(status='running',started_at=timestamp(),smoke=args.smoke,stage=3)
    write_json(args.output/'metadata/run_status.json',status)
    try:result=run(args)
    except BaseException as exc:
        status.update(status='failed',ended_at=timestamp(),error=str(exc));write_json(args.output/'metadata/run_status.json',status);raise
    status.update(status='smoke_completed' if args.smoke else 'completed',ended_at=timestamp())
    write_json(args.output/'metadata/run_status.json',status)
    print(json.dumps(dict(output=str(args.output),selected=result['selected'],domains=result['domains']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
