#!/usr/bin/env python3
"""Record the explicit 100-group Codex review; never change API labels."""
import argparse
import json
from pathlib import Path
from label_domains import timestamp,require,read_requests,existing_labels,review_sample
from jev_train.common import rows,write_json
from jev_train.domains import LABELS, REVIEW_FAILURE, label_acceptance


def finalize(directory,review_paths):
    path=directory/'metadata/run_status.json';state=json.loads(path.read_text())
    require(state['status']=='awaiting_review' and state['failure_count']==0,'requires zero-failure awaiting_review labels')
    preview=json.loads((Path(state['preview'])/'summary.json').read_text())
    requests=read_requests(Path(preview['recovery_inputs']['annotation_requests']))
    labels=existing_labels(directory/'labels.jsonl',requests)
    require(labels.keys()==requests.keys(),'label coverage differs')
    expected={r['task_group']:r for r in review_sample(labels,requests)};reviewed={}
    for review_path in review_paths:
        for row in rows(review_path):
            g=row['task_group'];require(g in expected and g not in reviewed,'unexpected/duplicate review task')
            require(row['reviewed_label'] in LABELS and isinstance(row['note'],str),'invalid review label/note')
            require(type(row['agrees']) is bool and row['agrees']==(row['reviewed_label']==expected[g]['label']),'review agreement differs from labels')
            reviewed[g]=row
    require(reviewed.keys()==expected.keys() and len(reviewed)==100,'review must cover exact deterministic 100 groups')
    agreement=sum(r['agrees'] for r in reviewed.values())/100
    state['review']=dict(count=100,agreement=agreement,required=.90,reviewed_at=timestamp(),
        method='Codex model-assisted review against closed definitions and fixed-source rules; labels shown; no human review claim.',
        items=[dict(reviewed[g],api_label=expected[g]['label']) for g in sorted(reviewed)])
    state['status']='completed' if agreement>=.90 else 'failed'
    state['ended_at']=timestamp()
    if agreement<.90:state['error']=REVIEW_FAILURE
    write_json(path,state)
    require(agreement>=.90,state.get('error','review gate failed'))
    print(json.dumps(dict(status=state['status'],agreement=agreement,disagreements=[r for r in state['review']['items'] if not r['agrees']]),ensure_ascii=False))


def accept_existing(directory,reason):
    """Record a user-authorized batch decision, without relabeling or changing the failed review."""
    path=directory/'metadata/run_status.json';state=json.loads(path.read_text())
    require(state['status']=='failed' and state['error']==REVIEW_FAILURE and state['failure_count']==0,
            'only the original review-threshold failure can be accepted')
    require('acceptance' not in state,'batch already has an acceptance decision')
    preview=json.loads((Path(state['preview'])/'summary.json').read_text())
    requests=read_requests(Path(preview['recovery_inputs']['annotation_requests']))
    labels=existing_labels(directory/'labels.jsonl',requests)
    require(labels.keys()==requests.keys() and len(labels)==state['completed_count']==state['item_count'],'label coverage differs')
    expected={r['task_group']:r for r in review_sample(labels,requests)}
    items=state['review']['items']
    require(len(items)==len({r['task_group'] for r in items})==100 and {r['task_group'] for r in items}==expected.keys(),'review coverage differs')
    for row in items:
        require(row['api_label']==expected[row['task_group']]['label'] and row['reviewed_label'] in LABELS
                and type(row['agrees']) is bool and row['agrees']==(row['reviewed_label']==row['api_label']),
                'saved review disagrees with unchanged labels')
    require(sum(r['agrees'] for r in items)/100==state['review']['agreement'],'review agreement differs')
    state['acceptance']=dict(authority='user',decision='accept_existing_labels',labels_directory=str(directory.resolve()),
        accepted_at=timestamp(),review_count=100,review_agreement=state['review']['agreement'],
        original_required=state['review']['required'],reason=reason)
    label_acceptance(state,directory)
    write_json(path,state)
    print(json.dumps(state['acceptance'],ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--review',type=Path,action='append')
    mode.add_argument('--accept-existing',action='store_true',help='Only after explicit user authorization for this label batch')
    parser.add_argument('--reason')
    args=parser.parse_args()
    if args.accept_existing:
        if not args.reason:parser.error('--accept-existing requires --reason')
        accept_existing(args.labels,args.reason)
    else:
        if args.reason:parser.error('--reason only applies to --accept-existing')
        finalize(args.labels,args.review)


if __name__=='__main__':main()
