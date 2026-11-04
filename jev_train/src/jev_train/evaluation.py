"""Shared lazy evaluation, durable parts, and paired set-hit comparisons."""
import json
import math
import time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from .data import make_item,collate,tool_name,SampleIndex,load_selected
from .loss import metrics,check_eval_logits

LIST_FIELDS = ('id','dataset','outcome','candidate_level','nopts','tool_names','transition','copy_prev_hit')
COPY_PREV_RULE = 'single_member_or_multi_equal'
SPLITS = ('val','test_id','test_ood')


def requested_splits(configuration):
    """Old full-eval configurations omit splits; new partial runs declare them."""
    splits = tuple(configuration['splits']) if 'splits' in configuration else SPLITS
    if splits not in (SPLITS, ('val',), ('test_id','test_ood')):
        raise ValueError('splits must be val, test_id test_ood, or all three in canonical order')
    return splits


def calibration_temperature(directory, configuration):
    """A test-only run reuses its own model's completed val calibration."""
    directory = Path(directory)
    state = json.loads((directory/'metadata/run_status.json').read_text())
    if state['status'] != 'completed' or 'val' not in requested_splits(state['configuration']):
        raise ValueError('calibration evaluation must have completed val')
    previous = state['configuration']
    for key in ('model','model_source','samples','seed','majority','copy_prev_rule'):
        if previous[key] != configuration[key]:
            raise ValueError(f'calibration identity differs: {key}')
    if previous['limits']['val'] != configuration['limits']['val']:
        raise ValueError('calibration val limit differs')
    temperature = json.loads((directory/'temperature.json').read_text())
    report = json.loads((directory/'eval.json').read_text())
    t = temperature['temperature']
    if type(t) not in (int,float) or not math.isfinite(t) or t <= 0 or temperature != report['temperature']:
        raise ValueError('invalid or inconsistent calibration temperature')
    completed_parts(directory, 'val')  # Exact saved shard and selected-ID coverage.
    return temperature


def transition_metadata(row):
    previous=row['prev_action_names'];transition=row['transition']
    if not isinstance(previous,list) or any(not isinstance(name,str) or not name for name in previous):raise ValueError('invalid prev_action_names')
    if transition not in {'first','repeat','change'}:raise ValueError('invalid transition')
    gold={tool_name(row['options'][i]) for i in row['gold']}
    previous=set(previous)
    return transition,bool(gold.intersection(previous)) if len(previous)==1 else bool(previous) and previous==gold


def empty_result():
    return dict(logits=torch.empty((0,255)),targets=torch.empty((0,255)),**{k:[] for k in LIST_FIELDS})


def combine(parts):
    parts=list(parts)
    if not parts:return empty_result()
    return {**{k:torch.cat([p[k] for p in parts]) for k in ('logits','targets')},
            **{k:[v for p in parts for v in p[k]] for k in LIST_FIELDS}}


def evaluate(model,index,entries,*,seed,max_tokens=32768,callback=None):
    """Bounded calls (normally <=200 rows); callbacks get per-batch CPU results."""
    pending=[];largest=0;results=[]
    device=next(model.parameters()).device
    def forward(batch_entries):
        items=[make_item(index.read(e['offset']),model.tok,32768,seed) for e in batch_entries]
        features=[transition_metadata(it['row']) for it in items]
        batch=collate(items,model.tok.pad_token_id)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.no_grad():logits=model(batch).cpu()
        check_eval_logits(logits)
        result={'logits':logits,'targets':batch['targets'].cpu(),
                'id':[it['ex_id'] for it in items], 'dataset':[it['task'] for it in items],
                'outcome':[it['row']['outcome'] for it in items],
                'candidate_level':[it['row']['candidate_level'] for it in items],
                'nopts':[it['nopts'][0] for it in items],
                'tool_names':[[tool_name(it['row']['options'][i]) for i in it['perms'][0]] for it in items],
                'transition':[value[0] for value in features],'copy_prev_hit':[value[1] for value in features]}
        if callback is not None:callback(result)
        else:results.append(result)
    model.eval()
    for entry in sorted(entries,key=lambda e:(e['n_prompt_tokens'],e['id'])):
        padded=((entry['n_prompt_tokens']+63)//64)*64
        if pending and max(largest,padded)*(len(pending)+1)>max_tokens:
            forward(pending);pending=[];largest=0
        pending.append(entry);largest=max(largest,padded)
    if pending:forward(pending)
    return combine(results) if callback is None else None


def plan_parts(entries,shard,shards,part_size=200):
    assigned=entries[shard::shards]
    assigned=sorted(assigned,key=lambda e:(e['n_prompt_tokens'],e['id']))
    return [assigned[i:i+part_size] for i in range(0,len(assigned),part_size)]


def load_part(path,entries):
    part=torch.load(path,map_location='cpu',weights_only=True)
    if part['id'] != [e['id'] for e in entries]:
        raise ValueError(f'{path}: part ids/order differ from the selected shard')
    n=len(entries)
    if any(len(part[k])!=n for k in LIST_FIELDS) or part['logits'].shape!=(n,255) or part['targets'].shape!=(n,255):
        raise ValueError(f'{path}: incomplete part')
    check_eval_logits(part['logits'])
    return part


def run_parts(model,index,entries,*,seed,shard,shards,directory,part_size=200):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    plan=plan_parts(entries,shard,shards,part_size)
    started=time.monotonic();new_count=0
    for number,group in enumerate(plan):
        path=directory/f'part_{number:06d}.pt'
        if path.exists():
            load_part(path,group)
            print(f'{directory}: reuse part {number+1}/{len(plan)}',flush=True)
            continue
        parts=[]
        evaluate(model,index,group,seed=seed,callback=parts.append)
        part=combine(parts)
        temporary=path.with_suffix('.pt.pending')
        torch.save(part,temporary);temporary.replace(path)
        new_count+=len(group)
        elapsed=time.monotonic()-started
        remaining=sum(map(len,plan[number+1:]))
        print(f'{directory}: part {number+1}/{len(plan)}, new rows={new_count}, ETA={remaining*elapsed/new_count:.0f}s',flush=True)


def read_parts(entries,*,shards,directory,part_size=200):
    parts=[];directory=Path(directory);all_expected=set()
    for shard in range(shards):
        plan=plan_parts(entries,shard,shards,part_size)
        expected={directory/str(shard)/f'part_{i:06d}.pt' for i in range(len(plan))}
        all_expected.update(expected)
        actual=set((directory/str(shard)).glob('part_*.pt'))
        if actual!=expected:raise ValueError(f'{directory}/{shard}: missing or unexpected parts')
        for i,group in enumerate(plan):parts.append(load_part(directory/str(shard)/f'part_{i:06d}.pt',group))
    if set(directory.glob('*/part_*.pt'))!=all_expected:raise ValueError(f'{directory}: unexpected shard parts')
    result=combine(parts)
    if len(set(result['id']))!=len(entries) or set(result['id'])!={e['id'] for e in entries}:
        raise ValueError('part union differs from selected sample ids')
    return result


def load_majority(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in value.items()):
        raise ValueError('majority file must map dataset names to nonempty tool names')
    return value


def training_majority_tools(train):
    freq=defaultdict(Counter)
    for row in train:
        if len(row['gold'])==1:
            name=tool_name(row['options'][row['gold'][0]])
            if name!='RESPOND':freq[row['dataset']][name]+=1
    return {d:min(c,key=lambda name:(-c[name],name)) for d,c in freq.items() if c}


def measured(dump,temperature=1.,*,train_majority=None,oracle=False):
    logits,targets,datasets,names=dump['logits'],dump['targets'],dump['dataset'],dump['tool_names'];metadata=[dict(outcome=o,candidate_level=c) for o,c in zip(dump['outcome'],dump['candidate_level'])];freq=defaultdict(Counter)
    if oracle:
        for i,dataset in enumerate(datasets):
            for j in torch.where(targets[i]>0)[0].tolist():
                if names[i][j]!='RESPOND':freq[dataset][names[i][j]]+=1
        majority={d:min(c,key=lambda name:(-c[name],name)) for d,c in freq.items() if c}
    else:
        if train_majority is None:raise ValueError('ID evaluation requires training majority statistics')
        majority=train_majority
    # Report-only: per-task tool sets (e.g. Toolathlon MCP servers) can legitimately exclude the
    # training majority tool from every evaluated state, so absence is counted rather than raised.
    available=Counter(d for d,options in zip(datasets,names) if majority.get(d) in options)
    key='oracle_majority_tool_acc' if oracle else 'train_majority_tool_acc'
    hits=[any(targets[i,j]>0 and name==majority.get(d) for j,name in enumerate(names[i])) for i,d in enumerate(datasets)]
    respond_hits=[any(targets[i,j]>0 and name=='RESPOND' for j,name in enumerate(names[i])) for i in range(len(datasets))]
    slices={k:[row[k] for row in metadata] for k in ('outcome','candidate_level')}
    slices['action_kind']=['TOOL_SET' if int((targets[i]>0).sum())>1 else 'RESPOND' if respond_hits[i] else 'TOOL' for i in range(len(datasets))]
    slices['transition']=dump['transition']
    if len(dump['transition'])!=len(datasets) or len(dump['copy_prev_hit'])!=len(datasets):raise ValueError('incomplete transition metadata')
    if any(v not in {'first','repeat','change'} for v in dump['transition']) or any(type(v) is not bool for v in dump['copy_prev_hit']):raise ValueError('invalid transition metadata')
    result=metrics(logits/temperature,targets,datasets,hits,slices=slices,respond_hits=respond_hits)
    for v in [result['overall'],*result['datasets'].values()]:
        if 'most_common_tool_acc' in v:v[key]=v.pop('most_common_tool_acc')
        for transition in ('first','repeat','change'):v.setdefault('by_transition',{}).setdefault(transition,{'count':0,'acc':None})
    result['overall']['copy_prev_acc']=sum(dump['copy_prev_hit'])/len(datasets) if datasets else None
    copy_by_dataset=defaultdict(list)
    for d,hit in zip(datasets,dump['copy_prev_hit']):copy_by_dataset[d].append(hit)
    for d,v in result['datasets'].items():
        v['most_common_tool']=majority.get(d)
        v['majority_tool_candidate_count']=available[d]
        v['copy_prev_acc']=sum(copy_by_dataset[d])/len(copy_by_dataset[d])
        if d not in majority:v[key]=None
    result['overall']['majority_tool_available_count']=sum(d in majority for d in datasets)
    if any(d not in majority for d in datasets):result['overall'][key]=None
    result['baseline_definition']={
        'train_majority_tool_acc':'Most frequent single-gold non-RESPOND tool per dataset in the actual selected training set; excludes TOOL_SET; ties by tool name; val/monitor/test_id only.',
        'oracle_majority_tool_acc':'Descriptive oracle within test_ood per dataset: most frequent non-RESPOND gold membership including TOOL_SET; ties by tool name. Not a learned baseline.'}
    diagnostics=prediction_diagnostics(dump)
    respond_gold=respond_pred=respond_correct=0
    for dataset,stats in diagnostics.items():
        result['datasets'][dataset].update(stats)
        respond=stats['respond']
        respond_gold+=respond['gold_count'];respond_pred+=respond['predicted_count'];respond_correct+=respond['true_positive_count']
    result['overall']['respond']={'precision':respond_correct/respond_pred if respond_pred else None,
                                  'recall':respond_correct/respond_gold if respond_gold else None}
    result['diagnostic_definition']={
        'copy_prev_acc':'Fraction of states hit by copying the previous action: one distinct previous name must belong to gold; multiple distinct previous names must exactly equal the gold set. RESPOND is an action; empty previous set misses. Computed from full untruncated history; partial overlap or a strict superset of gold does not hit for a multi-tool previous action.',
        'by_transition':'Set-hit model accuracy/count by stored transition: first=no previous assistant action; repeat=current gold set is a subset of previous action names; change=otherwise. Empty slices have count 0 and null accuracy. Evaluation states are not filtered.',
        'respond':'Gold/predicted state counts and top1 true positives for RESPOND; precision=TP/predicted_count, recall=TP/gold_count; zero denominator is null. Overall pools RESPOND counts only.',
        'tool_prediction':'Top 10 non-RESPOND tools per dataset by gold membership count, ties by tool name. Each distinct gold tool counts once per state, including TOOL_SET; top1 prediction counts once. TP means prediction is this tool and it belongs to the gold set. Recall=TP/gold_count; precision=TP/predicted_count; zero denominator is null.',
        'top_confusions':'Top 5 incorrect top1 (gold, predicted) pairs per dataset; only predictions outside the full gold set are errors. For multiple gold tools, use the lexicographically first gold name. Includes RESPOND; sort by descending count, then gold and predicted names. These are tool-level diagnostics, not a task-success measure.'}
    return result


def prediction_diagnostics(dump):
    counts={}
    for i,dataset in enumerate(dump['dataset']):
        gold,predicted,correct,confusions=counts.setdefault(dataset,(Counter(),Counter(),Counter(),Counter()))
        names=dump['tool_names'][i]
        members={names[j] for j in torch.where(dump['targets'][i]>0)[0].tolist()}
        prediction=names[int(dump['logits'][i].argmax())]
        gold.update(members);predicted[prediction]+=1
        if prediction in members:correct[prediction]+=1
        else:confusions[(min(members),prediction)]+=1
    result={}
    for dataset,(gold,predicted,correct,confusions) in counts.items():
        def scores(name):
            return {'gold_count':gold[name],'predicted_count':predicted[name],'true_positive_count':correct[name],
                    'recall':correct[name]/gold[name] if gold[name] else None,
                    'precision':correct[name]/predicted[name] if predicted[name] else None}
        tools=sorted((name for name in gold if name!='RESPOND'),key=lambda name:(-gold[name],name))[:10]
        result[dataset]={'respond':scores('RESPOND'),
                         'tool_prediction':{name:scores(name) for name in tools},
                         'top_confusions':[{'gold':g,'predicted':p,'count':n} for (g,p),n in sorted(confusions.items(),key=lambda pair:(-pair[1],*pair[0]))[:5]]}
    return result


def sign_test(improved,regressed):
    """Exact two-sided binomial test (p=0.5) on discordant paired states; None without discordance."""
    n=improved+regressed
    if n==0:return None
    return min(1.,2*sum(math.comb(n,k) for k in range(min(improved,regressed)+1))/2**n)


def completed_parts(directory,split):
    """Saved parts of a completed evaluation directory, with its selection seed."""
    directory=Path(directory)
    status=json.loads((directory/'metadata/run_status.json').read_text())
    if status['status']!='completed':raise ValueError(f'{directory}: comparison evaluation must be completed')
    configuration=status['configuration']
    if split not in requested_splits(configuration):raise ValueError(f'{directory}: requested split {split} absent')
    report=json.loads((directory/'eval.json').read_text())
    if set(report).intersection(SPLITS)!=set(requested_splits(configuration)):
        raise ValueError(f'{directory}: report splits differ from declared scope (legacy evaluations require all three)')
    if 'val' not in requested_splits(configuration):
        if (not isinstance(configuration['calibration_eval'],str) or not configuration['calibration_eval']
                or configuration['calibration_temperature']!=report['temperature']):
            raise ValueError(f'{directory}: test-only calibration provenance differs')
    if split not in report:raise ValueError(f'{directory}: requested split {split} absent from eval.json')
    index=SampleIndex(Path(configuration['samples'])/(split+'.jsonl'))
    entries=load_selected(index,configuration['limits'][split],configuration['seed'])
    result=read_parts(entries,shards=configuration['shards'],directory=directory/'shards'/split)
    if len(result['id'])!=report[split]['overall']['count']:raise ValueError(f'{directory}: comparison parts are incomplete')
    return result,status['configuration']['seed']


def paired_difference(current,reference,*,seed,reference_seed):
    if seed!=reference_seed:raise ValueError('paired comparison requires identical seeds')
    if len(set(current['id']))!=len(current['id']) or len(set(reference['id']))!=len(reference['id']) or set(current['id'])!=set(reference['id']):
        raise ValueError('paired comparison requires exactly matching unique sample ids')
    other={value:i for i,value in enumerate(reference['id'])}
    both=improved=regressed=neither=0
    for i,value in enumerate(current['id']):
        j=other[value]
        if current['nopts'][i]!=reference['nopts'][j] or not torch.equal(current['targets'][i],reference['targets'][j]):
            raise ValueError(f'{value}: paired targets/nopts differ')
        hit=bool(current['targets'][i,int(current['logits'][i].argmax())]>0)
        old_hit=bool(reference['targets'][j,int(reference['logits'][j].argmax())]>0)
        both+=hit and old_hit;improved+=hit and not old_hit;regressed+=old_hit and not hit;neither+=not hit and not old_hit
    n=len(current['id'])
    return {'count':n,'acc_difference':(improved-regressed)/n if n else None,'improved':improved,'regressed':regressed,'discordant':improved+regressed,'both_correct':both,'both_wrong':neither,'sign_test_p':sign_test(improved,regressed)}
