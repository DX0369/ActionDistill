"""Deterministic traversal of task groups, trajectories and states."""
import math
from collections import Counter,defaultdict,deque
from pathlib import Path
from .common import digest,rows


def round_robin_states(records,seed,order):
    groups=defaultdict(lambda:defaultdict(list))
    for row in records:groups[row['task_group']][row['trajectory_id']].append(row)
    queue=deque()
    for group in sorted(groups,key=lambda g:(digest(seed,'group',g),g)):
        trajectories=groups[group]
        queue.append(deque(deque(order(trajectories[t])) for t in sorted(trajectories,key=lambda t:(digest(seed,'trajectory',t),t))))
    while queue:
        trajectories=queue.popleft();states=trajectories.popleft()
        yield states.popleft()
        if states:trajectories.append(states)
        if trajectories:queue.append(trajectories)


def pool_sample(records,seed,per_trajectory,cap):
    from itertools import islice
    order=lambda r:islice(spread_order(r,seed,r[0]['dataset'],r[0]['trajectory_id']),per_trajectory)
    return list(islice(round_robin_states(records,seed,order),cap))


def spread_order(states,seed,dataset,trajectory):
    """Phase-shifted base-2 radical inverse; exact integer floor, no float rounding."""
    ordered=sorted(states,key=lambda r:(r['turn_index'],r['id']))
    phase=int(digest(seed,'phase',dataset,trajectory),16);scale=1<<256
    seen=set();i=0
    while len(seen)<len(ordered):
        numerator=0;denominator=1;j=i
        while j:
            numerator=2*numerator+(j&1);denominator*=2;j//=2
        period=scale*denominator
        position=((numerator*scale+phase*denominator)%period)*len(ordered)//period;i+=1
        if position not in seen:
            seen.add(position);yield ordered[position]


def validate_config(config):
    for key in ('quota_exponent','min_fraction','max_fraction','tool_fraction','failure_fraction'):
        value=config[key]
        if type(value) not in {float,int} or not math.isfinite(value) or not 0<value<=1:raise ValueError(f'invalid {key}')
    if config['min_fraction']>config['max_fraction']:raise ValueError('minimum exceeds maximum fraction')
    if type(config['per_trajectory_cap']) is not int or config['per_trajectory_cap']<=0:raise ValueError('invalid per_trajectory_cap')
    value=config['repeat_fraction']
    if value!='natural' and (type(value) not in {float,int} or not math.isfinite(value) or not 0<=value<=1):raise ValueError('invalid repeat_fraction')
    if config['loop_filter'] not in {'off','exact3'} or config['concise'] not in {'off','top_half'}:raise ValueError('invalid filter mode')


def load_pool(path):
    from .data import validate,tool_name
    result=[];seen=set();trajectories={}
    fields=('id','dataset','task_group','trajectory_id','turn_index','harness','action_kind',
            'outcome','transition','loop_state','trajectory_decisions','trajectory_position','n_prompt_tokens')
    for number,row in enumerate(rows(path),1):
        try:
            validate(row);record={k:row[k] for k in fields}
            for key in ('trajectory_id',):
                if not isinstance(record[key],str) or not record[key]:raise ValueError(f'invalid {key}')
            if record['harness'] is not None and (not isinstance(record['harness'],str) or not record['harness']):raise ValueError('invalid harness')
            if type(record['turn_index']) is not int or record['turn_index']<0:raise ValueError('invalid turn_index')
            if type(record['trajectory_decisions']) is not int or record['trajectory_decisions']<=0:raise ValueError('invalid trajectory_decisions')
            if type(record['trajectory_position']) is not int or not 0<=record['trajectory_position']<record['trajectory_decisions']:raise ValueError('invalid trajectory_position')
            if record['transition'] not in {'first','repeat','change'} or record['loop_state'] not in {'yes','no'}:raise ValueError('invalid history features')
            record['gold_names']=sorted({tool_name(row['options'][i]) for i in row['gold']})
            kind='TOOL_SET' if len(record['gold_names'])>1 else 'RESPOND' if record['gold_names']==['RESPOND'] else 'TOOL'
            if record['action_kind']!=kind:raise ValueError('action_kind disagrees with gold')
            if record['id'] in seen:raise ValueError(f"duplicate id: {record['id']}")
            seen.add(record['id'])
            key=(record['dataset'],record['trajectory_id'])
            identity=tuple(record[k] for k in ('task_group','outcome','trajectory_decisions'))
            if key in trajectories and trajectories[key]!=identity:raise ValueError(f'inconsistent trajectory metadata: {key}')
            trajectories[key]=identity;result.append(record)
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
    return result


def filter_pool(records,config,seed):
    selected=[r for r in records if config['loop_filter']=='off' or r['loop_state']=='no']
    report={'input':len(records),'after_loop_filter':len(selected)}
    if config['concise']=='top_half':
        groups=defaultdict(dict);order={'success':0,'unknown':1,'failure':2}
        for r in selected:groups[r['dataset'],r['task_group']][(r['dataset'],r['trajectory_id'])]=r
        keep=set()
        for group,trajectories in groups.items():
            ordered=sorted(trajectories,key=lambda key:(order[trajectories[key]['outcome']],trajectories[key]['trajectory_decisions'],digest(seed,'concise',*key),key))
            keep.update(ordered[:(len(ordered)+1)//2])
        selected=[r for r in selected if (r['dataset'],r['trajectory_id']) in keep]
    return selected,dict(report,after_concise_filter=len(selected))


def proportional_quotas(weights,capacities,count,minimum,maximum,seed):
    """q_d = clip(lambda*w_d, min, max/capacity), then largest remainder."""
    upper={d:min(maximum,capacities[d]) for d in weights}
    if any(n<minimum for n in upper.values()):raise ValueError('dataset capacity below minimum quota')
    if minimum*len(weights)>count or sum(upper.values())<count:raise ValueError('quota bounds cannot fill requested count')
    lo=0.;hi=max(upper[d]/weights[d] for d in weights)
    for _ in range(80):
        scale=(lo+hi)/2
        if sum(min(upper[d],max(minimum,scale*w)) for d,w in weights.items())<count:lo=scale
        else:hi=scale
    raw={d:min(upper[d],max(minimum,hi*w)) for d,w in weights.items()}
    quotas={d:min(upper[d],math.floor(v+1e-10)) for d,v in raw.items()}
    remaining=count-sum(quotas.values())
    order=sorted(weights,key=lambda d:(-(raw[d]-quotas[d]),digest(seed,'quota',d),d))
    for d in [d for d in order if quotas[d]<upper[d]][:remaining]:quotas[d]+=1
    if sum(quotas.values())!=count:raise ValueError('quota rounding failed')
    return quotas,upper


def budget_counts(records,dominant):
    return {'tool':sum(r['action_kind']=='TOOL' and r['gold_names']==[dominant] for r in records),
            'failure':sum(r['outcome']=='failure' for r in records),
            'repeat':sum(r['transition']=='repeat' for r in records)}


def budget_limits(config,quota):
    limits={key:math.floor(config[key+'_fraction']*quota+1e-12) for key in ('tool','failure')}
    if config['repeat_fraction']!='natural':limits['repeat']=math.floor(config['repeat_fraction']*quota+1e-12)
    return limits


def take_quota(ordered,quota,config,dominant):
    limits=budget_limits(config,quota);used=Counter();trajectories=Counter();selected=[]
    for row in ordered:
        trajectory=row['trajectory_id']
        if trajectories[trajectory]>=config['per_trajectory_cap']:continue
        flags={'tool':row['action_kind']=='TOOL' and row['gold_names']==[dominant],
               'failure':row['outcome']=='failure','repeat':row['transition']=='repeat'}
        if any(used[k]+flags[k]>cap for k,cap in limits.items()):continue
        selected.append(row);used.update(flags);trajectories[trajectory]+=1
        if len(selected)==quota:break
    return selected


def select(records,config,count,seed,reference_quotas=None):
    validate_config(config)
    if type(count) is not int or count<=0:raise ValueError('count must be positive')
    datasets=sorted({r['dataset'] for r in records})
    if not datasets:raise ValueError('pool is empty')
    filtered,filter_report=filter_pool(records,config,seed)
    pools={d:[] for d in datasets}
    for r in filtered:pools[r['dataset']].append(r)
    ordered={};dominant={};selection_seed=digest(seed,'selection')
    for d,pool in pools.items():
        freq=Counter(r['gold_names'][0] for r in pool if r['action_kind']=='TOOL')
        dominant[d]=min(freq,key=lambda name:(-freq[name],name)) if freq else None
        ordered[d]=list(round_robin_states(pool,seed,lambda r:spread_order(r,selection_seed,d,r[0]['trajectory_id'])))
    attempts=Counter();cache={}
    def probe(d,q):
        if (d,q) not in cache:
            choice=take_quota(ordered[d],q,config,dominant[d]);attempts[d]+=1
            cache[d,q]=choice if len(choice)==q else None
        return cache[d,q]
    choices={}
    if reference_quotas is not None:
        # Fixed per-dataset quotas replace sqrt(task groups) weights and the min/max fractions;
        # every other rule (order, per-trajectory cap, tool/failure/repeat budgets) is unchanged.
        if set(reference_quotas)!=set(datasets):raise ValueError(f'reference quotas must name exactly the pool datasets; missing {sorted(set(datasets)-set(reference_quotas))}, extra {sorted(set(reference_quotas)-set(datasets))}')
        if any(type(q) is not int or q<0 for q in reference_quotas.values()) or sum(reference_quotas.values())!=count:raise ValueError('reference quotas must be nonnegative integers summing to count')
        quotas=dict(reference_quotas);initial=dict(quotas);minimum=0;upper=dict(quotas)
        for d in datasets:
            choice=take_quota(ordered[d],quotas[d],config,dominant[d]) if quotas[d] else []
            attempts[d]+=1
            if len(choice)!=quotas[d]:
                raise ValueError(f'{d}: reference quota {quotas[d]} infeasible under per-trajectory cap/budgets {budget_limits(config,quotas[d])} (got {len(choice)}); quotas are not reassigned')
            choices[d]=choice
    else:
        minimum=math.ceil(config['min_fraction']*count);maximum=math.floor(config['max_fraction']*count)
        capacities={d:sum(min(n,config['per_trajectory_cap']) for n in Counter(r['trajectory_id'] for r in pool).values()) for d,pool in pools.items()}
        short={d:n for d,n in capacities.items() if n<minimum}
        if short:raise ValueError(f'filtered dataset capacity below minimum quota {minimum}: {short}')
        weights={d:len({r['task_group'] for r in pool})**config['quota_exponent'] for d,pool in pools.items()}
        quotas,upper=proportional_quotas(weights,capacities,count,minimum,maximum,seed)
        initial=dict(quotas)
        for d in datasets:
            for q in range(quotas[d],minimum-1,-1):
                choice=probe(d,q)
                if choice is not None:choices[d]=choice;quotas[d]=q;break
            else:raise ValueError(f'{d}: no feasible quota down to minimum with this traversal; constraints were not relaxed')
        if sum(quotas.values())>count:raise ValueError('deterministic quota allocation exceeds count; constraints were not relaxed')
        while sum(quotas.values())<count:
            remaining=count-sum(quotas.values());increments=[]
            for d in datasets:
                for q in range(quotas[d]+1,min(upper[d],quotas[d]+remaining)+1):
                    if probe(d,q) is not None:
                        increments.append((q/weights[d],digest(seed,'quota',d),d,q));break
            if not increments:raise ValueError(f'deterministic quota allocation cannot fill remaining {remaining}; not a proof of global infeasibility')
            _,_,d,q=min(increments);quotas[d]=q;choices[d]=probe(d,q)
    selected=[r for d in datasets for r in choices[d]]
    assert len(selected)==count and len({r['id'] for r in selected})==count
    for d,choice in choices.items():
        if not choice:continue
        actual=budget_counts(choice,dominant[d]);assert minimum<=len(choice)<=upper[d]
        assert all(actual[k]<=cap for k,cap in budget_limits(config,len(choice)).items())
        assert max(Counter(r['trajectory_id'] for r in choice).values())<=config['per_trajectory_cap']
    selected.sort(key=lambda r:(digest(seed,r['id']),r['id']))
    summary=selection_summary(selected,dominant)
    summary.update(filters=filter_report,initial_quotas=initial,final_quotas=quotas,quota_probe_counts=dict(attempts),
                   config=config,count=count,seed=seed,quota_source='reference' if reference_quotas is not None else 'sqrt_task_groups',
                   definition='Final-count budgets; dominant single TOOL fixed in each filtered dataset pool. Concise filtering within (dataset, task_group). Shared phase-shifted spread order with independent selection phase from digest(seed, "selection"). Deterministic greedy quota probing, no relaxed constraints or infeasibility proof.',
                   position_definition=POSITION_DEFINITION)
    return selected,summary


def selection_summary(records,dominant):
    def describe(selected):
        n=len(selected);lengths=sorted(r['n_prompt_tokens'] for r in selected)
        trajectory_counts=Counter((r['dataset'],r['trajectory_id']) for r in selected)
        dominant_count=sum(r['action_kind']=='TOOL' and r['gold_names']==[dominant[r['dataset']]] for r in selected)
        return {'count':n,'failure_fraction':sum(r['outcome']=='failure' for r in selected)/n,
                'repeat_fraction':sum(r['transition']=='repeat' for r in selected)/n,'dominant_tool_fraction':dominant_count/n,
                'transition':{v:sum(r['transition']==v for r in selected) for v in ('first','repeat','change')},
                'relative_position':position_summary(selected),
                'task_groups':len({r['task_group'] for r in selected}),'trajectories':len(trajectory_counts),
                'states_per_trajectory':dict(sorted(Counter(trajectory_counts.values()).items())),
                'n_prompt_tokens':{'total':sum(lengths),'p50':lengths[math.ceil(.5*n)-1],'p90':lengths[math.ceil(.9*n)-1]}}
    report={'overall':describe(records),'dominant_tools':dominant}
    for field in ('dataset','harness','action_kind','transition'):
        groups=defaultdict(list)
        for row in records:groups[row[field]].append(row)
        report['by_'+field]={key:describe(value) for key,value in sorted(groups.items(),key=lambda item:(item[0] is not None,item[0]))}
    return report


POSITION_DEFINITION='trajectory_position / trajectory_decisions, where position is the zero-based (turn_index, id) rank among all indexed states, before pool/selection/length filtering; 10 half-open bins [0,.1), ... [.9,1). Single-state trajectories have position 0 and are also counted separately. State-weighted, not trajectory-weighted.'


def position_summary(records):
    bins=[0]*10;count=first=single=0
    for row in records:
        position=row['trajectory_position'];length=row['trajectory_decisions']
        bins[10*position//length]+=1;count+=1;first+=position==0;single+=length==1
    return {'count':count,'bin_counts':bins,'bin_fractions':[n/count if count else None for n in bins],
            'first_state_count':first,'single_state_trajectory_count':single}


def load_train_ids(index,path,seed):
    ids=Path(path).read_text().splitlines()
    if not ids or any(not value for value in ids):raise ValueError('train_ids must contain nonempty ids')
    if len(set(ids))!=len(ids):raise ValueError('duplicate train_ids')
    available={r['id']:r for r in index.entries}
    missing=set(ids)-available.keys()
    if missing:raise ValueError(f'train_ids missing from train: {sorted(missing)[:5]}')
    return sorted((available[sid] for sid in ids),key=lambda r:(digest(seed,r['id']),r['id']))
