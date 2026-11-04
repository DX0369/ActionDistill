"""Deterministic training caps; repair earlier caps after later subsampling."""
import math
from collections import Counter,defaultdict,deque
from .common import digest


def select_hash(rows, count, seed):
    return sorted(rows,key=lambda r:(digest(seed,r['id']),r['id']))[:count]


def constrain(rows, predicate, fraction, seed):
    positive=[r for r in rows if predicate(r)];other=[r for r in rows if not predicate(r)]
    keep=min(len(positive),math.floor(len(other)*fraction/(1-fraction)+1e-9))
    return other+select_hash(positive,keep,seed)


def training_mix(rows, seed, failure_fraction=.3, tool_fraction=.4, cap=30000):
    report={'before':len(rows)}
    rows=constrain(rows,lambda r:r['outcome']=='failure',failure_fraction,(seed,'failure'))
    report['after_failure_cap']=len(rows)
    frequencies=Counter(r['action_names'][0] for r in rows if r['action_kind']=='TOOL')
    top=min(frequencies,key=lambda t:(-frequencies[t],t)) if frequencies else None
    report['dominant_single_tool']=top
    def tool(r):return top is not None and r['action_kind']=='TOOL' and r['action_names']==[top]
    def repair(current,stage):
        while True:
            n=len(current)
            freq=Counter(r['action_names'][0] for r in current if r['action_kind']=='TOOL')
            dominant=min(freq,key=lambda t:(-freq[t],t)) if freq else None
            current=constrain(current,lambda r:dominant is not None and r['action_kind']=='TOOL' and r['action_names']==[dominant],tool_fraction,(seed,stage,'tool'))
            current=constrain(current,lambda r:r['outcome']=='failure',failure_fraction,(seed,stage,'failure'))
            if len(current)==n:return current
    rows=repair(rows,'tool_cap')
    report['after_tool_cap']=len(rows)
    before=len(rows)
    if len(rows)>cap:
        trajectories=defaultdict(list)
        for r in rows:trajectories[r['trajectory_id']].append(r)
        queues=deque(deque(select_hash(trajectories[t],len(trajectories[t]),(seed,'trajectory'))) for t in sorted(trajectories,key=lambda t:digest(seed,t)))
        chosen=[]
        while queues and len(chosen)<cap:
            q=queues.popleft();chosen.append(q.popleft())
            if q:queues.append(q)
        rows=chosen
    sampled=len(rows);rows=repair(rows,'dataset_cap')
    report.update(after_dataset_cap=len(rows),dataset_cap_initial_sample=sampled,
                  cap_repair_dropped=sampled-len(rows),before_dataset_cap=before)
    n=len(rows)
    if n and (sum(r['outcome']=='failure' for r in rows)>failure_fraction*n+1e-8 or max(Counter(r['action_names'][0] for r in rows if r['action_kind']=='TOOL').values(),default=0)>tool_fraction*n+1e-8):
        raise AssertionError('training caps not satisfied')
    return sorted(rows,key=lambda r:digest(seed,r['id'])),report
