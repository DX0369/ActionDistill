"""B2 trajectory deduplication and the explicitly approved source reallocation."""
from collections import Counter, defaultdict

from .common import digest
from .curation import select_versions

SHORT_SOURCE = 'deepseek-v4-pro-0813-agentic'


def reallocate_shortfall(quotas, capacities):
    """Release only 0813's shortfall; one largest-remainder allocation, then stop on insufficiency."""
    if set(quotas) != set(capacities) or SHORT_SOURCE not in quotas:
        raise ValueError('quota/capacity source set mismatch')
    if any(type(v) is not int or v <= 0 for v in quotas.values()):
        raise ValueError('original quotas must be positive integers')
    if any(type(v) is not int or v < 0 for v in capacities.values()):
        raise ValueError('capacities must be nonnegative integers')
    short = {d for d,k in quotas.items() if k > capacities[d]}
    if short != {SHORT_SOURCE}:
        raise ValueError(f'approved reallocation requires only {SHORT_SOURCE} to be short: {sorted(short)}')
    released = quotas[SHORT_SOURCE] - capacities[SHORT_SOURCE]
    weights = {d:k for d,k in quotas.items() if d != SHORT_SOURCE}
    denominator = sum(weights.values())
    extra = {d:released*k//denominator for d,k in weights.items()}
    order = sorted(weights,key=lambda d:(-(released*weights[d]%denominator),d))
    for d in order[:released-sum(extra.values())]:
        extra[d] += 1
    result = {d:capacities[d] if d==SHORT_SOURCE else k+extra[d] for d,k in quotas.items()}
    overflow = {d:result[d]-capacities[d] for d in result if result[d]>capacities[d]}
    if overflow:
        raise ValueError(f'capacity insufficient after single reallocation; no further redistribution: {overflow}')
    if sum(result.values()) != sum(quotas.values()):
        raise ValueError('reallocation changed total quota')
    flows = [dict(source=SHORT_SOURCE,destination=d,count=extra[d],original_weight=weights[d])
             for d in sorted(extra) if extra[d]]
    return result, flows


def deduplicate(plans, seed):
    """Choose one C1-eligible trajectory per (dataset, task_group, model)."""
    winners, trajectories = {}, {}
    for sid, row in plans.items():
        for key in ('id', 'dataset', 'task_group', 'model', 'trajectory_id'):
            if key not in row or not isinstance(row[key], str) or not row[key]:
                raise ValueError(f'{sid}: missing/invalid canonical {key}')
        if row['id'] != sid or row['split'] != 'train':
            raise ValueError(f'{sid}: expected matching training ID')
        key = row['dataset'], row['task_group'], row['model']
        trajectory = row['dataset'], row['trajectory_id']
        if trajectory in trajectories and trajectories[trajectory] != row['task_group']:
            raise ValueError(f'{trajectory}: inconsistent task_group')
        trajectories[trajectory] = row['task_group']
        if row['drops']:
            continue
        rank = digest(seed, row['trajectory_id']), row['trajectory_id']
        if key not in winners or rank < winners[key]:
            winners[key] = rank
    kept = {sid: row for sid, row in plans.items()
            if not row['drops'] and
            row['trajectory_id'] == winners[row['dataset'], row['task_group'], row['model']][1]}
    chosen_trajectories = {(r['dataset'],r['trajectory_id']) for r in kept.values()}
    if any(not r['drops'] and (r['dataset'],r['trajectory_id']) in chosen_trajectories and sid not in kept
           for sid,r in plans.items()):
        raise ValueError('multi-model trajectory would be partially retained; cannot satisfy both dedup and all-states rules')
    return kept


def b2_capacity(priority, ordinary):
    """Maximum count respecting Python round(.25*k) and ordinary-only backfill."""
    lo, hi = 0, priority + ordinary
    while lo < hi:
        k = (lo + hi + 1) // 2
        if k - min(priority, round(.25 * k)) <= ordinary:
            lo = k
        else:
            hi = k - 1
    return lo


def population(rows):
    rows = list(rows)
    counts = Counter(r['task_group'] for r in rows)
    return dict(states=len(rows), trajectories=len({(r['dataset'], r['trajectory_id']) for r in rows}),
                task_groups=len(counts), teachers=len({r['model'] for r in rows}),
                states_per_task_group_histogram=dict(sorted(Counter(counts.values()).items())),
                maximum_states_per_task_group=max(counts.values(), default=0))


def preview(plans, quotas, seed, reference_ids):
    reference_ids = set(reference_ids)
    if not reference_ids <= plans.keys():
        raise ValueError('reference ID absent from recovered C1 candidates')
    eligible = {sid: r for sid, r in plans.items() if not r['drops']}
    kept = deduplicate(plans, seed)
    by_before, by_after = defaultdict(list), defaultdict(list)
    for r in eligible.values():
        by_before[r['dataset']].append(r)
    for r in kept.values():
        by_after[r['dataset']].append(r)
    if set(by_before) != set(quotas):
        raise ValueError('C1 source set differs from reference quotas')
    sources, shortfalls = {}, {}
    for dataset, quota in sorted(quotas.items()):
        after = by_after[dataset]
        priority = sum(r['short_turn'] or r['direct'] for r in after)
        ordinary = len(after) - priority
        capacity = b2_capacity(priority, ordinary)
        sources[dataset] = dict(before=population(by_before[dataset]), after=population(after),
            priority_available=priority, ordinary_available=ordinary, b2_capacity=capacity,
            original_quota=quota, final_quota=quota, priority_quota=round(.25*quota),
            shortfall=max(0, quota-capacity), redistribution=[])
        if quota > capacity:
            shortfalls[dataset] = sources[dataset]['shortfall']
    report = dict(before=population(eligible.values()), after=population(kept.values()),
                  sources=sources, shortfalls=shortfalls, redistribution=[],
                  selected=None, quota_policy='fixed_source_quotas_stop_on_shortfall')
    if shortfalls:
        return kept, [], report
    _, ids, _ = select_versions(kept, quotas, seed, apply_c4=False)
    selected = [kept[sid] for sid in ids]
    def composition(rows):
        count = len(rows)
        respond = sum(r['gold_names'] == ['RESPOND'] for r in rows)
        priority = sum(r['short_turn'] or r['direct'] for r in rows)
        return dict(count=count, respond=respond, respond_fraction=respond/count if count else None,
                    task_groups=len({r['task_group'] for r in rows}),
                    overlap_with_reference=sum(r['id'] in reference_ids for r in rows),
                    priority=priority, priority_fraction=priority/count if count else None,
                    short_turn=sum(r['short_turn'] for r in rows), direct=sum(r['direct'] for r in rows),
                    turn_calls_unknown=sum(r['turn_calls_unknown'] for r in rows),
                    n_prompt_tokens=sum(r['n_prompt_tokens'] for r in rows),
                    respond_next_event=dict(Counter(r['next_event'] for r in rows if r['gold_names']==['RESPOND'])))
    report['selected'] = dict(population(selected), **composition(selected))
    for dataset, stats in sources.items():
        stats['selected'] = composition([r for r in selected if r['dataset']==dataset])
    report['respond_gate_passed'] = .08 <= report['selected']['respond_fraction'] <= .20
    return kept, ids, report


def redistributed_preview(plans, original_quotas, seed, reference_ids):
    _, _, initial = preview(plans, original_quotas, seed, reference_ids)
    capacities = {d:r['b2_capacity'] for d,r in initial['sources'].items()}
    quotas, flows = reallocate_shortfall(original_quotas, capacities)
    kept, ids, result = preview(plans, quotas, seed, reference_ids)
    if result['shortfalls']:
        raise ValueError(f'post-reallocation selection infeasible: {result["shortfalls"]}')
    result.update(quota_policy='0813_capacity_then_original_weight_largest_remainder_once',
                  initial_shortfalls=initial['shortfalls'], redistribution=flows,
                  rounding='Integer largest remainder; ties by dataset name, matching load_selected.')
    reference = [plans[sid] for sid in reference_ids]
    result['reference'] = population(reference)
    for dataset, stats in result['sources'].items():
        stats.update(original_quota=original_quotas[dataset],
                     initial_shortfall=initial['sources'][dataset]['shortfall'],
                     redistribution=[r for r in flows if dataset in (r['source'],r['destination'])],
                     reference=population(r for r in reference if r['dataset']==dataset))
    return kept, ids, result
