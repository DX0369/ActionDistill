"""Closed task-deliverable taxonomy and hash-checked API inputs."""
import hashlib
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .common import digest

LABELS = {
    'repo_swe': '在已有代码仓库里修 bug、实现功能、重构、补测试、排查构建问题',
    'app_build': '从零或基于脚手架构建 web/app/UI/组件/小游戏',
    'data_file': '处理表格、CSV/JSON、数据库查询、文档解析，以及报表、统计、金融分析',
    'web_research': '浏览网页、搜索、收集信息并汇总',
    'office_assistant': '邮件、日程、文档协作、账号/工具/技能配置、定时任务，以及提示注入或安全对抗类指令的处理',
    'ml_experiment': '训练或评测模型、数据科学竞赛、实验调参',
    'other': '不属于以上任何一类，或无法判断',
}
FIXED = {'Open-SWE-Traces': 'repo_swe', 'swe-rebench-v2-glm-5.1-pi-traces': 'repo_swe',
         'mle-trajectories': 'ml_experiment', 'finance-agents-benchmark-traces': 'data_file'}
REVIEW_FAILURE = 'review agreement below 90%; taxonomy decision required; no relabeling or selection'


def label_acceptance(state, directory):
    """Keep the original review outcome; an explicit user decision may admit this batch."""
    review=state['review']
    if state['failure_count']!=0 or review['count']!=100 or not 0<=review['agreement']<=1:
        raise ValueError('domain labels require zero API failures and a valid 100-group review')
    if state['status']=='completed' and review['agreement']>=.90:
        return dict(decision='review_threshold',agreement=review['agreement'],required=.90)
    if 'acceptance' not in state:
        raise ValueError('domain review gate not passed and no explicit user acceptance')
    acceptance=state['acceptance']
    if (state['status']!='failed' or state['error']!=REVIEW_FAILURE or review['required']!=.90
            or acceptance['authority']!='user' or acceptance['decision']!='accept_existing_labels'
            or acceptance['labels_directory']!=str(Path(directory).resolve())
            or acceptance['review_count']!=review['count'] or acceptance['review_agreement']!=review['agreement']
            or acceptance['original_required']!=review['required']):
        raise ValueError('user acceptance does not match this label batch and original review')
    if not isinstance(acceptance['reason'],str) or not acceptance['reason'].strip():
        raise ValueError('user acceptance requires a reason')
    if datetime.fromisoformat(acceptance['accepted_at']).utcoffset() is None:
        raise ValueError('user acceptance timestamp requires a timezone')
    return acceptance


def clean_request(request, request_hash):
    if not isinstance(request, str) or not request:
        raise ValueError('missing initial request')
    if hashlib.sha256(request.encode('utf-8')).hexdigest() != request_hash:
        raise ValueError('initial request SHA256 differs from canonical request_hash')
    text = re.sub(r'<environment_context>.*?</environment_context>', '', request, flags=re.S)
    text = re.sub(r'^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-[^\]\n]*UTC\]\s*', '', text)
    text = re.sub(r'</?file(?:\s[^<>]*)?>', '', text)
    return text[:3000]


def fixed_label(datasets):
    labels = {FIXED[d] for d in datasets if d in FIXED}
    if len(labels) > 1:
        raise ValueError(f'conflicting fixed domains: {sorted(datasets)}')
    return next(iter(labels)) if labels else None


def annotation_requests(records, requests, seed):
    """One representative request per retained task group; split remains canonical."""
    groups = {}
    for row in records:
        group = groups.setdefault(row['task_group'], dict(split=row['split'], members=set(), trajectories=set()))
        if group['split'] != row['split']:
            raise ValueError(f'task_group crosses splits: {row["task_group"]}')
        group['members'].add(row['dataset'])
        group['trajectories'].add((row['dataset'], row['trajectory_id']))
    result = []
    for task_group, group in sorted(groups.items()):
        # Dataset is only a tie-break when two canonical trajectory_id strings coincide.
        key = min(group['trajectories'], key=lambda k: (digest(seed, k[1]), k[1], k[0]))
        request = requests[key]
        if request['task_group'] != task_group or request['split'] != group['split']:
            raise ValueError(f'request identity mismatch: {key}')
        members = sorted(group['members'])
        result.append(dict(task_group=task_group, split=group['split'], member_datasets=members,
            dataset=key[0], trajectory_id=key[1], request_sha256=request['request_hash'],
            request=request['request'], fixed_label=fixed_label(members)))
    return result


def parse_label(content):
    if not isinstance(content, str) or content.strip() not in LABELS:
        raise ValueError(f'API label is not an exact member of the closed taxonomy: {repr(content)[:160]}')
    return content.strip()


def apportion(weights, count):
    if type(count) is not int or count<0 or not weights or any(not math.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('invalid allocation weights/count')
    total = sum(weights.values())
    raw = {k:count*v/total for k,v in weights.items()}
    result = {k:math.floor(v) for k,v in raw.items()}
    order = sorted(weights,key=lambda k:(-(raw[k]-result[k]),k))
    for k in order[:count-sum(result.values())]:
        result[k]+=1
    if sum(result.values())!=count:
        raise ValueError('largest remainder total differs')
    return result


def capped_allocation(weights, count, capacities):
    """Redistribute only to positive-weight cells with B2-feasible spare capacity."""
    if set(weights)!=set(capacities) or any(type(v) is not int or v<0 for v in capacities.values()):
        raise ValueError('invalid cell capacities')
    if sum(capacities.values())<count:
        raise ValueError('global/cell capacity cannot fill requested count')
    initial = apportion(weights,count)
    result = {k:min(initial[k],capacities[k]) for k in weights}
    while (remaining:=count-sum(result.values())):
        active = {k:weights[k] for k in weights if result[k]<capacities[k]}
        if not active:
            raise ValueError('allocation has no eligible spare capacity')
        extra = apportion(active,remaining)
        for k,n in extra.items():
            result[k]+=min(n,capacities[k]-result[k])
    return result


def capped_equal_allocation(count, capacities):
    """Integer water filling; lexical ties only after saturated cells are fixed."""
    if (type(count) is not int or count <= 0 or not capacities
            or any(type(v) is not int or v < 0 for v in capacities.values())):
        raise ValueError('invalid equal-cap allocation inputs')
    if sum(capacities.values()) < count:
        raise ValueError('supported domain capacity cannot fill requested count')
    result = {}
    active = dict(capacities)
    remaining = count
    while active:
        proposed = apportion({key: 1 for key in active}, remaining)
        saturated = [key for key in active if capacities[key] < proposed[key]]
        if not saturated:
            result.update(proposed)
            break
        for key in saturated:
            result[key] = active.pop(key)
            remaining -= result[key]
    return dict(sorted(result.items()))


class BalanceCapacityError(ValueError):
    """A fixed quota plan fails C3; retain the plan without partial selections."""
    def __init__(self, report):
        self.report = report
        super().__init__('; '.join(f"{source}: priority budget {r['priority_quota']} < "
                                  f"mandatory priority {r['mandatory_priority']}"
                                  for source, r in report['c3_failures'].items()) + '; no relaxation')


def allocation_flows(before, after):
    """Deterministic bookkeeping of net transfers, after quotas have been fixed."""
    donors = [[k,before[k]-after[k]] for k in sorted(before) if before[k]>after[k]]
    receivers = [[k,after[k]-before[k]] for k in sorted(before) if before[k]<after[k]]
    result=[];i=j=0
    while i<len(donors) and j<len(receivers):
        n=min(donors[i][1],receivers[j][1])
        result.append(dict(source=donors[i][0],destination=receivers[j][0],count=n))
        donors[i][1]-=n;receivers[j][1]-=n
        if donors[i][1]==0:i+=1
        if receivers[j][1]==0:j+=1
    if i!=len(donors) or j!=len(receivers):
        raise ValueError('transfer totals differ')
    return result


def balanced_selection(plans, labels, source_weights, count, seed):
    from .dedup import deduplicate, b2_capacity, population
    from .curation import select_versions
    kept=deduplicate(plans,seed)
    cells=defaultdict(lambda:defaultdict(dict));groups=defaultdict(set)
    for sid,row in kept.items():
        domain=labels[row['task_group']]  # Missing label is a hard boundary failure.
        if domain not in LABELS:
            raise ValueError(f'invalid domain label: {domain}')
        cells[domain][row['dataset']][sid]=row
        groups[domain].add(row['task_group'])
    if {r['dataset'] for r in kept.values()}!=set(source_weights):
        raise ValueError('domain candidates differ from original source scope')
    domain_initial=apportion({d:math.sqrt(len(g)) for d,g in groups.items()},count)
    capacities={};priorities={}
    for domain, sources in cells.items():
        capacities[domain]={};priorities[domain]={}
        for dataset, rows_ in sources.items():
            p=sum(r['short_turn'] or r['direct'] for r in rows_.values())
            priorities[domain][dataset]=p
            capacities[domain][dataset]=b2_capacity(p,len(rows_)-p)
    domain_capacity={d:sum(c.values()) for d,c in capacities.items()}
    domain_final=capped_allocation(domain_initial,count,domain_capacity)
    chosen=[];cell_report={};domain_report={};local_flows={}
    for domain,sources in sorted(cells.items()):
        weights={ds:source_weights[ds] for ds in sources}
        initial=apportion(weights,domain_initial[domain])
        base_total=min(domain_initial[domain],domain_capacity[domain])
        base=capped_allocation(weights,base_total,capacities[domain])
        # Finish local redistribution before appending cross-domain seats. Repeating
        # largest-remainder from scratch can revoke a cell's earlier seat (Alabama paradox).
        quotas=dict(base)
        extra=domain_final[domain]-base_total
        if extra:
            active={ds:weights[ds] for ds in weights if base[ds]<capacities[domain][ds]}
            additions=capped_allocation(active,extra,{ds:capacities[domain][ds]-base[ds] for ds in active})
            for ds,n in additions.items():quotas[ds]+=n
        local_flows[domain]=allocation_flows(dict(initial,__other_domains__=0),
            dict(base,__other_domains__=domain_initial[domain]-base_total))
        cell_report[domain]={}
        domain_ids=[]
        for dataset,rows_ in sorted(sources.items()):
            k=quotas[dataset]
            ids=[] if k==0 else select_versions(rows_,{dataset:k},seed,apply_c4=False)[1]
            chosen.extend(ids);domain_ids.extend(ids)
            p=priorities[domain][dataset]
            cell_report[domain][dataset]=dict(original_source_weight=source_weights[dataset],
                task_groups=len({r['task_group'] for r in rows_.values()}), states=len(rows_),
                priority_available=p,ordinary_available=len(rows_)-p,b2_capacity=capacities[domain][dataset],
                initial_quota=initial[dataset],quota_after_local=base[dataset],
                cross_domain_addition=k-base[dataset],final_quota=k,
                priority_quota=round(.25*k),priority_selected=sum(kept[s]['short_turn'] or kept[s]['direct'] for s in ids),
                respond=sum(kept[s]['gold_names']==['RESPOND'] for s in ids),
                selected_task_groups=len({kept[s]['task_group'] for s in ids}))
        domain_report[domain]=dict(G_D=len(groups[domain]),K_D=domain_initial[domain],
            final_quota=domain_final[domain],b2_capacity=domain_capacity[domain],
            selected_task_groups=len({kept[s]['task_group'] for s in domain_ids}))
    if len(chosen)!=count or len(set(chosen))!=count:
        raise ValueError('balanced selection total/uniqueness failed')
    chosen.sort(key=lambda sid:(digest(seed,sid),sid))
    selected=[kept[sid] for sid in chosen]
    def composition(records):
        records=list(records);n=len(records)
        respond=sum(r['gold_names']==['RESPOND'] for r in records)
        priority=sum(r['short_turn'] or r['direct'] for r in records)
        return dict(population(records),count=n,respond=respond,respond_fraction=respond/n if n else None,
            priority=priority,priority_fraction=priority/n if n else None,
            short_turn=sum(r['short_turn'] for r in records),direct=sum(r['direct'] for r in records),
            n_prompt_tokens=sum(r['n_prompt_tokens'] for r in records),
            turn_calls_unknown=sum(r['turn_calls_unknown'] for r in records),
            respond_next_event=dict(Counter(r['next_event'] for r in records if r['gold_names']==['RESPOND'])))
    report=dict(domains=domain_report,cells=cell_report,source_quotas=dict(Counter(r['dataset'] for r in selected)),
        redistribution=dict(across_domains=allocation_flows(domain_initial,domain_final),within_domains=local_flows),
        selected=composition(selected),by_source={d:composition(r for r in selected if r['dataset']==d) for d in sorted(source_weights)},
        respond_gate_passed=.08<=sum(r['gold_names']==['RESPOND'] for r in selected)/count<=.20,
        definitions=dict(domain='sqrt(unique eligible task_group count) after deduplication; cross-source group counts once.',
            cells='Only observed source/domain cells participate; source weights are original B2 quotas. Capacity includes priority/ordinary constraints.',
            priority='At most Python round(.25*final_cell_quota) priority, remaining states only ordinary within the same cell.',
            redistribution='Within domain first by original source weights, then exhausted-domain deficits by original K_D; recheck every final cell.',
            rounding='Largest remainder, lexical label/source-name tie breaks.',
            flows='Net transfer bookkeeping matched lexically after final quotas are fixed.'))
    return chosen,report


def _select_balance_source(cells, quotas, seed):
    """One source's C3 budget, shared across its fixed domain quotas."""
    priority, ordinary, lower, upper = {}, {}, {}, {}
    order = lambda sid: (digest(seed, sid), sid)
    for domain, records in sorted(cells.items()):
        priority[domain] = sorted((sid for sid, r in records.items() if r['short_turn'] or r['direct']), key=order)
        ordinary[domain] = sorted((sid for sid, r in records.items() if not (r['short_turn'] or r['direct'])), key=order)
        k = quotas[domain]
        if k > len(records):
            raise ValueError(f'{domain}: state capacity below quota')
        lower[domain] = max(0, k-len(ordinary[domain]))
        upper[domain] = min(k, len(priority[domain]))
    budget = round(.25*sum(quotas.values()))
    requested = min(budget, sum(upper.values()))
    if sum(lower.values()) > requested:
        raise ValueError(f'per-source priority budget {budget} cannot meet mandatory priority {sum(lower.values())}; no relaxation')
    selected = {d: priority[d][:lower[d]] for d in cells}
    remaining = sorted(((sid, d) for d in cells for sid in priority[d][lower[d]:]), key=lambda pair: order(pair[0]))
    needed = requested-sum(lower.values())
    for sid, domain in remaining:
        if not needed:
            break
        if len(selected[domain]) < upper[domain]:
            selected[domain].append(sid)
            needed -= 1
    if needed:
        raise ValueError('priority allocation did not fill feasible source budget')
    counts = {d: len(ids) for d, ids in selected.items()}
    for domain in selected:
        selected[domain] += ordinary[domain][:quotas[domain]-counts[domain]]
    return selected, dict(priority_quota=budget, priority_selected=requested,
                         mandatory_priority=sum(lower.values()), priority_by_domain=counts)


def balance_selection(kept, labels, reference_ids, count, seed, *, domain_alpha, source_beta,
                      cap_domains=False):
    """Factorial allocation based on selected R00 states, with per-source C3 budgets.

    ``kept`` is the common C1-cleaned, trajectory-deduplicated pool. This is
    deliberately separate from the historical candidate-task-based experiment.
    """
    from .dedup import population
    if (type(domain_alpha) not in (int,float) or type(source_beta) not in (int,float)
            or domain_alpha not in (0, 1) or source_beta not in (.5, 1)):
        raise ValueError('unsupported balance exponents')
    if cap_domains and (domain_alpha, source_beta) != (0, .5):
        raise ValueError('domain caps are defined only for R11-cap')
    if (type(count) is not int or count <= 0 or not reference_ids
            or len(set(reference_ids)) != len(reference_ids)
            or (not cap_domains and len(reference_ids) != count)):
        raise ValueError('reference count/unique IDs differ')
    if not set(reference_ids) <= kept.keys():
        raise ValueError('reference ID absent from deduplicated pool')
    cells = defaultdict(lambda: defaultdict(dict))
    for sid, row in kept.items():
        for field in ('short_turn', 'direct', 'turn_calls_unknown'):
            if field not in row or type(row[field]) is not bool:
                raise ValueError(f'{sid}: missing/invalid {field}')
        if row['id'] != sid or row['split'] != 'train' or row['drops']:
            raise ValueError(f'{sid}: expected C1-eligible training state')
        if row['task_group'] not in labels or labels[row['task_group']] not in LABELS:
            raise ValueError(f'{sid}: missing/invalid task domain')
        cells[labels[row['task_group']]][row['dataset']][sid] = row
    reference = Counter((labels[kept[sid]['task_group']], kept[sid]['dataset']) for sid in reference_ids)
    domain_counts = Counter()
    for (domain, _), n in reference.items():
        domain_counts[domain] += n
    if set(domain_counts) != set(LABELS):
        raise ValueError('reference must cover all seven domains')
    domain_quotas = apportion({d: domain_counts[d]**domain_alpha for d in sorted(LABELS)}, count)
    domain_initial = domain_quotas.copy()
    domain_capacities = {d: sum(len(records) for s, records in cells[d].items()
                                if reference[d, s] > 0) for d in sorted(LABELS)}
    if cap_domains:
        domain_quotas = capped_equal_allocation(count, domain_capacities)
    quotas, cell_report, flows = {}, {}, {}
    for domain in sorted(LABELS):
        # Zero-reference cells stay excluded even when they have spare capacity.
        weights = {s: reference[domain, s]**source_beta for s in sorted(cells[domain]) if reference[domain, s] > 0}
        capacities = {s: len(cells[domain][s]) for s in weights}
        initial = apportion(weights, domain_quotas[domain])
        try:
            quotas[domain] = capped_allocation(weights, domain_quotas[domain], capacities)
        except ValueError as exc:
            raise ValueError(f'{domain}: insufficient supported capacity; no cross-domain backfill') from exc
        flows[domain] = allocation_flows(initial, quotas[domain])
        cell_report[domain] = {}
        for source, records in sorted(cells[domain].items()):
            p = sum(r['short_turn'] or r['direct'] for r in records.values())
            cell_report[domain][source] = dict(reference_count=reference[domain, source],
                eligible=source in weights, state_capacity=len(records), priority_available=p,
                ordinary_available=len(records)-p, initial_quota=initial.get(source, 0),
                final_quota=quotas[domain].get(source, 0), priority_selected=0)
    sources = sorted({s for domain in quotas.values() for s in domain})
    if cap_domains:
        # Check C3 only after both levels of quotas have been fixed. Do not use
        # C3-feasible capacities to silently change the approved allocation.
        planned_sources = {}
        for source in sources:
            supported = {d: cell_report[d][source] for d in quotas if source in quotas[d]}
            k = sum(c['final_quota'] for c in supported.values())
            lower = {d: max(0, c['final_quota']-c['ordinary_available']) for d, c in supported.items()}
            upper = sum(min(c['final_quota'], c['priority_available']) for c in supported.values())
            planned_sources[source] = dict(quota=k, priority_quota=round(.25*k),
                priority_available=sum(c['priority_available'] for c in supported.values()),
                ordinary_available=sum(c['ordinary_available'] for c in supported.values()),
                mandatory_priority=sum(lower.values()), mandatory_by_domain=lower,
                requested_priority=min(round(.25*k), upper),
                priority_shortfall=max(0, sum(lower.values())-min(round(.25*k), upper)),
                feasible=sum(lower.values()) <= min(round(.25*k), upper))
        cap_report = dict(reference_count=len(reference_ids), count=count,
            domains={d: dict(reference_count=domain_counts[d], initial_quota=domain_initial[d],
                             supported_capacity=domain_capacities[d], quota=domain_quotas[d])
                     for d in sorted(LABELS)},
            cells=cell_report, planned_sources=planned_sources,
            redistribution_between_domains=allocation_flows(domain_initial, domain_quotas),
            redistribution_within_domain=flows,
            c3_failures={s: r for s, r in planned_sources.items() if not r['feasible']})
        if cap_report['c3_failures']:
            # These are proposed quotas, not selected states or a usable sample.
            for cells_ in cap_report['cells'].values():
                for cell in cells_.values():
                    cell['priority_selected'] = None
            raise BalanceCapacityError(cap_report)
    selected, source_budgets = [], {}
    for source in sources:
        supported = {d: cells[d][source] for d in quotas if source in quotas[d]}
        source_quota = {d: quotas[d][source] for d in supported}
        try:
            picked, source_budgets[source] = _select_balance_source(supported, source_quota, seed)
        except ValueError as exc:
            raise ValueError(f'{source}: {exc}') from exc
        for domain, ids in picked.items():
            if len(ids) != source_quota[domain]:
                raise ValueError('cell quota not met')
            selected.extend(ids)
            cell_report[domain][source]['priority_selected'] = source_budgets[source]['priority_by_domain'][domain]
    selected.sort(key=lambda sid: (digest(seed, sid), sid))
    if len(selected) != count or len(set(selected)) != count:
        raise ValueError('balance selection count/uniqueness differs')
    def composition(records):
        records = list(records); n = len(records)
        respond = sum(r['gold_names'] == ['RESPOND'] for r in records)
        priority = sum(r['short_turn'] or r['direct'] for r in records)
        return dict(population(records), count=n, respond=respond, respond_fraction=respond/n if n else None,
            priority=priority, priority_fraction=priority/n if n else None,
            short_turn=sum(r['short_turn'] for r in records), direct=sum(r['direct'] for r in records),
            turn_calls_unknown=sum(r['turn_calls_unknown'] for r in records),
            n_prompt_tokens=sum(r['n_prompt_tokens'] for r in records),
            respond_next_event=dict(Counter(r['next_event'] for r in records if r['gold_names'] == ['RESPOND'])))
    records = [kept[sid] for sid in selected]
    by_source = {s: dict(composition(r for r in records if r['dataset'] == s), **source_budgets[s]) for s in sources}
    report = dict(domain_alpha=domain_alpha, source_beta=source_beta,
        domains={d: dict(reference_count=domain_counts[d], quota=domain_quotas[d],
                         selected=composition(r for r in records if labels[r['task_group']] == d)) for d in sorted(LABELS)},
        cells=cell_report, source_quotas=dict(sorted(Counter(r['dataset'] for r in records).items())),
        redistribution_within_domain=flows, selected=composition(records), by_source=by_source,
        overlap_with_R00=len(set(reference_ids) & set(selected)),
        respond_gate_passed=.08 <= sum(r['gold_names'] == ['RESPOND'] for r in records)/count <= .20)
    if cap_domains:
        report['cap_allocation'] = cap_report
    return selected, report
