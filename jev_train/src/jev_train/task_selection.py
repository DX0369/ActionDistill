"""Frozen-domain soft balance, task rotation and reply-weighted state selection."""
from collections import Counter, defaultdict, deque
from itertools import islice
from fractions import Fraction
import heapq
import math
import random

from .common import digest
from .curation import kind
from .domains import LABELS, capped_allocation
from .selection import spread_order
from .reply_origin import reply_priority


REPLY_CLASSES = {'verified_reply', 'framework_forced', 'unknown'}


def validate_inputs(records, labels, reply_classes):
    replies = set()
    for sid, row in records.items():
        for key in ('id', 'dataset', 'model', 'task_group', 'trajectory_id'):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f'{sid}: invalid {key}')
        if row['id'] != sid or row['split'] != 'train' or row['drops']:
            raise ValueError(f'{sid}: requires an eligible canonical train state')
        if type(row['turn_index']) is not int or row['turn_index'] < 0:
            raise ValueError(f'{sid}: invalid turn_index')
        if labels[row['task_group']] not in LABELS:
            raise ValueError(f'{sid}: invalid task domain')
        names = row['gold_names']
        if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
                or len(names) != len(set(names))):
            raise ValueError(f'{sid}: invalid gold_names')
        if kind(names) == 'RESPOND':
            replies.add(sid)
    if set(reply_classes) != replies:
        raise ValueError('reply classification must cover exactly the normalized RESPOND IDs')
    if any(value not in REPLY_CLASSES for value in reply_classes.values()):
        raise ValueError('invalid reply classification')


def domain_allocation(records, labels, reference, count, *, domain_uniform_mix=0.5):
    if (set(reference) != set(LABELS) or any(type(n) is not int or n < 0 for n in reference.values())
            or sum(reference.values()) <= 0):
        raise ValueError('invalid reference domain counts')
    capacities = Counter(labels[row['task_group']] for row in records.values())
    # Integer weights preserve the frozen .5 mixture, including rounding ties.
    mix = Fraction(str(domain_uniform_mix))
    weights = {domain: (mix.denominator-mix.numerator)*len(LABELS)*reference[domain]
               + mix.numerator*sum(reference.values()) for domain in LABELS}
    positive = {domain: weight for domain, weight in weights.items() if weight > 0}
    quotas = dict.fromkeys(LABELS, 0)
    quotas.update(capped_allocation(positive, count, {domain: capacities[domain] for domain in positive}))
    return {domain: dict(reference_count=reference[domain], capacity=capacities[domain],
                        weight=weights[domain]/sum(weights.values()), quota=quotas[domain])
            for domain in sorted(LABELS)}


def trajectory_order(states, reply_classes, seed, *, priority_ids=None, reply_priority_weight=2):
    """Integer-weighted R/O sampling without replacement; frozen default is two."""
    first = states[0]
    identity = (first['dataset'], first['model'], first['trajectory_id'])
    if priority_ids is None:
        priority_ids = {r['id'] for r in states if reply_classes.get(r['id']) == 'verified_reply'}
    preferred = [r for r in states if r['id'] in priority_ids]
    ordinary = [r for r in states if r['id'] not in priority_ids]
    streams = [deque(spread_order(group, digest(seed, 'task_reply', category),
                                  first['dataset'], first['trajectory_id']))
               for category, group in (('R', preferred), ('O', ordinary))]
    rng = random.Random(int(digest(seed, 'task_reply', 'draw', *identity), 16))
    r, o = streams
    while r or o:
        selected = r if rng.randrange(reply_priority_weight*len(r)+len(o)) < reply_priority_weight*len(r) else o
        yield selected.popleft()


def _round_robin(children):
    queue = deque(children)
    while queue:
        child = queue.popleft()
        try:
            yield next(child)
        except StopIteration:
            continue
        queue.append(child)


def domain_order(records, reply_classes, seed):
    """One state per task per round, then source -> teacher -> retained trajectory."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in records:
        groups[row['task_group']][row['dataset']][row['model']][row['trajectory_id']].append(row)

    def ordered(keys, *scope):
        return sorted(keys, key=lambda key: (digest(seed, 'task_reply', *scope, key), key))

    tasks = []
    for task in ordered(groups, 'task'):
        sources = []
        for source in ordered(groups[task], 'source', task):
            teachers = []
            for teacher in ordered(groups[task][source], 'teacher', task, source):
                trajectories = groups[task][source][teacher]
                teachers.append(_round_robin([
                    trajectory_order(trajectories[tid], reply_classes, seed)
                    for tid in ordered(trajectories, 'trajectory', task, source, teacher)]))
            sources.append(_round_robin(teachers))
        tasks.append(_round_robin(sources))
    yield from _round_robin(tasks)


def select(records, labels, reference, reply_classes, count, seed):
    if type(count) is not int or count <= 0:
        raise ValueError('count must be a positive integer')
    validate_inputs(records, labels, reply_classes)
    allocation = domain_allocation(records, labels, reference, count)
    domains = defaultdict(list)
    for row in records.values():
        domains[labels[row['task_group']]].append(row)
    ids = []
    for domain, plan in allocation.items():
        chosen = list(islice(domain_order(domains[domain], reply_classes, seed), plan['quota']))
        if len(chosen) != plan['quota']:
            raise ValueError(f'{domain}: task stream exhausted before quota')
        ids.extend(row['id'] for row in chosen)
    if len(ids) != count or len(ids) != len(set(ids)):
        raise ValueError('selection count or uniqueness differs')
    return ids, allocation


def source_allocation(records, labels, reference_cells, count, *, domain_uniform_mix=0.5,
                      source_weight_power=0.5):
    """V2: soft domain mixture, then sqrt(frozen R00 cell count) within domain."""
    if (set(reference_cells) != set(LABELS)
            or any(type(n) is not int or n < 0 for cells in reference_cells.values() for n in cells.values())):
        raise ValueError('invalid frozen source/domain reference counts')
    supported = {sid: r for sid, r in records.items()
                 if reference_cells[labels[r['task_group']]].get(r['dataset'], 0) > 0}
    reference = {d: sum(cells.values()) for d, cells in reference_cells.items()}
    allocation = domain_allocation(supported, labels, reference, count,
                                   domain_uniform_mix=domain_uniform_mix)
    capacities = Counter((labels[r['task_group']], r['dataset']) for r in records.values())
    for domain, plan in allocation.items():
        cells = reference_cells[domain]
        sources = sorted(set(cells) | {s for d, s in capacities if d == domain})
        weights = {s: math.sqrt(cells[s]) if source_weight_power == 0.5 else cells[s]**source_weight_power
                   for s in sources if cells.get(s, 0) > 0}
        quotas = (capped_allocation(weights, plan['quota'], {s: capacities[domain, s] for s in weights})
                  if weights else {})
        if not weights and plan['quota']:
            raise ValueError(f'{domain}: quota without a positive-reference source')
        plan['sources'] = {s: dict(reference_count=cells.get(s, 0), capacity=capacities[domain, s],
                                   weight=weights.get(s, 0), quota=quotas.get(s, 0)) for s in sources}
    return allocation


def source_quota_order(records, quotas, priority_ids, seed, domain, *, reply_priority_weight=2):
    """Rotate sources to their fixed quotas; choose their least-used shared task.

    Counts belong to canonical task_group, not (source, task). Teacher/trajectory
    pairs are the retained units, so different teachers are never merged.
    """
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in records:
        if quotas.get(row['dataset'], 0):
            groups[row['dataset']][row['task_group']][row['model'], row['trajectory_id']].append(row)
    streams, heaps = {}, {}
    counts = Counter()
    for source, tasks in groups.items():
        streams[source], heaps[source] = {}, []
        for task, units in tasks.items():
            if len({model for model, _ in units}) != len(units):
                raise ValueError(f'{source}/{task}: requires one retained trajectory per teacher')
            order = sorted(units, key=lambda pair: (digest(seed, 'task_reply_v2', 'unit', task, source, *pair), pair))
            streams[source][task] = _round_robin([
                trajectory_order(units[pair], {}, seed, priority_ids=priority_ids,
                                 reply_priority_weight=reply_priority_weight) for pair in order])
            heapq.heappush(heaps[source], (0, digest(seed, 'task_reply_v2', 'task', task), task))
    remaining = {s: n for s, n in quotas.items() if n}
    sources = deque(sorted(remaining, key=lambda s: (digest(seed, 'task_reply_v2', 'source', domain, s), s)))
    while sources:
        source = sources.popleft()
        heap = heaps[source]
        while heap:
            used, tie, task = heapq.heappop(heap)
            if used != counts[task]:
                heapq.heappush(heap, (counts[task], tie, task))
                continue
            try:
                row = next(streams[source][task])
            except StopIteration:
                continue
            counts[task] += 1
            heapq.heappush(heap, (counts[task], tie, task))
            yield row
            remaining[source] -= 1
            if remaining[source]:
                sources.append(source)
            break
        else:
            raise ValueError(f'{domain}/{source}: task stream exhausted before source quota')


def select_with_sources(records, labels, reference_cells, origins, count, seed, *, framework_boundary_ids=frozenset(),
                        domain_uniform_mix=0.5, source_weight_power=0.5, reply_priority_weight=2):
    """V2 selection; the historical v1 select() remains unchanged."""
    if type(count) is not int or count <= 0:
        raise ValueError('count must be a positive integer')
    validate_inputs(records, labels, {sid: r['category'] for sid, r in origins.items()})
    if not framework_boundary_ids <= origins.keys():
        raise ValueError('framework-boundary exclusion must name normalized RESPOND IDs')
    priority_ids = {sid for sid, origin in origins.items()
                    if reply_priority(origin, framework_boundary=sid in framework_boundary_ids)['eligible']}
    allocation = source_allocation(records, labels, reference_cells, count,
                                   domain_uniform_mix=domain_uniform_mix,
                                   source_weight_power=source_weight_power)
    domains = defaultdict(list)
    for row in records.values():
        domains[labels[row['task_group']]].append(row)
    ids = []
    for domain, plan in allocation.items():
        quotas = {s: cell['quota'] for s, cell in plan['sources'].items()}
        chosen = list(source_quota_order(domains[domain], quotas, priority_ids, seed, domain,
                                        reply_priority_weight=reply_priority_weight))
        actual = Counter(r['dataset'] for r in chosen)
        if any(actual[s] != n for s, n in quotas.items()) or len(chosen) != plan['quota']:
            raise ValueError(f'{domain}: selected source/domain counts differ from quotas')
        ids.extend(row['id'] for row in chosen)
    if len(ids) != count or len(ids) != len(set(ids)):
        raise ValueError('selection count or uniqueness differs')
    return ids, allocation
