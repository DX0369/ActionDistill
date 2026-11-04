"""Explicit C1 labels and C2/C3/C4 selection; never changes rendered inputs."""
from collections import defaultdict

from .common import digest
from .data import tool_name, validate


def kind(names):
    if not names or ('RESPOND' in names and names != ['RESPOND']):
        raise ValueError('invalid gold names')
    return 'RESPOND' if names == ['RESPOND'] else 'TOOL' if len(names) == 1 else 'TOOL_SET'


def c1_plan(row, feature):
    validate(row)
    for key in ('id', 'dataset', 'outcome'):
        if row[key] != feature[key]:
            raise ValueError(f'sample/feature disagreement: {row["id"]}/{key}')
    options = [tool_name(value) for value in row['options']]
    if len(set(options)) != len(options) or options[-1] != 'RESPOND':
        raise ValueError('candidate names must be unique and end with RESPOND')
    original = [options[i] for i in row['gold']]
    if kind(original) != feature['gold_kind']:
        raise ValueError('source gold kind differs from feature')
    if feature['next_event'] not in (None, 'user_turn_end', 'trajectory_end', 'intermediate_tool_continuation', 'unjudgeable'):
        raise ValueError('invalid next_event')
    if original == ['RESPOND']:
        if feature['next_event'] is None or type(feature['stop_reason_abnormal']) is not bool:
            raise ValueError('RESPOND requires next_event and D2 flag')
    elif feature['next_event'] is not None or feature['stop_reason_abnormal'] is not None:
        raise ValueError('RESPOND-only fields on a tool state')
    if type(feature['narration_successor']) is not bool:
        raise ValueError('narration_successor must be a bool')
    calls = feature['turn_total_calls']
    if calls is not None and (type(calls) is not int or calls < 0):
        raise ValueError('turn_total_calls must be a nonnegative integer or null')
    if feature['turn_end_kind'] not in (None, 'RESPOND', 'incomplete'):
        raise ValueError('invalid turn_end_kind')
    if feature['strict_loop'] is not None and type(feature['strict_loop']) is not bool:
        raise ValueError('strict_loop must be bool or null')
    d1 = feature['next_event'] == 'intermediate_tool_continuation'
    gold = original
    missing = []
    drops = []
    if d1:
        gold = feature['next_tool_names']
        if not isinstance(gold, list) or not gold or any(not isinstance(n, str) or not n or n == 'RESPOND' for n in gold) or len(set(gold)) != len(gold):
            raise ValueError('D1 requires distinct next tool names in source order')
        missing = [name for name in gold if name not in options]
        if missing:
            drops.append('C1_D1_missing_candidate')
    elif feature['next_tool_names'] is not None:
        raise ValueError('next_tool_names only applies to D1')
    if feature['narration_successor']:
        drops.append('C1_narration_successor')
    if feature['stop_reason_abnormal']:
        drops.append('C1_D2')
    if feature['next_event'] == 'unjudgeable':
        drops.append('C1_unjudgeable')
    return dict(id=row['id'], dataset=row['dataset'], outcome=row['outcome'], split=feature['split'],
                original_gold=original, gold_names=list(gold), gold=[options.index(n) for n in gold] if not missing else None,
                d1=d1, drops=drops, missing_candidates=missing, strict_loop=feature['strict_loop'],
                short_turn=calls in (1,2,3) and feature['turn_end_kind'] == 'RESPOND',
                direct=calls == 0 and gold == ['RESPOND'], turn_calls_unknown=calls is None,
                next_event=feature['next_event'])


def c4_drop(plan, loop_tool):
    if plan['strict_loop'] is not True:
        return False
    if not isinstance(loop_tool, str) or not loop_tool or loop_tool == 'RESPOND':
        raise ValueError(f'missing source loop tool: {plan["id"]}')
    return not plan['drops'] and plan['gold_names'] == [loop_tool]


def materialize(row, plan, restored_features=None):
    if plan['drops']:
        raise ValueError('cannot materialize a C1-discarded state')
    result = dict(row, gold=plan['gold'], short_turn=plan['short_turn'], direct=plan['direct'])
    if 'action_kind' in row:
        result['action_kind'] = kind(plan['gold_names'])
    if plan['d1']:
        if 'transition' in row:
            previous = set(row['prev_action_names'])
            result['transition'] = 'first' if not previous else 'repeat' if set(plan['gold_names']) <= previous else 'change'
        if 'loop_state' in row:
            if restored_features is None:
                raise ValueError('relabelled loop_state needs full source history')
            if restored_features['prev_action_names'] != row['prev_action_names'] or restored_features['transition'] != result['transition']:
                raise ValueError('restored transition/previous action mismatch')
            result['loop_state'] = restored_features['loop_state']
    validate(result)
    return result


def select_versions(plans, quotas, seed, *, apply_c4):
    by_dataset = defaultdict(list)
    for plan in plans.values():
        if plan['split'] == 'train' and not plan['drops']:
            by_dataset[plan['dataset']].append(plan)
    if set(by_dataset) != set(quotas):
        raise ValueError('C1 surviving dataset set differs from B0 quotas')
    b1, b2, preview = [], [], {}
    for dataset, k in sorted(quotas.items()):
        if type(k) is not int or k <= 0:
            raise ValueError('quota must be a positive integer')
        ordered = sorted(by_dataset[dataset], key=lambda r:(digest(seed,r['id']),r['id']))
        if len(ordered) < k:
            raise ValueError(f'{dataset}: B1 has {len(ordered)} candidates for quota {k}')
        chosen_b1 = ordered[:k]
        candidates = [r for r in ordered if not (apply_c4 and r['c4_drop'])]
        priority = [r for r in candidates if r['short_turn'] or r['direct']]
        ordinary = [r for r in candidates if not (r['short_turn'] or r['direct'])]
        requested = round(0.25*k)  # Python round: ties to even, exactly as specified.
        chosen_priority = priority[:requested]
        needed = k-len(chosen_priority)
        if len(ordinary) < needed:
            raise ValueError(f'{dataset}: B2 ordinary pool has {len(ordinary)} for {needed}; no quota relaxation')
        chosen_b2 = chosen_priority+ordinary[:needed]
        b1.extend(r['id'] for r in chosen_b1)
        b2.extend(r['id'] for r in chosen_b2)
        preview[dataset] = dict(k=k, priority_quota=requested, priority_available=len(priority),
                                short_turn_available=sum(r['short_turn'] for r in priority),
                                direct_available=sum(r['direct'] for r in priority),
                                priority_selected=len(chosen_priority), ordinary_selected=needed,
                                priority_fraction=len(chosen_priority)/k,
                                b1_respond=sum(r['gold_names']==['RESPOND'] for r in chosen_b1),
                                b2_respond=sum(r['gold_names']==['RESPOND'] for r in chosen_b2),
                                b2_short_turn=sum(r['short_turn'] for r in chosen_b2),
                                b2_direct=sum(r['direct'] for r in chosen_b2),
                                b1_unknown_turns=sum(r['turn_calls_unknown'] for r in chosen_b1),
                                b2_unknown_turns=sum(r['turn_calls_unknown'] for r in chosen_b2))
    for selected in (b1,b2):
        if len(selected) != sum(quotas.values()) or len(set(selected)) != len(selected):
            raise ValueError('selected count/ID uniqueness failed')
    return (sorted(b1,key=lambda sid:(digest(seed,sid),sid)),
            sorted(b2,key=lambda sid:(digest(seed,sid),sid)), preview)
