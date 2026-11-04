"""Small canonical sample/curation fixtures without raw-data or analysis imports."""

def sample(sid='s', gold=None):
    return dict(id=sid, dataset='d', model='teacher', task_group='g', outcome='success',
                candidate_level='S', context='[user] request', question='Next?',
                options=['run', 'other', 'RESPOND'], gold=[2] if gold is None else gold,
                n_prompt_tokens=12, messages_dropped=0, truncation_counts={})

def feature(row, **changes):
    kind = 'RESPOND' if row['gold'] == [2] else 'TOOL' if len(row['gold']) == 1 else 'TOOL_SET'
    result = dict(id=row['id'], dataset=row['dataset'], outcome=row['outcome'], split='train',
                  gold_kind=kind, next_event='trajectory_end' if kind == 'RESPOND' else None,
                  next_tool_names=None, stop_reason_abnormal=False if kind == 'RESPOND' else None,
                  narration_successor=False, turn_total_calls=4, turn_end_kind='RESPOND', strict_loop=False)
    result.update(changes)
    return result
