"""Append-only training attempts and the effective trajectory after a resume."""
import json
import math
from pathlib import Path


def read_training_history(path):
    train, data, resumes = [], [], []
    attempt_data = 0
    total = None
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith('\n'):
                raise ValueError(f'incomplete training log line {number}')
            event = json.loads(line)
            kind = event['event']
            if kind == 'resume':
                step = event['checkpoint_step']
                attempt = len(resumes) + 1
                if (type(step) is not int or not 0 < step <= len(train)
                        or event['previous_last_step'] != len(train)
                        or event['attempt'] != attempt
                        or event['metric_prefix'] != f'resume_{attempt}/'
                        or not event['checkpoint'] or not event['previous_status']['status']):
                    raise ValueError(f'invalid resume boundary on line {number}')
                train = train[:step]
                resumes.append(event)
                attempt_data = 0
            elif kind == 'data':
                attempt_data += 1
                if attempt_data != 1:
                    raise ValueError('multiple data events without a resume boundary')
                data.append(event['train_count'])
            elif kind == 'train':
                if total is None:
                    total = event['total']
                if (attempt_data != 1 or event['step'] != len(train) + 1
                        or event['total'] != total or not 0 < event['step'] <= total):
                    raise ValueError(f'train step coverage differs on line {number}')
                if not all(math.isfinite(event[k]) for k in ('loss', 'total_loss', 'lr', 'grad_norm')):
                    raise ValueError(f'nonfinite train event on line {number}')
                train.append(event)
    return dict(train=train, data=data, resumes=resumes, last_attempt_data=attempt_data)


def resume_boundary(history, checkpoint, state, previous_status, created_at):
    step = state['step']
    if not history['train'] or not 0 < step <= len(history['train']):
        raise ValueError('checkpoint step is not covered by the saved training log')
    if len(state['loss_history']) != step:
        raise ValueError('checkpoint loss history length differs')
    if state['loss_history'] != [row['loss'] for row in history['train'][:step]]:
        raise ValueError('checkpoint loss history differs from the effective training log')
    attempt = len(history['resumes']) + 1
    return dict(event='resume', attempt=attempt, metric_prefix=f'resume_{attempt}/',
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_step=step,
                previous_last_step=len(history['train']), previous_status=previous_status,
                created_at=created_at)
