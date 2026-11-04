#!/usr/bin/env python
"""Read-only training composition and deterministic loss-coefficient audit."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(WORKSPACE/'third_party/decider'),
               str(PROJECT.parent/'dataset_tool_inventory/src'), str(PROJECT/'scripts')]
from decider.train import batches_by_tokens
from jev_train.common import now, rows, write_json
from jev_train.data import SampleIndex, load_selected, tool_name
from jev_train.selection import load_train_ids
from jev_train.training_history import read_training_history
from train_jev import total_steps
from analyze_data_curation import (TurnView, source_units, classify_capture, BASE_FIELDS,
                                  termination_evidence, policy_flags)
from analyze_short_turn import classify as classify_respond


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def loss_exposure(entries, configuration, training_events):
    """Replay the effective schedule, not optimizer states or sample gradients.

    One global update averages WORLD_SIZE * accum micro-batch means. Each
    occurrence therefore has coefficient 1 / (WORLD_SIZE * accum * batch_size).
    Epoch rounding may consume the beginning of the next epoch, as in training.
    """
    require(entries and len({r['id'] for r in entries}) == len(entries),
            'training selection must contain distinct states')
    for r in entries:
        require(type(r['n_prompt_tokens']) is int and r['n_prompt_tokens'] > 0,
                'invalid n_prompt_tokens')
    for key in ('world_size', 'accum', 'max_tokens', 'seed'):
        require(type(configuration[key]) is int, f'invalid {key}')
    width = configuration['world_size'] * configuration['accum']
    require(width > 0 and configuration['max_tokens'] > 0, 'invalid batch budget')
    items = [{'ids': range(r['n_prompt_tokens'])} for r in entries]
    epoch = 0
    batches = batches_by_tokens(items, configuration['max_tokens'],
                                random.Random(configuration['seed']))
    expected = total_steps(len(batches), configuration['world_size'], configuration['accum'],
                           configuration['epochs'], configuration['max_steps'])
    require(len(training_events) == expected, 'completed step coverage differs from configuration')
    exposure = {r['id']: dict(occurrences=0, loss_coefficient=0.0) for r in entries}
    cursor = 0
    exposed_tokens = 0
    for step, event in enumerate(training_events, 1):
        require(event['step'] == step and event['total'] == expected, 'training event sequence differs')
        tokens = 0
        for _ in range(width):
            if cursor == len(batches):
                epoch += 1
                batches = batches_by_tokens(items, configuration['max_tokens'],
                                            random.Random(configuration['seed'] + epoch))
                cursor = 0
            batch = batches[cursor]
            cursor += 1
            coefficient = 1.0 / (width * len(batch))
            for i in batch:
                row = entries[i]
                exposure[row['id']]['occurrences'] += 1
                exposure[row['id']]['loss_coefficient'] += coefficient
                tokens += row['n_prompt_tokens']
        require(tokens == event['tokens'], f'training token schedule differs at step {step}')
        exposed_tokens += tokens
    coefficient_sum = sum(r['loss_coefficient'] for r in exposure.values())
    require(math.isclose(coefficient_sum, expected, rel_tol=1e-10), 'loss coefficients do not sum to updates')
    return exposure, dict(updates=expected, effective_prompt_tokens=exposed_tokens,
                          occurrences=sum(r['occurrences'] for r in exposure.values()),
                          loss_coefficient_sum=coefficient_sum,
                          all_step_token_counts_matched=True, final_epoch=epoch, final_cursor=cursor)


def read_run(directory):
    directory = Path(directory).resolve()
    state = read_json(directory/'metadata/run_status.json')
    require(state['status'] == 'completed', f'incomplete training: {directory}')
    c = state['configuration']
    samples = Path(c['samples'])
    cache = samples.parent/'index_cache'/samples.name/'train.json'
    require(cache.is_file(), f'read-only audit needs existing sample index: {cache}')
    index = SampleIndex(samples/'train.jsonl')
    selected = load_train_ids(index, c['train_ids'], c['seed']) if c['train_ids'] else load_selected(index, c['train_limit'], c['seed'])
    history = read_training_history(directory/'train.log')
    require(state['step'] == len(history['train']), 'final state and effective log disagree')
    require(all(n == len(selected) for n in history['data']), 'training selection count differs')
    exposure, validation = loss_exposure(selected, c, history['train'])
    return dict(directory=str(directory), configuration=c, index=index, selected=selected,
                exposure=exposure, validation=validation,
                resumed_attempts=len(history['resumes']))


def selected_rows(path, wanted, key='id'):
    found = {}
    for row in rows(path):
        identity = row[key]
        if identity in wanted:
            require(identity not in found, f'duplicate {key}: {identity} in {path}')
            found[identity] = row
    require(found.keys() == wanted, f'{path}: missing {len(wanted-found.keys())} requested {key}s')
    return found


def read_inputs(config, runs):
    ids = {r['id'] for run in runs.values() for r in run['selected']}
    canonical = selected_rows(PROJECT/config['canonical_states'], ids)
    features = selected_rows(PROJECT/config['state_features'], ids)
    domains = selected_rows(PROJECT/config['domains'], {r['task_group'] for r in canonical.values()}, 'task_group')
    targets = {}
    for label, run in runs.items():
        with run['index'].path.open('rb') as handle:
            for entry in sorted(run['selected'], key=lambda r:r['offset']):
                handle.seek(entry['offset'])
                row = json.loads(handle.readline())
                state = canonical[entry['id']]
                require(row['id'] == entry['id'], 'sample offset mismatch')
                for key in ('dataset', 'model', 'task_group', 'n_prompt_tokens', 'outcome', 'candidate_level'):
                    require(row[key] == state[key], f'{label}/{row["id"]}: {key} mismatch')
                names = [tool_name(option) for option in row['options']]
                require([names[i] for i in row['gold']] == state['gold_names'], 'normalized gold mismatch')
                require(not state['drops'] and state['split'] == 'train', 'selected state is not C1 eligible train')
                for key in ('short_turn', 'direct'):
                    require(row[key] == state[key], f'training priority flag mismatch: {key}')
                if row['id'] in targets:
                    require(targets[row['id']]['names'] == names, 'candidate order differs across selections')
                else:
                    targets[row['id']] = dict(state, names=names, populations=[])
        print(f'Validated actual samples: {label}/{len(run["selected"])}', flush=True)
    for sid, state in canonical.items():
        f = features[sid]
        require(all(f[k] == state[k] for k in ('dataset', 'outcome', 'split', 'next_event')), 'feature identity mismatch')
        require(domains[state['task_group']]['split'] == 'train', 'domain split differs')
    return canonical, features, domains, targets


def content_error(value):
    """Only explicit structured error values, never free-text keywords."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return False, None
    if isinstance(value, list):
        findings = [content_error(part['text']) for part in value
                    if isinstance(part, dict) and part.get('type') == 'text' and 'text' in part]
        return any(flag for flag, _ in findings), next((issue for _, issue in findings if issue), None)
    if not isinstance(value, dict) or 'error' not in value:
        return False, None
    error = value['error']
    if error is None or error is False:
        return False, None
    if error is True or isinstance(error, (str, dict, list)):
        return bool(error), None
    return False, 'uninterpreted_structured_error_value'


def explicit_error(record, content):
    flags = []
    for field in ('is_error', 'isError'):
        if field in record:
            if type(record[field]) is not bool:
                return False, 'invalid_explicit_error_flag'
            flags.append(record[field])
    flag, issue = content_error(content)
    root_flag, root_issue = content_error(record)
    return any(flags) or flag or root_flag, issue or root_issue


def raw_response(view, index):
    """Recover fields stripped by native adapters; ambiguous matches stay unknown."""
    message = view.raw[index]
    if 'tool_call_id' not in message:
        return False, 'response_without_call_id'
    cid = message['tool_call_id']
    family = view.unit['family']
    if family == 'native_pi':
        matches = [item['message'] for item in view.originals(index)
                   if item['type'] == 'message' and item['message']['role'] == 'toolResult'
                   and item['message']['toolCallId'] == cid]
        if len(matches) != 1:
            return False, 'raw_pi_response_not_unique'
        return explicit_error(matches[0], matches[0]['content'])
    if family == 'native_codex':
        matches = [item['payload'] for item in view.originals(index)
                   if item['type'] == 'response_item' and item['payload']['type'] in
                   {'function_call_output', 'custom_tool_call_output'} and item['payload']['call_id'] == cid]
        if len(matches) != 1:
            return False, 'raw_codex_response_not_unique'
        return explicit_error(matches[0], matches[0]['output'])
    if family in {'native_claude', 'anthropic'}:
        raw = ([item['message'] for item in view.originals(index) if 'message' in item]
               if family == 'native_claude' else view.unit['record']['trajectory'])
        if isinstance(raw, str):
            raw = json.loads(raw)
        matches = [part for item in raw if isinstance(item['content'], list) for part in item['content']
                   if part['type'] == 'tool_result' and part['tool_use_id'] == cid]
        if not matches or any(part != matches[0] for part in matches[1:]):
            return False, 'raw_anthropic_response_not_unique'
        return explicit_error(matches[0], matches[0]['content'])
    if family in {'openai', 'toolathlon'}:
        # These adapters retain the original message dictionary and its error fields.
        return explicit_error(message, message['content'])
    record = view.unit['record']
    if family == 'finance':
        candidates = [json.loads(line) for line in record['trace_jsonl'].splitlines()]
        matches = [item for item in candidates if item['type'] == 'tool' and item['tool_call_id'] == cid]
        if len(matches) != 1:
            return False, 'raw_finance_response_not_unique'
        return explicit_error(matches[0], matches[0]['result'])
    if family == 'atif':
        matches = [result for step in record['steps'] if step.get('observation') is not None
                   for result in step['observation']['results'] if result['source_call_id'] == cid]
        if len(matches) != 1:
            return False, 'raw_atif_response_not_unique'
        return explicit_error(matches[0], matches[0]['content'])
    return False, 'raw_response_mapping_unsupported'


def previous_error(view, index, prior):
    if prior['call_count_status'] != 'known':
        return dict(status='unknown', errors=0, issues=[prior['call_count_status']], evidence=[])
    if prior['previous_turn_tool_calls'] == 0:
        return dict(status='no_prior_call', errors=0, issues=[], evidence=[])
    boundaries = [i for i in range(index) if view.locator(i) == prior['user_boundary_source']]
    require(len(boundaries) == 1, 'known prior history has ambiguous user boundary')
    boundary = boundaries[0]
    ancestors = None
    if view.tree:
        ancestors, issue = view.tree.ancestry_to_user(index, boundary)
        require(issue is None, 'known prior history has invalid ancestry')
    abandoned = view.raw[index].get('_abandoned', frozenset())
    pending = Counter()
    issues, evidence = [], []
    calls = 0
    for i in range(boundary+1, index):
        raw, message = view.raw[i], view.messages[i]
        if raw.get('_uuids') and raw['_uuids'] <= abandoned:
            continue
        if ancestors is not None and not view.tree.message_nodes[i] & ancestors:
            continue
        for call in message.get('tool_calls', []):
            calls += 1
            if 'id' not in call:
                issues.append('call_without_id')
            else:
                pending[call['id']] += 1
                if pending[call['id']] > 1:
                    issues.append('duplicate_pending_call_id')
        if message['role'] != 'tool':
            continue
        if 'tool_call_id' not in raw or pending[raw['tool_call_id']] != 1:
            issues.append('unmatched_tool_response')
            continue
        pending[raw['tool_call_id']] -= 1
        error, issue = raw_response(view, i)
        if issue:
            issues.append(issue)
        if error:
            evidence.append(view.locator(i))
    require(calls == prior['previous_turn_tool_calls'], 'recovered call count differs from existing classifier')
    if any(pending.values()):
        issues.append('missing_tool_response')
    status = 'explicit_error' if evidence else 'unknown' if issues else 'no_explicit_error'
    return dict(status=status, errors=len(evidence), issues=sorted(set(issues)), evidence=evidence)


def restore_features(targets, features, config):
    restored = {}
    for unit in source_units(targets, config):
        view = TurnView(unit)
        descriptors = [dict((key, target[key]) for key in BASE_FIELDS) | {'populations': []}
                       for target in unit['targets'] if target['gold_names'] == ['RESPOND']]
        terminal = termination_evidence(dict(unit, targets=descriptors)) if descriptors else {}
        responds = {}
        for classified in classify_capture(unit['capture'], descriptors, unit['family'],
                                           unit['raw_records'], unit['prefix'], target_fields=BASE_FIELDS):
            evidence = terminal[classified['turn_index']]
            classified.update(termination_evidence=evidence, **policy_flags(classified, evidence))
            responds[classified['turn_index']] = classified
        for target in unit['targets']:
            sid, index = target['id'], target['turn_index']
            require(sid not in restored, f'duplicate reconstructed state: {sid}')
            prior = view.prior(index)
            require(prior['previous_turn_tool_calls'] == features[sid]['turn_calls_before'], 'prior calls differ from saved features')
            gold = target['gold_names']
            original = target['original_gold']
            message = view.messages[index]
            model = view.raw[index]['_model'] if '_model' in view.raw[index] else view.capture['context']['model']
            require(model == target['model'], 'raw teacher identity differs')
            original_names = list(dict.fromkeys(c['function']['name'] for c in message.get('tool_calls', []))) or ['RESPOND']
            require(original_names == original, 'raw teacher decision differs from saved original gold')
            label = 'RESPOND' if gold == ['RESPOND'] else 'TOOL' if len(gold) == 1 else 'TOOL_SET'
            row = dict(prior_calls=prior['previous_turn_tool_calls'], call_count_status=prior['call_count_status'],
                       error=previous_error(view, index, prior), action=label, semantic=label,
                       semantic_unknown_reason=None, termination_status=None, termination_value=None)
            if label == 'RESPOND':
                classified = responds[index]
                require(classified['category'] == features[sid]['next_event'], 'successor classification differs')
                require(classified['d2'] == features[sid]['stop_reason_abnormal'], 'termination D2 flag differs')
                semantic, reason = classify_respond(classified)
                row.update(semantic=semantic, semantic_unknown_reason=reason,
                           next_event=classified['category'], termination_status=classified['termination_status'],
                           termination_value=classified['termination_value'])
            else:
                row['next_event'] = None
            restored[sid] = row
    require(restored.keys() == targets.keys(), 'source reconstruction missed selected states')
    return restored


def call_bin(count):
    if count is None:
        return 'unknown'
    require(type(count) is int and count >= 0, 'invalid prior call count')
    return '0' if count == 0 else '1-3' if count <= 3 else '4+'


def describe(selected, arm, updates):
    if not selected:
        return dict(count=0, tasks=0, trajectories=0, prompt_tokens=0, mean_prompt_tokens=None,
                    loss_coefficient=0.0, fraction_of_run_loss=0.0, occurrences=0)
    coefficient = sum(row['arms'][arm]['loss_coefficient'] for row in selected)
    tokens = sum(row['n_prompt_tokens'] for row in selected)
    return dict(count=len(selected), tasks=len({row['task_group'] for row in selected}),
                trajectories=len({(row['dataset'], row['trajectory_id']) for row in selected}),
                prompt_tokens=tokens, mean_prompt_tokens=tokens/len(selected),
                loss_coefficient=coefficient, fraction_of_run_loss=coefficient/updates,
                occurrences=sum(row['arms'][arm]['occurrences'] for row in selected))


def aggregate(selected, arm, updates):
    result = describe(selected, arm, updates)
    result['by_semantic'] = {}
    for field in ('semantic', 'action', 'next_event', 'termination_status', 'error_status', 'prior_calls_bin'):
        groups = defaultdict(list)
        for row in selected:
            groups[row[field]].append(row)
        result['by_'+field] = {str(key): describe(value, arm, updates) for key, value in sorted(groups.items(), key=lambda kv:str(kv[0]))}
    fields = ('dataset', 'model', 'domain', 'semantic', 'prior_calls_bin', 'error_status')
    cells = defaultdict(list)
    for row in selected:
        cells[tuple(row[k] for k in fields)].append(row)
    result['cross_table'] = [dict(zip(fields, cell), **describe(value, arm, updates))
                             for cell, value in sorted(cells.items())]
    result['unknown_reasons'] = dict(Counter(reason for row in selected for reason in row['error']['issues']))
    result['semantic_unknown_reasons'] = dict(Counter(row['semantic_unknown_reason'] for row in selected
                                                     if row['semantic_unknown_reason'] is not None))
    result['d1_relabelled_count'] = sum(row['d1'] for row in selected)
    result['training_priority_count'] = sum(row['training_short_turn'] or row['training_direct'] for row in selected)
    result['training_short_turn_count'] = sum(row['training_short_turn'] for row in selected)
    return result


def make_summary(states, runs, pairs, smoke):
    groups, comparisons = {}, {}
    for label, run in runs.items():
        selected = [row for row in states if label in row['arms']]
        groups[label] = dict(run=run['directory'], training_validation=run['validation'],
                             configuration=run['configuration'], resumed_attempts=run['resumed_attempts'],
                             **aggregate(selected, label, run['validation']['updates']))
    by_id = {row['id']: row for row in states}
    for current, reference in pairs:
        current_ids = {row['id'] for row in states if current in row['arms']}
        reference_ids = {row['id'] for row in states if reference in row['arms']}
        sets = dict(retained=current_ids & reference_ids, added=current_ids-reference_ids,
                    removed=reference_ids-current_ids)
        comparison = {}
        for status, ids in sets.items():
            selected = [by_id[sid] for sid in sorted(ids)]
            arms = (current, reference) if status == 'retained' else (current,) if status == 'added' else (reference,)
            comparison[status] = {arm: aggregate(selected, arm, runs[arm]['validation']['updates']) for arm in arms}
        comparisons[current+':'+reference] = comparison
    return dict(smoke=smoke, states=len(states), groups=groups, membership_comparisons=comparisons,
                definitions=dict(semantic='Reuse analyze_short_turn.classify on normalized RESPOND only; missing/null termination retained, conflict unknown.',
                    prior_calls='Current user turn, reused TurnView.prior; bins 0, 1-3, 4+, unknown.',
                    error='Raw is_error/isError true or nonempty structured error; no text heuristics. Unknown mappings/missing responses explicit.',
                    error_priority='An observed explicit error takes precedence; incomplete-return reasons are retained separately.',
                    loss_coefficient='Sum 1/(world_size*accum*microbatch_size) over effective occurrences; normalized by global updates.',
                    interpretation='Loss coefficients are pre-optimizer linear factors, not gradient magnitude, causal influence, or Adam/learning-rate weighting.',
                    failure='All selected states retained; undecidable features grouped as unknown; asset/identity/regression mismatches fail.'))


def report(summary):
    lines = ['# 训练数据语义与损失系数审计', '',
             '本报告只读既有选样与原始轨迹，不修改标签、配比、模型或历史结果。',
             '严格短停复用既有离线分类；训练short_turn标记包含工具决策，两者分别报告。',
             '损失系数由原batch函数重放、逐更新token数核验；它不等于真实梯度贡献或因果效应。', '',
             '| 组 | 状态 | 独立任务 | RESPOND | 严格短停 | 短停条数占比 | 短停损失系数占比 | 更新数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, group in summary['groups'].items():
        respond = group['by_action'].get('RESPOND', {'count': 0})['count']
        stop = group['by_semantic'].get('short_turn', {'count': 0, 'fraction_of_run_loss': 0.0})
        lines.append(f'| {label} | {group["count"]} | {group["tasks"]} | {respond} | {stop["count"]} | '
                     f'{stop["count"]/group["count"]:.2%} | {stop["fraction_of_run_loss"]:.2%} | {group["training_validation"]["updates"]} |')
    lines += ['', '## 同一用户轮此前的工具返回', '',
              '未见显式错误不等于调用成功；文字错误、退出码和异常叙述不作推断。显式错误优先，但缺失返回原因仍单列。', '',
              '| 组 | 此前零调用 | 显式错误 | 未见显式错误 | 不可判 |', '|---|---:|---:|---:|---:|']
    for label, group in summary['groups'].items():
        counts = [group['by_error_status'].get(key, {'count': 0})['count']
                  for key in ('no_prior_call', 'explicit_error', 'no_explicit_error', 'unknown')]
        lines.append('| '+label+' | '+' | '.join(map(str, counts))+' |')
    lines += ['', '## 保留、移除和新增样本', '',
              '保留ID的已核验标签、长度和候选名称一致，但batch组成变化仍可改变损失系数。完整来源×教师×领域×语义×调用数×报错交叉表保存在summary.json。', '',
              '| 当前:参照 | 保留 | 新增 | 移除 | 新增严格短停 | 移除严格短停 |', '|---|---:|---:|---:|---:|---:|']
    for pair, values in summary['membership_comparisons'].items():
        current, reference = pair.split(':')
        retained, added, removed = values['retained'][current], values['added'][current], values['removed'][reference]
        n_added = added['by_semantic'].get('short_turn', {'count': 0})['count']
        n_removed = removed['by_semantic'].get('short_turn', {'count': 0})['count']
        lines.append(f'| {pair} | {retained["count"]} | {added["count"]} | {removed["count"]} | {n_added} | {n_removed} |')
    lines += ['', '这些描述用于选择后续匹配对照，不把组成相关性解释成在线成功率的因果贡献。',
              'states.jsonl保留逐ID分类、原始错误证据位置和各组损失系数，便于核验；不包含原始长文本。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/analysis_training_mechanism.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    require(not out.exists(), 'output already exists')
    require(out.is_relative_to(PROJECT/('tmp' if args.smoke else 'outputs')), 'wrong output root')
    config = read_json(args.config)
    out.mkdir(parents=True)
    (out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n'+shlex.join([sys.executable, *sys.argv])+'\n')
    status = dict(status='running', started_at=now(), smoke=args.smoke)
    write_json(out/'metadata/run_status.json', status)
    try:
        runs = {label: read_run(PROJECT/path) for label, path in config['runs'].items()}
        canonical, features, domains, targets = read_inputs(config, runs)
        if args.smoke:
            chosen = {}
            for sid, target in sorted(targets.items()):
                slot = (target['dataset'], target['gold_names'] == ['RESPOND'], target['next_event'],
                        call_bin(features[sid]['turn_calls_before']))
                chosen.setdefault(slot, sid)
            targets = {sid: targets[sid] for sid in chosen.values()}
        print(f'Restoring {len(targets)} selected states', flush=True)
        restored = restore_features(targets, features, read_json(PROJECT/config['build_config']))
        states = []
        for sid in sorted(restored):
            state, feature = canonical[sid], restored[sid]
            domain = domains[state['task_group']]
            row = {key: state[key] for key in ('id', 'dataset', 'model', 'task_group', 'trajectory_id', 'turn_index', 'source', 'n_prompt_tokens', 'd1')}
            row.update(feature, domain=domain['label'], domain_label_source=domain['label_source'],
                       error_status=feature['error']['status'], prior_calls_bin=call_bin(feature['prior_calls']),
                       training_short_turn=state['short_turn'], training_direct=state['direct'],
                       normalized_gold=state['gold_names'], original_gold=state['original_gold'],
                       arms={label: run['exposure'][sid] for label, run in runs.items() if sid in run['exposure']})
            states.append(row)
        summary = make_summary(states, runs, config['pairs'], args.smoke)
        with (out/'states.jsonl').open('w') as handle:
            for row in states:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(out/'summary.json', summary)
        (out/'report.md').write_text(report(summary))
        status.update(status='completed', ended_at=now(), states=len(states))
        write_json(out/'metadata/run_status.json', status)
    except BaseException as exc:
        status.update(status='failed', ended_at=now(), error_type=type(exc).__name__, error=str(exc))
        write_json(out/'metadata/run_status.json', status)
        raise


if __name__ == '__main__':
    main()
