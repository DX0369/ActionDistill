#!/usr/bin/env python3
"""Frozen-input R11-cap CPU preview. Never writes samples or selected-ID files."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from preview_dedup import PROJECT, SEED, require, timestamp
from preview_domains import load_prepared, load_labels
from jev_train.common import rows, write_json
from jev_train.dedup import deduplicate
from jev_train.domains import balance_selection, BalanceCapacityError, label_acceptance


def read_json(path):
    return json.loads(Path(path).read_text())


def selection_hash(ids):
    return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()


def load_frozen(config):
    require(config['seed'] == SEED, 'R11-cap seed differs')
    preview = (PROJECT/config['balance_preview']).resolve()
    state, old = read_json(preview/'metadata/run_status.json'), read_json(preview/'summary.json')
    require(state['status'] == 'completed' and not state['smoke'] and not old['smoke']
            and old['recipe'] == 'B2-balance-8k' and old['count'] == 8000 and old['seed'] == SEED,
            'requires the completed frozen four-arm preview')
    paths = {k: Path(old['inputs'][k]).resolve() for k in ('dedup_preview', 'baseline_samples', 'labels')}
    parent, plans, _, _, reference_ids = load_prepared(paths['dedup_preview'])
    require(parent['recovery_inputs'] == old['recovery_inputs']
            and parent['inputs']['samples'] == old['source_samples'], 'frozen recovery inputs differ')
    baseline = paths['baseline_samples']
    bs, bm = read_json(baseline/'metadata/run_status.json'), read_json(baseline/'summary.json')
    require(bs['status'] == 'completed' and not bs['smoke'] and bs['count'] == 8000
            and bm['recipe'] == 'B2-dedup-8k' and bm['count'] == 8000 and bm['seed'] == SEED
            and Path(bm['preview']).resolve() == paths['dedup_preview'], 'R00 baseline identity differs')
    require(reference_ids == (baseline/'train_ids.txt').read_text().splitlines()
            and len(reference_ids) == len(set(reference_ids)) == 8000, 'frozen ordered R00 IDs differ')
    labels, label_state = load_labels(paths['labels'], paths['dedup_preview'])
    kept = deduplicate(plans, SEED)
    ids, regression = balance_selection(kept, labels, reference_ids, 8000, SEED,
                                       domain_alpha=0, source_beta=.5, cap_domains=True)
    r11 = (PROJECT/config['r11_samples']).resolve()
    rs, rm = read_json(r11/'metadata/run_status.json'), read_json(r11/'summary.json')
    require(rs['status'] == 'completed' and not rs['smoke'] and rm['count'] == 8000
            and not rm['smoke'] and rm['recipe'] == 'B2-balance-R11-8k' and rm['seed'] == SEED
            and Path(rm['preview']).resolve() == preview
            and Path(rm['source']).resolve() == Path(old['source_samples']).resolve()/'train.jsonl'
            and rm['recovery_inputs'] == old['recovery_inputs'], 'R11 regression sample identity differs')
    require(ids == (r11/'train_ids.txt').read_text().splitlines(), 'R11-cap 8k ordered ID regression failed')
    require(selection_hash(ids) == old['groups']['R11']['selection_sha256'], 'frozen R11 digest differs')
    # Preserve not only IDs, but also the old writer's complete successful report.
    for key, value in regression.items():
        if key != 'cap_allocation':
            require(json.loads(json.dumps(value)) == old['groups']['R11'][key], f'R11 report regression: {key}')
    for d, cells in regression['cells'].items():
        for s, cell in cells.items():
            require(cell['reference_count'] == old['groups']['R00']['cells'][d][s]['reference_count'],
                    f'frozen reference count differs: {d}/{s}')
    inputs = dict(balance_preview=str(preview), r11_samples=str(r11), **{k: str(v) for k, v in paths.items()},
                  **old['recovery_inputs'], source_samples=old['source_samples'])
    return kept, labels, reference_ids, ids, regression, inputs, label_state


def load_semantics(directory, kept):
    state = read_json(directory/'metadata/run_status.json')
    require(state['status'] == 'completed' and not state['smoke'], 'mechanism cache incomplete')
    result = {}; seen = set()
    for row in rows(directory/'states.jsonl'):
        sid = row['id']
        require(sid not in seen, 'duplicate mechanism ID'); seen.add(sid)
        if sid not in kept:
            continue
        original = kept[sid]
        for field in ('dataset', 'model', 'task_group', 'trajectory_id', 'turn_index', 'n_prompt_tokens'):
            require(row[field] == original[field], f'{sid}: mechanism identity differs: {field}')
        require(row['normalized_gold'] == original['gold_names'], f'{sid}: mechanism gold differs')
        if original['gold_names'] == ['RESPOND']:
            require(row['next_event'] == original['next_event'], f'{sid}: mechanism successor differs')
            require(row['semantic'] in {'short_turn', 'short_end', 'long_turn_or_end',
                                       'no_prior_call', 'mid_commentary', 'unknown'},
                    f'{sid}: invalid cached RESPOND semantic')
            result[sid] = row
    require(len(seen) == state['states'], 'mechanism coverage differs')
    return result


def planned_exposure(records):
    """Predict the unchanged single-card accum=2, one-epoch R11 schedule."""
    from decider.train import batches_by_tokens
    from train_jev import total_steps
    require(all(type(r['n_prompt_tokens']) is int and r['n_prompt_tokens'] > 0 for r in records),
            'invalid prompt length')
    items = [{'ids': range(r['n_prompt_tokens'])} for r in records]
    batches = batches_by_tokens(items, 32768, random.Random(SEED))
    updates = total_steps(len(batches), 1, 2, 1, 0)
    schedule = list(batches)
    if len(schedule) < updates*2:
        schedule += batches_by_tokens(items, 32768, random.Random(SEED+1))[:1]
    coefficients = Counter(); occurrences = Counter(); tokens = 0
    for batch in schedule:
        for i in batch:
            coefficients[records[i]['id']] += 1/(2*len(batch))
            occurrences[records[i]['id']] += 1
            tokens += records[i]['n_prompt_tokens']
    require(math.isclose(sum(coefficients.values()), updates, rel_tol=1e-10), 'invalid planned coefficients')
    return coefficients, dict(kind='prediction_without_training_log_validation', world_size=1, accum=2,
        epochs=1, max_tokens=32768, seed=SEED, microbatches=len(batches), updates=updates,
        effective_prompt_tokens=tokens, occurrences=sum(occurrences.values()),
        loss_coefficient_sum=sum(coefficients.values()))


def diagnostics(ids, kept, r11_ids, semantics):
    records = [kept[sid] for sid in ids]
    coefficients, schedule = planned_exposure(records)
    lengths = sorted(r['n_prompt_tokens'] for r in records)
    trajectories = Counter((r['dataset'], r['trajectory_id']) for r in records)
    old_tasks = {kept[sid]['task_group'] for sid in r11_ids}
    tasks = {r['task_group'] for r in records}
    known, unknown = [], []
    for row in records:
        if row['gold_names'] != ['RESPOND'] or row['next_event'] == 'trajectory_end':
            continue
        sid = row['id']
        if sid not in semantics or semantics[sid]['semantic'] == 'unknown':
            unknown.append(sid)
        elif semantics[sid]['semantic'] == 'short_turn':
            known.append(sid)
    lower = sum(coefficients[sid] for sid in known)/schedule['updates']
    upper = lower + sum(coefficients[sid] for sid in unknown)/schedule['updates']
    return dict(overlap_with_R11_8k=len(set(ids) & set(r11_ids)), new_task_groups=len(tasks-old_tasks),
        retained_R11_task_groups=len(tasks & old_tasks),
        states_per_trajectory_histogram=dict(sorted(Counter(trajectories.values()).items())),
        maximum_states_per_trajectory=max(trajectories.values()),
        prompt_length=dict(method='nearest_rank', minimum=lengths[0], maximum=lengths[-1],
            mean=sum(lengths)/len(lengths),
            **{f'p{q}': lengths[math.ceil(q/100*len(lengths))-1] for q in (10,25,50,75,90,95,99)}),
        schedule=schedule,
        strict_short_stop=dict(known_positive=len(known), unknown=len(unknown),
            known_positive_task_groups=len({kept[sid]['task_group'] for sid in known}),
            complete=not unknown, count_lower_bound=len(known), count_upper_bound=len(known)+len(unknown),
            loss_coefficient_fraction_lower_bound=lower, loss_coefficient_fraction_upper_bound=upper,
            unknown_by_source=dict(sorted(Counter(kept[sid]['dataset'] for sid in unknown).items())),
            unknown_ids=unknown))


def build_preview(config, smoke=False):
    counts = [8000] if smoke else config['counts']
    require(counts and len(set(counts)) == len(counts)
            and all(type(n) is int and n > 0 for n in counts), 'invalid target counts')
    kept, labels, reference_ids, r11_ids, regression, inputs, label_state = load_frozen(config)
    cache = (PROJECT/config['mechanism']).resolve()
    semantics = load_semantics(cache, kept)
    result = dict(recipe='R11-cap', seed=SEED, inputs=dict(inputs, mechanism=str(cache)), smoke=smoke,
        ordered_R11_8k_regression=True, reference_count=len(reference_ids),
        regression_selection_sha256=selection_hash(r11_ids),
        regression_diagnostics=diagnostics(r11_ids, kept, r11_ids, semantics),
        label_acceptance=label_acceptance(label_state, Path(inputs['labels'])), targets={},
        definitions=dict(domain='Equal integer water filling over positive-reference raw capacities; lexical ties.',
            source='Frozen R00 8000 counts, sqrt(n_domain_source); old within-domain capped allocation; zero-reference excluded.',
            priority='C1+C2+C3, C4 off. Per-source round(.25*final quota), shared across domains; ordinary-only backfill.',
            failure='Fix raw-capacity quotas first, then check C3. No relaxation or joint reallocation on failure.',
            strict_short_stop='Normalized RESPOND, next user, 1-3 prior calls, no explicit abnormal/conflicting termination. Reuse frozen mechanism evidence only; missing cache is unknown.',
            loss_weight='Predicted sum of 1/(2*microbatch_size), divided by updates; not gradient magnitude or actual training.',
            provenance='Frozen historical prompts/labels only; old S grade does not prove request-time candidate completeness. No schema repair or raw reconstruction.',
            scope='CPU preview only; no sample/ID files, model, evaluation or service changes.'))
    for count in counts:
        try:
            ids, report = (r11_ids, regression) if count == 8000 else balance_selection(
                kept, labels, reference_ids, count, SEED, domain_alpha=0, source_beta=.5, cap_domains=True)
        except BalanceCapacityError as exc:
            result['targets'][str(count)] = dict(status='failed', failure_stage='fixed_quotas_C3',
                error=str(exc), allocation=exc.report, selected=None, diagnostics=None)
            continue
        result['targets'][str(count)] = dict(status='completed' if report['respond_gate_passed'] else 'failed',
            failure_stage=None if report['respond_gate_passed'] else 'RESPOND_gate',
            selection_sha256=selection_hash(ids), allocation=report,
            selected=report['selected'], diagnostics=diagnostics(ids, kept, r11_ids, semantics))
    return result


def report_markdown(result):
    lines = ['# R11-cap 冻结输入配额预览', '',
        '参考权重固定为原 R00 的 8,000 个状态；新入口在 8k 与原 R11 有序 ID、摘要和成功统计完全一致。',
        '按受支持原始容量分配领域及来源配额后才检查 C3；失败不放宽约束，也不输出部分选样。', '']
    for count, arm in result['targets'].items():
        lines += [f"## {count}：{arm['status']}", '',
                  '| 规模 | 领域 | 容量 | 配额 |', '|---:|---|---:|---:|']
        plan = arm['allocation']['cap_allocation'] if arm['selected'] is not None else arm['allocation']
        for domain, row in plan['domains'].items():
            lines.append(f"| {count} | {domain} | {row['supported_capacity']} | {row['quota']} |")
        lines += ['', '| 来源 | 配额 | 优先可用 | 普通可用 | C3上限 | 至少需优先 |',
            '|---|---:|---:|---:|---:|---:|']
        for source, row in plan['planned_sources'].items():
            lines.append(f"| {source} | {row['quota']} | {row['priority_available']} | {row['ordinary_available']} | {row['priority_quota']} | {row['mandatory_priority']} |")
        if arm['selected'] is None:
            lines += ['', arm['error'], '',
                '尚无合法选中集合；RESPOND、任务覆盖、长度、训练 token、预计步数及短停止损失份额均未计算。']
        else:
            r, d = arm['selected'], arm['diagnostics']
            lines += ['', f"RESPOND {r['respond']}/{r['count']}（{r['respond_fraction']:.4%}）；优先类 {r['priority']}；任务 {r['task_groups']}；轨迹 {r['trajectories']}。",
                f"prompt token {r['n_prompt_tokens']:,}；预计更新 {d['schedule']['updates']}；与 R11-8k 重合 {d['overlap_with_R11_8k']}。",
                f"严格短停已知 {d['strict_short_stop']['known_positive']}，未知 {d['strict_short_stop']['unknown']}；完整分布和系数上下界见 summary。"]
            if arm['status'] == 'failed':
                lines += ['RESPOND 比例未通过既有 8%–20% 检查；停止在预览阶段，不写样本、不放宽规则。']
        lines.append('')
    lines += ['严格短停仅复用冻结机制审计，未补原始轨迹；无证据的状态不当成负例。训练计划以原 R11 单卡 accum=2 为参照，未运行训练或核对新训练日志。',
        '沿用获准的85%复核一致率领域标签；冻结 Codex 候选仍有完整性/可见时段限制。没有加入新参数渲染，也未修复旧 canonical 样本。',
        '预览结束后停在用户审阅点。无 train.jsonl、train_ids.txt 或模型产物。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/selection/R11_cap.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='8k regression only, output under project tmp')
    args = parser.parse_args(); out = args.output.resolve()
    if args.smoke and not out.is_relative_to(PROJECT/'tmp'):
        parser.error('smoke output must be under project tmp')
    out.mkdir(parents=True, exist_ok=False); (out/'metadata').mkdir()
    (out/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable, *sys.argv])}\n')
    state = dict(status='running', started_at=timestamp(), smoke=args.smoke)
    write_json(out/'metadata/run_status.json', state)
    try:
        result = build_preview(read_json(args.config), args.smoke)
        result['config'] = str(args.config.resolve())
        write_json(out/'summary.json', result)
        (out/'report.md').write_text(report_markdown(result))
        failed = [n for n, arm in result['targets'].items() if arm['status'] == 'failed']
        state.update(status='failed' if failed else 'smoke_completed' if args.smoke else 'completed',
                     ended_at=timestamp(), failed_targets=failed, ordered_R11_8k_regression=True)
        write_json(out/'metadata/run_status.json', state)
    except BaseException as exc:
        state.update(status='failed', ended_at=timestamp(), error=str(exc))
        write_json(out/'metadata/run_status.json', state)
        raise
    print(json.dumps(dict(output=str(out), status=state['status'], failed_targets=failed), ensure_ascii=False))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
