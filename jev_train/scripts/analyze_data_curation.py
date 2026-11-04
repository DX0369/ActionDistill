#!/usr/bin/env python
"""Stage 1 only: source-derived state features; no sample edits or model calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from analyze_respond import (PROJECT, WORKSPACE, read_json, write_json, require,
                             SampleIndex, load_selected, validate, tool_name)
from analyze_teacher_loops import locate_missing, history_loop
from jev_train.build import codex_call_types
from jev_train.common import digest
from dataset_tool_inventory.common import iter_jsonl
from dataset_tool_inventory.schema import DecisionEvent
from dataset_tool_inventory.adapters.review_common import emit_trajectory, visible, has_text

from jev_train.reply_capture import (source_units, classify_capture, RawTree, count_prior,
                                     is_visible, user_kind, TARGET_FIELDS,
                                     termination_evidence, policy_flags)

BASE_FIELDS = ('id', 'dataset', 'model', 'trajectory_id', 'turn_index')
FEATURE_FIELDS = ('id', 'split', 'dataset', 'outcome', 'gold_kind', 'next_event',
                  'next_tool_names', 'stop_reason_abnormal', 'narration_successor',
                  'turn_calls_before', 'turn_total_calls', 'turn_end_kind', 'strict_loop')
REFERENCE_FIELDS = ('dataset', 'model', 'trajectory_id', 'turn_index', 'category', 'reason',
                    'previous_turn_tool_calls', 'call_count_status', 'next_sources',
                    'raw_phase', 'termination_evidence', 'd1', 'd2', 'termination_status',
                    'termination_value', 'sensitivity_extra')


def sample_metadata(raw):
    validate(raw)
    names = [tool_name(option) for option in raw['options']]
    require(len(set(names)) == len(names) and names[-1] == 'RESPOND', f"invalid option names: {raw['id']}")
    gold = [names[i] for i in raw['gold']]
    require('RESPOND' not in gold or gold == ['RESPOND'], 'mixed RESPOND/tool gold')
    kind = 'RESPOND' if gold == ['RESPOND'] else 'TOOL' if len(gold) == 1 else 'TOOL_SET'
    if 'action_kind' in raw:
        require(raw['action_kind'] == kind, 'gold/action kind mismatch')
    result = {k: raw[k] for k in ('id', 'dataset', 'model', 'outcome', 'candidate_level', 'task_group')}
    result.update(names=names, gold_names=gold, gold_kind=kind, populations=[])
    for key in ('trajectory_id', 'turn_index'):
        if key in raw:
            result[key] = raw[key]
    return result


def load_inputs(args, reference):
    path = args.samples/'train.jsonl'
    cache = path.parent.parent/'index_cache'/path.parent.name/'train.json'
    require(cache.is_file(), 'requires existing canonical training index cache')
    index = SampleIndex(path)
    b0 = {r['id'] for r in load_selected(index, 8000, 20261015)}
    require(len(index.entries) == 261663 and len(b0) == 8000, 'training population or B0 size changed')
    targets, split_ids, pool_ids = {}, defaultdict(set), set()
    def add(raw, population, output_split=None):
        row = sample_metadata(raw)
        sid = row['id']
        if sid in targets:
            previous = targets[sid]
            for key in row.keys()-{'populations'}:
                if key in previous:
                    require(previous[key] == row[key], f'conflicting sample metadata: {sid}/{key}')
                else:
                    previous[key] = row[key]
            row = previous
        else:
            targets[sid] = row
        require(population not in row['populations'], f'duplicate sample in {population}: {sid}')
        row['populations'].append(population)
        if output_split is not None:
            require('split' not in row, f'state overlaps canonical splits: {sid}')
            row['split'] = output_split
            split_ids[output_split].add(sid)
        return row

    for number, raw in iter_jsonl(path):
        row = add(raw, 'full_train', 'train')
        if row['id'] in b0:
            row['populations'].append('training')
        if number % 50000 == 0:
            print(f'loaded full train: {number}', flush=True)
    require(len(split_ids['train']) == 261663, 'full training row count changed')
    print('loaded full train: 261663', flush=True)
    for split in ('train', 'val', 'test_id', 'test_ood'):
        for _, raw in iter_jsonl(args.pool/f'{split}.jsonl'):
            if split == 'train':
                # Pool is only a regression population; never an output/train expansion.
                names = [tool_name(raw['options'][i]) for i in raw['gold']]
                if names != ['RESPOND']:
                    # Reuse canonical locators on the old/pool overlap, irrespective of label.
                    if raw['id'] in targets:
                        old = targets[raw['id']]
                        current = sample_metadata(raw)
                        for key in ('dataset', 'model', 'gold_names', 'names', 'outcome'):
                            require(old[key] == current[key], f'pool intersection differs: {raw["id"]}/{key}')
                        old.update(trajectory_id=raw['trajectory_id'], turn_index=raw['turn_index'])
                    continue
                row = add(raw, 'pool')
                pool_ids.add(row['id'])
            else:
                add(raw, split, split)
        print(f'loaded pool {split}', flush=True)
    for sid, row in reference.items():
        require(sid in targets, f'reference state absent from current input: {sid}')
        current = targets[sid]
        require(current['gold_kind'] == 'RESPOND', f'reference gold changed: {sid}')
        for key in BASE_FIELDS:
            if key in current:
                require(current[key] == row[key], f'reference identity differs: {sid}/{key}')
            else:
                current[key] = row[key]
    require(pool_ids == {sid for sid, row in reference.items() if 'pool' in row['populations']},
            'pool RESPOND IDs differ from reference')
    require({sid for sid in b0 if targets[sid]['gold_kind'] == 'RESPOND'} ==
            {sid for sid, row in reference.items() if 'training' in row['populations']}, 'B0 RESPOND IDs differ')
    for split in ('val', 'test_id', 'test_ood'):
        require({sid for sid in split_ids[split] if targets[sid]['gold_kind'] == 'RESPOND'} ==
                {sid for sid, row in reference.items() if split in row['populations']}, f'{split} RESPOND IDs differ')
    return targets, split_ids, b0


def regression(row, reference):
    if row['id'] not in reference:
        return False
    old = reference[row['id']]
    for key in REFERENCE_FIELDS:
        require(row[key] == old[key], f'classification regression mismatch: {row["id"]}/{key}')
    return True


class TurnView:
    """Walk the existing raw graph; ambiguous suffixes remain unknown."""
    def __init__(self, unit):
        self.unit = unit
        self.capture = unit['capture']
        self.raw = self.capture['messages']
        self.messages = [visible(m, self.capture['context']['strip_think']) for m in self.raw]
        self.tree = RawTree(unit['family'], unit['raw_records'], self.raw) if unit['family'] in ('native_pi', 'native_claude') else None
        self.raw_by_line = dict(unit['raw_records'])
        self.next_linear = {}
        following = []
        for i in reversed(range(len(self.raw))):
            self.next_linear[i] = following
            if is_visible(self.messages[i]):
                following = [i]
        self.suffixes = {}

    def locator(self, index):
        return self.raw[index].get('_source', self.capture['context']['source']+f'#normalized_messages[{index}]')

    def originals(self, index):
        if self.tree:
            return self.tree.originals(index)
        if self.unit['family'] == 'native_codex':
            return [self.raw_by_line[int(self.raw[index]['_source'].rsplit(':', 1)[1])]]
        return []

    def following(self, index):
        return self.tree.successors(index, self.messages) if self.tree else (self.next_linear[index], None)

    def prior(self, index):
        return count_prior(index, self.raw, self.messages, self.locator, self.originals,
                           self.tree, self.unit['prefix'])

    def suffix(self, index):
        """Return calls from index onward, last decision, and an unknown reason."""
        path = []
        current = index
        while current not in self.suffixes:
            path.append(current)
            following, issue = self.following(current)
            if issue:
                tail = (None, None, issue)
                break
            if not following:
                prefix = self.unit['prefix']
                if prefix and (not prefix['source_fully_captured'] or prefix['known_continuation']):
                    tail = (None, None, 'captured_prefix_or_known_continuation')
                else:
                    tail = (0, None, None)
                break
            require(len(following) == 1, 'multiple successors without explicit issue')
            nxt = following[0]
            require(nxt > current, 'non-forward normalized successor')
            if self.messages[nxt]['role'] == 'user':
                kind = user_kind(self.messages[nxt], self.originals(nxt))
                if kind == 'user_message':
                    tail = (0, None, None)
                    break
                if kind in ('framework_compact_summary', 'mixed_user_text_and_tool_result'):
                    tail = (None, None, kind)
                    break
                # Explicit framework envelopes are not a new user request, as in count_prior.
            current = nxt
        else:
            tail = self.suffixes[current]
        for position in reversed(path):
            count, last, issue = tail
            if issue is None:
                message = self.messages[position]
                count += len(message.get('tool_calls', []))
                if last is None and message['role'] == 'assistant' and is_visible(message) and not self.raw[position].get('_not_teacher_step'):
                    last = position
            tail = count, last, issue
            self.suffixes[position] = tail
        return self.suffixes[index]


def capture_responds(unit, view):
    cap = unit['capture']
    descriptors = []
    for index, message in enumerate(view.messages):
        if message['role'] != 'assistant' or message.get('tool_calls') or not has_text(message['content']) or view.raw[index].get('_not_teacher_step'):
            continue
        raw = view.raw[index]
        model = raw['_model'] if '_model' in raw else cap['context']['model']
        descriptors.append(dict(id=digest(cap['dataset'], cap['trajectory_id'], index), dataset=cap['dataset'],
                                model=model, trajectory_id=cap['trajectory_id'], turn_index=index, populations=[]))
    terminal = termination_evidence(dict(unit, targets=descriptors))
    result = {}
    for row in classify_capture(cap, descriptors, unit['family'], unit['raw_records'], unit['prefix'], target_fields=BASE_FIELDS):
        evidence = terminal[row['turn_index']]
        row.update(termination_evidence=evidence, **policy_flags(row, evidence))
        result[row['turn_index']] = row
    return result


def smoke_targets(targets, reference):
    chosen = {}
    for row in sorted(targets.values(), key=lambda r:r['id']):
        slots = [(row['dataset'], row['gold_kind'])]
        if row['id'] in reference:
            old = reference[row['id']]
            slots.append((row['dataset'], old['category'], old['d2']))
        for slot in slots:
            chosen.setdefault(slot, (row['dataset'], row['trajectory_id']))
    keys = set(chosen.values())
    return {sid:row for sid,row in targets.items() if (row['dataset'], row['trajectory_id']) in keys}


def rebuild(unit, reference):
    cap = unit['capture']
    view = TurnView(unit)
    responds = capture_responds(unit, view)
    predecessors = defaultdict(list)
    for index, row in responds.items():
        if row['d1']:
            following, issue = view.following(index)
            require(issue is None and len(following) == 1, 'D1 successor is not unique')
            predecessors[following[0]].append(index)
    call_types = codex_call_types(Path(cap['context']['source'])) if unit['family'] == 'native_codex' else None
    wanted = {row['id']:row for row in unit['targets']}
    found = set()
    for event in emit_trajectory(cap['dataset'], cap['trajectory_id'], cap['messages'], cap['tools'], cap['context']):
        if not isinstance(event, DecisionEvent):
            continue
        sid = digest(event.dataset, event.trajectory_id, event.turn_index)
        if sid not in wanted:
            continue
        require(sid not in found, f'duplicate rebuilt state: {sid}')
        found.add(sid)
        target, meta = wanted[sid], event.metadata
        index = meta['target_message_index']
        require(index == target['turn_index'], f'decision/message index disagreement: {sid}')
        names = list(event.candidate_tools)+['RESPOND']
        gold = ['RESPOND'] if event.action_kind == 'RESPOND' else list(event.action_names)
        require(names == target['names'] and set(gold) == set(target['gold_names']), f'gold/candidate order changed: {sid}')
        require(event.model == target['model'] and event.action_kind == target['gold_kind'], f'teacher/action changed: {sid}')
        require(all(meta[k] == target[k] for k in ('outcome', 'candidate_level')), f'outcome/level changed: {sid}')
        prior = view.prior(index)
        reasons = {}
        before = prior['previous_turn_tool_calls']
        if before is None:
            reasons['turn_calls_before'] = prior['call_count_status']
        count, last, suffix_issue = view.suffix(index)
        total = before+count if before is not None and count is not None else None
        if total is None:
            reasons['turn_total_calls'] = suffix_issue or prior['call_count_status']
        if suffix_issue:
            end_kind = None
            reasons['turn_end_kind'] = suffix_issue
        else:
            final = responds.get(last)
            if final is not None and final['termination_status'] == 'conflict':
                end_kind = None
                reasons['turn_end_kind'] = 'conflicting_terminal_evidence'
            else:
                end_kind = ('RESPOND' if final is not None and final['category'] in ('user_turn_end', 'trajectory_end')
                            and not final['d2'] else 'incomplete')
        loop, argument_forms = None, []
        if before is not None:
            past = history_loop(meta['_history'], call_types)
            if past['current_turn_call_count'] == before:
                loop, argument_forms = past['loop_history'], past['argument_forms']
            else:
                reasons['strict_loop'] = 'framework_user_boundary_differs_from_loop_history'
        else:
            reasons['strict_loop'] = prior['call_count_status']
        narration = False
        if predecessors[index]:
            history = [message for message in meta['_history'] if is_visible(message)]
            narration = bool(history) and any(history[-1] == view.messages[i] for i in predecessors[index])
            require(narration, f'D1 successor history does not end with its narration: {sid}')
        classified = responds[index] if target['gold_kind'] == 'RESPOND' else None
        if classified is not None:
            require(classified['previous_turn_tool_calls'] == before, f'prior call count differs: {sid}')
            regression(classified, reference)
        next_tools = list(dict.fromkeys(classified['next_tool_names'])) if classified and classified['d1'] else None
        feature = dict(id=sid, split=target.get('split'), dataset=target['dataset'], outcome=target['outcome'],
                       gold_kind=target['gold_kind'], next_event=classified['category'] if classified else None,
                       next_tool_names=next_tools, stop_reason_abnormal=classified['d2'] if classified else None,
                       narration_successor=narration, turn_calls_before=before, turn_total_calls=total,
                       turn_end_kind=end_kind, strict_loop=loop)
        require(tuple(feature) == FEATURE_FIELDS, 'state feature schema drift')
        if classified and classified['category'] == 'unjudgeable':
            reasons['next_event'] = classified['reason']
        diagnostics = dict(reasons=reasons, argument_forms=argument_forms, family=unit['family'],
                           d1_missing_candidates=sorted(set(next_tools or [])-set(names)),
                           terminal_status=classified['termination_status'] if classified else None,
                           source=event.source)
        yield feature, classified, diagnostics
    require(found == wanted.keys(), f'missing reconstructed states: {sorted(wanted.keys()-found)[:5]}')


def add_statistics(bucket, feature):
    bucket['count'] = bucket.get('count', 0)+1
    for key in FEATURE_FIELDS[3:]:
        value = feature[key]
        if key == 'next_tool_names':
            if value is None:
                continue
            for name in value:
                counter = bucket.setdefault(key, Counter())
                counter[name] += 1
        else:
            counter = bucket.setdefault(key, Counter())
            counter[json.dumps(value, ensure_ascii=False)] += 1


def summarize_report(summary):
    lines = ['# 状态级特征与回归核对（第一阶段）\n',
             '只恢复源特征，不改标签、渲染、选样或模型；未进入第二阶段。\n',
             '## 回归\n', '```json', json.dumps(summary['validation'], ensure_ascii=False, indent=2), '```',
             '\n## 切分分布\n', '| split | 状态 | RESPOND | D1 | D2 | narration_successor | strict_loop | 整轮未知 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, stats in summary['by_split'].items():
        def count(field, value):
            return stats[field].get(json.dumps(value), 0)
        lines.append(f'| {split} | {stats["count"]} | {count("gold_kind", "RESPOND")} | '
                     f'{count("next_event", "intermediate_tool_continuation")} | {count("stop_reason_abnormal", True)} | '
                     f'{count("narration_successor", True)} | {count("strict_loop", True)} | {count("turn_total_calls", None)} |')
    lines.extend(['\n## 逐来源（train）\n', '| dataset | 状态 | RESPOND | D1 | D2 | strict_loop | 1–3调用且RESPOND结尾 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for dataset, stats in summary['by_split_dataset'].get('train', {}).items():
        lines.append(f'| {dataset} | {stats["count"]} | {stats["gold_kind"].get(chr(34)+"RESPOND"+chr(34),0)} | '
                     f'{stats["next_event"].get(chr(34)+"intermediate_tool_continuation"+chr(34),0)} | '
                     f'{stats["stop_reason_abnormal"].get("true",0)} | {stats["strict_loop"].get("true",0)} | '
                     f'{summary["short_turn_observations"].get(dataset,0)} |')
    lines.append('\n## 口径与局限\n')
    lines.extend('- '+line for line in summary['definitions'].values())
    lines.append('\n全部字段分布、未知原因及对应ID见summary.json；逐状态记录见state_features.jsonl。')
    return '\n'.join(lines)+'\n'


DEFINITIONS = dict(
    inputs='训练仅旧capped的261663条；val/test_id/test_ood复用已有逐ID等价的pool保留集。pool train的RESPOND仅用于回归，不扩充训练人群。',
    identity='每个输出状态均与原始教师、动作、候选顺序、outcome、candidate_level核对；旧训练定位复用历史review，仅以旧Codex步号作定位上界。',
    d1='沿用classify_capture的紧邻可见事件定义；next_tool_names按原调用顺序去重。narration_successor来自唯一实际后继并确认其可见历史末尾为D1，非简单步号加一。',
    d2='按用户确认仅对RESPOND，当前消息明确且唯一的error/length才为true；工具动作记null表示不适用。缺失/null终止元数据并不证明正常，单独计数。',
    turns='在完整已发布原始轨迹而非抽样状态中计数，并行调用逐条计；复用原用户边界/父链检查。turn_total_calls含当前状态的调用以及同轮后续调用。',
    end='turn_end_kind=RESPOND表示源轮最后决策为C1后仍有效的RESPOND；incomplete表示已观察到轮界/记录结束但没有保留的RESPOND结尾，不推断任务是否完成。分支/后缀不明则null。',
    loops='strict_loop仅看当前用户轮此前最近三次调用的名字和规范参数，与当前gold无关，包含报错后同参重试；C4错误豁免留第二阶段确认。',
    arguments='复用已批准的参数比较：JSON规范化、空串视为{}、损坏JSON按原文、Codex custom按原始类型文本精确比较；特殊形式单独计数。',
    unknown='不可可靠恢复的特征记null并按字段/原因/ID列出；非RESPOND的next_event/stop_reason_abnormal及非D1的next_tool_names为不适用null。',
    regression='旧summary仅覆盖B0、pool和保留集的21231个唯一RESPOND，不含全train所有RESPOND；这些ID逐项复现原语义字段，新出现的train状态另行统计。',
    limits='未来调用数只用于数据整理，不是推理可用特征；原始历史与渲染截断后的可见历史不同。无父链的Codex采用现有顺序恢复，不声称存在Codex parent树。',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('samples', 'pool', 'reference', 'build-config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.output/'metadata'
    metadata.mkdir()
    (metadata/'command.txt').write_text(shlex.join([sys.executable, *sys.argv])+'\n')
    started = datetime.now().astimezone().isoformat(timespec='seconds')
    write_json(metadata/'run_status.json', dict(status='running', started_at=started))
    try:
        previous = read_json(args.reference/'summary.json')
        reference = {r['id']:r for r in previous['states']}
        require(len(reference) == len(previous['states']) == 21231, 'reference ID coverage changed')
        targets, split_ids, b0 = load_inputs(args, reference)
        config = read_json(args.build_config)
        locate_missing(targets, config)
        if args.smoke:
            targets = smoke_targets(targets, reference)
        expected = {sid for sid, r in targets.items() if 'split' in r}
        summary = dict(created_at=started, definitions=DEFINITIONS,
                       sources={k:str(getattr(args,k).resolve()) for k in ('samples','pool','reference','build_config')},
                       by_split={}, by_split_dataset={}, unknown={}, terminal_status={},
                       argument_forms={}, d1_missing_candidates={}, short_turn_observations={}, validation={})
        seen, compared, b0_flags = set(), set(), Counter()
        families = Counter()
        with (args.output/'state_features.jsonl').open('w', encoding='utf-8') as handle:
            for unit in source_units(targets, config):
                for feature, classified, diagnostic in rebuild(unit, reference):
                    sid = feature['id']
                    require(sid not in seen, f'duplicate reconstructed ID: {sid}')
                    seen.add(sid)
                    if sid in reference:
                        require(classified is not None, 'reference RESPOND disappeared')
                        compared.add(sid)
                    if sid in b0 and classified:
                        b0_flags.update(respond=1, d1=int(classified['d1']), d2=int(classified['d2']),
                                        both=int(classified['d1'] and classified['d2']))
                    if feature['split'] is None:
                        continue
                    families[diagnostic['family']] += 1
                    handle.write(json.dumps(feature, ensure_ascii=False)+'\n')
                    split, dataset = feature['split'], feature['dataset']
                    add_statistics(summary['by_split'].setdefault(split, {}), feature)
                    add_statistics(summary['by_split_dataset'].setdefault(split, {}).setdefault(dataset, {}), feature)
                    for field, reason in diagnostic['reasons'].items():
                        summary['unknown'].setdefault(split, {}).setdefault(dataset, {}).setdefault(field, {}).setdefault(reason, []).append(sid)
                    if diagnostic['terminal_status'] is not None:
                        counter = summary['terminal_status'].setdefault(split, {}).setdefault(dataset, Counter())
                        counter[diagnostic['terminal_status']] += 1
                    for form in diagnostic['argument_forms']:
                        summary['argument_forms'].setdefault(split, {}).setdefault(dataset, {}).setdefault(form, []).append(sid)
                    if diagnostic['d1_missing_candidates']:
                        summary['d1_missing_candidates'][sid] = diagnostic['d1_missing_candidates']
                    if split == 'train' and feature['turn_end_kind'] == 'RESPOND' and feature['turn_total_calls'] in (1,2,3):
                        summary['short_turn_observations'][dataset] = summary['short_turn_observations'].get(dataset,0)+1
        require(seen == targets.keys(), 'rebuilt ID coverage differs')
        require(compared == reference.keys() & targets.keys(), 'regression coverage differs')
        if not args.smoke:
            require(dict(b0_flags) == dict(respond=1039, d1=469, d2=37, both=3), f'B0 regression failed: {b0_flags}')
            require(expected == set.union(*split_ids.values()), 'output coverage differs')
        summary['validation'] = dict(smoke=args.smoke, output_states=len(expected), rebuilt_states=len(seen),
                                     reference_states_compared=len(compared), all_reference_fields_equal=True,
                                     b0=dict(b0_flags), format_families=dict(families),
                                     full_training_rows=len(split_ids['train']))
        write_json(args.output/'summary.json', summary)
        (args.output/'report.md').write_text(summarize_report(summary), encoding='utf-8')
    except Exception as exc:
        write_json(metadata/'run_status.json', dict(status='failed', started_at=started, error=repr(exc)))
        raise
    write_json(metadata/'run_status.json', dict(status='smoke_completed' if args.smoke else 'completed',
                                               started_at=started, completed_at=datetime.now().astimezone().isoformat(timespec='seconds')))


if __name__ == '__main__':
    main()
