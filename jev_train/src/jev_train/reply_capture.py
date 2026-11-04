"""Recover source reply boundaries and terminal evidence with the original protocol."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json

from dataset_tool_inventory.adapters import review_formats as formats
from dataset_tool_inventory.adapters.review_common import has_text, json_value, visible
from dataset_tool_inventory.common import iter_jsonl

from . import REPO as WORKSPACE
from .build import setup_entry
from .common import digest


TARGET_FIELDS = ('id', 'dataset', 'model', 'trajectory_id', 'turn_index', 'transition',
                 'trajectory_position', 'trajectory_decisions')


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def requested_records(path, indices):
    """Read only selected records, preserving their canonical 1-based file index."""
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        file = pq.ParquetFile(path)
        offset = 0
        for group in range(file.num_row_groups):
            size = file.metadata.row_group(group).num_rows
            wanted = sorted(i for i in indices if offset < i <= offset+size)
            if wanted:
                # Bounded batches rather than converting the full row group to Python.
                start = offset
                for batch in file.iter_batches(row_groups=[group], batch_size=8):
                    for i in wanted:
                        if start < i <= start+batch.num_rows:
                            yield i, batch.slice(i-start-1, 1).to_pylist()[0]
                    start += batch.num_rows
                    if start >= max(wanted):
                        break
            offset += size
    elif path.suffix in ('.jsonl', '.gz'):
        for index, row in iter_jsonl(path):
            if index in indices:
                yield index, row
            if index >= max(indices):
                break
    elif path.suffix == '.json':
        require(indices == {1}, 'JSON file must use canonical record index 1')
        yield 1, read_json(path)
    else:
        raise ValueError(f'unsupported source storage: {path}')


def text_of(message):
    value = message['content']
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return '\n'.join(block['text'] for block in value
                     if block.get('type') in ('text', 'input_text', 'output_text'))


def is_visible(message):
    return message['role'] != 'assistant' or bool(message.get('tool_calls')) or has_text(message['content'])


def user_kind(message, originals=()):
    """Do not infer user intent from promptSource or from ordinary prose."""
    for raw in originals:
        if raw.get('isCompactSummary') is True:
            return 'framework_compact_summary'
    for raw in originals:
        if raw.get('isMeta') is True:
            return 'framework_isMeta'
    for raw in originals:
        blocks = raw.get('message', {}).get('content')
        if isinstance(blocks, list) and any(p.get('type') == 'tool_result' for p in blocks) and any(
                p.get('type') == 'text' for p in blocks):
            return 'mixed_user_text_and_tool_result'
    if text_of(message).lstrip().startswith(('<task-notification>', '<local-command-caveat>',
                                            '<local-command-stdout>', '<command-name>')):
        return 'framework_explicit_envelope'
    return 'user_message'


@contextmanager
def capture_normalized():
    """Reuse the adapters up to emit_trajectory without materializing every prefix."""
    captured = []
    previous = formats.emit_trajectory
    def capture(dataset, trajectory_id, messages, tools, context, *, target_index=None):
        captured.append(dict(dataset=dataset, trajectory_id=trajectory_id, messages=messages,
                             tools=tools, context=context, target_index=target_index))
        return iter(())
    formats.emit_trajectory = capture
    try:
        yield captured
    finally:
        formats.emit_trajectory = previous


class RawTree:
    """Uncontracted raw parent graph; normalized Claude messages can span several nodes."""
    def __init__(self, family, raw_records, messages):
        self.family = family
        self.records = dict(raw_records)
        self.nodes, self.children, self.message_nodes, self.node_messages = {}, defaultdict(list), {}, defaultdict(set)
        self.problem = None
        node_field, parent_field = ('uuid', 'parentUuid') if family == 'native_claude' else ('id', 'parentId')
        for line, row in raw_records:
            if node_field not in row:
                continue
            node = row[node_field]
            if node in self.nodes:
                self.problem = 'duplicate_raw_node'
            self.nodes[node] = (line, row, row.get(parent_field))
            self.children[row.get(parent_field)].append(node)
        for index, message in enumerate(messages):
            if family == 'native_claude':
                nodes = set(message.get('_uuids', ()))
            else:
                line = int(message['_source'].rsplit(':', 1)[1])
                nodes = {self.records[line][node_field]} if node_field in self.records[line] else set()
            self.message_nodes[index] = nodes
            for node in nodes:
                self.node_messages[node].add(index)
        done = set()
        for node in self.nodes:
            branch = set()
            current = node
            while current in self.nodes and current not in done:
                if current in branch:
                    self.problem = 'raw_parent_cycle'
                    break
                branch.add(current)
                current = self.nodes[current][2]
            done.update(branch)

    def originals(self, index):
        return [self.nodes[node][1] for node in self.message_nodes[index] if node in self.nodes]

    def successors(self, index, messages):
        if self.problem:
            return [], self.problem
        own = self.message_nodes[index]
        if not own:
            return [], 'missing_raw_node'
        pending = [child for node in own for child in self.children[node]]
        seen, found = set(), set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            visible_indices = {i for i in self.node_messages[node] if i != index and is_visible(messages[i])}
            if visible_indices:
                found.update(visible_indices)
            else:
                pending.extend(self.children[node])
        following = sorted(found)
        if any(i < index for i in following):
            return following, 'merged_fragment_order_ambiguous'
        if len(following) > 1:
            return following, 'multiple_tree_successors'
        if not following and any(is_visible(m) for m in messages[index+1:]):
            return [], 'no_connected_successor_with_later_records'
        if following:
            sides = {r['isSidechain'] for r in self.originals(index) if 'isSidechain' in r}
            next_sides = {r['isSidechain'] for r in self.originals(following[0]) if 'isSidechain' in r}
            if sides and next_sides and sides != next_sides:
                return following, 'sidechain_mismatch'
        return following, None

    def ancestry_to_user(self, target, boundary):
        if self.problem:
            return set(), self.problem
        own, user = self.message_nodes[target], self.message_nodes[boundary]
        if not own or not user:
            return set(), 'parent_chain_missing_before_user'
        ancestors = set(own)
        # Fragment parents that are inside the same message are not independent roots.
        parents = {self.nodes[n][2] for n in own if n in self.nodes and self.nodes[n][2] not in own}
        for parent in parents:
            branch = set()
            while parent not in user:
                if parent not in self.nodes:
                    return set(), 'parent_chain_missing_before_user'
                if parent in branch:
                    return set(), 'raw_parent_cycle'
                branch.add(parent)
                ancestors.add(parent)
                parent = self.nodes[parent][2]
            ancestors.add(parent)
        return ancestors, None


def classify_next(following, messages, originals, issue=None, prefix=None):
    if issue:
        return 'unjudgeable', issue
    if not following:
        if prefix and (not prefix['source_fully_captured'] or prefix['known_continuation']):
            return 'unjudgeable', 'captured_prefix_or_known_continuation'
        return 'trajectory_end', 'end_of_observed_published_trajectory'
    require(len(following) == 1, 'ambiguous successors lack an explicit issue')
    index = following[0]
    message = messages[index]
    if message['role'] == 'user':
        kind = user_kind(message, originals(index))
        return ('user_turn_end', 'next_user_message') if kind == 'user_message' else ('unjudgeable', kind)
    if message['role'] == 'assistant':
        return ('intermediate_tool_continuation', 'next_assistant_tool_call') if message.get('tool_calls') else ('unjudgeable', 'assistant_text_next')
    return 'unjudgeable', message['role']+'_next'


def count_prior(index, raw_messages, messages, locator, originals, tree=None, prefix=None):
    if prefix and (prefix['continuation'] or prefix['trace_part'] > 1):
        return dict(previous_turn_tool_calls=None, call_count_status='continued_source_part_missing_prior_turn')
    abandoned = raw_messages[index].get('_abandoned', frozenset())
    history = [i for i in range(index) if not (raw_messages[i].get('_uuids') and raw_messages[i]['_uuids'] <= abandoned)]
    ignored = 0
    boundary = None
    for i in reversed(history):
        if messages[i]['role'] != 'user':
            continue
        kind = user_kind(messages[i], originals(i))
        if kind == 'user_message':
            boundary = i
            break
        if kind in ('framework_compact_summary', 'mixed_user_text_and_tool_result'):
            return dict(previous_turn_tool_calls=None, call_count_status=kind)
        ignored += 1
    if boundary is None:
        return dict(previous_turn_tool_calls=None, call_count_status='no_user_boundary_in_record')
    if tree:
        ancestors, issue = tree.ancestry_to_user(index, boundary)
        if issue:
            return dict(previous_turn_tool_calls=None, call_count_status=issue)
    calls = []
    for i in history:
        if i <= boundary or not messages[i].get('tool_calls'):
            continue
        if tree and not tree.message_nodes[i] & ancestors:
            return dict(previous_turn_tool_calls=None, call_count_status='tool_message_outside_target_ancestry')
        if tree and tree.message_nodes[i] and max(tree.nodes[n][0] for n in tree.message_nodes[i]) >= min(
                tree.nodes[n][0] for n in tree.message_nodes[index]):
            return dict(previous_turn_tool_calls=None, call_count_status='tool_fragment_after_target')
        calls.extend(c['function']['name'] for c in messages[i]['tool_calls'])
    return dict(previous_turn_tool_calls=len(calls), user_boundary_source=locator(boundary),
                previous_turn_call_names=dict(Counter(calls)), framework_user_messages_ignored=ignored,
                call_count_status='known')


def classify_capture(capture, targets, family, raw_records=(), prefix=None, *, target_fields=TARGET_FIELDS):
    raw_messages, context = capture['messages'], capture['context']
    messages = [visible(message, context['strip_think']) for message in raw_messages]
    tree = RawTree(family, raw_records, raw_messages) if family in ('native_pi', 'native_claude') else None
    raw_by_line = dict(raw_records)
    def locator(index):
        return raw_messages[index].get('_source', context['source']+f'#normalized_messages[{index}]')
    def originals(index):
        if tree:
            return tree.originals(index)
        if family == 'native_codex':
            return [raw_by_line[int(raw_messages[index]['_source'].rsplit(':', 1)[1])]]
        return []
    for target in sorted(targets, key=lambda r: r['turn_index']):
        index = target['turn_index']
        require(0 <= index < len(messages), f"target index outside published messages: {target['id']}")
        current = messages[index]
        require(current['role'] == 'assistant' and not current.get('tool_calls') and has_text(current['content']),
                f"not a source RESPOND: {target['id']}")
        require(not raw_messages[index].get('_not_teacher_step'), 'target is a framework message')
        teacher = raw_messages[index]['_model'] if '_model' in raw_messages[index] else context['model']
        require(teacher == target['model'], f"source teacher differs: {target['id']}")
        require(digest(capture['dataset'], capture['trajectory_id'], index) == target['id'], 'source ID differs')
        if 'source' in target and not prefix:
            require(target['source'] == raw_messages[index].get('_source', context['source']), 'saved raw locator differs')
        row = {key: target[key] for key in target_fields}
        row.update(populations=target['populations'], source=locator(index), family=family,
                   target_excerpt=text_of(current)[:180], raw_phase={'present': False},
                   raw_channel={'present': False}, previous_turn_tool_calls=None)
        if family == 'native_codex':
            payload = originals(index)[0]['payload']
            for key in ('phase', 'channel'):
                row['raw_'+key] = dict(present=True, value=payload[key]) if key in payload else dict(present=False)
        if family == 'native_pi':
            row['raw_stopReason'] = [r['message']['stopReason'] for r in originals(index) if 'stopReason' in r['message']]
        if tree:
            following, issue = tree.successors(index, messages)
        else:
            following = next(([i] for i in range(index+1, len(messages)) if is_visible(messages[i])), [])
            issue = None
        row.update(next_sources=[locator(i) for i in following], next_roles=[messages[i]['role'] for i in following])
        if family == 'native_codex':
            start = int(raw_messages[index]['_source'].rsplit(':', 1)[1])
            end = int(raw_messages[following[0]]['_source'].rsplit(':', 1)[1]) if following else max(raw_by_line)+1
            row['end_markers_before_next_message'] = [dict(source=context['source']+f':{line}', type=r['payload']['type'])
                for line, r in raw_records if start < line < end and r['type'] == 'event_msg'
                and r['payload']['type'] in ('task_complete', 'turn_aborted')]
        if len(following) == 1:
            message = messages[following[0]]
            row.update(next_excerpt=text_of(message)[:180], next_tool_names=[c['function']['name'] for c in message.get('tool_calls', [])])
        row['category'], row['reason'] = classify_next(following, messages, originals, issue, prefix)
        if prefix:
            row['prefix_capture'] = prefix
        row.update(count_prior(index, raw_messages, messages, locator, originals, tree, prefix))
        yield row


def source_units(targets, config, missing_sources=None):
    entries = {e['name']: setup_entry(e, WORKSPACE/config['review_run'])
               for e in read_json(WORKSPACE/config['inventory_config'])['datasets']}
    grouped = defaultdict(list)
    for row in targets.values():
        grouped[(row['dataset'], row['trajectory_id'])].append(row)
    files, native, prefix_datasets = defaultdict(set), [], set()
    for ds, trajectory in sorted(grouped):
        entry = entries[ds]
        root = WORKSPACE/config['data_root']/entry['path']
        if entry['review_adapter'].startswith('native_'):
            native.append((ds, trajectory, root/trajectory))
        elif 'trajectory_field' in entry:
            prefix_datasets.add(ds)
        else:
            relative, number = trajectory.rsplit(':', 1)
            files[(ds, root/relative)].add(int(number))
    for number, (ds, trajectory, path) in enumerate(native, 1):
        entry = entries[ds]
        if not path.exists() and missing_sources is not None:
            for target in grouped[(ds, trajectory)]:
                missing_sources[target['id']] = f'missing source file: {path}'
            continue
        raw_records = list(iter_jsonl(path))
        with capture_normalized() as captures:
            list(formats.native_events(ds, WORKSPACE/config['data_root']/entry['path'], entry, path, entry['review_adapter']))
        require(len(captures) == 1, 'native source must yield one trajectory')
        yield dict(capture=captures[0], targets=grouped[(ds, trajectory)], family=entry['review_adapter'],
                   raw_records=raw_records, record=None, prefix=None)
        if number % 250 == 0:
            print(f'rebuilt {number}/{len(native)} native source files', flush=True)
    for (ds, path), indices in sorted(files.items()):
        entry = entries[ds]
        root = WORKSPACE/config['data_root']/entry['path']
        if not path.exists() and missing_sources is not None:
            for index in indices:
                trajectory = f'{path.relative_to(root).as_posix()}:{index}'
                for target in grouped[(ds, trajectory)]:
                    missing_sources[target['id']] = f'missing source file: {path}'
            continue
        found = set()
        selected = (((index, record) for _, index, record in formats.records(path, entry, max(indices)) if index in indices)
                    if entry['storage'] == 'tau2' else requested_records(path, indices))
        for index, record in selected:
            found.add(index)
            with capture_normalized() as captures:
                list(formats.record_events(ds, WORKSPACE/config['data_root']/entry['path'], entry, entry['review_adapter'], path, index, record))
            require(len(captures) == 1, 'record must yield one trajectory')
            capture = captures[0]
            yield dict(capture=capture, targets=grouped[(ds, capture['trajectory_id'])], family=entry['review_adapter'],
                       raw_records=(), record=record, prefix=None)
        if missing_sources is None:
            require(found == indices, f'missing requested source records: {path}')
        else:
            for index in indices-found:
                trajectory = f'{path.relative_to(root).as_posix()}:{index}'
                for target in grouped[(ds, trajectory)]:
                    missing_sources[target['id']] = f'missing source record: {path}:{index}'
        print(f'rebuilt {len(found)} records: {ds}/{path.name}', flush=True)
    for ds in sorted(prefix_datasets):
        entry = entries[ds]
        root = WORKSPACE/config['data_root']/entry['path']
        wanted = {trajectory for d, trajectory in grouped if d == ds}
        longest, supervised = {}, defaultdict(set)
        for path in sorted(root.glob(entry['glob'])):
            for number, record in iter_jsonl(path):
                trajectory = record[entry['trajectory_field']]
                if trajectory not in wanted:
                    continue
                messages = record[entry['messages_field']]
                require(record['n_messages'] == len(messages) and record[entry['target_field']] == len(messages)-1,
                        'invalid cumulative prefix boundary')
                supervised[trajectory].add(record[entry['target_field']])
                if trajectory in longest:
                    previous = longest[trajectory][2]
                    overlap = min(len(previous['messages']), len(messages))
                    require(previous['messages'][:overlap] == messages[:overlap], 'cumulative source prefixes differ')
                    require(all(previous[k] == record[k] for k in ('original_n_messages', 'assistant_steps')), 'prefix completion metadata differs')
                    if len(previous['messages']) >= len(messages):
                        continue
                longest[trajectory] = (path, number, record)
        if missing_sources is None:
            require(longest.keys() == wanted, 'missing cumulative source trajectory')
        else:
            for trajectory in wanted-longest.keys():
                for target in grouped[(ds, trajectory)]:
                    missing_sources[target['id']] = f'missing cumulative source trajectory: {ds}/{trajectory}'
        # Match source-file traversal for representative examples. Classification
        # and the state ID are independent of this presentation order.
        for trajectory in sorted(longest, key=lambda key: longest[key][:2]):
            path, number, record = longest[trajectory]
            require({r['turn_index'] for r in grouped[(ds, trajectory)]} <= supervised[trajectory], 'target not supervised in a saved prefix')
            prefix = dict(source_fully_captured=record['n_messages'] == record['original_n_messages'] and record['assistant_step'] == record['assistant_steps'],
                          known_continuation=record['trace_part'] < record['trace_parts'], continuation=record['continuation'],
                          **{k: record[k] for k in ('n_messages', 'original_n_messages', 'trace_part', 'trace_parts')},
                          longest_source=f'{path}:{number}')
            with capture_normalized() as captures:
                list(formats.record_events(ds, root, entry, entry['review_adapter'], path, number, record))
            require(len(captures) == 1, 'prefix must yield one trajectory')
            yield dict(capture=captures[0], targets=grouped[(ds, trajectory)], family=entry['review_adapter'],
                       raw_records=(), record=record, prefix=prefix)


def field_evidence(raw, fields, source):
    value = raw
    for field in fields:
        if value is None:
            return dict(source=source, field='.'.join(map(str, fields)), present=False)
        if isinstance(field, int):
            require(isinstance(value, list), f'invalid terminal metadata container: {source} {fields}')
            if len(value) <= field:
                return dict(source=source, field='.'.join(map(str, fields)), present=False)
            value = value[field]
        else:
            require(isinstance(value, dict), f'invalid terminal metadata container: {source} {fields}')
            if field not in value:
                return dict(source=source, field='.'.join(map(str, fields)), present=False)
            value = value[field]
    require(value is None or isinstance(value, str), f'invalid terminal reason: {source} {fields}')
    return dict(source=source, field='.'.join(map(str, fields)), present=True, value=value)


def termination_evidence(unit):
    """Map each normalized current message to its own raw finish field, never an episode outcome."""
    capture, family, record = unit['capture'], unit['family'], unit['record']
    messages, source = capture['messages'], capture['context']['source']
    mapping = {}
    if family.startswith('native_'):
        raw = dict(unit['raw_records'])
        for target in unit['targets']:
            index = target['turn_index']
            if not 0 <= index < len(messages):
                continue
            message = messages[index]
            if family == 'native_claude':
                nodes = message.get('_uuids', frozenset())
                records = [(line, row) for line, row in unit['raw_records'] if row.get('uuid') in nodes]
                mapping[index] = [field_evidence(row, ('message', 'stop_reason'), f'{source}:{line}') for line, row in records]
            else:
                line = int(message['_source'].rsplit(':', 1)[1])
                fields = ('message', 'stopReason') if family == 'native_pi' else ('payload', 'finish_reason')
                mapping[index] = [field_evidence(raw[line], fields, f'{source}:{line}')]
        return mapping
    if family == 'anthropic':
        cursor = 0
        for index, raw in enumerate(json_value(record['trajectory'])):
            normalized = formats.anthropic_messages([raw])
            if raw['role'] == 'assistant':
                mapping[cursor] = [field_evidence(raw, ('stop_reason',), f'{source}#trajectory[{index}]')]
            cursor += len(normalized)
    elif family == 'finance':
        for index, line in enumerate(record['trace_jsonl'].splitlines(), 1):
            raw = json.loads(line)
            if raw['type'] == 'model':
                mapping[index] = [field_evidence(raw, ('finish_reason',), f'{source}#trace_jsonl:{index}')]
    elif family == 'atif':
        cursor = 0
        for index, raw in enumerate(record['steps']):
            if raw['source'] == 'agent':
                mapping[cursor] = [field_evidence(raw, ('extra', 'stop_reason'), f'{source}#steps[{index}]')]
            cursor += 1
            if raw.get('observation') is not None:
                cursor += len(raw['observation']['results'])
    elif family == 'tau2':
        cursor = int(isinstance(record['_info']['environment_info']['policy'], str))
        for index, raw in enumerate(record['messages']):
            normalized = formats.tau2_messages(dict(_info=dict(environment_info=dict(policy=None)), messages=[raw]))
            if raw['role'] == 'assistant':
                mapping[cursor] = [field_evidence(raw, ('raw_data', 'choices', 0, 'finish_reason'), f'{source}#messages[{index}]')]
            cursor += len(normalized)
    elif family in ('openai', 'toolathlon', 'hermes'):
        if family == 'hermes':
            raw_messages, offset, raw_name = record['conversations'], 0, 'conversations'
        else:
            raw_messages, offset, raw_name = json_value(record['messages']), int(family == 'toolathlon'), 'messages'
        for index, raw in enumerate(raw_messages):
            role = raw['from'] if family == 'hermes' else raw['role']
            if role in ('assistant', 'gpt'):
                mapping[index+offset] = [field_evidence(raw, ('finish_reason',), f'{source}#{raw_name}[{index}]')]
    else:
        raise ValueError(f'unsupported termination evidence format: {family}')
    return mapping


def termination_status(evidence):
    require(evidence, 'no raw current-message mapping for terminal metadata')
    values = {row['value'] for row in evidence if row['present'] and row['value'] is not None}
    if len(values) > 1:
        return 'conflict', None
    if values:
        return 'known', next(iter(values))
    return ('null' if any(row['present'] for row in evidence) else 'missing'), None


def policy_flags(row, evidence):
    status, value = termination_status(evidence)
    d1 = row['category'] == 'intermediate_tool_continuation'
    d2 = status == 'known' and value in ('error', 'length')
    sensitive = row['raw_phase'] == dict(present=True, value='commentary') or row['reason'] == 'assistant_text_next'
    return dict(d1=d1, d2=d2, excluded=d1 or d2, termination_status=status,
                termination_value=value, sensitivity_extra=not (d1 or d2) and sensitive)
