"""Role-tagged Jev input; budget the actual decider prompt, never its context only."""
import json
import random
from collections import Counter
from copy import deepcopy
from decider.data import Example, Q
from decider.prompt import build
from dataset_tool_inventory.review_aggregate import truncate_history, IMAGE_BLOCK_TYPES, IMAGE_PLACEHOLDER
from dataset_tool_inventory.adapters.review_formats import TAU2_GREETING

QUESTION = 'Which action should the agent take next?'
RESPOND = 'RESPOND: reply to the user directly without calling a tool'


def initial_prefix(history):
    """Return the boundary before the first non-framework assistant message."""
    skipped = 0
    for i, message in enumerate(history):
        if message['role'] == 'assistant':
            if message.get('content') == TAU2_GREETING:
                skipped += 1
            else:
                return i, skipped
    return len(history), skipped


class HistoryTruncator:
    """Reuse only the identical prefix; truncation is independent per message."""
    def __init__(self, tokenizer, head, tail):
        self.tokenizer, self.head, self.tail = tokenizer, head, tail
        self.raw, self.messages, self.counts = [], [], []

    def __call__(self, history):
        shared = 0
        for old, new in zip(self.raw, history):
            if old != new:
                break
            shared += 1
        self.raw = self.raw[:shared]
        self.messages = self.messages[:shared]
        self.counts = self.counts[:shared]
        for message in history[shared:]:
            # Snapshot also handles a source adapter mutating a reused object.
            snapshot = deepcopy(message)
            rendered, counts = truncate_visible_history([snapshot], self.tokenizer, self.head, self.tail)
            self.raw.append(snapshot)
            self.messages.append(rendered[0])
            self.counts.append(counts)
        counts = Counter()
        for item in self.counts:
            counts.update(item)
        return list(self.messages), counts


def text(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError('content must be text or text-block list')
    pieces = []
    for block in content:
        if block['type']=='tool_reference':
            if not isinstance(block['tool_name'],str) or not block['tool_name']:
                raise ValueError('tool_reference requires nonempty canonical tool_name')
            pieces.append('[tool_reference] '+json.dumps(block,ensure_ascii=False))
            continue
        if block['type'] not in {'text', 'input_text', 'output_text'}:
            raise ValueError(f'unsupported visible content block: {block["type"]}')
        if not isinstance(block['text'], str):
            raise ValueError('text block text must be string')
        pieces.append(block['text'])
    return '\n'.join(pieces)


def truncate_visible_history(history, tokenizer, head, tail):
    normalized = []; images = 0
    for message in history:
        if message['role']=='tool' and isinstance(message.get('content'),list):
            blocks = []
            for block in message['content']:
                if block['type'] in IMAGE_BLOCK_TYPES:
                    blocks.append({'type':'text','text':IMAGE_PLACEHOLDER}); images += 1
                else:
                    blocks.append(block)
            message = dict(message,content=text(blocks))
        normalized.append(message)
    result, counts = truncate_history(normalized, tokenizer, head, tail)
    counts['images_replaced'] += images
    return result, counts


def render_history(history, omitted=0, omission_index=0):
    lines = []
    for index, message in enumerate(history):
        if omitted and index == omission_index:
            lines.append(f'[... {omitted} earlier messages omitted ...]')
        role = message['role']
        if role not in {'system', 'developer', 'user', 'assistant', 'tool'}:
            raise ValueError(f'unsupported role: {role}')
        label = f'tool: {message["name"]}' if role == 'tool' and 'name' in message else role
        content = text(message.get('content'))
        if content:
            lines.append(f'[{label}] {content}')
        for call in message.get('tool_calls', []):
            f = call['function']
            # Truncated historical argument strings may intentionally cease to be JSON.
            argument = f['arguments'] if 'arguments' in f else None
            encoded = argument if isinstance(argument, str) else json.dumps(argument, ensure_ascii=False)
            lines.append(f'[assistant → tool_call] {f["name"]}({encoded})')
    if omitted and omission_index == len(history):
        lines.append(f'[... {omitted} earlier messages omitted ...]')
    return '\n'.join(lines)


def option(tool, description_chars=300):
    name = tool['name']
    if not isinstance(name, str) or not name:
        raise ValueError('tool name must be nonempty string')
    if ': ' in name or ' (params:' in name:
        raise ValueError('tool name contains a reserved option delimiter')
    out = name
    if 'description' in tool:
        if not isinstance(tool['description'], str):
            raise ValueError('tool description must be string')
        if tool['description']:
            out += ': ' + tool['description'][:description_chars]
    if 'parameters' in tool:
        schema = tool['parameters']
        if not isinstance(schema, dict):
            raise ValueError('tool parameters must be object')
        # JSON Schema properties is optional; an absent properties means no named params.
        if 'properties' in schema:
            if not isinstance(schema['properties'], dict):
                raise ValueError('schema.properties must be object')
            out += ' (params: ' + ', '.join(schema['properties']) + ')'
    return out


class CanonicalOrder(random.Random):
    def shuffle(self, sequence):
        pass


def prompt_item(context, question, options, gold, tokenizer, rng=None):
    ex = Example(context, [Q(question, options, gold[0])], task='')
    # This bound cannot cut Context; enforce the complete final prompt afterwards.
    budget = len(tokenizer.encode('Context:\n' + context, add_special_tokens=False))
    return build(ex, tokenizer, rng if rng is not None else CanonicalOrder(0),
                 max_options=255, max_ctx_tokens=max(1, budget), layout='state_first')


def render_event(event, tokenizer, max_len=32768, description_chars=300, head=2048, tail=2048, history_truncator=None, length_margin=0, *, allow_duplicate_candidates=False):
    if type(length_margin) is not int or not 0 <= length_margin < max_len:
        raise ValueError('length_margin must be a nonnegative integer smaller than max_len')
    budget = max_len - length_margin
    tools = event.metadata['_candidate_definitions']
    names = [t['name'] for t in tools]
    if allow_duplicate_candidates and event.action_kind != 'RESPOND':
        raise ValueError('duplicate candidates are only supported with the inference RESPOND placeholder')
    if (not allow_duplicate_candidates and len(names) != len(set(names))) or 'RESPOND' in names:
        raise ValueError('duplicate candidate identity or reserved RESPOND tool name')
    if tuple(names) != event.candidate_tools:
        raise ValueError('candidate definitions differ from candidate names')
    if len(tools) + 1 > 255:
        return None, 'too_many_options'
    options = [option(t, description_chars) for t in tools] + [RESPOND]
    kind = event.action_kind
    if kind == 'RESPOND':
        if event.action_names:
            raise ValueError('RESPOND cannot carry tool labels')
        gold = [len(names)]
    elif kind in {'TOOL', 'TOOL_SET'}:
        if not event.action_names or set(event.action_names) - set(names):
            raise ValueError('gold action missing from candidates')
        gold = [names.index(n) for n in event.action_names]
        if (kind == 'TOOL' and len(gold) != 1) or (kind == 'TOOL_SET' and len(gold) < 2):
            raise ValueError('action kind and distinct gold count disagree')
    else:
        raise ValueError(f'ineligible action kind: {kind}')
    if any('arguments' not in c['function'] for m in event.metadata['_history'] for c in m.get('tool_calls', [])):
        return None, 'missing_historical_tool_arguments'
    history, counts = (history_truncator(event.metadata['_history']) if history_truncator is not None
                       else truncate_visible_history(event.metadata['_history'], tokenizer, head, tail))
    boundary, skipped = initial_prefix(history)
    counts['skipped_framework_greeting'] = skipped
    if not any(m['role'] == 'user' for m in history[:boundary]):
        return None, 'missing_initial_request'
    protected = {i for i, m in enumerate(history) if m['role'] == 'system' or i < boundary}
    units = []; i = boundary
    while i < len(history):
        if i in protected:
            i += 1; continue
        unit = [i]; i += 1
        if history[unit[0]]['role'] == 'assistant':
            while i < len(history) and history[i]['role'] == 'tool':
                unit.append(i); i += 1
        units.append(unit)
    def attempt(k):
        removed = {index for unit in units[:k] for index in unit}
        kept = [m for i, m in enumerate(history) if i not in removed]
        context = render_history(kept, len(removed), boundary)
        item = prompt_item(context, QUESTION, options, gold, tokenizer)
        return context, len(item['ids']), len(removed)
    context, n_tokens, dropped = attempt(0)
    if n_tokens > budget:
        if not units or attempt(len(units))[1] > budget:
            return None, 'over_length_after_drop'
        low, high = 1, len(units)
        while low < high:
            middle = (low + high) // 2
            if attempt(middle)[1] <= budget: high = middle
            else: low = middle + 1
        context, n_tokens, dropped = attempt(low)
    assert len(prompt_item(context, QUESTION, options, gold, tokenizer)['ids']) == n_tokens <= budget
    return {'context': context, 'question': QUESTION, 'options': options, 'gold': gold,
            'n_prompt_tokens': n_tokens, 'messages_dropped': dropped,
            'truncation_counts': dict(counts)}, None
