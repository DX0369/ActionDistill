"""Narrow positive evidence for reply request origin; no task-completion claims."""
import json
import re


HERMES_FINAL = ("You've reached the maximum number of tool-calling iterations allowed. "
                "Please provide a final response summarizing what you've found and accomplished "
                "so far, without calling any more tools.")

ZCLAW_BACKGROUND_DIRECTIVE = (
    'Summarize this naturally for the user. Keep it brief (1-2 sentences). Flow it into the conversation naturally.\n'
    'Do not mention technical details like tokens, stats, or that this was a background task.\n'
    'You can respond with NO_REPLY if no announcement is needed (e.g., internal task with no user-facing result).')
ZCLAW_INTERNAL_HEADER = ('OpenClaw runtime context (internal):\n'
    'This context is runtime-generated, not user-authored. Keep internal details private.')
ZCLAW_INTERNAL_DIRECTIVE = (
    'A completed subagent task is ready for user delivery. Convert the result above into your normal assistant '
    "voice and send that user-facing update now. Keep this internal context private (don't mention "
    'system/log/stats/session details or announce type).')


def zclaw_framework_message(content):
    """Two reviewed complete notification templates; quoted mentions do not match."""
    text = exact_text(content)
    if text is None:
        return None
    text = re.sub(r'^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\] ', '', text)
    if (text.startswith('A background task "') and '" just completed successfully.' in text
            and text.endswith(ZCLAW_BACKGROUND_DIRECTIVE)):
        return dict(rule='zclaw_background_task_completion',
                    markers=['A background task "', '" just completed successfully.', ZCLAW_BACKGROUND_DIRECTIVE])
    if (text.startswith(ZCLAW_INTERNAL_HEADER) and '\n[Internal task completion event]\n' in text
            and text.endswith(ZCLAW_INTERNAL_DIRECTIVE)):
        return dict(rule='zclaw_internal_task_completion',
                    markers=[ZCLAW_INTERNAL_HEADER, '[Internal task completion event]', ZCLAW_INTERNAL_DIRECTIVE])
    return None


def hermes_budget_pressure(content):
    """Diagnostic only. Non-exhausted pressure never changes reply eligibility."""
    evidence, issues = [], []
    for match in re.finditer(r'<tool_response>(.*?)</tool_response>', content, re.S):
        try:
            response = json.loads(match[1])
        except json.JSONDecodeError:
            issues.append('unparseable_tool_response')
            continue
        if not isinstance(response, dict) or not isinstance(response.get('content'), dict):
            continue
        payload = response['content']
        if '_budget_warning' not in payload:
            continue
        warning = payload['_budget_warning']
        if not isinstance(warning, str):
            raise ValueError('Hermes content._budget_warning must be a string when present')
        # Reviewed BUDGET and BUDGET WARNING envelopes use these two spellings.
        budget = re.search(r'\bIteration (\d+)/(\d+)\. (?:Only )?(\d+) (?:iterations|iteration\(s\)) left\b', warning)
        if budget is None:
            issues.append('unrecognized_budget_warning')
            continue
        used, limit, remaining = map(int, budget.groups())
        if not (0 < used < limit and remaining == limit-used):
            continue
        if 'Provide your final response NOW. No more tool calls unless absolutely critical.' in warning:
            evidence.append(dict(rule='hermes_nonexhausted_final_now', warning=warning))
        elif 'Start consolidating your work.' in warning:
            evidence.append(dict(rule='hermes_nonexhausted_consolidation', warning=warning))
    return evidence, issues


def reply_priority(origin, *, framework_boundary=False):
    """V2 preference uses verifiable turn boundaries, independently of provenance.

    The caller supplies a C1-normalized RESPOND. Unknown request origin remains
    unknown; a missing finish field is not evidence of successful completion.
    """
    for field in ('category', 'next_event', 'termination_status', 'termination_value',
                  'call_count_status', 'user_boundary_source', 'prior_calls'):
        origin[field]  # Required canonical evidence fields, including explicit nulls.
    if origin['category'] not in {'verified_reply', 'framework_forced', 'unknown'}:
        raise ValueError('invalid reply origin category')
    if origin['next_event'] not in {'user_turn_end', 'trajectory_end', 'unjudgeable'}:
        raise ValueError('invalid reply successor')
    status = origin['termination_status']
    if status not in {'known', 'missing', 'null', 'conflict'}:
        raise ValueError('invalid reply termination status')
    if origin['call_count_status'] not in {'known', 'parent_chain_missing_before_user',
                                         'no_user_boundary_in_record', 'continued_source_part_missing_prior_turn'}:
        raise ValueError('invalid prior-boundary status')
    if origin['category'] == 'framework_forced':
        return dict(eligible=False, rule='explicit_framework_finalization')
    if origin['call_count_status'] != 'known':
        return dict(eligible=False, rule='unverifiable_turn_boundary')
    if (not isinstance(origin['user_boundary_source'], str) or not origin['user_boundary_source']
            or type(origin['prior_calls']) is not int or origin['prior_calls'] < 0):
        raise ValueError('known reply boundary requires source and nonnegative prior_calls')
    if origin['next_event'] == 'unjudgeable':
        return dict(eligible=False, rule='unverifiable_successor')
    if status == 'conflict':
        return dict(eligible=False, rule='termination_evidence_conflict')
    # Preserve the accepted D2 spelling and scope; do not infer extra stop labels.
    if status == 'known' and origin['termination_value'] in {'error', 'length'}:
        raise ValueError('C1 reply priority received a D2-abnormal state')
    if framework_boundary:
        return dict(eligible=False, rule='zclaw_framework_boundary')
    return dict(eligible=True, rule='verifiable_reply_boundary')


def exact_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
            isinstance(part, dict) and part['type'] in ('text', 'input_text')
            and isinstance(part['text'], str) for part in content):
        return ''.join(part['text'] for part in content)
    return None  # Non-text inputs cannot be proved equal by this rule.


def classify_origin(view, index, prior, termination_status):
    """The caller has checked C1 gold, successor, source ID and teacher identity."""
    result = dict(category='unknown', rule='unverified_input_origin',
                  user_boundary_source=None, evidence=[])
    if prior['call_count_status'] != 'known':
        return dict(result, rule='unknown_boundary:'+prior['call_count_status'])
    boundary_source = prior['user_boundary_source']
    boundaries = [i for i in range(index) if view.locator(i) == boundary_source]
    if len(boundaries) != 1:
        raise ValueError('known reply boundary does not map to one source message')
    boundary = boundaries[0]
    result['user_boundary_source'] = boundary_source
    family, record = view.unit['family'], view.unit['record']
    content = exact_text(view.raw[boundary]['content'])
    if family == 'hermes' and content == HERMES_FINAL:
        raw = record['conversations']
        if len(raw) != len(view.raw) or raw[boundary]['from'] != 'human' or raw[boundary]['value'] != content:
            raise ValueError('Hermes source/normalized boundary differs')
        if boundary > 0 and raw[boundary-1]['from'] == 'tool':
            for match in re.finditer(r'<tool_response>(.*?)</tool_response>', raw[boundary-1]['value'], re.S):
                try:
                    response = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return dict(result, rule='hermes_budget_warning_unparseable')
                if not isinstance(response, dict) or not isinstance(response.get('content'), dict):
                    continue
                warning = response['content'].get('_budget_warning')
                if not isinstance(warning, str):
                    continue
                exhausted = re.search(r'\bIteration (\d+)/(\d+)\. Only 0 iteration\(s\) left\b', warning)
                if exhausted and int(exhausted[1]) == int(exhausted[2]) > 0:
                    return dict(result, category='framework_forced', rule='hermes_budget_exhausted_summary',
                                evidence=[dict(source=boundary_source, field='value', value=HERMES_FINAL),
                                          dict(source=view.locator(boundary-1), field='content._budget_warning', value=warning)])
        return dict(result, rule='hermes_final_instruction_without_exhaustion_evidence')
    if termination_status not in ('known', 'missing', 'null'):
        return dict(result, rule='unverified_termination:'+termination_status)
    if prior['framework_user_messages_ignored']:
        return dict(result, rule='framework_input_inside_turn')
    if content is None:
        return dict(result, rule='nontext_boundary')
    if family in ('toolathlon', 'finance'):
        if family == 'finance':
            task, field = record['instructions'], 'instructions'
        else:
            config = json.loads(record['config']) if isinstance(record['config'], str) else record['config']
            task, field = config['task_str'], 'config.task_str'
        if not isinstance(task, str) or not task:
            raise ValueError(f'{family}: task input must be a nonempty string')
        if content == task:
            return dict(result, category='verified_reply', rule='task_field_exact_'+family,
                        evidence=[dict(source=view.capture['context']['source'], field=field,
                                       exact_match=True)])
        return dict(result, rule='task_field_boundary_mismatch')
    if family == 'native_codex':
        line = int(boundary_source.rsplit(':', 1)[1])
        raw = dict(view.unit['raw_records'])
        item = raw[line]
        if (item['type'] != 'response_item' or item['payload']['type'] != 'message'
                or item['payload']['role'] != 'user'):
            raise ValueError('Codex normalized user boundary differs from original')
        if exact_text(item['payload']['content']) != content:
            raise ValueError('Codex original boundary text differs')
        # The native event must immediately follow the message, at the same timestamp.
        # Missing/reordered/ambiguous event pairs stay unknown, never text-matched elsewhere.
        lines = sorted(raw)
        following = lines[lines.index(line)+1:lines.index(line)+2]
        if following:
            event = raw[following[0]]
            if event['type'] == 'event_msg' and event['payload']['type'] == 'user_message':
                if any('timestamp' not in row or not isinstance(row['timestamp'],str)
                       or not row['timestamp'].strip() for row in (item,event)):
                    return dict(result, rule='codex_pair_time_unverified')
                if event['payload']['message'] == content and event['timestamp'] == item['timestamp']:
                    return dict(result, category='verified_reply', rule='codex_user_event_exact',
                                evidence=[dict(source=boundary_source, field='payload.content', exact_match=True),
                                          dict(source=boundary_source.rsplit(':', 1)[0]+f':{following[0]}',
                                               field='payload.message', exact_match=True)])
        return dict(result, rule='codex_user_event_pair_unverified')
    return result
