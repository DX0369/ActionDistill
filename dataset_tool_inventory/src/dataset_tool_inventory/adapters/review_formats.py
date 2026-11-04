"""Explicit format adapters for review, preserving missing provenance as issues."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..common import iter_jsonl
from ..schema import DecisionEvent, IssueEvent
from .review_common import (
    definitions,
    emit_trajectory,
    field,
    json_value,
    outcome,
    records,
)

_CALL_RE = re.compile(r"(?:^\[|,\s*)([^\[\]\(\),]+?)\s*\(")
_JSON_LIST_MARKERS = (
    "Here is a list of functions in JSON format that you can invoke:",
    "以下是可以调用的工具的JSON格式列表:",
)
_CUSTOM_MARKER = "Here are the tools you can use:"

# tau2 orchestrator.DEFAULT_FIRST_AGENT_MESSAGE: fixed text, not a teacher step.
TAU2_GREETING = "Hi! How can I help you today?"
# Codex built-in tool items; the tool name is the item type without "_call".
CODEX_BUILTIN_CALLS = {"web_search_call": "web_search", "tool_search_call": "tool_search",
                       "image_generation_call": "image_generation"}


def context(entry, source, record=None):
    record = {} if record is None else record
    model = field(record, entry["model_field"]) if "model_field" in entry else entry.get("model_constant")
    return {"source": source, "model": model, "harness": field(record, entry["harness_field"]) if "harness_field" in entry else entry["harness"],
            "domain": field(record, entry["domain_field"]) if "domain_field" in entry else None,
            "task_id": field(record, entry["task_field"]) if "task_field" in entry else None,
            "outcome": outcome(record, entry.get("outcome")),
            "candidate_complete": entry.get("candidate_complete", False),
            "candidate_source": entry.get("candidate_source", "prior_calls"),
            "candidate_level": entry.get("candidate_level"),
            "model_source": entry.get("model_field", entry.get("model_source")),
            "strip_think": entry.get("strip_think", True),
            "harness_tool_sets": entry.get("harness_tool_sets"),
            "quality_flags": list(entry.get("quality_flags", []))}


CLAUDE_META = ("_source", "_after_compaction", "_uuids", "_abandoned")


def anthropic_messages(raw):
    messages = []
    for item in raw:
        role = field(item, "role")
        content = field(item, "content")
        # An explicit null is an empty message (counted as UNKNOWN if it is the action).
        if content is None: content = []
        if isinstance(content, str):
            messages.append({"role": role, "content": content,
                             **({"_model": item["model"]} if "model" in item else {}),
                             **({k: item[k] for k in CLAUDE_META if k in item})})
            continue
        if not isinstance(content, list):
            raise ValueError("Anthropic content must be a block list")
        text = []; calls = []; results = []
        for part in content:
            kind = field(part, "type")
            if kind in {"thinking", "redacted_thinking"}:
                continue
            if kind == "tool_use":
                calls.append({"id": field(part, "id"), "function": {
                    "name": field(part, "name"), "arguments": field(part, "input")}})
            elif kind == "tool_result":
                results.append({"role": "tool", "tool_call_id": field(part, "tool_use_id"),
                                "content": field(part, "content")})
            else:
                text.append(part)
        if role == "assistant" or text:
            m = {"role": role, "content": text}
            if calls: m["tool_calls"] = calls
            if "model" in item: m["_model"] = item["model"]
            m.update({k: item[k] for k in CLAUDE_META if k in item})
            messages.append(m)
        if "_uuids" in item:
            for result in results: result["_uuids"] = item["_uuids"]
        messages.extend(results)
    return messages


def hermes_messages(raw):
    messages = []
    roles = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    for row in raw:
        role = field(row, "from")
        if role not in roles: raise ValueError(f"unknown ShareGPT role: {role}")
        text = field(row, "value")
        if not isinstance(text, str): raise ValueError("ShareGPT value must be string")
        if role == "gpt":
            # Reasoning may quote <tool_call>; remove it before parsing calls.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            if "<think>" in text or "</think>" in text:
                raise ValueError("unclosed reasoning block")
            calls = []
            for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
                call = json.loads(match.group(1))
                calls.append({"function": {"name": field(call, "name"),
                                            "arguments": field(call, "arguments")}})
            remaining = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
            if "<tool_call>" in remaining or "</tool_call>" in remaining:
                raise ValueError("unclosed tool_call block")
            m = {"role": "assistant", "content": remaining}
            if calls: m["tool_calls"] = calls
        else:
            m = {"role": roles[role], "content": text}
        messages.append(m)
    return messages


def otel_messages(raw):
    result = []
    for item in raw:
        role = field(item, "role"); text = []; calls = []; results = []
        for part in field(item, "parts"):
            kind = field(part, "type")
            if kind in {"reasoning", "thinking", "reasoning_text"}: continue
            if kind == "text":
                text.append(field(part, "content"))
            elif kind == "tool_call":
                calls.append({"id": field(part, "id"), "function": {
                    "name": field(part, "name"), "arguments": field(part, "arguments")}})
            elif kind == "tool_call_response":
                results.append({"role": "tool", "tool_call_id": field(part, "id"),
                                "content": field(part, "result")})
            else: raise ValueError(f"unsupported OTel part: {kind}")
        if text or calls or role == "assistant":
            m = {"role": role, "content": "\n".join(text)}
            if calls: m["tool_calls"] = calls
            result.append(m)
        result.extend(results)
    return result


def claude_abandoned(records):
    """For each assistant message (first fragment uuid), the record uuids it
    cannot see: subtrees of siblings it did not follow at real branch points.

    A branch point is real when its children start two or more distinct turns
    (different assistant message ids, or user messages that are not tool
    results), e.g. a user editing and resending a message. Parallel tool calls
    (one message id, its fragments and their tool_result rows) are not branches.
    A broken parent chain stops the walk, so nothing above it is removed."""
    parent = {r["uuid"]: r.get("parentUuid") for r in records if r.get("uuid")}
    children = defaultdict(list)
    for r in records:
        if r.get("uuid"): children[r.get("parentUuid")].append(r)
    def turn_starts(rows):
        mids = {(r.get("message") or {}).get("id") for r in rows if r.get("type") == "assistant"}
        texts = [r for r in rows if r.get("type") == "user" and not any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in ((r.get("message") or {}).get("content") or []) if isinstance((r.get("message") or {}).get("content"), list))]
        return len(mids) + len(texts)
    real = {p for p, rows in children.items()
            if turn_starts([r for r in rows if r.get("type") in {"user", "assistant"}]) >= 2}
    if not real: return {}
    def subtree(u):
        out = set(); stack = [u]
        while stack:
            v = stack.pop(); out.add(v)
            stack.extend(r["uuid"] for r in children.get(v, []))
        return out
    by_uuid = {r["uuid"]: r for r in records if r.get("uuid")}
    def is_tool_result(r):
        content = (r.get("message") or {}).get("content")
        return isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    result = {}
    for r in records:
        if r.get("type") != "assistant" or not r.get("uuid"): continue
        ancestors = set(); u = r["uuid"]
        while u is not None and u in parent and u not in ancestors:
            ancestors.add(u); u = parent[u]
        points = real & (ancestors | ({None} if u is None else set()))
        own_ids = {(by_uuid[a].get("message") or {}).get("id") for a in ancestors if by_uuid[a].get("type") == "assistant"}
        hidden = set()
        for p in points:
            for child in children[p]:
                if child["uuid"] in ancestors: continue
                # Sibling turns only: parallel tool results and fragments of an
                # ancestor message stay visible.
                if child.get("type") == "user" and not is_tool_result(child) or (
                        child.get("type") == "assistant" and (child.get("message") or {}).get("id") not in own_ids):
                    hidden |= subtree(child["uuid"])
        if hidden: result[r["uuid"]] = frozenset(hidden)
    return result


def native_claude(path):
    """Group all fragments by message.id, freezing input at its first fragment;
    history follows the parentUuid tree (see claude_abandoned)."""
    records = []; session_id = None; flags = set()
    for line, item in iter_jsonl(path):
        if "sessionId" in item:
            if session_id is not None and session_id != item["sessionId"]:
                raise ValueError("multiple sessionId values in one native session file")
            session_id = item["sessionId"]
        records.append((line, item))
    if session_id is None: raise ValueError("native session missing sessionId")
    abandoned = claude_abandoned([item for _, item in records])
    messages = []; by_id = {}; compacted = False
    for line, item in records:
        if item.get("isSidechain") is True: flags.add("subagent_sidechain")
        if item.get("type") == "system" and item.get("subtype") == "compact_boundary":
            compacted = True
        if item.get("type") not in {"assistant", "user"}: continue
        m = dict(field(item, "message")); m["_source"] = f"{path}:{line}"
        if compacted: m["_after_compaction"] = True
        if item.get("uuid"): m["_uuids"] = frozenset({item["uuid"]})
        if item["type"] == "assistant":
            message_id = field(m, "id")
            model = field(m, "model")
            if message_id in by_id:
                old = by_id[message_id]
                if old["model"] != model: raise ValueError("same message.id has conflicting models")
                old["content"].extend(field(m, "content"))
                old["_uuids"] = old.get("_uuids", frozenset()) | m.get("_uuids", frozenset())
            else:
                if item.get("uuid") in abandoned: m["_abandoned"] = abandoned[item["uuid"]]
                by_id[message_id] = m; messages.append(m)
        else: messages.append(m)
    return session_id, anthropic_messages(messages), sorted(flags)


def native_pi(path, entry):
    messages = []; header = None; flags = set(); compacted = False
    for line, item in iter_jsonl(path):
        if item.get("type") == "session": header = item
        if item.get("type") in {"compaction", "branch_summary"}:
            compacted = True
        if item.get("type") != "message": continue
        raw = field(item, "message"); role = field(raw, "role")
        content = field(raw, "content"); text = []; calls = []
        if not isinstance(content, list): raise ValueError("pi content must be block list")
        m = {"role": "tool" if role == "toolResult" else role, "content": text,
             "_source": f"{path}:{line}"}
        if compacted: m["_after_compaction"] = True
        for part in content:
            kind = field(part, "type")
            if kind == "thinking": continue
            if kind == "toolCall":
                calls.append({"id": field(part, "id"), "function": {
                    "name": field(part, "name"), "arguments": field(part, "arguments")}})
            else: text.append(part)
        if role == "assistant": m["_model"] = raw.get("model")
        if "responseModel" in raw: m["_response_model"] = raw["responseModel"]
        if role == "toolResult":
            m["tool_call_id"] = field(raw, "toolCallId"); m["name"] = field(raw, "toolName")
        if calls: m["tool_calls"] = calls
        if role in {"system", "developer", "user"}:
            for part in text:
                if part.get("type") == "text" and "Available tools:" in part.get("text", ""):
                    block = part["text"].split("Available tools:", 1)[1].lstrip("\n").split("\n\n", 1)[0]
                    names = re.findall(r"^[-*]\s+([A-Za-z_][\w.-]*)\s*:\s*(.*)$", block, re.M)
                    if not names: raise ValueError("Available tools marker without parseable declaration")
                    m["_tools"] = [{"name": n, "description": desc} for n, desc in names]
        messages.append(m)
    if header is None: raise ValueError("native pi missing session header")
    return messages, header, sorted(flags)


def native_codex(path):
    messages = []; model = None; session_id = None; flags = ["response_item_consecutive_calls_grouped"]
    tools = []; compacted = False; pending_call = None
    for line, item in iter_jsonl(path):
        kind = field(item, "type")
        if kind == "session_meta": session_id = field(item, "payload.id")
        if kind == "turn_context":
            model = field(item, "payload.model"); pending_call = None
        if kind == "compacted":
            compacted = True; pending_call = None
        if kind != "response_item": continue
        payload = field(item, "payload"); source = f"{path}:{line}"
        part = field(payload, "type")
        if part == "reasoning": continue
        is_call = part in {"function_call", "custom_tool_call"}
        m = None
        if part == "tool_schema":
            tools.append({"name": field(payload, "name"), **field(payload, "schema")})
        elif part == "message":
            if payload.get("role") == "assistant" and payload.get("channel") == "analysis": continue
            m = {"role": field(payload, "role"), "content": field(payload, "content"), "_model": model}
        elif part in {"function_call", "custom_tool_call"}:
            arguments = field(payload, "arguments" if part == "function_call" else "input")
            m = {"role": "assistant", "content": None, "_model": model,
                 "tool_calls": [{"id": field(payload, "call_id"), "function": {
                     "name": field(payload, "name"), "arguments": arguments}}]}
        elif part in CODEX_BUILTIN_CALLS:
            call = {"function": {"name": CODEX_BUILTIN_CALLS[part], "arguments": {
                k: v for k, v in payload.items() if k not in {"type", "status", "metadata", "id", "call_id"}}}}
            if "call_id" in payload: call["id"] = payload["call_id"]
            m = {"role": "assistant", "content": None, "_model": model, "tool_calls": [call]}
        elif part in {"function_call_output", "custom_tool_call_output"}:
            m = {"role": "tool", "tool_call_id": field(payload, "call_id"), "content": field(payload, "output")}
        elif part == "tool_search_output":
            m = {"role": "tool", "tool_call_id": field(payload, "call_id"),
                 "content": json.dumps(field(payload, "tools"), ensure_ascii=False)}
        else:
            raise ValueError(f"unsupported canonical Codex response_item type: {part}")
        if m is not None:
            m["_source"] = source
            if compacted: m["_after_compaction"] = True
            if is_call and pending_call is not None:
                pending_call["tool_calls"].extend(m["tool_calls"])
            else:
                messages.append(m)
            pending_call = messages[-1] if is_call else None
        else:
            pending_call = None
    if session_id is None: raise ValueError("Codex missing session_meta.payload.id")
    return session_id, messages, flags, (definitions(tools) if tools else None)


def snapshots(root, entry):
    if "snapshot_json_block" in entry:
        source = root / entry["snapshot_json_block"]["file"]
        blocks = list(re.finditer(r"```json\s*\n(.*?)\n```", source.read_text(encoding="utf-8"), re.S))
        index = entry["snapshot_json_block"]["index"]
        if not 0 <= index < len(blocks): raise ValueError("missing configured dataset schema snapshot block")
        return definitions(json.loads(blocks[index].group(1)))
    if "snapshot_names" not in entry: return None
    return [{"name": name} for name in entry["snapshot_names"]]


@lru_cache(maxsize=None)
def tau2_official_tools(path):
    return json.loads(Path(path).read_text())


def harness_tool_sets(dataset, root, entry, limit, min_trajectories):
    """First pass: tools called in at least `min_trajectories` trajectories of
    each (harness, domain) form that harness's candidate set."""
    seen = defaultdict(lambda: defaultdict(set))
    for event in scan(dataset, root, entry, limit):
        if isinstance(event, DecisionEvent):
            for name in event.action_names:
                seen[(event.harness, event.domain)][name].add(event.trajectory_id)
    return {key: sorted(n for n, ids in tools.items() if len(ids) >= min_trajectories)
            for key, tools in seen.items()}


def scan(dataset, root, entry, limit=None):
    family = entry["review_adapter"]
    if family in {"excluded", "unresolved"}:
        yield IssueEvent(dataset, family + "_dataset", entry["reason"], source=str(root))
        return
    if family in {"native_claude", "native_pi", "native_codex"}:
        paths = sorted(root.glob(entry["glob"]))
        if not paths:
            yield IssueEvent(dataset, "missing_input", "configured glob matched no session files", source=str(root))
            return
        if limit is not None: paths = paths[:limit]
        units = ((str(path), lambda path=path: native_events(dataset, root, entry, path, family)) for path in paths)
    else:
        units = ((f"{path}:{index}", lambda path=path, index=index, record=record:
                  record_events(dataset, root, entry, family, path, index, record))
                 for path, index, record in records(root, entry, limit))
    for source, build in units:
        try:
            # A trajectory is released only after it parses completely, so a
            # late error never leaves a partial prefix in the statistics.
            events = list(build())
        except (ValueError, OSError) as exc:
            yield IssueEvent(dataset, "record_parse_error", str(exc), source=source)
            continue
        yield from events


def native_events(dataset, root, entry, path, family):
    source = str(path)
    ctx = context(entry, source); tools = snapshots(root, entry)
    from_snapshot = tools is not None
    trajectory_id = path.relative_to(root).as_posix()
    if family == "native_claude":
        session_id, messages, flags = native_claude(path)
        ctx["source_session_id"] = session_id
    elif family == "native_codex":
        session_id, messages, flags, session_tools = native_codex(path)
        ctx["source_session_id"] = session_id
        if session_tools is not None:
            # The session's own tool_schema items take precedence over the README snapshot.
            tools = session_tools
            from_snapshot = False
            # Collected across the trajectory, not captured for each request.
            ctx["candidate_level"] = "N"
            ctx["candidate_complete"] = False
            ctx["candidate_source"] = "recovered session tool_schema items (trajectory-level)"
    else:
        messages, header, flags = native_pi(path, entry)
        if "id" in header: ctx["source_session_id"] = header["id"]
        if "task_id" in header: ctx["task_id"] = header["task_id"]
        if "native_header_harness_field" in entry:
            ctx["harness"] = field(header, entry["native_header_harness_field"])
    if from_snapshot:
        # A recovered schema remains N even when it includes full parameters.
        ctx["candidate_level"] = "N"
    ctx["quality_flags"].extend(flags)
    yield from emit_trajectory(dataset, trajectory_id, messages, tools, ctx)


def toolace_call_names(value, declared):
    """Resolve names against the declaration so that names containing
    parentheses are not cut at their first "("."""
    names = []
    for match in _CALL_RE.finditer(value):
        start = match.start(1)
        full = next((n for n in sorted(declared, key=len, reverse=True)
                     if value.startswith(n, start) and value[start + len(n):].lstrip().startswith("(")), None)
        names.append(full if full is not None else match.group(1).strip())
    return names


def tau2_messages(record):
    messages = []
    policy = field(record, "_info.environment_info.policy")
    if isinstance(policy, str): messages.append({"role": "system", "content": policy})
    for index, raw in enumerate(field(record, "messages")):
        role = field(raw, "role")
        # User-simulator device tools (telecom) are invisible to the agent.
        if role == "tool" and field(raw, "requestor") == "user": continue
        m = {"role": role, "content": raw.get("content")}
        if role == "user" and raw.get("tool_calls") and m["content"] is None: continue
        if role == "assistant" and raw.get("tool_calls"):
            m["tool_calls"] = [{"id": field(c, "id"), "function": {
                "name": field(c, "name"), "arguments": field(c, "arguments")}}
                               for c in field(raw, "tool_calls")]
        if role == "tool": m["tool_call_id"] = field(raw, "id")
        if index == 0 and role == "assistant" and m["content"] == TAU2_GREETING and not raw.get("tool_calls"):
            m["_not_teacher_step"] = True
        messages.append(m)
    return messages


def record_events(dataset, root, entry, family, path, index, record):
    source = f"{path}:{index}"
    effective = dict(entry)
    if "file_model_map" in entry:
        relative = path.relative_to(root).as_posix()
        if relative not in entry["file_model_map"]: raise ValueError(f"unmapped file model: {relative}")
        effective["model_constant"] = entry["file_model_map"][relative]
    ctx = context(effective, source, record)
    if "source_record_id_field" in entry:
        original_id = field(record, entry["source_record_id_field"])
        if type(original_id) not in {str, int} or original_id == "":
            raise ValueError("canonical source record identifier must be a nonempty string or integer")
        ctx["source_record_id"] = original_id
        ctx["source_record_id_field"] = entry["source_record_id_field"]
    if "trajectory_field" in entry:
        identity = field(record, entry["trajectory_field"])
        if type(identity) not in {str, int} or identity == "":
            raise ValueError("canonical trajectory identifier must be a nonempty string or integer")
        trajectory_id = str(identity)
    else:
        trajectory_id = f"{path.relative_to(root).as_posix() if root.is_dir() else path.name}:{index}"
    if "tools_field" in entry:
        tools = definitions(field(record, entry["tools_field"]))
    else:
        tools = snapshots(root, entry)
        if tools is not None:
            ctx["candidate_level"] = "N"
    if family == "otel":
        for turn, span in enumerate(field(record, "spans")):
            attrs = field(span, "attributes")
            if attrs.get("gen_ai.operation.name") != "chat": continue
            if "gen_ai.input.messages" not in attrs:
                yield IssueEvent(dataset, "uncaptured_chat_span", "chat span has no captured input; no state emitted", trajectory_id, source + f"#span={turn}")
                continue
            # OTel attributes use literal dotted keys, not nested paths.
            def attr(key):
                if key not in attrs: raise ValueError(f"missing canonical OTel attribute: {key}")
                return attrs[key]
            inputs = json_value(attr("gen_ai.input.messages"))
            output = json_value(attr("gen_ai.output.messages"))
            if output is None:
                output = [{"role": "assistant", "parts": []}]
                yield IssueEvent(dataset, "missing_current_output", "chat span output is null", trajectory_id, source)
            out = otel_messages(output)
            if len(out) != 1 or out[0]["role"] != "assistant":
                raise ValueError("OTel output must contain exactly one assistant message")
            history = otel_messages(inputs)
            system = attrs.get("gen_ai.system_instructions")
            if system is not None:
                system = json_value(system) if not isinstance(system, str) or system.startswith(("[", "{")) else system
                if isinstance(system, list):
                    pieces = []
                    for part in system:
                        if field(part, "type") != "text" or not isinstance(field(part, "content"), str):
                            raise ValueError("unsupported OTel system instruction part")
                        pieces.append(part["content"])
                    system = "\n".join(pieces)
                if not isinstance(system, str):
                    raise ValueError("unsupported non-string OTel system instructions")
                history.insert(0, {"role": "system", "content": system})
            subctx = dict(ctx, model=attr("gen_ai.request.model"),
                          response_model=attrs.get("gen_ai.response.model"),
                          model_source="spans.attributes.gen_ai.request.model",
                          decision_index=turn,
                          source=source + f"#span={turn}")
            defs = json_value(attr("gen_ai.tool.definitions"))
            tools = definitions(defs) if defs is not None else None
            yield from emit_trajectory(dataset, trajectory_id, history + out, tools, subctx,
                                       target_index=len(history))
        return
    if family == "mcp":
        def react(text):
            value = json.loads(text)
            if "action" in value:
                action = field(value, "action")
                name = field(action, "server") + "::" + field(action, "tool")
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"function": {"name": name, "arguments": field(action, "arguments")}}]}
            if "answer" in value:
                return {"role": "assistant", "content": field(value, "answer")}
            raise ValueError("ReAct response has neither action nor answer")
        counter = 0
        for chunk in field(record, "trajectory"):
            for stored in field(chunk, "records"):
                data = field(stored, "data")
                if data.get("type") != "llm": continue
                response = field(data, "response")
                react_protocol = isinstance(response, str)
                history = []
                for message in field(data, "messages"):
                    if react_protocol and message.get("role") == "assistant" and isinstance(message.get("content"), str):
                        history.append(react(message["content"]))
                    else: history.append(message)
                try:
                    if isinstance(response, str):
                        current = react(response)
                    else:
                        choices = field(response, "choices")
                        if not isinstance(choices, list) or len(choices) != 1:
                            raise ValueError("captured FunctionCall response must have exactly one choice")
                        current = field(choices[0], "message")
                        if not isinstance(current, dict) or current.get("role") != "assistant":
                            raise ValueError("captured response message must have assistant role")
                except ValueError as exc:
                    current = {"role": "assistant", "content": None}
                    yield IssueEvent(dataset, "invalid_current_response", str(exc), trajectory_id, source)
                subctx = dict(ctx, model=field(data, "config.model_name"), decision_index=counter,
                              model_source="trajectory.records.data.config.model_name",
                              source=source + f"#llm={counter}")
                yield from emit_trajectory(dataset, trajectory_id, history + [current], None,
                                           subctx, target_index=len(history))
                counter += 1
        return
    if family == "hermes":
        messages = hermes_messages(field(record, "conversations"))
    elif family == "anthropic":
        messages = anthropic_messages(json_value(field(record, "trajectory")))
    elif family == "finance":
        messages = [{"role": "user", "content": field(record, "instructions")}]
        for line in field(record, "trace_jsonl").splitlines():
            event = json.loads(line)
            if event["type"] == "model":
                m = {"role": "assistant", "content": event.get("content")}
                calls = event.get("tool_calls", [])
                if calls: m["tool_calls"] = [{"id": field(c, "id"), "function": {
                    "name": field(c, "name"), "arguments": field(c, "arguments")}} for c in calls]
                messages.append(m)
            elif event["type"] == "tool":
                messages.append({"role": "tool", "tool_call_id": field(event, "tool_call_id"),
                                 "name": field(event, "name"), "content": field(event, "result")})
            else: raise ValueError(f"unsupported finance event type: {event['type']}")
    elif family == "toolathlon":
        config = json_value(field(record, "config"))
        ctx["outcome"] = outcome(json_value(field(record, "task_status")), entry["task_outcome"])
        tools = definitions(field(json_value(field(record, "tool_calls")), "tools"))
        messages = json_value(field(record, "messages"))
        prompt = field(config, "system_prompts.agent")
        if not isinstance(prompt, str): raise ValueError("agent system prompt must be string")
        messages = [{"role": "system", "content": prompt}] + messages
    elif family == "tau2":
        messages = tau2_messages(record)
        ctx["task_scenario_hash"] = field(record, "_task_scenario_hash")
        declared = field(record, "_info.environment_info.tool_defs")
        tools = definitions(declared) if declared is not None else None
        if tools is None and "tau2_official_tools" in entry:
            official = tau2_official_tools(entry["tau2_official_tools"])
            if ctx["domain"] in official["domains"]:
                # Results files omit tool_defs; the domain tools are fixed in tau2 itself.
                tools = definitions(official["domains"][ctx["domain"]])
                ctx["candidate_source"] = f"tau2 official domain tools @ {official['commit']}"
    elif family == "atif":
        messages = []
        agent = field(record, "agent")
        tools = definitions(agent["tool_definitions"]) if "tool_definitions" in agent else None
        for step in field(record, "steps"):
            origin = field(step, "source")
            if origin not in {"agent", "user", "system"}: raise ValueError(f"unsupported ATIF source: {origin}")
            m = {"role": "assistant" if origin == "agent" else origin,
                 "content": field(step, "message")}
            if origin == "agent":
                m["_model"] = field(step, "model_name")
                if "tool_calls" in step and step["tool_calls"] is not None:
                    m["tool_calls"] = [{"id": field(c, "tool_call_id"), "function": {
                        "name": field(c, "function_name"), "arguments": field(c, "arguments")}}
                                       for c in step["tool_calls"]]
            messages.append(m)
            observation = step.get("observation")
            if observation is not None:
                for result in field(observation, "results"):
                    messages.append({"role": "tool", "tool_call_id": field(result, "source_call_id"),
                                     "content": field(result, "content")})
    elif family == "toolace":
        system = field(record, "system")
        marker = next((m for m in _JSON_LIST_MARKERS if m in system), None)
        if marker is None:
            # Per-record custom formats and missing declarations are counted, not converted.
            issue = "custom_format_tool_declaration" if _CUSTOM_MARKER in system else "missing_tool_declaration"
            yield IssueEvent(dataset, issue, "no standard JSON tool list; record not converted", trajectory_id, source)
            return
        start = system.find("[", system.find(marker))
        tools = definitions(json.JSONDecoder().raw_decode(system[start:])[0])
        declared_names = [t["name"] for t in tools]
        messages = [{"role": "system", "content": system}]
        for row in field(record, "conversations"):
            role = field(row, "from"); value = field(row, "value")
            roles = {"user": "user", "assistant": "assistant", "tool": "tool"}
            if role not in roles: raise ValueError(f"unsupported ToolACE role: {role}")
            m = {"role": roles[role], "content": value}
            if role == "assistant" and value.lstrip().startswith("["):
                names = toolace_call_names(value, declared_names)
                if not names: raise ValueError("unparseable ToolACE call syntax")
                m["tool_calls"] = [{"function": {"name": n}} for n in names]
            messages.append(m)
    elif family == "openai":
        messages = json_value(field(record, entry["messages_field"]))
    else: raise ValueError(f"unknown review adapter: {family}")
    target = field(record, entry["target_field"]) if "target_field" in entry else None
    if target is not None and (type(target) is not int or not 0 <= target < len(messages)
                               or messages[target].get("role") != "assistant"):
        raise ValueError("target_message_index must address an assistant message")
    yield from emit_trajectory(dataset, trajectory_id, messages, tools, ctx, target_index=target)
