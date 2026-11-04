"""Shared boundaries for the dataset review; never infer future candidates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from ..common import iter_json_array, iter_jsonl, open_text
from ..schema import DecisionEvent, IssueEvent, ToolEvent, TrajectoryEvent


def field(record, path):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing canonical field: {path}")
        value = value[key]
    return value


def json_value(value):
    return json.loads(value) if isinstance(value, str) else value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def user_hash(content):
    if isinstance(content, str): text = content
    elif isinstance(content, list):
        text = "\n".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    else: return fingerprint(content)
    return fingerprint(" ".join(text.split()))


def records(root, entry, limit=None):
    """One record at a time. A JSON object file is one trajectory, not a corpus."""
    count = 0
    paths = sorted(root.glob(entry["glob"])) if root.is_dir() else [root]
    if not paths: raise ValueError(f"no input files match {entry['glob']}")
    for path in paths:
        fmt = entry["storage"]
        if fmt == "jsonl":
            iterator = iter_jsonl(path)
        elif fmt == "json_array":
            iterator = enumerate(iter_json_array(path), 1)
        elif fmt == "json":
            with open_text(path) as handle:
                item = json.load(handle)
            iterator = [(1, item)]
        elif fmt == "tau2":
            import ijson
            with path.open("rb") as handle:
                info = next(ijson.items(handle, "info", use_float=True), None)
            if info is None:
                raise ValueError("tau2 Results missing info")
            with path.open("rb") as handle:
                # Embedded task definitions let contamination checks compare scenarios exactly.
                scenarios = {str(field(t, "id")): fingerprint(field(t, "user_scenario"))
                             for t in ijson.items(handle, "tasks.item", use_float=True)}
            def simulations():
                with path.open("rb") as handle:
                    for index, simulation in enumerate(ijson.items(handle, "simulations.item", use_float=True), 1):
                        task = str(field(simulation, "task_id"))
                        if task not in scenarios: raise ValueError(f"simulation task {task} missing from embedded tasks")
                        yield index, {**simulation, "_info": info, "_task_scenario_hash": scenarios[task]}
            iterator = simulations()
        elif fmt == "parquet":
            import pyarrow.parquet as pq
            def rows():
                index = 0
                # Rows can be large (full trajectories); a modest batch keeps memory bounded.
                for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        index += 1
                        yield index, row
            iterator = rows()
        else:
            raise ValueError(f"unknown storage: {fmt}")
        for index, item in iterator:
            if limit is not None and count >= limit:
                return
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{index}: record must be an object")
            count += 1
            yield path, index, item


def outcome(record, spec):
    if spec is None:
        return "unknown"
    value = field(record, spec["field"])
    for mapping in spec["values"]:
        expected = mapping["value"]
        same_type = type(value) is type(expected)
        numeric_protocol = (spec.get("value_type") == "number" and type(value) in {int, float}
                            and type(expected) in {int, float})
        if (same_type or numeric_protocol) and value == expected:
            return mapping["outcome"]
    raise ValueError(f"unmapped outcome value at {spec['field']}: {value!r}")


def definitions(raw):
    raw = json_value(raw)
    if not isinstance(raw, list):
        raise ValueError("tools must be a list")
    tools = []
    for tool in raw:
        if isinstance(tool, str) and tool.lstrip().startswith("{"):
            tool = json.loads(tool)
        if isinstance(tool, str):
            if not tool: raise ValueError("declared tool name must be nonempty")
            tools.append({"name": tool})
            continue
        if not isinstance(tool, dict):
            raise ValueError("tool definition must be an object or declared name")
        body = tool["function"] if "function" in tool else tool
        if not isinstance(body, dict) or not isinstance(body.get("name"), str) or not body["name"]:
            raise ValueError("tool definition missing name")
        body = dict(body)
        if "parameters" in body:
            body["parameters"] = json_value(body["parameters"])
            if not isinstance(body["parameters"], dict):
                raise ValueError("tool parameters must be an object")
        tools.append(body)
    if len({t["name"] for t in tools}) != len(tools):
        raise ValueError("duplicate names in candidate declaration")
    return tools


def clean_content(content, *, strip_think=False):
    if content is None:
        return None
    if isinstance(content, str):
        if strip_think:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            if "<think>" in content or "</think>" in content:
                raise ValueError("unclosed reasoning block")
        return content
    if isinstance(content, list):
        result = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("content block must be an object")
            if block.get("type") in {"thinking", "redacted_thinking", "reasoning"}:
                continue
            if strip_think and block.get("type") in {"text", "input_text", "output_text"}:
                block = dict(block)
                block["text"] = clean_content(field(block, "text"), strip_think=True)
            result.append(block)
        return result
    raise ValueError("content must be string, block list, or null")


def visible(message, strip_think=False):
    if not isinstance(message, dict) or not isinstance(message.get("role"), str):
        raise ValueError("message missing canonical role")
    if message["role"] not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported normalized role: {message['role']}")
    content = message.get("content")
    if message["role"] == "tool" and isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=False)
    result = {"role": message["role"], "content": clean_content(
        content, strip_think=strip_think and message["role"] == "assistant")}
    for key in ("name", "tool_call_id"):
        if key in message:
            result[key] = message[key]
    if "tool_calls" in message:
        calls = message["tool_calls"]
        # OpenAI chat messages explicitly allow null for no tool calls.
        if calls is None: calls = []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        result["tool_calls"] = []
        for call in calls:
            f = field(call, "function")
            name = field(f, "name")
            if not isinstance(name, str) or not name:
                raise ValueError("function.name must be a nonempty string")
            c = {"type": "function", "function": {"name": name}}
            if "id" in call:
                c["id"] = call["id"]
            if "arguments" in f:
                c["function"]["arguments"] = f["arguments"]
            result["tool_calls"].append(c)
    return result


def without_ids(message):
    message = {k: v for k, v in message.items() if k != "tool_call_id"}
    if "tool_calls" in message:
        message["tool_calls"] = [{k: v for k, v in c.items() if k != "id"} for c in message["tool_calls"]]
    return message


def has_text(content):
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in {"text", "output_text", "input_text"}
        and isinstance(p.get("text"), str) and bool(p["text"].strip()) for p in content)


def emit_trajectory(dataset, trajectory_id, messages, tools, context, *, target_index=None):
    """Emit before updating historical observations. Only prior calls enter I."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    history = []
    history_nodes = []  # source record ids per history message (Claude tree logs)
    observed = {}
    call_names = {}
    call_schemas = {}
    declared = tools
    # H: no declaration, but the harness tool set is known from the whole
    # dataset (tools used in >= N trajectories). It is the same for every
    # trajectory of the harness, so it reveals nothing about this trajectory's future.
    union = context.get("harness_tool_sets")
    harness_tools = None if union is None else union.get((context.get("harness"), context.get("domain")), [])
    level = ((context.get("candidate_level") or ("S" if all("parameters" in t for t in tools) else "N"))
             if tools is not None else "I" if harness_tools is None else "H")
    candidate_source = context.get("candidate_source", "prior_calls") if level != "H" else "harness_union_plus_prior_calls"
    first_user = None
    normalized_trace = []
    trace_models = []
    reported_missing_model = False
    history_char_sum = 0
    return_message_count = 0
    unidentified_returns = 0
    skipped_non_teacher = 0
    for index, raw in enumerate(messages):
        message = visible(raw, context.get("strip_think", False))
        normalized_trace.append(message)
        if message["role"] == "user" and first_user is None:
            first_user = message.get("content")
        model = raw["_model"] if "_model" in raw else context.get("model")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError("teacher model must be a nonempty string when provided")
        if message["role"] == "assistant": trace_models.append(model)
        base = dict(dataset=dataset, trajectory_id=trajectory_id, model=model,
                    harness=context.get("harness"), domain=context.get("domain"),
                    source=raw.get("_source", context["source"]))
        decision_index = context.get("decision_index", index)
        if "_tools" in raw:
            declared = raw["_tools"]
            level = "S" if all("parameters" in t for t in declared) else "N"
        if declared is not None:
            candidates = list(declared)
        elif harness_tools is not None:
            # Session-specific tools (below the dataset threshold) join only after their first use here.
            candidates = [{"name": n} for n in harness_tools] + [t for n, t in observed.items() if n not in harness_tools]
        else:
            candidates = list(observed.values())
        calls = message.get("tool_calls", [])
        if message["role"] == "assistant":
            names = [c["function"]["name"] for c in calls]
            unique = tuple(dict.fromkeys(names))
            kind = "TOOL_SET" if len(unique) > 1 else "TOOL" if names else "RESPOND" if has_text(message.get("content")) else "UNKNOWN"
            complete = declared is not None and context.get("candidate_complete", True)
            if raw.get("_not_teacher_step"):
                skipped_non_teacher += 1
            elif target_index is None or index == target_index:
                view = history
                abandoned = raw.get("_abandoned")
                if abandoned:
                    # Drop messages that lie only on branches the user abandoned
                    # (edited/resent turns) relative to this message's ancestry.
                    view = [h for h, nodes in zip(history, history_nodes) if not (nodes and nodes <= abandoned)]
                    if declared is None:
                        prior = dict.fromkeys(c["function"]["name"] for h in view for c in h.get("tool_calls", []))
                        base_names = harness_tools or []
                        candidates = ([{"name": n} for n in base_names] if harness_tools is not None else []) + \
                                     [{"name": n} for n in prior if n not in base_names]
                if model is None and not reported_missing_model:
                    yield IssueEvent(dataset, "missing_teacher_model", "no recorded or dataset-declared teacher identity",
                                     trajectory_id, base["source"])
                    reported_missing_model = True
                if declared is not None:
                    for tool in candidates:
                        yield ToolEvent(**base, event="declared", raw_name=tool["name"],
                                        turn_index=decision_index, schema=tool)
                for call in calls:
                    definition = next((t for t in candidates if t["name"] == call["function"]["name"]), None)
                    yield ToolEvent(**base, event="called", raw_name=call["function"]["name"],
                                    turn_index=decision_index, call_id=call.get("id"), schema=definition)
                metadata = {"outcome": context.get("outcome", "unknown"),
                            "source_session_id": context.get("source_session_id"),
                            "source_record_id": context.get("source_record_id"),
                            "response_model": raw.get("_response_model") if "_response_model" in raw else context.get("response_model"),
                            "task_id": context.get("task_id"), "candidate_level": level,
                            "candidate_count": len(candidates), "call_count": len(calls),
                            "call_counts": dict(Counter(names)),
                            "history_chars": (2 + history_char_sum + max(0, len(history) - 1) * 2 if view is history
                                              else len(json.dumps(view, ensure_ascii=False))),
                            "history_message_count": len(view), "target_message_index": index,
                            "abandoned_branch_messages_removed": len(history) - len(view),
                            "first_user_hash": user_hash(first_user),
                            "candidate_source": candidate_source,
                            "quality_flags": context.get("quality_flags", []),
                            "model_source": context.get("model_source"),
                            "after_context_compaction": bool(raw.get("_after_compaction")),
                            "task_scenario_hash": context.get("task_scenario_hash"),
                            "_history": list(view), "_candidate_definitions": candidates}
                yield DecisionEvent(**base, turn_index=decision_index,
                                    candidate_tools=tuple(t["name"] for t in candidates),
                                    action_kind=kind, action_names=unique,
                                    candidate_set_complete=complete, metadata=metadata)
            for call in calls:
                name = call["function"]["name"]
                observed.setdefault(name, {"name": name})
                if "id" in call:
                    call_names[call["id"]] = name
                    call_schemas[call["id"]] = next((t for t in candidates if t["name"] == name), None)
        elif message["role"] == "tool":
            return_message_count += 1
            call_id = message.get("tool_call_id")
            name = message.get("name") if "name" in message else call_names.get(call_id)
            if isinstance(name, str) and name and target_index is None:
                yield ToolEvent(**base, event="returned", raw_name=name, turn_index=index,
                                call_id=call_id, schema=call_schemas.get(call_id))
            elif not isinstance(name, str) or not name:
                unidentified_returns += 1
        history.append(message)
        history_nodes.append(frozenset(raw.get("_uuids", ())))
        history_char_sum += len(json.dumps(message, ensure_ascii=False))
    unique_models = set(trace_models)
    if unidentified_returns:
        yield IssueEvent(dataset, "unidentified_tool_returns", "visible tool results lack a resolvable tool identity; no pairing invented",
                         trajectory_id, context["source"], {"visible_messages": unidentified_returns})
    yield TrajectoryEvent(dataset, trajectory_id, model=next(iter(unique_models)) if len(unique_models) == 1 else None,
                          harness=context.get("harness"), source=context["source"],
                          metadata={"outcome": context.get("outcome", "unknown"),
                                    "source_session_id": context.get("source_session_id"),
                                    "source_record_id": context.get("source_record_id"),
                                    "source_record_id_field": context.get("source_record_id_field"),
                                    "task_id": context.get("task_id"),
                                    "first_user_hash": user_hash(first_user),
                                    "models": sorted(m for m in unique_models if m is not None),
                                    "trace_scope": "captured_request_prefix" if target_index is not None else "published_record",
                                    "visible_tool_return_message_count": return_message_count,
                                    "non_teacher_steps_skipped": skipped_non_teacher,
                                    # Call IDs and model labels differ between republished copies.
                                    "trajectory_hash": fingerprint([without_ids(m) for m in normalized_trace])})
