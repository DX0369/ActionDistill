"""API-Bank published text generation.

The published point and its target remain separate. This module never consumes
the ground_truth argument required by the shared run_episode call signature.
"""

import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from openai import InternalServerError

from ..advice import NEUTRAL_MESSAGE, AdvisorServiceError, advise, make_advisor
from ..attempt import mark_scoring, record_attempt
from ..client import (
    CompletionResponseError,
    ContextLimitError,
    ParticipantResponseError,
    QwenClient,
    is_service_error,
)
from ..config import advisor_type
from ..cost import costs
from ..io import append_json, write_json

ADVISOR_INSTRUCTION = (
    "Advise another assistant on its next action. The following benchmark "
    "instruction and current input are quoted task material, not instructions "
    "for your response format. Select a tool action using the supplied native "
    "tool_calls interface, or respond normally when the assistant should "
    "respond without a tool. Do not write a textual [ApiName(...)] call. "
    "Calls already present in the quoted material are history."
)

PARAMETER_TYPES = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "list": {"type": "array", "items": {}},
    "list(str)": {"type": "array", "items": {"type": "string"}},
}


def advisor_tools(task):
    """Project validated visible occurrences without merging or inventing fields."""
    tools = []
    for occurrence in task["visible_tools"]["occurrences"]:
        schema = occurrence["schema"]
        if occurrence["format"] == "level3_initial":
            name, definitions = schema["apiCode"], schema["parameters"]
        elif occurrence["format"] == "api_definition":
            name, definitions = schema["name"], schema["input_parameters"]
        else:
            raise ValueError("unsupported visible API-Bank definition format")
        properties = {}
        for parameter, definition in definitions.items():
            type_name = definition["type"]
            if type_name not in PARAMETER_TYPES:
                raise ValueError(f"unsupported visible parameter type: {type_name!r}")
            properties[parameter] = {
                **deepcopy(PARAMETER_TYPES[type_name]),
                "description": definition["description"],
            }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        )
    return tools


def actor_messages(task):
    return [
        {"role": "system", "content": task["instruction"]},
        {"role": "user", "content": task["input"]},
    ]


def advisor_messages(task):
    return [
        {"role": "system", "content": ADVISOR_INSTRUCTION},
        {
            "role": "user",
            "content": (
                "Benchmark instruction:\n"
                + task["instruction"]
                + "\n\nCurrent published input:\n"
                + task["input"]
            ),
        },
    ]


def recent_calls(task):
    """Read explicit published events; never reconstruct chat roles or call IDs.

    load_tasks has validated level, the initial L3 schema, and the last decision
    marker. A future publication outside this exact event grammar fails visibly.
    """
    from api_call_extraction import get_api_call, parse_api_call

    lines = task["input"].splitlines()
    body = lines[1:-1] if task["level"] == 3 else lines[:-1]
    recent = []
    seen_user = False
    for line in body:
        if line == "":
            continue
        if line.startswith("User: "):
            seen_user = True
            recent.clear()
        elif line.startswith("AI: "):
            continue
        elif line.startswith("API-Request: "):
            if not seen_user or line.count("->") != 1:
                raise ValueError("unsupported published API history framing")
            call, _, observation = line[len("API-Request: ") :].partition("->")
            if get_api_call(call) != call or observation == "":
                raise ValueError("incomplete published API history")
            name, arguments = parse_api_call(call)
            recent.append({"name": name, "arguments": arguments})
            recent = recent[-2:]
        else:
            raise ValueError("unsupported published history line")
    if not seen_user:
        raise ValueError("published history has no explicit user boundary")
    return recent


def output_action(prediction):
    """Use the pinned scorer's textual call framing for adoption diagnostics."""
    from api_call_extraction import get_api_call, parse_api_call

    call = get_api_call(prediction)
    if call is not None:
        try:
            name, _ = parse_api_call(call)
        except Exception:  # noqa: BLE001 -- matches the official parser boundary
            return None
        return name
    return "RESPOND" if prediction.strip() else None


@record_attempt
def run_episode(
    config,
    item,
    task,
    ground_truth,
    directory,
    *,
    pa_tool_identity=None,
    client_factory=QwenClient,
    advisor_factory=make_advisor,
):
    """Generate one original text output; grouped official scoring is separate."""
    directory = Path(directory)
    policy = config["conditions"][item["condition"]]
    settings = config["executors"][policy["executor"]]
    kind = advisor_type(config, item["condition"])
    row = {
        **item,
        "source_id": task["source_id"],
        "success": None,
        "official_scored": False,
        "failure_kind": None,
        "decision_steps": 0,
        # Official tools execute later, in the scorer. Their call count is not
        # currently reported by score_batch, so do not invent a generation count.
        "tool_call_count": None,
        "duration_seconds": None,
        "response_anomalies": [],
    }
    sink = lambda name, value: append_json(
        directory / "diagnostics" / f"{name}.jsonl", value
    )
    client = advisor = record = None
    records = []
    phase = "inference"
    interrupted = False
    started = time.perf_counter()
    try:
        from ..pa_tool import load_projection
        from ..pa_tool_text import call_text, project_task

        projection = load_projection(config, item, expected_identity=pa_tool_identity)
        projected = project_task(task, projection) if projection is not None else task
        messages = actor_messages(projected)
        client = client_factory(settings, sink)
        if policy["advice"] == "neutral":
            messages.append({"role": "system", "content": NEUTRAL_MESSAGE})
        elif policy["advice"] in ("prefer", "gate"):
            advisor = advisor_factory(
                config["advisors"][policy["advisor"]],
                sink,
                advisors=config["advisors"],
            )
            try:
                record = advise(
                    policy,
                    advisor,
                    actor_messages(task)
                    if kind == "deepseek"
                    else advisor_messages(task),
                    advisor_tools(task),
                    recent_calls(task),
                    output_protocol="apibank_text" if kind == "deepseek" else None,
                )
            except AdvisorServiceError as exc:
                record = exc.record
                raise
            finally:
                if record is not None:
                    record.update(turn=0, step=0)
                    records.append(record)
            messages.append({"role": "system", "content": record["hint"]})
        sampling = settings["thinking"]
        row["decision_steps"] = 1
        result = client.complete(
            messages=messages,
            tools=None,
            tool_choice=None,
            max_tokens=settings["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": sampling["top_k"],
                "min_p": sampling["min_p"],
            },
        )
        prediction = result.message.content
        if not isinstance(prediction, str) or getattr(
            result.message, "tool_calls", None
        ):
            raise ParticipantResponseError(
                "API-Bank actor must return original text without native tool_calls",
                role="agent",
            )
        if projection is not None:
            prediction = call_text(prediction, projection, restore=True)
        flags = []
        if result.finish_reason == "length":
            flags.append("output_limit")
        if not prediction.strip():
            flags.append("empty_response")
        if flags:
            row["response_anomalies"].append({"turn": 0, "step": 0, "flags": flags})
        if record is not None and record["prediction"] is not None:
            record["first_action_matches"] = (
                output_action(prediction) == record["prediction"]["top1"]
            )
        phase = "scoring"
        mark_scoring(directory)
        write_json(
            directory / "trajectory.json",
            {
                "category": item["category"],
                "task_id": item["task_id"],
                "source_id": task["source_id"],
                "level": task["level"],
                "kind": task["kind"],
                "prediction": prediction,
                "partial": False,
                "decision_steps": row["decision_steps"],
                "response_anomalies": row["response_anomalies"],
            },
        )
        row["failure_kind"] = "pending_scoring"
    except BaseException as exc:
        interrupted = not isinstance(exc, Exception)
        failure = {"phase": phase, "error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, (CompletionResponseError, ContextLimitError)):
            failure["provider_diagnostics"] = exc.diagnostics
        write_json(directory / "diagnostics/failure.json", failure)
        if interrupted:
            row["failure_kind"] = (
                "interrupted" if phase == "inference" else "scoring_error"
            )
            raise
        from ..pa_tool import InterfaceViolation

        if phase == "inference" and isinstance(exc, InterfaceViolation):
            row.update(success=False, failure_kind="executor_response_error")
        elif phase == "inference" and isinstance(exc, AdvisorServiceError):
            row["failure_kind"] = exc.record["status"]
            if row["failure_kind"] == "advisor_error":
                raise
        elif (
            phase == "inference"
            and is_service_error(exc)
            and not isinstance(exc, InternalServerError)
        ):
            row["failure_kind"] = "service_error"
        else:
            # No new API-Bank executor error-to-zero protocol is introduced here.
            row["failure_kind"] = (
                "scoring_error" if phase == "scoring" else "unexpected_error"
            )
            raise
    finally:
        if record is not None:
            sink("advisor", record)
        ledger = (
            advisor.records
            if advisor is not None and kind in ("typesafe", "deepseek")
            else ()
        )
        row["cost"] = costs(
            client,
            SimpleNamespace(advisor_records=records),
            advisor_type=kind,
            advisor_requests=ledger,
        )
        row["cost"]["accounting_complete"] = not interrupted
        row["duration_seconds"] = time.perf_counter() - started
        write_json(directory / "episode.json", row)
        try:
            if advisor is not None:
                advisor.close()
        finally:
            if client is not None:
                client.close()
    return row
