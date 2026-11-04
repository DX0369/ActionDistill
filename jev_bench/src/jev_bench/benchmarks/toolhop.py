"""ToolHop official execution/scoring with a Qwen3 engineering smoke entry point."""

import ast
import importlib.util
import json
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
    QwenClient,
    is_service_error,
)
from ..config import WORKSPACE_ROOT, advisor_type, fields, integer
from ..cost import costs
from ..io import append_json, write_json


def configure(config, runtime):
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    for filename in ("data/ToolHop.json", "code/evaluation_closed.py"):
        (source / filename).resolve(strict=True)


def visible_tools(definitions):
    """Drop dictionary keys: upstream keys contain hidden subquestions."""
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("ToolHop requires a nonempty tools object")
    tools = []
    for tool in definitions.values():
        fields(tool, ("name", "description", "parameters"))
        if not isinstance(tool["name"], str) or not tool["name"]:
            raise ValueError("tool names must be nonempty strings")
        if not isinstance(tool["description"], str):
            raise TypeError("tool descriptions must be strings")
        parameters = tool["parameters"]
        fields(
            parameters, ("type", "properties", "required"), ("additionalProperties",)
        )
        if parameters["type"] != "object" or not isinstance(
            parameters["properties"], dict
        ):
            raise ValueError("tool parameters require object properties")
        required = parameters["required"]
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or len(set(required)) != len(required)
            or not set(required) <= parameters["properties"].keys()
        ):
            raise ValueError("invalid required tool parameters")
        # Preserve JSON Schema unions and nested definitions verbatim.
        for definition in parameters["properties"].values():
            if not isinstance(definition, dict) or "type" not in definition:
                raise ValueError("parameter definitions require their canonical type")
            types = definition["type"]
            if not (
                isinstance(types, str)
                and types
                or isinstance(types, list)
                and types
                and all(isinstance(value, str) and value for value in types)
            ):
                raise ValueError("invalid parameter type")
        tools.append(deepcopy(tool))
    return tools


def load_tasks(config):
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    rows = json.loads((source / "data/ToolHop.json").read_text())
    if not isinstance(rows, list):
        raise TypeError("ToolHop data must be an array")
    tasks, answers = {}, {}
    for row in rows:
        # Additional published annotation fields are never copied into inputs.
        for field in ("id", "question", "tools", "functions", "answer"):
            if field not in row:
                raise KeyError(field)
        integer(row["id"])
        if not isinstance(row["question"], str) or not row["question"]:
            raise ValueError("question must be nonempty text")
        if not isinstance(row["answer"], str):
            raise TypeError("answer must be the published string")
        functions = row["functions"]
        if (
            not isinstance(functions, list)
            or not functions
            or any(not isinstance(code, str) or not code for code in functions)
        ):
            raise ValueError("functions must contain source strings")
        identity = ("tasks", str(row["id"]))
        if identity in tasks:
            raise ValueError("duplicate ToolHop ID")
        tasks[identity] = {
            "id": row["id"],
            "question": row["question"],
            "tools": visible_tools(row["tools"]),
        }
        answers[identity] = {"answer": row["answer"], "functions": functions}
    selected = config["task_ids"]["tasks"]
    if selected != "all":
        if not set(selected) <= {identity for _, identity in tasks}:
            raise ValueError("unknown ToolHop ID")
        tasks = {key: value for key, value in tasks.items() if key[1] in selected}
        answers = {key: answers[key] for key in tasks}
    return tasks, answers


def check_tool_dependencies(functions):
    """Detect missing imports before upstream turns them into ordinary tool feedback."""
    modules = set()
    for code in functions:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
    missing = sorted(name for name in modules if importlib.util.find_spec(name) is None)
    if missing:
        raise ModuleNotFoundError(
            "ToolHop tool dependencies are missing: " + ", ".join(missing)
        )
    for name in sorted(modules):
        importlib.import_module(name)


def run_fixture(config, task, target, actor, *, scenario, max_turns):
    """Run a scripted actor through upstream execution AND its embedded scorer.

    Each call gets a fresh module and private sample. The actor sees only the
    official messages and tool schemas, never source code or score targets.
    Use in an isolated CPU worker.
    """
    return _run_official(
        config, task, target, actor, scenario=scenario, max_turns=max_turns
    )


def _run_official(
    config,
    task,
    target,
    actor,
    *,
    scenario,
    max_turns,
    on_messages=None,
    on_inference_complete=None,
    on_timeout=None,
):
    if scenario not in ("Direct", "Free", "Mandatory"):
        raise ValueError("explicit official scenario required")
    integer(max_turns, 1)
    if scenario != "Direct":
        check_tool_dependencies(target["functions"])
    source = WORKSPACE_ROOT / config["benchmark"]["path"] / "code/evaluation_closed.py"
    spec = importlib.util.spec_from_file_location("toolhop_cpu_official", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    from func_timeout import FunctionTimedOut

    timeout_events = []

    def timeout_observation(call):
        event = {
            "event": "tool_execution_timeout",
            "tool_call_id": call["id"],
            "tool_name": call["function"]["name"],
            "timeout_seconds": 10,
        }
        timeout_events.append(event)
        if on_timeout is not None:
            on_timeout(event)
        # FunctionTimedOut's string/args can contain all hidden function source.
        # Only this fixed, named observation may enter actor-visible history.
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"error":"tool_execution_timeout"}',
        }

    tree = ast.parse(source.read_text())
    (feedback,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_feedback"
    ]
    (call_try,) = [node for node in ast.walk(feedback) if isinstance(node, ast.Try)]
    call_try.handlers.insert(
        0,
        ast.ExceptHandler(
            type=ast.Name(id="_FunctionTimedOut", ctx=ast.Load()),
            name=None,
            body=ast.parse("res.append(_timeout_observation(tool_call))").body,
        ),
    )
    official.__dict__.update(
        _FunctionTimedOut=FunctionTimedOut, _timeout_observation=timeout_observation
    )
    exec(  # noqa: S102 -- pinned upstream function plus the approved timeout handler
        compile(
            ast.fix_missing_locations(ast.Module(body=[feedback], type_ignores=[])),
            str(source),
            "exec",
        ),
        official.__dict__,
    )

    visible = []

    def scripted_request(messages, args, tools=None):
        nonlocal visible
        visible = deepcopy(messages)
        if on_messages is not None:
            on_messages(visible)
        response = actor(deepcopy(messages), deepcopy(tools))
        if not isinstance(response, dict) or response.get("role") != "assistant":
            raise ValueError("actor must return an assistant message")
        if "content" not in response or (
            response["content"] is not None and not isinstance(response["content"], str)
        ):
            raise ValueError("assistant content must be explicit string/null")
        if "tool_calls" in response and response["tool_calls"] is not None:
            if not isinstance(response["tool_calls"], list):
                raise ValueError("tool_calls must be a list")
            for call in response["tool_calls"]:
                fields(call, ("id", "type", "function"))
                fields(call["function"], ("name", "arguments"))
                if (
                    call["type"] != "function"
                    or any(
                        not isinstance(value, str) or not value
                        for value in (
                            call["id"],
                            call["function"]["name"],
                        )
                    )
                    or not isinstance(call["function"]["arguments"], str)
                ):
                    raise ValueError("malformed tool call")
        visible.append(deepcopy(response))
        if on_messages is not None:
            on_messages(visible)
        return deepcopy(response)

    official.test_closed = scripted_request
    original_feedback = official.get_feedback

    def tracked_feedback(calls, functions):
        feedback = original_feedback(calls, functions)
        visible.extend(deepcopy(feedback))
        if on_messages is not None:
            on_messages(visible)
        return feedback

    official.get_feedback = tracked_feedback
    if on_inference_complete is not None:
        # Upstream scores inline after returning from this loop. Persist the
        # boundary before scoring, without serializing the private sample.
        if scenario != "Free":
            raise ValueError("live ToolHop boundary requires Free mode")
        original_free = official.sample_process_free

        def tracked_free(sample, args):
            result = original_free(sample, args)
            if result is None:
                raise ValueError("official ToolHop loop returned no result")
            on_inference_complete(
                {
                    "id": task["id"],
                    "messages": deepcopy(result["messages"]),
                    "turn": result["turn"],
                    "partial": False,
                }
            )
            return result

        official.sample_process_free = tracked_free
    sample = deepcopy(
        {
            "id": task["id"],
            "question": task["question"],
            # Keys are ignored upstream; never reconstruct the hidden subquestions.
            "tools": {str(index): tool for index, tool in enumerate(task["tools"])},
            "functions": target["functions"],
            "answer": target["answer"],
        }
    )
    result = official.sample_process(
        sample, SimpleNamespace(scenario=scenario, max_turns=max_turns)
    )
    if result is None:
        raise ValueError("official ToolHop fixture returned no result")
    output = {
        "messages": result["messages"],
        "answer_correct": result["answer_correct"],
        "tool_error_counts": {"tool_execution_timeout": len(timeout_events)},
        "tool_error_events": timeout_events,
    }
    if "turn" in result:
        output["turn"] = result["turn"]
    return output


class ToolHopActor:
    def __init__(self, client, settings, policy, advisor, sink, projection=None):
        self.client, self.settings, self.policy = client, settings, policy
        self.advisor, self.sink = advisor, sink
        self.advisor_records = []
        self.anomalies = []
        self.recent_calls = []
        self.decision_steps = 0
        self.projection = projection

    def __call__(self, history, tools):
        messages = deepcopy(history)
        step = self.decision_steps
        record = None
        if self.policy["advice"] == "neutral":
            messages.append({"role": "system", "content": NEUTRAL_MESSAGE})
        elif self.policy["advice"] in ("prefer", "gate"):
            try:
                record = advise(
                    self.policy, self.advisor, messages, tools, self.recent_calls
                )
            except AdvisorServiceError as exc:
                exc.record.update(turn=0, step=step)
                self.advisor_records.append(exc.record)
                self.sink("advisor", exc.record)
                raise
            record.update(turn=0, step=step)
            self.advisor_records.append(record)
            messages.append({"role": "system", "content": record["hint"]})
        try:
            self.decision_steps += 1
            sampling = self.settings["thinking"]
            if self.projection is not None:
                messages, tools = self.projection.request(messages, tools)
            result = self.client.complete(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.settings["max_tokens"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": sampling["top_k"],
                    "min_p": sampling["min_p"],
                },
            )
            if self.projection is not None:
                result = self.projection.response(result)
            message = result.message
            response = {"role": message.role, "content": message.content}
            if message.tool_calls is not None:
                response["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            reasoning = getattr(message, "reasoning", None)
            if isinstance(reasoning, str) and reasoning:
                response["reasoning"] = reasoning
            flags = []
            if result.finish_reason == "length":
                flags.append("output_limit")
            if not message.tool_calls and not (
                isinstance(message.content, str) and message.content.strip()
            ):
                flags.append("empty_response")
            if flags:
                self.anomalies.append({"turn": 0, "step": step, "flags": flags})
            if record is not None and record["prediction"] is not None:
                actual = (
                    message.tool_calls[0].function.name
                    if message.tool_calls
                    else "RESPOND"
                    if isinstance(message.content, str) and message.content.strip()
                    else None
                )
                record["first_action_matches"] = actual == record["prediction"]["top1"]
            if self.policy["advice"] == "gate":
                for call in message.tool_calls or []:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except ValueError:
                        continue
                    if isinstance(arguments, dict):
                        self.recent_calls.append(
                            {"name": call.function.name, "arguments": arguments}
                        )
                self.recent_calls = self.recent_calls[-2:]
            return response
        finally:
            if record is not None:
                self.sink("advisor", record)


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
    if config["protocol"]["mode"] == "cpu_only":
        raise ValueError("ToolHop CPU preview cannot execute an episode")
    directory = Path(directory)
    policy = config["conditions"][item["condition"]]
    settings = config["executors"][policy["executor"]]
    row = {
        **item,
        "success": None,
        "official_scored": False,
        "failure_kind": None,
        "decision_steps": 0,
        "tool_call_count": 0,
        "duration_seconds": None,
        "response_anomalies": [],
        "tool_error_counts": {"tool_execution_timeout": 0},
    }
    started = time.perf_counter()
    client = actor = advisor = None
    phase = "inference"
    interrupted = False
    pending_feedback = False
    sink = lambda kind, value: append_json(
        directory / "diagnostics" / f"{kind}.jsonl", value
    )

    def save_messages(messages):
        nonlocal pending_feedback
        pending_feedback = bool(messages and messages[-1].get("tool_calls"))
        row["tool_call_count"] = sum(message["role"] == "tool" for message in messages)
        write_json(
            directory / "trajectory.json",
            {"id": task["id"], "messages": messages, "partial": True},
        )

    def inference_complete(trajectory):
        nonlocal phase
        phase = "scoring"
        mark_scoring(directory)
        write_json(directory / "trajectory.json", trajectory)

    def timeout_event(event):
        row["tool_error_counts"]["tool_execution_timeout"] += 1
        sink("tool_errors", event)

    try:
        client = client_factory(settings, sink)
        advisor = (
            advisor_factory(
                config["advisors"][policy["advisor"]], sink, advisors=config["advisors"]
            )
            if "advisor" in policy
            else None
        )
        from ..pa_tool import load_projection

        actor = ToolHopActor(
            client,
            settings,
            policy,
            advisor,
            sink,
            load_projection(config, item, expected_identity=pa_tool_identity),
        )
        result = _run_official(
            config,
            task,
            ground_truth,
            actor,
            scenario="Free",
            max_turns=9,
            on_messages=save_messages,
            on_inference_complete=inference_complete,
            on_timeout=timeout_event,
        )
        score = result["answer_correct"]
        if type(score) is not int or score not in (0, 1):
            raise ValueError("official ToolHop answer_correct must be 0 or 1")
        write_json(directory / "score.json", {"answer_correct": score})
        row.update(
            success=bool(score),
            official_scored=True,
            failure_kind=None if score else "model_task_failure",
        )
    except BaseException as exc:
        interrupted = not isinstance(exc, Exception)
        if pending_feedback:
            # Upstream returns observations only after the whole call batch.
            # A partial batch can have executed tools absent from this snapshot.
            row["tool_call_count"] = None
        failure = {"phase": phase, "error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, (CompletionResponseError, ContextLimitError)):
            failure["provider_diagnostics"] = exc.diagnostics
        write_json(directory / "diagnostics/failure.json", failure)
        if isinstance(exc, KeyboardInterrupt):
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
            row["failure_kind"] = (
                "scoring_error" if phase == "scoring" else "unexpected_error"
            )
            raise
    finally:
        if actor is not None:
            row.update(
                decision_steps=actor.decision_steps, response_anomalies=actor.anomalies
            )
        row["cost"] = costs(
            client,
            actor,
            advisor_type=advisor_type(config, item["condition"]),
            advisor_requests=advisor.records if advisor is not None else (),
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


def score_saved(config, item, task, ground_truth, directory):
    """Run only the pinned inline scorer, supplying the saved completed loop."""
    directory = Path(directory)
    score_path = directory / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
    else:
        trajectory = json.loads((directory / "trajectory.json").read_text())
        if trajectory.get("partial") is True:
            raise ValueError("ToolHop scoring requires complete inference")
        source = (
            WORKSPACE_ROOT / config["benchmark"]["path"] / "code/evaluation_closed.py"
        )
        module = ast.parse(source.read_text())
        scorer = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "sample_process"
        )
        namespace = {"sample_process_free": lambda sample, args: sample}
        exec(  # noqa: S102 -- pinned official scorer without running inference
            compile(ast.Module(body=[scorer], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        result = namespace["sample_process"](
            {"messages": trajectory["messages"], "answer": ground_truth["answer"]},
            SimpleNamespace(scenario="Free"),
        )
        score = {"answer_correct": result["answer_correct"]}
        write_json(score_path, score)
    if type(score["answer_correct"]) is not int or score["answer_correct"] not in (
        0,
        1,
    ):
        raise ValueError("invalid official ToolHop score")
    return {"success": bool(score["answer_correct"])}
