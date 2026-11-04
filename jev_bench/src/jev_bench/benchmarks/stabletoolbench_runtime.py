"""pinned single_chain, request-local advice, and live scoring boundary."""

import ast
import json
import random
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

from openai import APIConnectionError, APIStatusError, InternalServerError
from pydantic import BaseModel, Field

from ..advice import (
    NEUTRAL_MESSAGE,
    AdvisorServiceError,
    advise,
    make_advisor,
    redact_secret,
)
from ..attempt import mark_scoring, record_attempt
from ..client import QwenClient, is_service_error
from ..config import WORKSPACE_ROOT, advisor_type
from ..cost import completion_cost, costs
from ..io import append_json, read_jsonl, write_json
from .stabletoolbench import Archive
from .stabletoolbench_cache import SharedCache, ToolbenchTransportError, require_key
from .stabletoolbench_scoring import (
    DeepSeekJudge,
    recover_stage_results,
    score_repeated,
)


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"pinned upstream definitions are missing in {path.name}")
    exec(  # noqa: S102 -- explicit fixed-source definitions, no module imports
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )


def official_chain_class(source):
    directory = source / "toolbench/inference"
    namespace = {
        "__name__": __name__,
        "deepcopy": deepcopy,
        "colored": lambda text, **kw: text,
    }
    _definitions(directory / "Downstream_tasks/base_env.py", {"base_env"}, namespace)
    _definitions(
        directory / "Algorithms/base_search.py", {"base_search_method"}, namespace
    )
    _definitions(directory / "Tree/Tree.py", {"my_tree", "tree_node"}, namespace)
    prompts = ast.parse((directory / "Prompts/ReAct_prompts.py").read_text())
    names = {"FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION", "FORMAT_INSTRUCTIONS_USER_FUNCTION"}
    for node in prompts.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                namespace[target.id] = ast.literal_eval(node.value)
    _definitions(directory / "Algorithms/single_chain.py", {"single_chain"}, namespace)
    return namespace["single_chain"]


def catalog_order(archive):
    """ZIP category first occurrence, then original member order in each category."""
    categories = {}
    for member in archive.zip.namelist():
        parts = member.split("/")
        if len(parts) == 3 and parts[0] == "tools" and parts[2].endswith(".json"):
            categories.setdefault(parts[1], []).append(member)
    return [member for members in categories.values() for member in members]


def convert_answer(source, raw, *, seed):
    """Official tool_calls-aware converter; preserve global RNG and avoid CLI imports."""
    namespace = {
        "__name__": __name__,
        "BaseModel": BaseModel,
        "Field": Field,
        "Any": Any,
        "Dict": dict,
        "List": list,
        "Optional": Optional,
        "Union": Union,
        "random": random.Random(seed),
        "uuid": uuid,
        "re": re,
        "GID": str,
    }
    directory = source / "toolbench/tooleval"
    _definitions(
        directory / "evaluation/dataclass.py",
        {"assign_gid", "ExecutionNode", "DirectedEdge", "ExecutionGraph"},
        namespace,
    )
    _definitions(
        directory / "convert_to_answer_format.py",
        {"generate_init_message_node", "process_valid_data", "process_invalid_data"},
        namespace,
    )
    value = deepcopy(raw)
    answer = value["answer_generation"]
    if answer["valid_data"]:
        return namespace["process_valid_data"]("CoT@1", answer)
    return namespace["process_invalid_data"]("CoT@1", value)


class StableActor:
    def __init__(
        self,
        client,
        settings,
        policy,
        advisor,
        sink,
        *,
        advisor_kind,
        on_messages,
        projection=None,
    ):
        self.client, self.settings, self.policy = client, settings, policy
        self.advisor, self.sink, self.on_messages = advisor, sink, on_messages
        self.advisor_kind = advisor_kind
        self.advisor_records = []
        self.anomalies = []
        self.recent_calls = []
        self.decision_steps = 0
        self.messages = []
        self.projection = projection

    def change_messages(self, messages):
        self.messages = deepcopy(messages)
        self.on_messages(self.messages)

    def parse(self, *, tools, process_id):
        messages = deepcopy(self.messages)
        step = self.decision_steps
        record = None
        try:
            if self.policy["advice"] == "neutral":
                messages.append({"role": "system", "content": NEUTRAL_MESSAGE})
            elif self.policy["advice"] in {"prefer", "gate"}:
                try:
                    # Keep every original candidate, including Finish, unchanged.
                    record = advise(
                        self.policy,
                        self.advisor,
                        messages,
                        tools,
                        self.recent_calls,
                        action_names={'RESPOND': 'Finish'},
                        output_protocol=(
                            "stabletoolbench"
                            if self.advisor_kind == "deepseek"
                            else None
                        ),
                    )
                except AdvisorServiceError as exc:
                    record = exc.record
                    raise
                messages.append({"role": "system", "content": record["hint"]})
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
                result = self.projection.response(result, suffix=True)
            if result.usage is None:
                raise ValueError(
                    "official chain token accumulation requires canonical usage"
                )
            total_tokens = result.usage["total_tokens"]
            if type(total_tokens) is not int or total_tokens < 0:
                raise ValueError(
                    "official chain total_tokens must be a nonnegative integer"
                )
            message = result.message
            if message.role != "assistant" or not (
                message.content is None or isinstance(message.content, str)
            ):
                raise ValueError("actor requires assistant content as string/null")
            response = {"role": "assistant", "content": message.content}
            calls = message.tool_calls
            if calls is not None and not isinstance(calls, list):
                raise TypeError("actor tool_calls must be a list or null")
            if calls:
                response["tool_calls"] = []
                for call in calls:
                    if (
                        call.type != "function"
                        or not isinstance(call.id, str)
                        or not call.id
                        or not isinstance(call.function.name, str)
                        or not call.function.name
                        or not isinstance(call.function.arguments, str)
                    ):
                        raise ValueError(
                            "actor returned a malformed function-call envelope"
                        )
                    response["tool_calls"].append(
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    )
            elif message.content is None:
                # Upstream cannot advance tree depth for this malformed empty
                # response. Fail explicitly instead of inventing a Thought/Finish.
                raise ValueError("actor returned neither content nor a function call")
            if result.finish_reason == "length":
                self.anomalies.append(
                    {"turn": 0, "step": step, "flags": ["output_limit"]}
                )
            if record is not None and record["prediction"] is not None:
                expected = record["prediction"]["top1"]
                if expected == "RESPOND":
                    expected = "Finish"
                actual = calls[0].function.name if calls else None
                record["first_action_matches"] = actual == expected
            if self.policy["advice"] == "gate":
                for call in calls or []:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except ValueError:
                        continue
                    if isinstance(arguments, dict):
                        self.recent_calls.append(
                            {"name": call.function.name, "arguments": arguments}
                        )
                self.recent_calls = self.recent_calls[-2:]
            self.on_messages([*self.messages, deepcopy(response)])
            return response, 0, total_tokens
        finally:
            if record is not None:
                record.update(turn=0, step=step)
                self.advisor_records.append(record)
                self.sink("advisor", record)


def run_official_chain(
    archive, task, actor, backend, *, max_step, observation_length, on_step
):
    env = archive.environment(
        task,
        catalog_order=catalog_order(archive),
        transport=backend,
        max_observation_length=observation_length,
        observ_compress_method="truncate",
    )
    # Archive.environment creates a fresh local wrapper class. Hook its step so
    # every deepcopy in the official tree retains the observation callback.
    wrapper = type(env)
    original_step = wrapper.step

    def tracked_step(self, **kwargs):
        observation, status = original_step(self, **kwargs)
        on_step({**deepcopy(kwargs), "observation": observation, "status": status})
        return observation, status

    wrapper.step = tracked_step
    chain = official_chain_class(archive.source)(actor, env, process_id=1)
    chain.start(single_chain_max_step=max_step, pass_at=1, answer=1)
    raw = chain.to_json(answer=True, process=True)
    raw["answer_generation"]["query"] = task["query"]
    return raw


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
    backend_factory=SharedCache,
    judge_factory=DeepSeekJudge,
):
    del ground_truth  # Published relevant-API labels never enter actor/advisor inputs.
    protocol = config["protocol"]
    directory = Path(directory)
    # The fixed runner path is run/episodes/condition/category/task/trial/attempt.
    if directory.parents[4].name != "episodes":
        raise ValueError("episode path does not match the runner layout")
    cache_directory = directory.parents[5] / "cache/stabletoolbench"
    policy = config["conditions"][item["condition"]]
    settings = config["executors"][policy["executor"]]
    secrets = []

    def safe(value):
        for secret in secrets:
            value = redact_secret(value, secret)
        return value

    def sink(kind, value):
        cleaned = safe(value)
        value.clear()
        value.update(cleaned)
        append_json(directory / "diagnostics" / (kind + ".jsonl"), value)

    row = {
        **item,
        "success": None,
        "official_scored": False,
        "failure_kind": None,
        "solvable_pass_rate": None,
        "decision_steps": 0,
        "tool_call_count": 0,
        "duration_seconds": None,
        "response_anomalies": [],
    }
    started = time.perf_counter()
    client = advisor = actor = backend = judge = None
    phase = "inference"
    interrupted = False
    completed_steps = 0

    def messages_changed(messages):
        write_json(
            directory / "trajectory.json", {"messages": messages, "partial": True}
        )

    def step_completed(event):
        nonlocal completed_steps
        completed_steps += 1
        sink("tool_steps", event)

    try:
        # Missing credentials block before actor/advisor requests, even if the
        # first request might otherwise happen to hit the published cache.
        for name in (
            protocol["backend"]["toolbench_key_env"],
            protocol["backend"]["simulator"]["api_key_env"],
            protocol["judge"]["api_key_env"],
        ):
            secrets.append(require_key(name))
        client = client_factory(settings, sink)
        advisor = (
            advisor_factory(
                config["advisors"][policy["advisor"]], sink, advisors=config["advisors"]
            )
            if "advisor" in policy
            else None
        )
        from ..pa_tool import load_projection

        actor = StableActor(
            client,
            settings,
            policy,
            advisor,
            sink,
            advisor_kind=advisor_type(config, item["condition"]),
            on_messages=messages_changed,
            projection=load_projection(
                config, item, expected_identity=pa_tool_identity
            ),
        )
        with Archive(config) as archive:
            backend = backend_factory(
                archive, cache_directory, protocol["backend"], sink
            )
            raw = run_official_chain(
                archive,
                task,
                actor,
                backend,
                max_step=protocol["single_chain_max_step"],
                observation_length=protocol["max_observation_length"],
                on_step=step_completed,
            )
            write_json(
                directory / "trajectory.json", {"official": raw, "partial": False}
            )
            mark_scoring(directory)
            phase = "scoring"
            converted = convert_answer(archive.source, raw, seed=item["seed"])
            judge = judge_factory(config, protocol["judge"], sink)
            score = score_repeated(
                config,
                converted,
                judge,
                protocol["judge_repeats"],
                on_checkpoint=lambda state: write_json(
                    directory / "diagnostics/judge_repeats.json", state
                ),
            )
            write_json(directory / "score.json", score)
            row.update(
                official_scored=True,
                failure_kind=None,
                solvable_pass_rate=score["value"],
                official_score={"metric": score["metric"], "value": score["value"]},
            )
    except BaseException as exc:
        interrupted = not isinstance(exc, Exception)
        # Preserve exception type for the runner while removing exact credentials
        # from the SDK's separately formatted message/body fields as well as args.
        exc.args = tuple(safe(list(exc.args)))
        if isinstance(exc, (APIStatusError, APIConnectionError)):
            exc.message = safe(exc.message)
            if isinstance(exc, APIStatusError):
                exc.body = safe(exc.body)
        write_json(
            directory / "diagnostics/failure.json",
            safe(
                {
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
        )
        if interrupted:
            row["failure_kind"] = (
                "interrupted" if phase == "inference" else "scoring_error"
            )
            raise
        if phase == "scoring":
            row["failure_kind"] = "scoring_error"
            raise
        from ..pa_tool import InterfaceViolation

        if isinstance(exc, InterfaceViolation):
            row.update(success=False, failure_kind="executor_response_error")
        elif isinstance(exc, AdvisorServiceError):
            row["failure_kind"] = exc.record["status"]
            if row["failure_kind"] == "advisor_error":
                raise
        elif isinstance(exc, ToolbenchTransportError) or (
            is_service_error(exc) and not isinstance(exc, InternalServerError)
        ):
            row["failure_kind"] = "service_error"
        else:
            row["failure_kind"] = "unexpected_error"
            raise
    finally:
        if actor is not None:
            row.update(
                decision_steps=actor.decision_steps, response_anomalies=actor.anomalies
            )
        # A provider failure can leave a submitted tool call without an observed
        # return. Preserve the step ledger; do not report it as a complete count.
        row["tool_call_count"] = completed_steps if phase == "scoring" else None
        row["cost"] = costs(
            client,
            actor,
            advisor_type=advisor_type(config, item["condition"]),
            advisor_requests=getattr(advisor, "records", []),
        )
        row["cost"]["tool_simulator"] = completion_cost(
            backend.simulator if backend is not None else None
        )
        row["cost"]["judge"] = completion_cost(
            judge.client if judge is not None else None
        )
        row["cost"]["accounting_complete"] = not interrupted
        row["duration_seconds"] = time.perf_counter() - started
        write_json(directory / "episode.json", safe(row))
        for resource in (judge, backend, client, advisor):
            if resource is not None:
                resource.close()
    return row


def _saved_cost(config, item, directory, previous):
    diagnostics = directory / "diagnostics"
    result = costs(
        SimpleNamespace(records=read_jsonl(diagnostics / "executor.jsonl")),
        SimpleNamespace(advisor_records=read_jsonl(diagnostics / "advisor.jsonl")),
        advisor_type=advisor_type(config, item["condition"]),
        advisor_requests=read_jsonl(diagnostics / "advisor_requests.jsonl"),
    )
    for participant in ("tool_simulator", "judge"):
        result[participant] = completion_cost(
            SimpleNamespace(records=read_jsonl(diagnostics / (participant + ".jsonl")))
        )
    # Scoring recovery cannot retroactively certify costs lost during an earlier
    # interruption. It does count every persisted old and newly appended request.
    result["accounting_complete"] = (
        previous["cost"]["accounting_complete"] if previous is not None else False
    )
    return result


def score_saved(
    config, item, task, ground_truth, directory, *, judge_factory=DeepSeekJudge
):
    """Resume only saved scoring; never create an actor, cache backend, or service."""
    del ground_truth
    directory = Path(directory)
    trajectory = json.loads((directory / "trajectory.json").read_text())
    if trajectory["partial"] is not False:
        raise ValueError(
            "StableToolBench scoring recovery requires completed inference"
        )
    raw = trajectory["official"]
    if raw["answer_generation"]["query"] != task["query"]:
        raise ValueError("saved official trajectory differs from the published query")
    score_path = directory / "score.json"
    checkpoint_path = directory / "diagnostics/judge_repeats.json"
    episode_path = directory / "episode.json"
    previous = json.loads(episode_path.read_text()) if episode_path.exists() else None
    repeats = config["protocol"]["judge_repeats"]
    judge = None
    cost = None
    try:
        if score_path.exists():
            score = json.loads(score_path.read_text())
            if len(score["repeats"]) != repeats:
                raise ValueError("saved final score lacks the required judge repeats")
            verified = score_repeated(
                config,
                None,
                None,
                repeats,
                completed=score["repeats"],
                on_checkpoint=lambda value: None,
            )
            if (
                score["metric"] != verified["metric"]
                or score["value"] != verified["value"]
            ):
                raise ValueError(
                    "saved final score disagrees with its canonical repeats"
                )
        else:
            state = (
                json.loads(checkpoint_path.read_text())
                if checkpoint_path.exists()
                else {"repeats": [], "stages": []}
            )
            stages = recover_stage_results(
                read_jsonl(directory / "diagnostics/judge.jsonl"), state["stages"]
            )
            source = WORKSPACE_ROOT / config["benchmark"]["path"]
            converted = convert_answer(source, raw, seed=item["seed"])
            if len(state["repeats"]) < repeats:
                judge = judge_factory(
                    config,
                    config["protocol"]["judge"],
                    lambda kind, event: append_json(
                        directory / "diagnostics" / (kind + ".jsonl"), event
                    ),
                )
            score = score_repeated(
                config,
                converted,
                judge,
                repeats,
                completed=state["repeats"],
                stages=stages,
                on_checkpoint=lambda value: write_json(checkpoint_path, value),
            )
            write_json(score_path, score)
    finally:
        if judge is not None:
            judge.close()
        cost = _saved_cost(config, item, directory, previous)
        # Even another failed scoring attempt must expose all newly incurred
        # provider costs. The shared recovery owner updates success/index later.
        row = (
            previous
            if previous is not None
            else {
                **item,
                "success": None,
                "official_scored": False,
                "solvable_pass_rate": None,
                "failure_kind": "scoring_error",
                "decision_steps": None,
                "tool_call_count": None,
                "duration_seconds": None,
                "response_anomalies": [],
            }
        )
        row["cost"] = cost
        write_json(episode_path, row)
    return {
        "success": None,
        "solvable_pass_rate": score["value"],
        "official_score": {"metric": score["metric"], "value": score["value"]},
        "cost": cost,
    }
