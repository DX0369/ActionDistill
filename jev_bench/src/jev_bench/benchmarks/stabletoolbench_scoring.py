"""Official SoPR with scripted fixtures or an explicit DeepSeek judge.

Live settings and repetition count are fixed in the smoke configuration.
SoPR retains Unsure; it is not the boolean success used by BFCL.
"""

import ast
import importlib
import json
import re
import sys
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Optional, Union
from unittest.mock import patch

import yaml
from openai import APIConnectionError, APIStatusError

from ..advice import redact_secret
from ..client import CompletionClient
from ..config import WORKSPACE_ROOT, fields, integer
from .stabletoolbench_cache import require_key


@contextmanager
def fixture_evaluator(config, request):
    """Build the official evaluator without loading any API pool or credentials."""
    source = (WORKSPACE_ROOT / config["benchmark"]["path"]).resolve(strict=True)
    sys.path.insert(0, str(source))
    try:
        rtl = importlib.import_module(
            "toolbench.tooleval.evaluators.registered_cls.rtl"
        )
        base = importlib.import_module(
            "toolbench.tooleval.evaluators.registered_cls.tooleval"
        )
        if not Path(rtl.__file__).resolve().is_relative_to(source):
            raise ValueError("ToolEval was imported from another checkout")
        with patch.object(
            base, "OpenaiPoolRequest", return_value=SimpleNamespace(request=request)
        ):
            evaluator = rtl.ReinforceToolLearningEvaluator(
                str(
                    source
                    / "toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default"
                )
            )
        yield evaluator
    finally:
        sys.path.remove(str(source))


def score_fixture(config, example, evaluator):
    """Preserve official Finish gate and AnswerStatus, including Unsure.

    get_steps is loaded without its module's global RNG reset and graph imports.
    The unmodified upstream function traverses the first branch, as in SoPR.
    """
    fields(example, ("query", "available_tools", "answer"))
    if not isinstance(example["query"], str) or not isinstance(
        example["available_tools"], list
    ):
        raise TypeError("invalid converted query/tools")
    answer = example["answer"]
    fields(answer, ("method", "total_steps", "final_answer", "answer_details"))
    integer(answer["total_steps"])
    if not isinstance(answer["method"], str) or not isinstance(
        answer["final_answer"], str
    ):
        raise TypeError("converted answer text fields must be strings")
    if not isinstance(answer["answer_details"], list) or not answer["answer_details"]:
        raise ValueError("converted answer requires a nonempty node list")
    nodes = list(answer["answer_details"])
    while nodes:
        node = nodes.pop()
        fields(node, ("role", "message", "next"))
        if not isinstance(node["role"], str) or not isinstance(node["next"], list):
            raise TypeError("converted nodes require a role and explicit next list")
        if not isinstance(node["message"], (str, dict)):
            raise TypeError("converted node message must be text or an object")
        if node["role"] == "tool" and isinstance(node["message"], dict):
            fields(node["message"], ("name", "arguments", "response"))
            if (
                not isinstance(node["message"]["name"], str)
                or not node["message"]["name"]
            ):
                raise ValueError("converted tool messages require a nonempty name")
        nodes.extend(node["next"])
    source = (
        WORKSPACE_ROOT / config["benchmark"]["path"] / "toolbench/tooleval/utils.py"
    )
    tree = ast.parse(source.read_text())
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_steps"
    ]
    namespace = {}
    exec(  # noqa: S102 -- fixed upstream function, no module-level RNG reset
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    isolated = deepcopy(example)
    steps, final_step = namespace["get_steps"](isolated)
    if "'name': 'Finish'" not in final_step:
        status, reason = "Unsolved", "official_missing_finish"
    else:
        result, reason = evaluator.check_is_solved(
            {name: isolated[name] for name in ("query", "available_tools")},
            isolated["answer"],
            return_reason=True,
        )
        status = result.value
        if not isinstance(reason, str):
            raise TypeError("official judge reason must be a string")
    return {
        "answer_status": status,
        "reason": reason,
        "answer_steps": steps,
        "final_step": final_step,
    }


class DeepSeekJudge:
    """Official prompts and state transitions; explicit non-thinking transport.

    No SDK or hidden whole-evaluation retry. A failed stage remains unscored.
    The separate judge_repeats setting counts independent evaluations only.
    """

    def __init__(self, config, settings, sink, *, client_factory=CompletionClient):
        self.settings = settings
        self.key = require_key(settings["api_key_env"])
        self.repeat = None
        self.function_name = None
        self.saved_calls = {}
        self.on_call = None

        def safe_sink(kind, event):
            cleaned = redact_secret(event, self.key)
            event.clear()
            event.update(cleaned)
            event.update(repeat=self.repeat, judge_function=self.function_name)
            sink(kind, event)

        self.client = client_factory(
            settings,
            safe_sink,
            api_key=self.key,
            participant="judge",
        )
        source = WORKSPACE_ROOT / config["benchmark"]["path"]
        directory = (
            source / "toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default"
        )
        definition = yaml.safe_load((directory / "config.yaml").read_text())
        self.functions = {
            item["name"]: item for item in definition["completions_kwargs"]["functions"]
        }
        template = (directory / "template.txt").read_text()
        self.templates = {}
        for block in re.findall(r"<function>(.*?)</function>", template, re.DOTALL):
            name = re.findall(r"<name>(.*?)</name>", block, re.DOTALL)[0]
            self.templates[name] = re.findall(
                r"<description>(.*?)</description>", block, re.DOTALL
            )[0]

        # Reuse the exact official state machine without constructing an API pool
        # or accepting a process-wide EVAL_MODEL override of the frozen config.
        path = source / "toolbench/tooleval/evaluators/registered_cls/rtl.py"
        tree = ast.parse(path.read_text())
        enum = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AnswerStatus"
        )
        evaluator = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ReinforceToolLearningEvaluator"
        )
        check = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.FunctionDef) and node.name == "check_is_solved"
        )
        namespace = {
            "Enum": Enum,
            "json": json,
            "Dict": dict,
            "Any": Any,
            "Union": Union,
            "Optional": Optional,
        }
        exec(  # noqa: S102 -- fixed enum and check_is_solved, no upstream API pool
            compile(ast.Module(body=[enum, check], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.check_is_solved = MethodType(namespace["check_is_solved"], self)

    def close(self):
        self.client.close()

    def begin_repeat(self, index, stages, on_call):
        self.repeat = index
        self.saved_calls = {
            row["function"]: deepcopy(row["result"])
            for row in stages
            if row["repeat"] == index
        }
        self.on_call = on_call

    def function_call(self, func_name, func_args, *, return_reason=False):
        if self.repeat is None:
            raise ValueError("judge repeat must be identified before its first stage")
        if func_name in self.saved_calls:
            return validate_judge_reply(
                deepcopy(self.saved_calls[func_name]), return_reason
            )
        self.function_name = func_name
        function = deepcopy(self.functions[func_name])
        if return_reason:
            function["parameters"]["required"].append("reason")
            function["parameters"]["properties"]["reason"] = {
                "type": "string",
                "description": "explain your answer.",
            }
        request = {
            "model": self.settings["model"],
            "messages": [
                {
                    "role": "user",
                    "content": self.templates[func_name].format(**func_args),
                }
            ],
            "tools": [{"type": "function", "function": function}],
            "tool_choice": {"type": "function", "function": {"name": func_name}},
            "temperature": self.settings["temperature"],
            "max_tokens": self.settings["max_tokens"],
            "extra_body": {"thinking": {"type": self.settings["thinking"]}},
        }
        try:
            result = self.client.request(request)
        except BaseException as exc:
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            if isinstance(exc, (APIConnectionError, APIStatusError)):
                exc.message = redact_secret(exc.message, self.key)
                if isinstance(exc, APIStatusError):
                    exc.body = redact_secret(exc.body, self.key)
            raise
        calls = result.message.tool_calls
        if (
            result.finish_reason != "tool_calls"
            or not isinstance(calls, list)
            or len(calls) != 1
        ):
            raise ValueError("judge must finish with exactly one function call")
        call = calls[0]
        if call.type != "function" or call.function.name != func_name:
            raise ValueError("judge called a function other than the forced function")
        reply = validate_judge_reply(json.loads(call.function.arguments), return_reason)
        reply = redact_secret(reply, self.key)
        self.saved_calls[func_name] = deepcopy(reply)
        self.on_call(
            {"repeat": self.repeat, "function": func_name, "result": deepcopy(reply)}
        )
        return reply


def validate_judge_reply(reply, return_reason=True):
    if not isinstance(reply, dict):
        raise TypeError("judge arguments must be a JSON object")
    status = reply["answer_status"]
    if not isinstance(status, str) or status not in {"Solved", "Unsolved", "Unsure"}:
        raise ValueError("judge returned an invalid canonical answer_status")
    if return_reason and not isinstance(reply["reason"], str):
        raise TypeError("judge reason must be a string")
    return reply


def recover_stage_results(records, saved_stages):
    """Recover a validated response persisted before its stage checkpoint.

    The ledger's explicit repeat/function identifies the exact scoring stage.
    Failed or malformed responses remain evidence and cost, never a saved label.
    """
    stages = deepcopy(saved_stages)
    by_key = {}
    for stage in stages:
        key = stage["repeat"], stage["function"]
        if key in by_key:
            raise ValueError("duplicate persisted judge stage")
        validate_judge_reply(stage["result"])
        by_key[key] = stage["result"]
    ledger_keys = set()
    for record in records:
        if record["error"] is not None:
            continue
        key = record["repeat"], record["judge_function"]
        raw = record["response"]
        try:
            choices = raw["choices"]
            if len(choices) != 1 or choices[0]["finish_reason"] != "tool_calls":
                continue
            message = choices[0]["message"]
            calls = message["tool_calls"]
            if message["role"] != "assistant" or len(calls) != 1:
                continue
            call = calls[0]
            if call["type"] != "function" or call["function"]["name"] != key[1]:
                continue
            reply = validate_judge_reply(json.loads(call["function"]["arguments"]))
        except (KeyError, TypeError, ValueError):
            # These are rejected provider outputs, not valid scores to replay.
            continue
        if key in ledger_keys:
            raise ValueError(
                "multiple valid provider responses exist for one judge stage"
            )
        ledger_keys.add(key)
        if key in by_key:
            if by_key[key] != reply:
                raise ValueError("judge checkpoint and provider response disagree")
        else:
            stages.append({"repeat": key[0], "function": key[1], "result": reply})
            by_key[key] = reply
    return stages


def score_repeated(
    config,
    example,
    evaluator,
    repeats,
    *,
    completed=(),
    stages=(),
    on_checkpoint,
):
    integer(repeats, 1)
    values = {"Solved": 1.0, "Unsolved": 0.0, "Unsure": 0.5}
    results = deepcopy(list(completed))
    stages = deepcopy(list(stages))
    if len(results) > repeats:
        raise ValueError("saved judge repeats exceed the frozen budget")
    for index, result in enumerate(results):
        integer(result["repeat"])
        if result["repeat"] != index:
            raise ValueError("saved judge repeats must be an ordered unique prefix")
        if (
            not isinstance(result["reason"], str)
            or type(result["value"]) not in (int, float)
            or result["value"] != values[result["answer_status"]]
        ):
            raise ValueError("saved judge status/reason/value is invalid")
    seen = set()
    for stage in stages:
        integer(stage["repeat"])
        key = stage["repeat"], stage["function"]
        if stage["repeat"] >= repeats or stage["function"] not in {
            "check_answer_status",
            "parse_answer_status",
        }:
            raise ValueError("saved judge stage is outside the frozen protocol")
        if key in seen:
            raise ValueError("duplicate saved judge stage")
        seen.add(key)
        validate_judge_reply(stage["result"])
    stage_by_key = {
        (stage["repeat"], stage["function"]): stage["result"] for stage in stages
    }
    for (index, name), result in stage_by_key.items():
        if name == "parse_answer_status":
            first = stage_by_key[index, "check_answer_status"]
            if first["answer_status"] != "Unsure":
                raise ValueError(
                    "saved detailed judge stage lacks its official Unsure trigger"
                )

    def checkpoint():
        on_checkpoint(deepcopy({"repeats": results, "stages": stages}))

    def save_stage(stage):
        stages.append(stage)
        checkpoint()

    for index in range(len(results), repeats):
        evaluator.begin_repeat(index, stages, save_stage)
        result = score_fixture(config, example, evaluator)
        result["value"] = values[result["answer_status"]]
        result["repeat"] = index
        results.append(result)
        # Persist completed replicates before attempting another provider call.
        checkpoint()
    return {
        "metric": "solvable_pass_rate",
        "value": sum(row["value"] for row in results) / repeats,
        "repeats": results,
    }
