import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_bench.benchmarks.bfcl import load_tasks, run_episode
from jev_bench.client import CompletionResult, QwenClient
from jev_bench.config import CATEGORIES

from tests.support import PROJECT, completion, config, prediction, sdk


class ScriptedClient:
    """A deterministic test actor, not a model or a measured benchmark result."""

    def __init__(self, settings, sink, actions):
        self.model, self.sink, self.actions = settings["model"], sink, actions
        self.records, self.requests, self.done = [], [], set()

    def complete(self, **request):
        self.requests.append(deepcopy(request))
        turn = sum(message["role"] == "user" for message in request["messages"]) - 1
        calls = []
        if turn not in self.done:
            self.done.add(turn)
            definitions = {
                tool["function"]["name"]: tool["function"] for tool in request["tools"]
            }
            for index, action in enumerate(self.actions[turn]):
                expression = ast.parse(action, mode="eval").body
                name = expression.func.id
                parameters = list(definitions[name]["parameters"]["properties"])
                arguments = {
                    key: ast.literal_eval(arg)
                    for key, arg in zip(parameters, expression.args)
                }
                arguments.update(
                    (kw.arg, ast.literal_eval(kw.value)) for kw in expression.keywords
                )
                calls.append(
                    {
                        "id": f"call_{turn}_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                )
        raw = completion(calls=calls)
        record = {
            "request": request,
            "response": raw.model_dump(mode="json"),
            "usage": raw.usage.model_dump(),
            "latency_seconds": 0.0,
            "error": None,
        }
        self.records.append(record)
        self.sink("executor", record)
        return CompletionResult(
            raw.choices[0].message, record["usage"], 0, "stop", record["response"]
        )

    def close(self):
        pass


class BFCLTests(unittest.TestCase):
    def test_remote_advisor_factory_cost_streams_and_close(self):
        from jev_bench.io import read_jsonl

        from tests.support import sdk
        from tests.test_remote_advisors import deepseek_response, settings

        identity = ("multi_turn_base", "multi_turn_base_0")
        value = deepcopy(self.config)
        value["conditions"]["native"].update(advice="prefer", advisor="remote")
        value["advisors"] = {"remote": settings("deepseek")}
        for reason in ("stop", "length"):
            closed = []
            endpoint = sdk(lambda reason=reason, **_: deepseek_response(reason))
            endpoint.close = lambda closed=closed: closed.append(True)
            with (
                self.subTest(reason=reason),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "cpu-only-fixture"}),
                patch("jev_bench.client.OpenAI", return_value=endpoint),
                redirect_stdout(io.StringIO()),
            ):
                output = Path(path) / "episode"
                arguments = (
                    value,
                    {
                        "condition": "native",
                        "category": identity[0],
                        "task_id": identity[1],
                        "trial": 0,
                        "attempt": 1,
                    },
                    self.tasks[identity],
                    self.answers[identity],
                    output,
                )
                row = run_episode(
                    *arguments,
                    client_factory=lambda settings, sink: ScriptedClient(
                        settings, sink, self.answers[identity]
                    ),
                )
                self.assertTrue(row["official_scored"])
                self.assertEqual(
                    row["cost"]["advisor"]["advisor_format_error_neutral"],
                    row["cost"]["advisor"]["requests"] if reason == "length" else 0,
                )
                requests = read_jsonl(output / "diagnostics/advisor_requests.jsonl")
                decisions = read_jsonl(output / "diagnostics/advisor.jsonl")
                self.assertEqual(len(requests), len(decisions))
                self.assertEqual(row["cost"]["advisor"]["requests"], len(requests))
                self.assertEqual(
                    row["cost"]["advisor"]["completion_tokens"]["observed_total"],
                    5 * len(requests),
                )
                self.assertEqual(closed, [True])

    def test_overlength_uses_neutral_without_a_prediction_or_false_match(self):
        from jev_bench.advice import NEUTRAL_MESSAGE, HTTPResponseError
        from jev_bench.io import read_jsonl

        identity = ("multi_turn_base", "multi_turn_base_0")
        value = deepcopy(self.config)
        value["conditions"]["native"].update(advice="prefer", advisor="fixture")
        value["advisors"]["fixture"] = {"type": "jev_local"}
        clients = []

        def factory(settings, sink):
            client = ScriptedClient(settings, sink, self.answers[identity])
            clients.append(client)
            return client

        advisor = SimpleNamespace(
            last_response=None,
            last_usage=None,
            records=[],
            close=lambda: None,
            predict=lambda *args: (_ for _ in ()).throw(
                HTTPResponseError(
                    422,
                    {
                        "error": {
                            "type": "ValueError",
                            "message": "over_length_after_drop",
                        }
                    },
                )
            ),
        )
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            redirect_stdout(io.StringIO()),
        ):
            output = Path(path) / "episode"
            row = run_episode(
                value,
                {
                    "condition": "native",
                    "category": identity[0],
                    "task_id": identity[1],
                    "trial": 0,
                    "attempt": 1,
                },
                self.tasks[identity],
                self.answers[identity],
                output,
                client_factory=factory,
                advisor_factory=lambda settings, sink, **kwargs: advisor,
            )
            records = read_jsonl(output / "diagnostics/advisor.jsonl")
        self.assertTrue(row["official_scored"])
        self.assertTrue(row["success"])
        self.assertEqual(
            row["cost"]["advisor"]["over_length_after_drop_neutral"],
            len(clients[0].requests),
        )
        self.assertEqual(len(records), len(clients[0].requests))
        for record, request in zip(records, clients[0].requests, strict=True):
            self.assertIsNone(record["prediction"])
            self.assertIsNone(record["first_action_matches"])
            self.assertEqual(
                request["messages"][-1], {"role": "system", "content": NEUTRAL_MESSAGE}
            )
            self.assertEqual(
                sum(
                    message.get("content") == NEUTRAL_MESSAGE
                    for message in request["messages"]
                ),
                1,
            )

    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.tasks, cls.answers = load_tasks(cls.config)

    def test_all_800_and_official_loader(self):
        from bfcl_eval.utils import load_dataset_entry

        value = deepcopy(self.config)
        value["task_ids"] = dict.fromkeys(CATEGORIES, "all")
        tasks, answers = load_tasks(value)
        self.assertEqual(len(tasks), 800)
        self.assertEqual(tasks.keys(), answers.keys())
        for category in CATEGORIES:
            official = load_dataset_entry(category, include_language_specific_hint=True)
            for task in official:
                self.assertEqual(tasks[(category, task["id"])], task)
                self.assertTrue(
                    all(
                        tool["description"].endswith(
                            " Note that the provided function is in Python 3 syntax."
                        )
                        for tool in task["function"]
                    )
                )

    def test_executor_timeout_remains_unresolved(self):
        import httpx
        from openai import APITimeoutError

        identity = ("multi_turn_base", "multi_turn_base_0")
        item = {
            "condition": "native",
            "category": identity[0],
            "task_id": identity[1],
            "trial": 0,
            "attempt": 1,
        }

        def timed_out(**request):
            raise APITimeoutError(
                request=httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
            )

        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            patch("jev_bench.client.OpenAI", return_value=sdk(timed_out)),
        ):
            row = run_episode(
                self.config,
                item,
                self.tasks[identity],
                self.answers[identity],
                Path(path) / "episode",
            )
        self.assertIsNone(row["success"])
        self.assertFalse(row["official_scored"])
        self.assertEqual(row["failure_kind"], "service_error")
        self.assertEqual(row["cost"]["executor"]["requests"], 1)
        self.assertEqual(row["cost"]["executor"]["total_tokens"]["missing"], 1)

    def test_scoring_boundary_and_advisor_interrupt_preserve_unresolved_evidence(self):
        from jev_bench.attempt import mark_scoring

        identity = ("multi_turn_base", "multi_turn_base_0")
        item = {
            "condition": "native",
            "category": identity[0],
            "task_id": identity[1],
            "trial": 0,
            "attempt": 1,
        }

        def interrupt_mark(directory):
            mark_scoring(directory)
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            output = Path(path) / "scoring"
            with (
                patch(
                    "jev_bench.benchmarks.bfcl.mark_scoring", side_effect=interrupt_mark
                ),
                redirect_stdout(io.StringIO()),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_episode(
                    self.config,
                    item,
                    self.tasks[identity],
                    self.answers[identity],
                    output,
                    client_factory=lambda settings, sink: ScriptedClient(
                        settings, sink, self.answers[identity]
                    ),
                )
            row = json.loads((output / "episode.json").read_text())
            self.assertEqual(row["failure_kind"], "scoring_error")
            self.assertFalse(row["cost"]["accounting_complete"])

            value = deepcopy(self.config)
            value["conditions"]["native"].update(advice="prefer", advisor="fixture")
            value["advisors"]["fixture"] = {"type": "jev_local"}
            advisor = SimpleNamespace(
                last_usage=None,
                records=[],
                close=lambda: None,
                predict=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
            output = Path(path) / "advisor"
            with self.assertRaises(KeyboardInterrupt), redirect_stdout(io.StringIO()):
                run_episode(
                    value,
                    item,
                    self.tasks[identity],
                    self.answers[identity],
                    output,
                    client_factory=lambda settings, sink: ScriptedClient(
                        settings, sink, self.answers[identity]
                    ),
                    advisor_factory=lambda settings, sink, **kwargs: advisor,
                )
            row = json.loads((output / "episode.json").read_text())
            self.assertEqual(row["failure_kind"], "interrupted")
            self.assertFalse(row["cost"]["accounting_complete"])

    def test_four_official_loops_and_scores(self):
        for category, number in zip(CATEGORIES, [0, 0, 0, 100]):
            task_id = f"{category}_{number}"
            task = self.tasks[(task_id.rsplit("_", 1)[0], task_id)]
            pristine = deepcopy(task)
            clients = []

            def factory(settings, sink, task_id=task_id, clients=clients):
                client = ScriptedClient(
                    settings, sink, self.answers[(task_id.rsplit("_", 1)[0], task_id)]
                )
                clients.append(client)
                return client

            with (
                self.subTest(category=category),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            ):
                item = {
                    "condition": "native",
                    "category": category,
                    "task_id": task_id,
                    "trial": 0,
                    "attempt": 1,
                }
                for repetition in range(2):
                    output = Path(path) / str(repetition)
                    with redirect_stdout(io.StringIO()):
                        row = run_episode(
                            self.config,
                            item,
                            task,
                            self.answers[(task_id.rsplit("_", 1)[0], task_id)],
                            output,
                            client_factory=factory,
                        )
                    score = json.loads((output / "score.json").read_text())
                    self.assertTrue(row["official_scored"])
                    self.assertIs(row["success"], score["valid"])
                    self.assertTrue(row["success"], score)
                    self.assertEqual(task, pristine)
                if category == "multi_turn_miss_func":
                    for request in clients[0].requests:
                        turn = (
                            sum(
                                message["role"] == "user"
                                for message in request["messages"]
                            )
                            - 1
                        )
                        names = [tool["function"]["name"] for tool in request["tools"]]
                        self.assertEqual("sort" in names, turn >= 3)

    def test_official_failure_stays_in_denominator(self):
        task_id = "multi_turn_base_0"
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            redirect_stdout(io.StringIO()),
        ):
            row = run_episode(
                self.config,
                {
                    "condition": "native",
                    "task_id": task_id,
                    "category": "multi_turn_base",
                    "trial": 0,
                    "attempt": 1,
                },
                self.tasks[(task_id.rsplit("_", 1)[0], task_id)],
                self.answers[(task_id.rsplit("_", 1)[0], task_id)],
                Path(path) / "episode",
                client_factory=lambda settings, sink: ScriptedClient(
                    settings, sink, [[], [], [], []]
                ),
            )
            self.assertFalse(row["success"])
            self.assertTrue(row["official_scored"])
            self.assertEqual(row["failure_kind"], "model_task_failure")


class RequestParityTests(unittest.TestCase):
    def test_nonfinite_and_malformed_arguments_reach_official_decoder(self):
        config()
        from jev_bench.benchmarks.handler import GuidanceHandler

        for mode in ("none", "prefer", "gate"):
            handler = GuidanceHandler(
                condition="test",
                namespace="test",
                client=SimpleNamespace(model="qwen3-4b"),
                settings=config()["executors"]["qwen4b"],
                policy={"advice": mode},
                advisor_client=None,
                event_sink=lambda *args: None,
            )
            for arguments in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', "{", "[]"):
                message = (
                    completion(
                        calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": arguments},
                            }
                        ]
                    )
                    .choices[0]
                    .message
                )
                result = CompletionResult(message, None, 0, "stop", {})
                with self.subTest(mode=mode, arguments=arguments):
                    parsed = handler._parse_query_response_FC(result)
                    self.assertEqual(parsed["model_responses"], [{"echo": arguments}])
                    if arguments in ("{", "[]"):
                        with self.assertRaises((ValueError, AttributeError)):
                            handler.decode_execute(parsed["model_responses"], False)
                    else:
                        from bfcl_eval.model_handler.utils import (
                            convert_to_function_call,
                        )

                        self.assertEqual(
                            handler.decode_execute(parsed["model_responses"], False),
                            convert_to_function_call([{"echo": arguments}]),
                        )
                        if mode == "gate":
                            from jev_bench.advice import choose_hint

                            handler._parse_query_response_FC(result)
                            policy = {
                                "advice": "gate",
                                "gate": {
                                    "rule": "respond_prob_ge",
                                    "threshold": 0.1,
                                    "loop_guard": True,
                                },
                            }
                            self.assertEqual(
                                choose_hint(
                                    policy, prediction(), handler.turn_tool_calls
                                )[1]["triggered"],
                                "loop",
                            )

    def test_hint_policies_send_expected_request_without_mutating_history(self):
        from jev_bench.benchmarks.handler import GuidanceHandler

        settings = config()["executors"]["qwen4b"]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]
        original = {
            "message": [
                {"role": "user", "content": "Echo my request"},
                {
                    "role": "assistant",
                    "content": "Working",
                    "reasoning": "Visible reasoning",
                },
            ],
            "tools": tools,
        }
        neutral = "Continue with the user's request and decide the next action yourself."
        prefer_tool = (
            "Recommended next action: call the tool `echo` first. Construct its arguments yourself "
            "from the available information. You may depart from this recommendation if it conflicts "
            "with the current request or cannot be executed with valid arguments."
        )
        prefer_respond = (
            "Recommended next action: reply to the user without calling a tool. You may depart from "
            "this recommendation if the current request still requires a tool call."
        )
        cases = [
            ("none", None, (0.8, 0.2), False, None),
            ("neutral", None, (0.8, 0.2), False, neutral),
            ("prefer", None, (0.8, 0.2), False, prefer_tool),
            ("prefer", None, (0.1, 0.9), False, prefer_respond),
            ("gate", "respond_prob_ge", (0.8, 0.2), False, neutral),
            ("gate", "respond_prob_ge", (0.95, 0.05), False, prefer_tool),
            ("gate", "top1_prob_lt", (0.8, 0.2), True, neutral),
        ]
        for mode, rule, probabilities, loop, hint in cases:
            captured = []
            mock_sdk = sdk(
                lambda **kwargs: captured.append(deepcopy(kwargs)) or completion()
            )
            with patch("jev_bench.client.OpenAI", return_value=mock_sdk):
                client = QwenClient(settings, lambda *args: None)
            policy = {"advice": mode}
            if mode == "gate":
                policy["gate"] = {
                    "rule": rule,
                    "threshold": 0.1 if rule == "respond_prob_ge" else 0.3,
                    "loop_guard": True,
                }
            pred = prediction(values=probabilities)
            handler = GuidanceHandler(
                condition=mode,
                namespace="expected_request",
                client=client,
                settings=settings,
                policy=policy,
                advisor_client=SimpleNamespace(
                    last_usage=None, predict=lambda *args, pred=pred: deepcopy(pred)
                ),
                event_sink=lambda *args: None,
            )
            if loop:
                handler.turn_tool_calls = [{"name": "echo", "arguments": {"text": "x"}}] * 2
            messages = deepcopy(original["message"])
            if hint is not None:
                messages.append({"role": "system", "content": hint})
            expected = {
                "model": "qwen3-4b",
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 8192,
                "temperature": 0.6,
                "top_p": 0.95,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": 20,
                    "min_p": 0,
                },
            }
            history = deepcopy(original)
            with self.subTest(mode=mode, rule=rule, loop=loop):
                for _ in range(2):
                    handler._query_FC(history)
                    self.assertEqual(history["message"], original["message"])
                    self.assertEqual(history["tools"], tools)
                self.assertEqual(captured, [expected, expected])
