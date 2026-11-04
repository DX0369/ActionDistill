"""CPU-only SDK fixtures through the real ToolHop loop and embedded scorer."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from jev_bench.advice import (
    NEUTRAL_MESSAGE,
    OPTION_ORDER,
    PREFER_TOOL,
    HTTPResponseError,
)
from jev_bench.benchmarks.toolhop import load_tasks, run_episode
from jev_bench.config import PROJECT_ROOT, load_config
from jev_bench.io import read_jsonl, write_json
from jev_bench.stats import summarize
from openai import APIConnectionError
from openai.types.chat import ChatCompletion


def response(content=None, calls=None):
    return ChatCompletion.model_validate(
        {
            "id": "cpu",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-4b",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": calls,
                        "reasoning": "CPU reasoning fixture",
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def call(name, arguments):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT_ROOT / "configs/toolhop_advisor_smoke.json")
        self.tasks, self.answers = load_tasks(self.config)
        temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tmp", prefix="toolhop_runtime_"
        )
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "attempt"
        self.requests = []
        self.advisor_inputs = []
        for name in ("socket.socket.connect", "socket.getaddrinfo"):
            blocker = patch(
                name, side_effect=AssertionError("CPU test attempted network")
            )
            blocker.start()
            self.addCleanup(blocker.stop)

    def predict(self, url, timeout, payload):
        self.advisor_inputs.append(deepcopy(payload))
        settings = self.config["advisors"]["a38k"]
        names = [tool["name"] for tool in payload["tools"]] + ["RESPOND"]
        values = [0.7] + [0.1 / (len(names) - 2)] * (len(names) - 2) + [0.2]
        return {
            "model": settings["model"],
            "temperature_file": settings["temperature_file"],
            "temperature": json.loads(Path(settings["temperature_file"]).read_text())[
                "temperature"
            ],
            "prompt_budget": 32704,
            "option_order": OPTION_ORDER,
            "duplicate_candidates": "positional",
            "probabilities": [
                {"name": name, "probability": value}
                for name, value in zip(names, values)
            ],
            "top1": names[0],
            "n_prompt_tokens": 30,
            "truncated": False,
            "messages_dropped": 0,
            "truncation_counts": {
                "images_replaced": 0,
                "skipped_framework_greeting": 0,
            },
            "latency_seconds": 0.01,
        }

    def run_episode(self, replies, *, condition="native", identity="0", advisor=None):
        replies = iter(replies)

        def create(**request):
            self.requests.append(deepcopy(request))
            result = next(replies)
            if isinstance(result, BaseException):
                raise result
            return result

        sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=lambda: None,
        )
        item = {
            "condition": condition,
            "category": "tasks",
            "task_id": identity,
            "trial": 0,
            "seed": 20261015,
            "attempt": 1,
        }
        with (
            patch("jev_bench.client.OpenAI", return_value=sdk),
            patch("jev_bench.advice.json_request", side_effect=advisor or self.predict),
        ):
            return run_episode(
                self.config,
                item,
                self.tasks[("tasks", identity)],
                self.answers[("tasks", identity)],
                self.directory,
            )

    def test_score_recovery_reuses_saved_loop_without_tools_or_models(self):
        from jev_bench.benchmarks.toolhop import score_saved

        row = self.run_episode([response("<answer>4</answer>")])
        expected = row["success"]
        (self.directory / "score.json").unlink()
        with (
            patch("jev_bench.client.OpenAI", side_effect=AssertionError("no model")),
            patch(
                "jev_bench.benchmarks.toolhop.run_fixture",
                side_effect=AssertionError("no tools"),
            ),
        ):
            result = score_saved(
                self.config,
                row,
                self.tasks["tasks", "0"],
                self.answers["tasks", "0"],
                self.directory,
            )
            self.assertEqual(result["success"], expected)
            self.assertEqual(
                score_saved(
                    self.config,
                    row,
                    self.tasks["tasks", "0"],
                    self.answers["tasks", "0"],
                    self.directory,
                ),
                result,
            )

    def test_native_sdk_to_official_execution_and_score(self):
        row = self.run_episode(
            [
                response(
                    calls=[call("extract_first_name", '{"full_name":"Thomas Mawson"}')]
                ),
                response(
                    calls=[
                        call(
                            "count_letters",
                            '{"input":"Thomas","ignore_position":["first","last"]}',
                        )
                    ]
                ),
                response("<answer>4</answer>"),
            ]
        )
        self.assertIs(row["success"], True)
        self.assertTrue(row["official_scored"])
        self.assertEqual((row["decision_steps"], row["tool_call_count"]), (3, 2))
        self.assertEqual(row["cost"]["executor"]["requests"], 3)
        self.assertEqual(
            len(read_jsonl(self.directory / "diagnostics/executor.jsonl")), 3
        )
        for request in self.requests:
            self.assertEqual(
                (request["tool_choice"], request["max_tokens"]), ("auto", 8192)
            )
            self.assertEqual(
                request["extra_body"],
                {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": 20,
                    "min_p": 0,
                },
            )
            self.assertEqual((request["temperature"], request["top_p"]), (0.6, 0.95))
        self.assertEqual(
            self.requests[1]["messages"][1]["reasoning"], "CPU reasoning fixture"
        )
        trajectory = json.loads((self.directory / "trajectory.json").read_text())
        self.assertEqual(set(trajectory), {"id", "messages", "turn", "partial"})
        self.assertIs(trajectory["partial"], False)
        self.assertEqual(trajectory["turn"], 2)
        for source in self.answers[("tasks", "0")]["functions"]:
            self.assertNotIn(source, json.dumps(self.requests))
        plan = [
            {
                key: row[key]
                for key in ("condition", "category", "task_id", "trial", "seed")
            }
        ]
        metrics = summarize([row], plan, [], benchmark="toolhop")
        self.assertTrue(metrics["coverage_complete"])
        self.assertEqual(metrics["conditions"]["native"]["official_correct"], 1)
        self.assertNotIn("accuracy", metrics["conditions"]["native"])

    def test_hints_are_request_local_with_duplicate_candidates(self):
        row = self.run_episode(
            [
                response("one"),
            ],
            condition="prefer",
            identity="44",
        )
        tools = self.tasks[("tasks", "44")]["tools"]
        names = [tool["name"] for tool in tools]
        self.assertLess(len(set(names)), len(names))
        self.assertEqual(self.advisor_inputs[0]["tools"], tools)
        self.assertEqual(
            [tool["function"] for tool in self.requests[0]["tools"]], tools
        )
        record = read_jsonl(self.directory / "diagnostics/advisor.jsonl")[0]
        self.assertEqual(len(record["prediction"]["raw_probabilities"]), len(tools) + 1)
        self.assertEqual(
            self.requests[0]["messages"][-1]["content"],
            PREFER_TOOL.format(name=record["prediction"]["top1"]),
        )
        trajectory = json.loads((self.directory / "trajectory.json").read_text())
        self.assertFalse(
            any(message["role"] == "system" for message in trajectory["messages"])
        )
        self.assertEqual(row["cost"]["advisor"]["requests"], 1)

    def test_gate_and_named_neutral_fallback_do_not_persist(self):
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                self.directory = self.directory.parent / f"gate_{fallback}"
                self.requests.clear()
                self.advisor_inputs.clear()

                def predict(url, timeout, payload, fallback=fallback):
                    if fallback:
                        raise HTTPResponseError(
                            422,
                            {
                                "error": {
                                    "type": "ValueError",
                                    "message": "over_length_after_drop",
                                }
                            },
                        )
                    return self.predict(url, timeout, payload)

                self.run_episode(
                    [response(calls=[call("unknown", "")]), response("done")],
                    condition="G1",
                    advisor=predict,
                )
                for request in self.requests:
                    self.assertEqual(
                        sum(
                            message["role"] == "system"
                            for message in request["messages"]
                        ),
                        1,
                    )
                    self.assertEqual(
                        request["messages"][-1]["content"], NEUTRAL_MESSAGE
                    )
                records = read_jsonl(self.directory / "diagnostics/advisor.jsonl")
                self.assertEqual(len(records), 2)
                if fallback:
                    self.assertTrue(
                        all(
                            record["prediction"] is None
                            and record["gate"] is None
                            and record["first_action_matches"] is None
                            for record in records
                        )
                    )
                else:
                    self.assertTrue(
                        all(
                            record["gate"]["triggered"] == "confidence"
                            for record in records
                        )
                    )
                    self.assertTrue(
                        all(
                            not any(
                                message["role"] == "system"
                                for message in payload["messages"]
                            )
                            for payload in self.advisor_inputs
                        )
                    )
                self.assertIn(
                    "an error occured", self.requests[1]["messages"][-2]["content"]
                )

    def test_free_nine_turn_limit_and_loop_guard_use_parsed_calls(self):
        name = self.tasks[("tasks", "0")]["tools"][0]["name"]
        row = self.run_episode(
            [response(calls=[call(name, "{}")]) for _ in range(9)], condition="G2"
        )
        records = read_jsonl(self.directory / "diagnostics/advisor.jsonl")
        self.assertEqual(
            [record["gate"]["triggered"] for record in records[:3]],
            ["none", "none", "loop"],
        )
        self.assertEqual((row["decision_steps"], row["tool_call_count"]), (9, 9))
        self.assertTrue(row["official_scored"])
        self.assertFalse(row["success"])
        self.assertEqual(
            json.loads((self.directory / "trajectory.json").read_text())["turn"], 8
        )

    def test_service_error_stays_unresolved_and_counts_failed_request(self):
        error = APIConnectionError(
            request=httpx.Request("POST", "http://127.0.0.1:8001/v1/chat/completions")
        )
        row = self.run_episode([error])
        self.assertIsNone(row["success"])
        self.assertEqual(row["failure_kind"], "service_error")
        self.assertEqual(row["cost"]["executor"]["requests"], 1)
        self.assertFalse((self.directory / "score.json").exists())
        self.assertTrue(
            json.loads((self.directory / "trajectory.json").read_text())["partial"]
        )

    def test_advisor_failure_is_logged_once_without_executor_request(self):
        row = self.run_episode(
            [],
            condition="prefer",
            advisor=lambda *args: (_ for _ in ()).throw(
                OSError("fixture service down")
            ),
        )
        self.assertIsNone(row["success"])
        self.assertEqual(row["failure_kind"], "service_error")
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(
            len(read_jsonl(self.directory / "diagnostics/advisor.jsonl")), 1
        )

    def test_score_write_failure_retains_complete_inference(self):
        def write(path, value):
            if path.name == "score.json":
                raise OSError("score fixture failure")
            write_json(path, value)

        with (
            patch("jev_bench.benchmarks.toolhop.write_json", side_effect=write),
            self.assertRaisesRegex(OSError, "score fixture failure"),
        ):
            self.run_episode([response("<answer>4</answer>")])
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(
            json.loads((self.directory / "trajectory.json").read_text())["partial"]
        )
        self.assertEqual(
            json.loads((self.directory / "diagnostics/attempt.json").read_text())[
                "phase"
            ],
            "scoring",
        )
        self.assertEqual(
            json.loads((self.directory / "episode.json").read_text())["failure_kind"],
            "scoring_error",
        )

    def test_partial_tool_batch_does_not_claim_zero_executed_calls(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception):
                self.directory = self.directory.parent / exception.__name__
                completed = []

                def timer(
                    timeout,
                    function,
                    args=(),
                    kwargs=None,
                    exception=exception,
                    completed=completed,
                ):
                    if completed:
                        raise exception("CPU interruption")
                    result = function(*args, **(kwargs or {}))
                    completed.append(result)
                    return result

                with (
                    patch("func_timeout.dafunc.func_timeout", side_effect=timer),
                    self.assertRaises(exception),
                ):
                    self.run_episode(
                        [
                            response(
                                calls=[
                                    call("count_letters", '{"input":"four"}'),
                                    call("count_letters", '{"input":"five"}'),
                                ]
                            )
                        ]
                    )
                self.assertEqual(completed, [4])
                row = json.loads((self.directory / "episode.json").read_text())
                self.assertIsNone(row["tool_call_count"])
                self.assertIsNone(row["success"])
                self.assertFalse(row["cost"]["accounting_complete"])
                self.assertTrue(
                    json.loads((self.directory / "trajectory.json").read_text())[
                        "partial"
                    ]
                )
                self.assertFalse((self.directory / "score.json").exists())


if __name__ == "__main__":
    unittest.main()
