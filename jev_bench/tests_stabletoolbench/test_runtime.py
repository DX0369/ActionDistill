"""CPU tests for the official chain, judge stages and saved scoring recovery."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_bench.benchmarks.stabletoolbench_runtime import (
    StableActor,
    convert_answer,
    official_chain_class,
    score_saved,
)
from jev_bench.benchmarks.stabletoolbench_scoring import (
    DeepSeekJudge,
    recover_stage_results,
    score_repeated,
)
from jev_bench.client import CompletionResult
from jev_bench.config import PROJECT_ROOT, WORKSPACE_ROOT
from jev_bench.io import append_json, read_jsonl, write_json

CONFIG = {
    "benchmark": {"path": "third_party/StableToolBench"},
    "conditions": {"native": {"advice": "none", "executor": "qwen"}},
    "advisors": {},
    "protocol": {
        "judge_repeats": 3,
        "judge": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-flash",
            "api_key_env": "STABLE_TEST_DEEPSEEK_KEY",
            "thinking": "disabled",
            "temperature": 0.2,
            "max_tokens": 1000,
            "timeout_seconds": 120,
        },
    },
}
SOURCE = WORKSPACE_ROOT / CONFIG["benchmark"]["path"]
USAGE = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
SETTINGS = {
    "max_tokens": 8192,
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
}


def raw_answer():
    arguments = '{"return_type":"give_answer","final_answer":"fixture"}'
    return {
        "answer_generation": {
            "valid_data": True,
            "query": "fixture query",
            "final_answer": arguments,
            "function": [
                {
                    "type": "function",
                    "function": {
                        "name": "Finish",
                        "description": "finish",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "train_messages": [
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "fixture query"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "finish-1",
                                "type": "function",
                                "function": {
                                    "name": "Finish",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                ]
            ],
        }
    }


def provider_record(index, function, status, *, failure=False):
    arguments = json.dumps({"answer_status": status, "reason": "scripted"})
    return {
        "repeat": index,
        "judge_function": function,
        "request": {},
        "usage": deepcopy(USAGE),
        "latency_seconds": 0.01,
        "error": {"type": "RuntimeError", "message": "injected"} if failure else None,
        "response": None
        if failure
        else {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "judge-call",
                                "type": "function",
                                "function": {
                                    "name": function,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                }
            ]
        },
    }


class ScriptedClient:
    def __init__(self, settings, sink, *, api_key, participant, plan, calls):
        self.sink, self.participant = sink, participant
        self.plan, self.calls = plan, calls
        self.records = []
        self.closed = False

    def close(self):
        self.closed = True

    def request(self, request):
        function = request["tool_choice"]["function"]["name"]
        expected, status = self.plan.pop(0)
        if function != expected:
            raise AssertionError((function, expected))
        self.calls.append(function)
        failure = isinstance(status, Exception)
        event = provider_record(
            0, function, "Solved" if failure else status, failure=failure
        )
        event["request"] = deepcopy(request)
        self.records.append(event)
        # DeepSeekJudge supplies the current repeat/function metadata here.
        self.sink(self.participant, event)
        if failure:
            raise status
        message = event["response"]["choices"][0]["message"]
        call = message["tool_calls"][0]
        return CompletionResult(
            message=SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        type="function",
                        function=SimpleNamespace(**call["function"]),
                    )
                ],
            ),
            usage=deepcopy(USAGE),
            latency_seconds=0.01,
            finish_reason="tool_calls",
            raw_response=deepcopy(event["response"]),
        )


def judge_factory(plan, calls):
    def construct(config, settings, sink):
        def client(settings, sink, **kwargs):
            return ScriptedClient(settings, sink, plan=plan, calls=calls, **kwargs)

        return DeepSeekJudge(config, settings, sink, client_factory=client)

    return construct


class StableRuntimeTests(unittest.TestCase):
    def setUp(self):
        for name in ("socket.socket.connect", "socket.getaddrinfo"):
            blocker = patch(
                name, side_effect=AssertionError("CPU test attempted network")
            )
            blocker.start()
            self.addCleanup(blocker.stop)
        key = patch.dict(os.environ, {"STABLE_TEST_DEEPSEEK_KEY": "CPU_ONLY_TEST_KEY"})
        key.start()
        self.addCleanup(key.stop)

    def test_complete_runtime_and_saved_scoring_use_continuous_summary(self):
        from jev_bench.benchmarks.stabletoolbench import load_tasks
        from jev_bench.benchmarks.stabletoolbench_runtime import run_episode
        from jev_bench.config import load_config
        from jev_bench.runner import directory_for, validate_attempts
        from jev_bench.scoring import recover_scores
        from jev_bench.stats import summarize

        config = load_config(PROJECT_ROOT / "configs/stabletoolbench_main_smoke.json")
        config["protocol"]["judge"]["api_key_env"] = "STABLE_TEST_DEEPSEEK_KEY"
        tasks, answers = load_tasks(config)
        identity = next(iter(tasks))
        planned = {"condition": "native", "category": identity[0], "task_id": identity[1], "trial": 0, "seed": 42}
        item = {**planned, "attempt": 1}
        closed = []
        class Actor:
            def __init__(self, settings, sink):
                self.records, self.sink = [], sink
            def complete(self, **request):
                arguments = '{"return_type":"give_answer","final_answer":"fixture"}'
                record = {"request": request, "usage": deepcopy(USAGE), "latency_seconds": 0.01, "error": None, "response": {}}
                self.records.append(record)
                self.sink("executor", record)
                return CompletionResult(SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(
                    type="function", id="finish", function=SimpleNamespace(name="Finish", arguments=arguments)
                )]), deepcopy(USAGE), 0.01, "tool_calls", {})
            def close(self):
                closed.append("actor")
        class Backend:
            def __init__(self, *args):
                self.simulator = SimpleNamespace(records=[])
            def __call__(self, payload):
                raise AssertionError("Finish must not call the backend")
            def close(self):
                closed.append("backend")
        calls = []
        factory = judge_factory([("check_answer_status", "Solved")]*3, calls)
        env = {config["protocol"]["backend"]["toolbench_key_env"]: "cpu-key",
               config["protocol"]["backend"]["simulator"]["api_key_env"]: "cpu-key"}
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as folder, patch.dict(os.environ, env):
            root = Path(folder)
            directory = directory_for(root, item)
            row = run_episode(config, item, tasks[identity], answers[identity], directory,
                              client_factory=Actor, backend_factory=Backend, judge_factory=factory)
            self.assertIsNone(row["success"])
            self.assertEqual(row["solvable_pass_rate"], 1)
            self.assertEqual(row["tool_call_count"], 1)
            self.assertEqual(closed, ["backend", "actor"])
            validate_attempts([row], [planned])
            self.assertEqual(summarize([row], [planned], [], benchmark="stabletoolbench")["conditions"]["native"]["cost_all_attempts"]["judge"]["requests"], 3)
            # Simulate a crash after score.json, before episode/index completion.
            row.update(official_scored=False, failure_kind="scoring_error")
            row.pop("official_score")
            row.pop("solvable_pass_rate")
            write_json(directory / "episode.json", row)
            append_json(root / "episodes.jsonl", row)
            with patch("jev_bench.client.OpenAI", side_effect=AssertionError("saved score must not call providers")):
                self.assertEqual(recover_scores(config, [planned], tasks, answers, root, code={}), 1)
            restored = read_jsonl(root / "episodes.jsonl")[0]
            self.assertEqual(restored["cost"], row["cost"])
            self.assertTrue(restored["official_scored"])
            self.assertEqual(len(calls), 3)

    def test_respond_hint_maps_to_finish_without_rewriting_candidates_or_history(self):
        from jev_bench.advice import PREFER_TOOL

        tools = raw_answer()["answer_generation"]["function"]
        seen = []
        class Advisor:
            def predict(self, messages, candidates):
                seen.append(deepcopy(candidates))
                return {"top1":"RESPOND", "probabilities":[{"name":"Finish","probability":0.1},{"name":"RESPOND","probability":0.9}]}
            last_usage = None
        client = SimpleNamespace(complete=lambda **kw: CompletionResult(SimpleNamespace(
            role="assistant", content=None, tool_calls=[SimpleNamespace(type="function",id="f",function=SimpleNamespace(name="Finish",arguments='{}'))]
        ), USAGE,0.01,"tool_calls",{}))
        actor=StableActor(client, SETTINGS, {"advice":"prefer"}, Advisor(),lambda *a: None,advisor_kind="jev_local",on_messages=lambda v:None)
        original=[{"role":"user","content":"fixture"}]
        actor.change_messages(original)
        with patch.object(client,"complete",wraps=client.complete) as request:
            actor.parse(tools=tools,process_id=1)
        record=actor.advisor_records[0]
        self.assertEqual(record["prediction"]["top1"],"RESPOND")
        self.assertEqual(record["hint"],PREFER_TOOL.format(name="Finish"))
        self.assertTrue(record["first_action_matches"])
        self.assertEqual(seen,[tools])
        self.assertEqual(request.call_args.kwargs["messages"][-1]["content"],record["hint"])
        self.assertEqual(actor.messages,original)

    def test_deepseek_thought_is_neutral_and_never_counts_as_finish_adoption(self):
        from jev_bench.advice import NEUTRAL_MESSAGE, PREFER_TOOL, DeepSeekClient
        from jev_bench.cost import costs
        from jev_bench.stats import aggregate_cost
        from openai.types.chat import ChatCompletion

        tools = raw_answer()["answer_generation"]["function"]
        finish = {"id": "f", "type": "function", "function": {"name": "Finish", "arguments": "{}"}}
        for thought in (True, False):
            raw = {
                "id": "fixture", "object": "chat.completion", "created": 0,
                "model": "deepseek-fixture",
                "choices": [{"index": 0, "finish_reason": "stop" if thought else "tool_calls",
                             "message": {"role": "assistant", "content": "I need more information" if thought else None,
                                         "tool_calls": [] if thought else [finish]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            requests, saved = [], []

            def create(requests=requests, raw=raw, **request):
                requests.append(deepcopy(request))
                return ChatCompletion(**raw)

            settings = {
                "type": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                "api_key_env": "DEEPSEEK_API_KEY", "timeout_seconds": 120, "max_tokens": 16384,
                "reasoning_effort": "high",
            }
            with (
                self.subTest(thought=thought),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "cpu-key"}),
                patch("jev_bench.client.OpenAI", return_value=SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)),
            ):
                advisor = DeepSeekClient(settings, lambda kind, row, saved=saved: saved.append((kind, deepcopy(row))))
                self.addCleanup(advisor.close)
                client = SimpleNamespace(complete=lambda **kw: CompletionResult(SimpleNamespace(
                    role="assistant", content=None, tool_calls=[SimpleNamespace(
                        type="function", id="f", function=SimpleNamespace(name="Finish", arguments="{}"))]
                ), USAGE, 0.01, "tool_calls", {}))
                actor = StableActor(client, SETTINGS, {"advice": "prefer"}, advisor, lambda *args: None,
                                    advisor_kind="deepseek", on_messages=lambda value: None)
                original = [{"role": "user", "content": "fixture"}]
                actor.change_messages(original)
                actor.parse(tools=tools, process_id=1)
            record = actor.advisor_records[0]
            self.assertEqual(requests[0]["tools"], tools)
            self.assertEqual(actor.messages, original)
            if thought:
                self.assertEqual(record["raw_action"], "Thought")
                self.assertEqual(record["fallback"], "stable_thought_neutral")
                self.assertIsNone(record["prediction"])
                self.assertIsNone(record["first_action_matches"])
                self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
            else:
                self.assertEqual(record["prediction"]["top1"], "Finish")
                self.assertEqual(record["hint"], PREFER_TOOL.format(name="Finish"))
                self.assertTrue(record["first_action_matches"])
            ledger = json.loads(json.dumps([row for kind, row in saved if kind == "advisor_requests"]))
            cost = costs(None, actor, advisor_type="deepseek", advisor_requests=ledger)
            self.assertEqual(cost["advisor"]["stable_thought_neutral"], int(thought))
            self.assertEqual(cost["advisor"]["advisor_format_error_neutral"], 0)
            self.assertEqual(cost["advisor"]["provider_requests"], 1)
            summary = aggregate_cost([{"cost": cost, "duration_seconds": 0, "decision_steps": 1,
                                       "tool_call_count": 1, "response_anomalies": []}])
            self.assertEqual(summary["advisor"]["stable_thought_neutral"], int(thought))

    def test_empty_string_advances_official_tree_but_null_empty_fails(self):
        class Env:
            task_description = "fixture"
            input_description = "query"

            def __init__(self):
                self.functions = []

            def check_success(self):
                return 0

            def to_json(self):
                return {}

        class Client:
            def complete(self, **kwargs):
                return CompletionResult(
                    SimpleNamespace(role="assistant", content="", tool_calls=None),
                    deepcopy(USAGE),
                    0.0,
                    "stop",
                    {},
                )

        actor = StableActor(
            Client(),
            SETTINGS,
            {"advice": "none"},
            None,
            lambda *args: None,
            advisor_kind=None,
            on_messages=lambda value: None,
        )
        chain = official_chain_class(SOURCE)(actor, Env(), process_id=1)
        chain.start(single_chain_max_step=12, pass_at=1, answer=1)
        self.assertEqual(actor.decision_steps, 12)
        self.assertEqual(chain.terminal_node[-1].get_depth(), 12)
        self.assertTrue(chain.terminal_node[-1].pruned)
        self.assertEqual(chain.total_tokens, 36)
        missing = CompletionResult(
            SimpleNamespace(role="assistant", content=None, tool_calls=None),
            deepcopy(USAGE),
            0.0,
            "stop",
            {},
        )
        with (
            patch.object(actor.client, "complete", return_value=missing),
            self.assertRaisesRegex(ValueError, "neither content"),
        ):
            actor.parse(tools=[], process_id=1)
        missing.usage = None
        with (
            patch.object(actor.client, "complete", return_value=missing),
            self.assertRaisesRegex(ValueError, "canonical usage"),
        ):
            actor.parse(tools=[], process_id=1)

    def test_unsure_stage_resume_does_not_repeat_successful_first_stage(self):
        example = convert_answer(SOURCE, raw_answer(), seed=42)
        state = {"repeats": [], "stages": []}
        records = []

        def checkpoint(value):
            state.clear()
            state.update(deepcopy(value))

        calls = []
        judge = judge_factory(
            [
                ("check_answer_status", "Unsure"),
                ("parse_answer_status", RuntimeError("injected")),
            ],
            calls,
        )(
            CONFIG,
            CONFIG["protocol"]["judge"],
            lambda kind, event: records.append(deepcopy(event)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                score_repeated(CONFIG, example, judge, 3, on_checkpoint=checkpoint)
        finally:
            judge.close()
        self.assertEqual(state["repeats"], [])
        self.assertEqual(len(state["stages"]), 1)
        resumed_calls = []
        resumed = judge_factory(
            [
                ("parse_answer_status", "Solved"),
                ("check_answer_status", "Unsolved"),
                ("check_answer_status", "Solved"),
            ],
            resumed_calls,
        )(
            CONFIG,
            CONFIG["protocol"]["judge"],
            lambda kind, event: records.append(deepcopy(event)),
        )
        try:
            result = score_repeated(
                CONFIG,
                example,
                resumed,
                3,
                completed=state["repeats"],
                stages=recover_stage_results(records, state["stages"]),
                on_checkpoint=checkpoint,
            )
        finally:
            resumed.close()
        self.assertEqual(
            resumed_calls,
            ["parse_answer_status", "check_answer_status", "check_answer_status"],
        )
        self.assertAlmostEqual(result["value"], 2 / 3)
        self.assertEqual(len(records), 5)  # includes the failed first detailed request

    def test_score_saved_keeps_inference_and_counts_failed_and_new_judge_calls(self):
        item = {
            "condition": "native",
            "category": "G1_instruction",
            "task_id": "fixture",
            "trial": 0,
            "seed": 42,
            "attempt": 1,
        }
        task = {"query": "fixture query"}
        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tmp", prefix="stable-draft-"
        ) as temporary:
            directory = Path(temporary)
            write_json(
                directory / "trajectory.json",
                {"official": raw_answer(), "partial": False},
            )
            before = (directory / "trajectory.json").read_bytes()
            completed = {
                "repeat": 0,
                "answer_status": "Solved",
                "reason": "scripted",
                "value": 1.0,
            }
            stage = {
                "repeat": 0,
                "function": "check_answer_status",
                "result": {"answer_status": "Solved", "reason": "scripted"},
            }
            write_json(
                directory / "diagnostics/judge_repeats.json",
                {"repeats": [completed], "stages": [stage]},
            )
            append_json(
                directory / "diagnostics/judge.jsonl",
                provider_record(0, "check_answer_status", "Solved"),
            )
            write_json(
                directory / "episode.json",
                {
                    **item,
                    "success": None,
                    "official_scored": False,
                    "failure_kind": "scoring_error",
                    "cost": {"accounting_complete": True},
                },
            )
            first_calls = []
            with (
                patch(
                    "jev_bench.benchmarks.stabletoolbench_runtime.QwenClient",
                    side_effect=AssertionError("actor rerun"),
                ),
                patch(
                    "jev_bench.benchmarks.stabletoolbench_runtime.SharedCache",
                    side_effect=AssertionError("tool rerun"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected"),
            ):
                score_saved(
                    CONFIG,
                    item,
                    task,
                    {},
                    directory,
                    judge_factory=judge_factory(
                        [
                            ("check_answer_status", "Unsolved"),
                            ("check_answer_status", RuntimeError("injected")),
                        ],
                        first_calls,
                    ),
                )
            partial = json.loads((directory / "episode.json").read_text())
            self.assertEqual(partial["cost"]["judge"]["requests"], 3)
            state = json.loads(
                (directory / "diagnostics/judge_repeats.json").read_text()
            )
            self.assertEqual(len(state["repeats"]), 2)
            second_calls = []
            outcome = score_saved(
                CONFIG,
                item,
                task,
                {},
                directory,
                judge_factory=judge_factory(
                    [
                        ("check_answer_status", "Solved"),
                    ],
                    second_calls,
                ),
            )
            self.assertEqual(second_calls, ["check_answer_status"])
            self.assertEqual(outcome["cost"]["judge"]["requests"], 4)
            self.assertEqual(
                outcome["cost"]["judge"]["total_tokens"]["observed_total"], 12
            )
            self.assertEqual(outcome["cost"]["judge"]["total_tokens"]["known"], 4)
            self.assertEqual(
                outcome["solvable_pass_rate"], outcome["official_score"]["value"]
            )
            self.assertAlmostEqual(outcome["solvable_pass_rate"], 2 / 3)
            self.assertEqual((directory / "trajectory.json").read_bytes(), before)
            self.assertEqual(len(read_jsonl(directory / "diagnostics/judge.jsonl")), 4)
            # A final saved score is read-only with respect to providers.
            repeated = score_saved(
                CONFIG,
                item,
                task,
                {},
                directory,
                judge_factory=lambda *a: self.fail("judge rerun"),
            )
            self.assertEqual(repeated["cost"]["judge"]["requests"], 4)

    def test_valid_ledger_stage_survives_a_missing_checkpoint(self):
        records = [provider_record(0, "check_answer_status", "Solved")]
        stages = recover_stage_results(records, [])
        self.assertEqual(len(stages), 1)
        calls = []
        judge = judge_factory(
            [
                ("check_answer_status", "Unsolved"),
                ("check_answer_status", "Solved"),
            ],
            calls,
        )(
            CONFIG,
            CONFIG["protocol"]["judge"],
            lambda kind, event: records.append(deepcopy(event)),
        )
        try:
            result = score_repeated(
                CONFIG,
                convert_answer(SOURCE, raw_answer(), seed=42),
                judge,
                3,
                stages=stages,
                on_checkpoint=lambda value: None,
            )
        finally:
            judge.close()
        self.assertEqual(calls, ["check_answer_status", "check_answer_status"])
        self.assertEqual([row["repeat"] for row in records], [0, 1, 2])
        self.assertAlmostEqual(result["value"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
