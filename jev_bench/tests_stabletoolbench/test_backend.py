"""Real pinned cache resolver with scripted transport and simulator, no network."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from jev_bench.benchmarks.stabletoolbench import Archive, load_tasks
from jev_bench.benchmarks.stabletoolbench_cache import (
    SharedCache,
    ToolbenchTransportError,
)
from jev_bench.benchmarks.stabletoolbench_runtime import (
    StableActor,
    catalog_order,
    convert_answer,
    run_official_chain,
)
from jev_bench.client import CompletionResult
from jev_bench.config import PROJECT_ROOT, load_config

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class Simulator:
    def __init__(self, settings, sink, **kwargs):
        self.requests = []
        self.records = []

    def request(self, request):
        self.requests.append(request)
        return CompletionResult(
            SimpleNamespace(content=json.dumps({"error": "", "response": "simulated"})),
            USAGE,
            0.01,
            "stop",
            {},
        )

    def close(self):
        pass


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            PROJECT_ROOT / "configs/stabletoolbench_main_smoke.json"
        )
        temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "tmp", prefix="stable_backend_"
        )
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(
                os.environ,
                {"TOOLBENCH_KEY": "cpu-toolbench", "DEEPSEEK_API_KEY": "cpu-deepseek"},
            )
        )
        self.archive = self.stack.enter_context(Archive(self.config))
        self.events = []
        self.settings = self.config["protocol"]["backend"]
        self.settings["toolbench_key_env"] = "TOOLBENCH_KEY"
        self.api = {
            "category": "Data",
            "tool_name": "open_brewery_db",
            "api_name": "breweries",
            "tool_input": "{}",
            "strip": "",
        }
        # Use a real released cache member and its exact request key.
        tool, category, api, _ = self.archive.prepare(SimpleNamespace(**self.api))
        member = f"tool_response_cache/{category}/{tool}/{api}.json"
        self.member = member
        self.published = json.loads(self.archive.zip.read(member))

    def backend(self, post):
        result = SharedCache(
            self.archive,
            self.directory / "cache",
            self.settings,
            lambda kind, event: self.events.append((kind, deepcopy(event))),
            client_factory=Simulator,
            post=post,
        )
        self.addCleanup(result.close)
        return result

    def test_real_success_overlay_is_shared_and_transport_errors_are_not_simulated(
        self,
    ):
        calls = []

        def post(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status_code=200, json=lambda: {"error": "", "response": "fixture"}
            )

        first, second = self.backend(post), self.backend(post)
        query = {**self.api, "tool_input": '{"cpu_fixture":1,"order":2}'}
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda backend: backend(query), (first, second)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(values[0], values[1])
        self.assertEqual(first.simulator.requests + second.simulator.requests, [])
        self.assertEqual(json.loads(self.archive.zip.read(self.member)), self.published)
        query["tool_input"] = '{"order":2,"cpu_fixture":1}'
        second(query)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(json.loads(second._overlay_file.read_text())), 2)
        broken = self.backend(
            lambda *a, **kw: (_ for _ in ()).throw(requests.Timeout("fixture"))
        )
        with self.assertRaises(ToolbenchTransportError):
            broken({**self.api, "tool_input": '{"cpu_fixture":3}'})
        self.assertEqual(broken.simulator.requests, [])

    def test_published_hit_and_official_miss_simulator_contract(self):
        import ast

        key = next(iter(self.published))
        value = ast.literal_eval(key)
        hit = self.backend(
            lambda *a, **kw: self.fail("published hit must not use backend")
        )
        self.assertEqual(
            hit({**self.api, "tool_input": json.dumps(value)}), self.published[key]
        )
        simulated = self.backend(lambda *a, **kw: SimpleNamespace(status_code=503))
        result = simulated({**self.api, "tool_input": '{"cpu_fixture":"miss"}'})
        self.assertEqual(result["response"], "simulated")
        request = simulated.simulator.requests[0]
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(request["temperature"], 0)
        self.assertNotIn("cpu-toolbench", json.dumps(self.events))
        self.assertEqual(
            simulated({**self.api, "tool_input": "invalid"})["error"],
            "Tool input parse error...\n",
        )
        self.assertEqual(len(simulated.simulator.requests), 1)
        simulated._overlay_file.parent.mkdir(parents=True, exist_ok=True)
        simulated._overlay_file.write_text("[]")
        with self.assertRaises(TypeError):
            simulated({**self.api, "tool_input": '{"cpu_fixture":"corrupt"}'})

    def test_official_chain_tools_then_finish_preserves_raw_arguments(self):
        tasks, _ = load_tasks(self.config)
        task = next(iter(tasks.values()))
        env = self.archive.environment(
            task,
            catalog_order=catalog_order(self.archive),
            transport=lambda payload: {"error": "", "response": "fixture"},
            max_observation_length=1024,
            observ_compress_method="truncate",
        )
        first = env.functions[0]["function"]["name"]
        finish = '{"return_type":"give_answer","final_answer":"scripted result"}'
        requests_seen = []

        class Client:
            def complete(self, **request):
                requests_seen.append(request)
                name, arguments = (
                    (first, "{}") if len(requests_seen) == 1 else ("Finish", finish)
                )
                message = SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=str(len(requests_seen)),
                            type="function",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                )
                return CompletionResult(message, USAGE, 0.01, "tool_calls", {})

        settings = self.config["executors"]["qwen3_4b"]
        actor = StableActor(
            Client(),
            settings,
            {"advice": "none"},
            None,
            lambda *a: None,
            advisor_kind=None,
            on_messages=lambda v: None,
        )
        events = []
        with patch("jev_bench.benchmarks.stabletoolbench.time.sleep") as sleep:
            raw = run_official_chain(
                self.archive,
                task,
                actor,
                lambda p: {"error": "", "response": "观" * 1025},
                max_step=12,
                observation_length=1024,
                on_step=events.append,
            )
        sleep.assert_called_once_with(2)
        converted = convert_answer(self.archive.source, raw, seed=20261015)
        self.assertEqual(converted["answer"]["final_answer"], finish)
        self.assertEqual(len(events), 2)
        self.assertEqual(
            events[0]["observation"],
            json.dumps({"error": "", "response": "观" * 1025})[:1024] + "...",
        )
        self.assertEqual(requests_seen[0]["tools"], env.functions)
        self.assertTrue(
            any(t["function"]["name"] == "Finish" for t in requests_seen[0]["tools"])
        )

    def test_official_request_wait_and_rate_limit_backoff(self):
        tasks, _ = load_tasks(self.config)
        task = next(iter(tasks.values()))
        events = []
        responses = iter(
            [
                {"error": "Rate limit per minute error...", "response": ""},
                {"error": "", "response": "fixture"},
            ]
        )

        def transport(payload):
            events.append("request")
            return next(responses)

        env = self.archive.environment(
            task,
            catalog_order=catalog_order(self.archive),
            transport=transport,
            max_observation_length=1024,
            observ_compress_method="truncate",
            sleep=lambda seconds: events.append(("sleep", seconds)),
        )
        first = env.functions[0]["function"]["name"]
        self.assertEqual(env.step(action_name=first, action_input="{}")[1], 10)
        self.assertEqual(env.step(action_name=first, action_input="{}")[1], 0)
        self.assertEqual(
            events,
            [("sleep", 2), "request", ("sleep", 10), ("sleep", 2), "request"],
        )


if __name__ == "__main__":
    unittest.main()
