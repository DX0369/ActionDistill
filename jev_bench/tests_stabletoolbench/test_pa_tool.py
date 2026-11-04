import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_bench.benchmarks.stabletoolbench import Archive, load_tasks
from jev_bench.benchmarks.stabletoolbench_runtime import (
    StableActor,
    catalog_order,
    convert_answer,
    run_official_chain,
)
from jev_bench.config import PROJECT_ROOT, load_config
from jev_bench.pa_tool import Projection
from tests.pa_fixtures import freeze, result, rows_for


class PaStableTests(unittest.TestCase):
    def test_original_name_failure_is_terminal_without_a_fake_sopr(self):
        from jev_bench.benchmarks.stabletoolbench_runtime import run_episode
        from jev_bench.runner import directory_for, pending_plan
        from jev_bench.stats import summarize

        config = load_config(PROJECT_ROOT / "configs/stabletoolbench_main_smoke.json")
        tasks, answers = load_tasks(config)
        identity = next(iter(tasks))
        task = tasks[identity]
        with Archive(config) as archive:
            env = archive.environment(
                task,
                catalog_order=catalog_order(archive),
                transport=lambda _: None,
                max_observation_length=1024,
                observ_compress_method="truncate",
            )
            tools = deepcopy(env.functions)

        class Client:
            def __init__(self, settings, sink):
                self.records, self.sink = [], sink

            def complete(self, **request):
                value = result([(tools[0]["function"]["name"], "{}")])
                record = {
                    "request": request,
                    "response": value.raw_response,
                    "usage": value.usage,
                    "latency_seconds": 0,
                    "error": None,
                }
                self.records.append(record)
                self.sink("executor", record)
                return value

            def close(self):
                pass

        class Backend:
            simulator = SimpleNamespace(records=[])

            def __call__(self, payload):
                raise AssertionError("invalid interface must not execute a tool")

            def close(self):
                pass

        protocol = config["protocol"]
        keys = {
            settings[field]: "cpu-fixture-secret"
            for settings, field in (
                (protocol["backend"], "toolbench_key_env"),
                (protocol["backend"]["simulator"], "api_key_env"),
                (protocol["judge"], "api_key_env"),
            )
        }
        planned = {
            "condition": "native",
            "category": identity[0],
            "task_id": identity[1],
            "trial": 0,
            "seed": 20261015,
        }
        item = {**planned, "attempt": 1}
        with (
            tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory,
            patch.dict(os.environ, keys),
        ):
            root = Path(directory)
            p = freeze(config, item, tools, root / "mapping")
            target = directory_for(root, item)
            row = run_episode(
                config,
                item,
                task,
                answers[identity],
                target,
                client_factory=Client,
                backend_factory=lambda *args: Backend(),
                judge_factory=lambda *args: self.fail("failure must not call judge"),
                pa_tool_identity=p.asset_identity,
            )
            self.assertEqual(row["failure_kind"], "executor_response_error")
            self.assertFalse(row["official_scored"])
            self.assertIsNone(row["solvable_pass_rate"])
            self.assertFalse((target / "score.json").exists())
            self.assertEqual(pending_plan([planned], [row]), [])
            summary = summarize([row], [planned], [], benchmark="stabletoolbench")
            self.assertTrue(summary["execution_coverage_complete"])
            self.assertFalse(summary["coverage_complete"])
            self.assertEqual(summary["conditions"]["native"]["protocol_failures"], 1)

    def test_official_chain_restores_tools_and_keeps_finish(self):
        config = load_config(PROJECT_ROOT / "configs/stabletoolbench_main_smoke.json")
        tasks, _ = load_tasks(config)
        task = next(iter(tasks.values()))
        requests = []
        finish = '{"return_type":"give_answer","final_answer":"scripted result"}'
        with Archive(config) as archive:
            env = archive.environment(
                task,
                catalog_order=catalog_order(archive),
                transport=lambda _: None,
                max_observation_length=1024,
                observ_compress_method="truncate",
            )
            p = Projection(env.functions, rows_for(env.functions))

            class Client:
                def complete(self, **request):
                    requests.append(deepcopy(request))
                    return result(
                        [("pa_tool_0", "{}")]
                        if len(requests) == 1
                        else [("Finish", finish)]
                    )

            actor = StableActor(
                Client(),
                config["executors"]["qwen3_4b"],
                {"advice": "none"},
                None,
                lambda *_: None,
                advisor_kind=None,
                on_messages=lambda _: None,
                projection=p,
            )
            events = []
            with patch("jev_bench.benchmarks.stabletoolbench.time.sleep"):
                raw = run_official_chain(
                    archive,
                    task,
                    actor,
                    lambda _: {"error": "", "response": "fixture"},
                    max_step=12,
                    observation_length=1024,
                    on_step=events.append,
                )
            converted = convert_answer(archive.source, raw, seed=20261015)
            self.assertEqual(converted["answer"]["final_answer"], finish)
            self.assertEqual(len(events), 2)
            self.assertEqual(requests[0]["tools"][-1], env.functions[-1])
            self.assertEqual(requests[0]["tools"][0]["function"]["name"], "pa_tool_0")
            replies = [m for m in requests[1]["messages"] if m["role"] == "tool"]
            self.assertEqual(replies[0]["name"], "pa_tool_0")
            self.assertIn(env.functions[0]["function"]["name"], str(raw))
