import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.bfcl import costs
from jev_bench.io import write_json
from jev_bench.runner import execute, pending_plan

from tests.support import PROJECT, config


class ImmediatePool:
    """Exercise real coordinator persistence without executing model code."""

    callback = None

    def __init__(self, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(type(self).callback(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 -- mirror Future's worker exception transport
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class CoordinatorTests(unittest.TestCase):
    def test_pa_asset_change_after_preflight_fails_before_model_request(self):
        from unittest.mock import Mock

        from jev_bench.benchmarks import run_episode
        from jev_bench.benchmarks.bfcl import load_tasks

        from tests.pa_fixtures import freeze, tool
        from tests.support import sdk

        value = config()
        tasks, answers = load_tasks(value)
        plan = [
            {
                "condition": "native",
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
        ]
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            p = freeze(value, plan[0], [tool()], root / "mapping")
            state = {"pa_tool_assets": {"native": p.asset_identity}}
            mapping = root / "mapping/mapping.json"
            changed = json.loads(mapping.read_text())
            changed["scopes"]["fixture"][0]["name"] = "changed_after_preflight"
            write_json(mapping, changed)
            complete = Mock(side_effect=AssertionError("must not call model"))
            ImmediatePool.callback = run_episode
            with (
                patch("jev_bench.runner.ProcessPoolExecutor", ImmediatePool),
                patch("jev_bench.client.OpenAI", return_value=sdk(complete)),
                self.assertRaisesRegex(ValueError, "frozen identity"),
            ):
                execute(
                    value,
                    plan,
                    tasks,
                    answers,
                    root / "run",
                    state,
                )
            complete.assert_not_called()
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["pa_tool_assets"], {"native": p.asset_identity})

    def test_worker_interrupt_is_recoverable_and_stops_dispatch(self):
        plan = [
            {
                "condition": "native",
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
        ]
        state = {}

        def interrupted(config, item, task, answer, directory):
            row = {
                **item,
                "success": None,
                "official_scored": False,
                "failure_kind": "interrupted",
            }
            write_json(Path(directory) / "episode.json", row)
            raise KeyboardInterrupt()

        ImmediatePool.callback = interrupted
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            patch("jev_bench.runner.ProcessPoolExecutor", ImmediatePool),
        ):
            with self.assertRaises(KeyboardInterrupt):
                execute(
                    config(),
                    plan,
                    {("multi_turn_base", "multi_turn_base_0"): {}},
                    {("multi_turn_base", "multi_turn_base_0"): []},
                    path,
                    state,
                )
            self.assertEqual(state["status"], "interrupted")
            rows = [
                json.loads(line)
                for line in (Path(path) / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(pending_plan(plan, rows)[0]["attempt"], 2)

    def test_service_error_then_explicit_resume_preserves_attempts(self):
        value = config()
        plan = [
            {
                "condition": "native",
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
        ]
        state = {"config": value, "plan": plan}

        def fake_episode(config, item, task, answer, directory):
            success = None if item["attempt"] == 1 else True
            row = {
                **item,
                "success": success,
                "official_scored": success is True,
                "failure_kind": "service_error" if success is None else None,
                "cost": costs(None, None, advisor_type=None),
                "duration_seconds": 1,
                "decision_steps": 0,
                "tool_call_count": None,
                "response_anomalies": [],
            }
            write_json(Path(directory) / "episode.json", row)
            return row

        ImmediatePool.callback = fake_episode
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            patch("jev_bench.runner.ProcessPoolExecutor", ImmediatePool),
        ):
            rows = execute(
                value,
                plan,
                {("multi_turn_base", "multi_turn_base_0"): {}},
                {("multi_turn_base", "multi_turn_base_0"): []},
                path,
                state,
            )
            self.assertEqual(state["status"], "partial")
            self.assertEqual(pending_plan(plan, rows)[0]["attempt"], 2)
            rows = execute(
                value,
                plan,
                {("multi_turn_base", "multi_turn_base_0"): {}},
                {("multi_turn_base", "multi_turn_base_0"): []},
                path,
                state,
            )
            self.assertEqual(state["phase"], "episodes_complete")
            self.assertEqual(state["status"], "running")
            self.assertEqual([row["attempt"] for row in rows], [1, 2])
            self.assertEqual(
                len((Path(path) / "episodes.jsonl").read_text().splitlines()), 2
            )

    def test_fatal_cost_is_indexed_but_never_resumed(self):
        value = config()
        plan = [
            {
                "condition": "native",
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
        ]
        state = {"config": value, "plan": plan}

        def fake_episode(config, item, task, answer, directory):
            row = {
                **item,
                "success": None,
                "official_scored": False,
                "failure_kind": "scoring_error",
                "cost": costs(None, None, advisor_type=None),
                "duration_seconds": 12,
                "decision_steps": 1,
                "tool_call_count": 1,
                "response_anomalies": [],
            }
            write_json(Path(directory) / "episode.json", row)
            raise ValueError("official scoring failed")

        ImmediatePool.callback = fake_episode
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            patch("jev_bench.runner.ProcessPoolExecutor", ImmediatePool),
        ):
            with self.assertRaisesRegex(ValueError, "official scoring failed"):
                execute(
                    value,
                    plan,
                    {("multi_turn_base", "multi_turn_base_0"): {}},
                    {("multi_turn_base", "multi_turn_base_0"): []},
                    path,
                    state,
                )
            self.assertEqual(state["status"], "failed")
            self.assertEqual(
                state["error"]["scoring_attempts"], [{**plan[0], "attempt": 1}]
            )
            row = json.loads((Path(path) / "episodes.jsonl").read_text())
            self.assertEqual(row["duration_seconds"], 12)
            with self.assertRaisesRegex(ValueError, "cannot be resumed"):
                pending_plan(plan, [row])

    def test_mixed_worker_failures_never_acquire_scoring_only_evidence(self):
        value = config()
        value["executors"]["other"] = {
            **value["executors"]["qwen4b"],
            "base_url": "http://127.0.0.1:8005/v1",
        }
        value["conditions"]["other"] = {"executor": "other", "advice": "none"}
        plan = [
            {
                "condition": condition,
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
            for condition in ("native", "other")
        ]
        inputs = {("multi_turn_base", "multi_turn_base_0"): {}}

        def fail(config, item, task, answer, directory):
            if item["condition"] == "native":
                write_json(
                    Path(directory) / "episode.json",
                    {
                        **item,
                        "success": None,
                        "official_scored": False,
                        "failure_kind": "scoring_error",
                    },
                )
            raise RuntimeError("fixture fatal")

        ImmediatePool.callback = fail
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            patch("jev_bench.runner.ProcessPoolExecutor", ImmediatePool),
        ):
            state = {}
            with self.assertRaises(RuntimeError):
                execute(value, plan, inputs, inputs, path, state)
            self.assertEqual(state["status"], "failed")
            self.assertNotIn("scoring_attempts", state["error"])
