import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.toolhop import load_tasks, run_episode, score_saved
from jev_bench.config import PROJECT_ROOT, load_config
from jev_bench.io import read_jsonl
from tests.pa_fixtures import alias_message, freeze
from tests.support import sdk
from tests_toolhop.test_runtime import call, response


class PaToolHopTests(unittest.TestCase):
    def test_unexposed_original_name_is_terminal_failure_not_a_tool_retry(self):
        from jev_bench.runner import pending_plan, validate_attempts

        config = load_config(PROJECT_ROOT / "configs/toolhop_main_smoke.json")
        tasks, answers = load_tasks(config)
        task = tasks["tasks", "0"]
        planned = {
            "condition": "native",
            "category": "tasks",
            "task_id": "0",
            "trial": 0,
            "seed": 20261015,
        }
        item = {**planned, "attempt": 1}
        tools = [{"type": "function", "function": t} for t in task["tools"]]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory:
            root = Path(directory)
            p = freeze(config, item, tools, root / "mapping")
            with patch(
                "jev_bench.client.OpenAI",
                return_value=sdk(
                    lambda **_: response(
                        calls=[
                            call("extract_first_name", '{"full_name":"Thomas Mawson"}')
                        ]
                    )
                ),
            ):
                outcome = run_episode(
                    config,
                    item,
                    task,
                    answers["tasks", "0"],
                    root / "episode",
                    pa_tool_identity=p.asset_identity,
                )
            self.assertEqual(outcome["failure_kind"], "executor_response_error")
            self.assertFalse(outcome["success"])
            self.assertFalse(outcome["official_scored"])
            self.assertEqual(outcome["cost"]["executor"]["requests"], 1)
            validate_attempts([outcome], [planned])
            self.assertEqual(pending_plan([planned], [outcome]), [])

    def test_tool_execution_and_score_recovery_use_canonical_calls(self):
        config = load_config(PROJECT_ROOT / "configs/toolhop_main_smoke.json")
        tasks, answers = load_tasks(config)
        task = tasks["tasks", "0"]
        item = {
            "condition": "native",
            "category": "tasks",
            "task_id": "0",
            "trial": 0,
            "seed": 20261015,
            "attempt": 1,
        }
        tools = [{"type": "function", "function": t} for t in task["tools"]]
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory:
            root = Path(directory)
            p = freeze(config, item, tools, root / "mapping")
            messages = [
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
            for message in messages:
                raw = message.model_dump(mode="json")
                raw["choices"][0]["message"] = alias_message(
                    raw["choices"][0]["message"], tools, p.rows
                )
                message.choices[0].message = type(
                    message.choices[0].message
                ).model_validate(raw["choices"][0]["message"])
            requests = []

            def create(**request):
                requests.append(deepcopy(request))
                return messages.pop(0)

            with patch("jev_bench.client.OpenAI", return_value=sdk(create)):
                outcome = run_episode(
                    config,
                    item,
                    task,
                    answers["tasks", "0"],
                    root / "episode",
                    pa_tool_identity=p.asset_identity,
                )
            self.assertTrue(outcome["success"])
            self.assertEqual(outcome["tool_call_count"], 2)
            trajectory = json.loads((root / "episode/trajectory.json").read_text())
            self.assertIn("extract_first_name", str(trajectory))
            ledger = read_jsonl(root / "episode/diagnostics/executor.jsonl")
            self.assertTrue(
                ledger[0]["response"]["choices"][0]["message"]["tool_calls"][0][
                    "function"
                ]["name"].startswith("pa_tool_")
            )
            self.assertTrue(
                all(
                    t["function"]["name"].startswith("pa_tool_")
                    for t in requests[0]["tools"]
                )
            )
            (root / "episode/score.json").unlink()
            with patch(
                "jev_bench.client.OpenAI",
                side_effect=AssertionError("no inference during recovery"),
            ):
                self.assertTrue(
                    score_saved(
                        config, item, task, answers["tasks", "0"], root / "episode"
                    )["success"]
                )
