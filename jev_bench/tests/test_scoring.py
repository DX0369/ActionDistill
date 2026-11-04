import fcntl
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jev_bench.benchmarks.bfcl import score_saved
from jev_bench.cost import costs
from jev_bench.io import append_json, read_jsonl, write_json
from jev_bench.runner import directory_for, run_lock
from jev_bench.scoring import recover_scores

from tests.support import PROJECT, config


class ScoringTests(unittest.TestCase):
    def test_missing_marker_is_explicit_and_does_not_reconstruct_ownership(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            value, plan, inputs, directory, _row = self.fixture(root)
            marker = directory / "diagnostics/attempt.json"
            marker.unlink()
            index = (root / "episodes.jsonl").read_bytes()
            with (
                patch("jev_bench.scoring.adapter") as adapter,
                self.assertRaisesRegex(ValueError, "lacks ownership evidence"),
            ):
                recover_scores(value, plan, inputs, inputs, root, code={})
            adapter.assert_not_called()
            self.assertFalse(marker.exists())
            self.assertEqual((root / "episodes.jsonl").read_bytes(), index)

    def test_official_nonfinite_tool_state_is_a_scored_failure(self):
        value = config()
        task = {
            "id": "multi_turn_base_cpu_fixture",
            "initial_config": {"VehicleControlAPI": {"engineState": "running"}},
            "involved_classes": ["VehicleControlAPI"],
        }
        gold = [["setCruiseControl(speed=50, activate=True, distanceToNextVehicle=10)"]]
        for distance in ("NaN", "Infinity"):
            with (
                self.subTest(distance=distance),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            ):
                root = Path(path)
                result = [
                    [
                        [
                            {
                                "setCruiseControl": json.dumps(
                                    {
                                        "speed": 50,
                                        "activate": True,
                                        "distanceToNextVehicle": distance,
                                    }
                                )
                            }
                        ]
                    ]
                ]
                write_json(
                    root / "trajectory.json",
                    {"id": task["id"], "result": result, "inference_log": []},
                )
                outcome = score_saved(
                    value, {"category": "multi_turn_base"}, task, gold, root
                )
                self.assertFalse(outcome["success"])
                score = json.loads((root / "score.json").read_text())
                self.assertEqual(
                    score["error"]["error_type"], "multi_turn:instance_state_mismatch"
                )
                self.assertFalse(
                    math.isfinite(
                        score["error"]["details"]["model_instance_state"][
                            "distanceToNextVehicle"
                        ]
                    )
                )
                with self.assertRaises(ValueError):
                    write_json(root / "strict.json", score)
                with patch(
                    "bfcl_eval.eval_checker.eval_runner._evaluate_single_multi_turn_entry",
                    side_effect=AssertionError("must reuse saved score"),
                ):
                    self.assertEqual(
                        score_saved(
                            value, {"category": "multi_turn_base"}, task, gold, root
                        ),
                        outcome,
                    )

    def fixture(self, root):
        value = config()
        planned = {
            "condition": "native",
            "category": "multi_turn_base",
            "task_id": "cpu",
            "trial": 0,
            "seed": 1,
        }
        item = {**planned, "attempt": 1}
        directory = directory_for(root, item)
        write_json(
            directory / "diagnostics/attempt.json",
            {"item": item, "advisor_type": None, "phase": "scoring"},
        )
        write_json(
            directory / "diagnostics/failure.json", {"error": "interrupted scoring"}
        )
        write_json(directory / "trajectory.json", {"result": "complete fixture"})
        row = {
            **item,
            "success": None,
            "official_scored": False,
            "failure_kind": "scoring_error",
            "cost": costs(None, None, advisor_type=None),
        }
        write_json(directory / "episode.json", row)
        append_json(root / "episodes.jsonl", row)
        inputs = {(planned["category"], planned["task_id"]): {}}
        return value, [planned], inputs, directory, row

    def test_recovery_preserves_attempt_cost_and_repairs_interrupted_index_write(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            value, plan, inputs, directory, original = self.fixture(root)
            scorer = Mock(return_value={"success": True, "tool_call_count": 1})
            failure = (directory / "diagnostics/failure.json").read_bytes()
            with (
                run_lock(root),
                patch(
                    "jev_bench.scoring.adapter",
                    return_value=SimpleNamespace(score_saved=scorer),
                ),
            ):
                self.assertEqual(
                    recover_scores(
                        value,
                        plan,
                        inputs,
                        inputs,
                        root,
                        code={"commit": "cpu-fixture"},
                    ),
                    1,
                )
                (row,) = read_jsonl(root / "episodes.jsonl")
                self.assertEqual(row["cost"], original["cost"])
                self.assertEqual(row["attempt"], 1)
                self.assertTrue(row["official_scored"])
                self.assertEqual(
                    (directory / "diagnostics/failure.json").read_bytes(), failure
                )
                self.assertEqual(
                    recover_scores(
                        value,
                        plan,
                        inputs,
                        inputs,
                        root,
                        code={"commit": "cpu-fixture"},
                    ),
                    0,
                )
                (root / "episodes.jsonl").write_text(json.dumps(original) + "\n")
                self.assertEqual(
                    recover_scores(
                        value,
                        plan,
                        inputs,
                        inputs,
                        root,
                        code={"commit": "cpu-fixture"},
                    ),
                    0,
                )
                self.assertEqual(read_jsonl(root / "episodes.jsonl"), [row])
                scorer.assert_called_once()

    def test_missing_trajectory_and_active_worker_cannot_be_resampled(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            value, plan, inputs, directory, _ = self.fixture(root)
            with (
                run_lock(root),
                (directory / "diagnostics/attempt.json").open() as marker,
            ):
                fcntl.flock(marker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "still active"):
                    recover_scores(
                        value,
                        plan,
                        inputs,
                        inputs,
                        root,
                        code={"commit": "cpu-fixture"},
                    )
                fcntl.flock(marker, fcntl.LOCK_UN)
                (directory / "trajectory.json").unlink()
                with self.assertRaisesRegex(ValueError, "complete saved trajectory"):
                    recover_scores(
                        value,
                        plan,
                        inputs,
                        inputs,
                        root,
                        code={"commit": "cpu-fixture"},
                    )
