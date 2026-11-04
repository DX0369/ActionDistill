import unittest
from copy import deepcopy

from jev_bench.benchmarks.bfcl import costs
from jev_bench.config import CATEGORIES
from jev_bench.runner import validate_attempts
from jev_bench.stats import exact_permutation, paired, summarize


def fixture(sizes=(1, 1, 1, 3), trials=1):
    plan, rows = [], []
    for category, size in zip(CATEGORIES, sizes):
        for index in range(size):
            for trial in range(trials):
                for condition in ("A", "B"):
                    item = {
                        "condition": condition,
                        "task_id": f"{category}_{index}",
                        "category": category,
                        "trial": trial,
                    }
                    plan.append(item)
                    success = (condition == "A") == (category != CATEGORIES[-1])
                    rows.append(
                        {
                            **item,
                            "attempt": 1,
                            "success": success,
                            "official_scored": True,
                            "failure_kind": None if success else "model_task_failure",
                            "cost": costs(None, None, advisor_type=None),
                            "duration_seconds": 1.0,
                            "decision_steps": 2,
                            "tool_call_count": 1,
                            "response_anomalies": [],
                        }
                    )
    return plan, rows


class StatsTests(unittest.TestCase):
    def test_complete_single_category_smoke_does_not_invent_four_category_accuracy(
        self,
    ):
        plan, rows = fixture(sizes=(0, 0, 1, 0))
        result = summarize(rows, plan, [], benchmark="bfcl")
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["resolved"], 2)
        self.assertEqual(result["paired_comparisons"], [])
        for condition, accuracy in (("A", 1.0), ("B", 0.0)):
            summary = result["conditions"][condition]
            self.assertIsNone(summary["official_multi_turn_accuracy"])
            self.assertEqual(summary["categories"][CATEGORIES[2]]["accuracy"], accuracy)
            for category in (CATEGORIES[0], CATEGORIES[1], CATEGORIES[3]):
                self.assertEqual(summary["categories"][category]["planned"], 0)
                self.assertIsNone(summary["categories"][category]["accuracy"])

    def test_complete_four_category_zero_score_is_not_missing(self):
        plan, rows = fixture(sizes=(1, 1, 1, 1))
        for row in rows:
            row.update(success=False, failure_kind="model_task_failure")
        result = summarize(rows, plan, [], benchmark="bfcl")
        self.assertTrue(result["coverage_complete"])
        for condition in ("A", "B"):
            self.assertEqual(
                result["conditions"][condition]["official_multi_turn_accuracy"], 0.0
            )

    def test_interface_failure_withholds_category_mean_but_keeps_scored_evidence(self):
        cases = (
            ("apibank", "level_1_api", "api_accuracy"),
            ("apibank", "level_1_response", "rouge_l_f"),
            ("toolsandbox", "scenarios", "similarity"),
            ("stabletoolbench", "G1_instruction", "solvable_pass_rate"),
        )
        for benchmark, category, metric in cases:
            with self.subTest(benchmark=benchmark, category=category):
                plan = [
                    {
                        "condition": "pa_tool",
                        "category": category,
                        "task_id": str(i),
                        "trial": 0,
                    }
                    for i in range(2)
                ]
                template = fixture()[1][0]
                rows = [{**deepcopy(template), **item} for item in plan]
                rows[0]["official_score"] = {"metric": metric, "value": 1.0}
                if metric != "api_accuracy":
                    rows[0].update(success=None, **{metric: 1.0})
                if benchmark == "toolsandbox":
                    rows[0].update(
                        milestone_similarity=1.0, minefield_similarity=0.0, turn_count=2
                    )
                rows[1].update(
                    success=False,
                    official_scored=False,
                    failure_kind="executor_response_error",
                )
                result = summarize(rows, plan, [], benchmark=benchmark)
                self.assertTrue(result["execution_coverage_complete"])
                self.assertFalse(result["coverage_complete"])
                condition = result["conditions"]["pa_tool"]
                self.assertEqual(
                    condition["scores"][category],
                    {"metric": metric, "count": 1, "mean": None},
                )
                self.assertEqual(
                    condition["coverage_by_category"][category],
                    {
                        "planned": 2,
                        "official_scored": 1,
                        "missing_scores": 1,
                        "protocol_failures": 1,
                    },
                )
                self.assertEqual(rows[0]["official_score"]["value"], 1.0)
                self.assertNotIn("official_score", rows[1])

    def test_descriptive_adapters_do_not_use_boolean_comparisons(self):
        for benchmark in ("stabletoolbench", "toolsandbox", "apibank"):
            with self.subTest(benchmark=benchmark):
                summary = summarize([], [], [], benchmark=benchmark)
                self.assertFalse(summary["coverage_complete"])
                self.assertEqual(summary["purpose"], "engineering_smoke_only")
                self.assertEqual(summary["comparisons"], [])
                with self.assertRaisesRegex(ValueError, "boolean"):
                    paired({}, [], "a", "b", benchmark=benchmark)

    def test_category_weighting_and_exact_probability(self):
        plan, rows = fixture()
        latest = validate_attempts(rows, plan)
        result = paired(latest, plan, "A", "B", benchmark="bfcl")
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["p_exact_task_permutation"], 22 / 64)
        self.assertEqual(result["bootstrap_95_percentile"], [0.5, 0.5])
        self.assertEqual(result["bootstrap_draws"], 10000)
        self.assertEqual(result["seed"], 20261015)
        reverse = paired(latest, plan, "B", "A", benchmark="bfcl")
        self.assertEqual(reverse["difference"], -0.5)
        self.assertEqual(reverse["bootstrap_95_percentile"], [-0.5, -0.5])
        self.assertEqual(
            reverse["p_exact_task_permutation"], result["p_exact_task_permutation"]
        )

    def test_bootstrap_retains_trials_is_reproducible_and_reverses_sign(self):
        intervals = []
        for trials in (1, 3):
            plan, rows = fixture(sizes=(2, 2, 2, 2), trials=trials)
            for row in rows:
                row["success"] = row["condition"] == "A" and row["task_id"].endswith(
                    "_0"
                )
                row["failure_kind"] = None if row["success"] else "model_task_failure"
            latest = validate_attempts(rows, plan)
            result = paired(latest, plan, "A", "B", benchmark="bfcl")
            self.assertEqual(result, paired(latest, plan, "A", "B", benchmark="bfcl"))
            intervals.append(result["bootstrap_95_percentile"])
            reverse = paired(latest, plan, "B", "A", benchmark="bfcl")
            self.assertEqual(
                reverse["bootstrap_95_percentile"],
                [-value for value in reversed(result["bootstrap_95_percentile"])],
            )
        # Eight sampled tasks, not 24 independent trials: the interval stays wide.
        self.assertEqual(intervals, [[0.125, 0.875], [0.125, 0.875]])

    def test_repeats_are_not_independent_units(self):
        self.assertEqual(exact_permutation([3, 3, 0, 0]), 0.5)
        self.assertEqual(exact_permutation([0, 0]), 1.0)
        for values in ([0.5], [True], [1.9, 1.9]):
            with self.assertRaises(ValueError):
                exact_permutation(values)

    def test_missing_rows_never_produce_complete_scores(self):
        plan, rows = fixture()
        result = summarize(rows[:-1], plan, [["A", "B"]], benchmark="bfcl")
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(result["conditions"]["A"]["official_multi_turn_accuracy"])
        self.assertEqual(result["paired_comparisons"], [])
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1], plan, [], benchmark="bfcl")

    def test_resume_cost_includes_failed_attempt(self):
        plan, rows = fixture()
        failed = deepcopy(rows[0])
        failed.update(
            success=None,
            official_scored=False,
            failure_kind="service_error",
            duration_seconds=10,
        )
        rows[0]["attempt"] = 2
        result = summarize([failed] + rows, plan, [["A", "B"]], benchmark="bfcl")
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(
            result["conditions"]["A"]["official_multi_turn_accuracy"], 0.75
        )
        self.assertEqual(
            result["conditions"]["A"]["cost_all_attempts"]["duration_seconds"][
                "observed_total"
            ],
            16,
        )
        self.assertEqual(
            result["conditions"]["A"]["failures_all_attempts"]["service_error"], 1
        )

    def test_zero_usage_is_distinct_from_missing_usage(self):
        from types import SimpleNamespace

        data = costs(
            SimpleNamespace(
                records=[
                    {"usage": None, "latency_seconds": 1.0},
                    {
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "latency_seconds": 1.0,
                    },
                ]
            ),
            None,
            advisor_type=None,
        )
        self.assertEqual(
            data["executor"]["prompt_tokens"],
            {"observed_total": 0, "known": 1, "missing": 1},
        )
        self.assertEqual(data["advisor"]["requests"], 0)
