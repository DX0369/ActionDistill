import importlib.util
import itertools
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from jev_bench.benchmarks.bfcl import costs
from jev_bench.config import FORMAL_TASK_COUNTS, load_config, validate
from jev_bench.extended_stats import cluster_bootstrap, cluster_permutation, holm
from jev_bench.runner import make_plan, validate_attempts
from jev_bench.stats import summarize

from tests.support import PROJECT

spec = importlib.util.spec_from_file_location(
    "formal_config", PROJECT / "scripts/prepare_formal_config.py"
)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


def fixture(category, left, right, groups=None):
    plan, rows = [], []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        for condition, value in (("ours", a), ("native", b)):
            item = {
                "condition": condition,
                "category": category,
                "task_id": str(index),
                "trial": 0,
            }
            if category.startswith("level_"):
                item["source_id"] = {"file": f"dialog_{groups[index]}", "id": index}
            plan.append(item)
            binary = category == "tasks" or category.endswith("_api")
            row = {
                **item,
                "attempt": 1,
                "official_scored": True,
                "success": bool(value) if binary else None,
                "failure_kind": "model_task_failure" if binary and not value else None,
                "cost": costs(None, None, advisor_type=None),
                "duration_seconds": 1.0,
                "decision_steps": 1,
                "tool_call_count": 0,
                "response_anomalies": [],
            }
            if not binary:
                metric = (
                    "rouge_l_f"
                    if category.endswith("_response")
                    else "solvable_pass_rate"
                )
                row.update(
                    official_score={"metric": metric, "value": value}, **{metric: value}
                )
            if category == "tasks":
                row["tool_error_counts"] = {"tool_execution_timeout": 0}
            rows.append(row)
    return plan, rows


class FormalConfigurationTests(unittest.TestCase):
    def test_core_benchmarks_keep_their_protocol_and_deployment_settings(self):
        conditions = ["native", "official_jev", "actionopd", "star"]
        for name in ("bfcl_mt", "tau2"):
            with self.subTest(benchmark=name):
                original = load_config(PROJECT / f"configs/{name}_main_smoke.json")
                saved = deepcopy(original)
                value = configuration.prepare(original, conditions)
                self.assertEqual(original, saved)
                self.assertEqual(list(value["conditions"]), conditions)
                self.assertEqual(value.get("protocol"), original.get("protocol"))
                self.assertEqual(value["executors"], original["executors"])
                self.assertEqual(value["seed"], original["seed"])
                self.assertEqual(value["output_root"], "outputs")
                self.assertEqual(value["trials"], 1)
                self.assertEqual(value["comparisons"], [])
                self.assertEqual(
                    value["task_ids"], dict.fromkeys(original["task_ids"], "all")
                )
                # Official Jev still uses the existing local renderer, not an Ours arm.
                self.assertEqual(
                    value["advisors"]["typesafe"], original["advisors"]["typesafe"]
                )
                self.assertEqual(
                    value["advisors"]["ours_jev"], original["advisors"]["ours_jev"]
                )

    def test_full_scope_and_engineering_boundaries(self):
        for name in FORMAL_TASK_COUNTS:
            with self.subTest(benchmark=name):
                original = load_config(PROJECT / f"configs/{name}_main_smoke.json")
                saved = deepcopy(original)
                value = configuration.prepare(original, ["native"])
                self.assertEqual(original, saved)
                self.assertEqual(
                    value["task_ids"], dict.fromkeys(original["task_ids"], "all")
                )
                self.assertEqual(value["output_root"], "outputs")
                self.assertEqual(value["protocol"]["mode"], "formal")
                self.assertEqual(value["comparisons"], [])
                for field, change in (
                    ("output_root", "tmp"),
                    ("trials", 2),
                    ("task_ids", original["task_ids"]),
                ):
                    bad = deepcopy(value)
                    bad[field] = change
                    with self.assertRaises(ValueError):
                        validate(bad)
                with self.assertRaisesRegex(ValueError, "formal input coverage"):
                    make_plan(value, {})

    def test_final_ours_and_pa_assets_are_explicit(self):
        template = load_config(PROJECT / "configs/toolhop_main_smoke.json")
        with self.assertRaisesRegex(ValueError, "explicit model"):
            configuration.prepare(template, ["native", "ours"])
        with self.assertRaisesRegex(ValueError, "frozen mapping"):
            configuration.prepare(template, ["pa_tool"])
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            mapping = Path(directory) / "mapping.json"
            mapping.write_text(
                "{}\n"
            )  # Config creation checks the path; execution validates contents/coverage.
            settings = template["advisors"]["ours_jev"]
            conditions = [
                "native",
                "official_jev",
                "ours",
                "actionopd",
                "star",
                "pa_tool",
                "deepseek_advisor",
            ]
            value = configuration.prepare(
                template,
                conditions,
                model=settings["model"],
                temperature=settings["temperature_file"],
                gate="G2",
                pa_mapping=mapping,
            )
            self.assertEqual(
                value["comparisons"],
                [["ours", name] for name in conditions if name != "ours"],
            )
            self.assertEqual(
                value["conditions"]["ours"]["gate"]["rule"], "top1_prob_lt"
            )
            self.assertEqual(
                value["conditions"]["pa_tool"]["pa_tool_mapping"], str(mapping)
            )
            self.assertEqual(
                template["conditions"]["ours"]["gate"]["rule"], "respond_prob_ge"
            )

    def test_plan_freezes_canonical_sources_and_rejects_changed_attempt_metadata(self):
        template = load_config(PROJECT / "configs/apibank_main_smoke.json")
        value = configuration.prepare(template, ["native"])
        tasks = {}
        for category, count in FORMAL_TASK_COUNTS["apibank"].items():
            for index in range(count):
                source = (
                    {"sample_id": index % 50, "api_id": index}
                    if category.startswith("level_3_")
                    else {"file": "dialog", "id": index}
                )
                tasks[category, str(index)] = {"source_id": source}
        plan = make_plan(value, tasks)
        self.assertEqual(len(plan), 1307)
        self.assertEqual(
            plan[0]["source_id"],
            tasks[plan[0]["category"], plan[0]["task_id"]]["source_id"],
        )
        row = {
            **plan[0],
            "attempt": 1,
            "official_scored": False,
            "success": None,
            "failure_kind": "interrupted",
        }
        validate_attempts([row], plan)
        del row["source_id"]
        with self.assertRaises(KeyError):
            validate_attempts([row], plan)
        row["source_id"] = {"file": "different", "id": 0}
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_attempts([row], plan)


class FormalStatisticsTests(unittest.TestCase):
    def test_toolhop_exact_paired_result_and_protocol_failure(self):
        plan, rows = fixture("tasks", [1] * 6, [0] * 6)
        rows[0]["tool_error_counts"]["tool_execution_timeout"] = 2
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="toolhop", formal=True
        )
        pair = result["comparisons"][0]
        self.assertEqual(pair["difference"], 1)
        self.assertEqual(pair["p_value"], 2 / 64)
        self.assertEqual(pair["bootstrap_95_percentile"], [1, 1])
        self.assertEqual(
            result["conditions"]["ours"]["tool_execution_timeout"]["observed_total"], 2
        )
        row = next(row for row in rows if row["condition"] == "native")
        row.update(official_scored=False, failure_kind="executor_response_error")
        counted = summarize(
            rows, plan, [["ours", "native"]], benchmark="toolhop", formal=True
        )
        self.assertTrue(counted["coverage_complete"])
        self.assertEqual(counted["comparisons"][0]["difference"], 1)
        self.assertEqual(
            counted["conditions"]["native"]["coverage_by_category"]["tasks"][
                "missing_scores"
            ],
            1,
        )

    def test_api_clusters_keep_point_weights_and_missing_never_reweights(self):
        plan, rows = fixture(
            "level_1_response", [1.0, 1.0, 1.0, 0.0], [0.0] * 4, groups=[0, 0, 0, 1]
        )
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="apibank", formal=True
        )
        pair = next(
            p for p in result["comparisons"] if p["category"] == "level_1_response"
        )
        self.assertEqual(pair["clusters"], 2)
        self.assertEqual(pair["difference"], 0.75)
        self.assertEqual(pair["bootstrap_95_percentile"], [0.0, 1.0])
        self.assertEqual(result["multiplicity"]["family_size"], 6)
        bad = rows[0]
        bad.update(
            official_scored=False, success=False, failure_kind="executor_response_error"
        )
        del bad["official_score"]
        del bad["rouge_l_f"]
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="apibank", formal=True
        )
        pair = next(
            p for p in result["comparisons"] if p["category"] == "level_1_response"
        )
        self.assertIsNone(pair["difference"])
        self.assertEqual(pair["missing_pairs"], 1)
        self.assertIsNone(
            result["conditions"]["ours"]["scores"]["level_1_response"]["mean"]
        )

    def test_continuous_exact_permutation_matches_enumeration(self):
        differences = [0.25, -0.125, 0.5, 0.0]
        expected = (
            sum(
                abs(sum(s * x for s, x in zip(signs, differences)))
                >= abs(sum(differences))
                for signs in itertools.product((-1, 1), repeat=4)
            )
            / 16
        )
        result = cluster_permutation(differences)
        self.assertEqual(result["p_value"], expected)
        self.assertEqual(
            result["p_value"], cluster_permutation([-x for x in differences])["p_value"]
        )
        self.assertEqual(cluster_permutation([0.0, 0.0])["p_value"], 1.0)

    def test_monte_carlo_plus_one_and_seed_are_fixed(self):
        values = [0.125] * 21
        result = cluster_permutation(values)
        self.assertEqual(result["p_method"], "monte_carlo_cluster_permutation")
        self.assertEqual(result["permutation_draws"], 100000)
        self.assertGreaterEqual(result["p_value"], 1 / 100001)
        self.assertEqual(result, cluster_permutation(values))

    def test_bootstrap_keeps_zero_clusters_and_unequal_sizes(self):
        values, sizes = [3.0, 0.0, -1.0], [3, 1, 1]
        rng = np.random.default_rng(20261015)
        indices = rng.integers(3, size=(10000, 3))
        expected = np.quantile(
            np.asarray(values)[indices].sum(axis=1)
            / np.asarray(sizes)[indices].sum(axis=1),
            [0.025, 0.975],
        ).tolist()
        self.assertEqual(cluster_bootstrap(values, sizes), expected)
        self.assertEqual(
            cluster_bootstrap([-x for x in values], sizes),
            [-x for x in reversed(expected)],
        )

    def test_holm_preserves_missing_members_of_declared_family(self):
        comparisons = [{"p_value": p} for p in (0.01, None, 0.03, 0.2)]
        holm(comparisons)
        self.assertEqual([x["p_holm"] for x in comparisons], [0.04, None, 0.09, 0.4])

    def test_stable_query_score_is_used_once_and_missing_source_is_explicit(self):
        plan, rows = fixture("G1_instruction", [1.0, 0.5], [0.5, 0.0])
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="stabletoolbench", formal=True
        )
        pair = result["comparisons"][0]
        self.assertEqual(pair["points"], 2)
        self.assertEqual(pair["difference"], 0.5)
        plan, rows = fixture("level_1_api", [1], [0], groups=[0])
        del plan[0]["source_id"]
        with self.assertRaises(KeyError):
            summarize(
                rows, plan, [["ours", "native"]], benchmark="apibank", formal=True
            )
