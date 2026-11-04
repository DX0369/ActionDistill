import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.attempt import mark_scoring, record_attempt, recover_interrupted
from jev_bench.config import CATEGORIES, load_config, validate
from jev_bench.io import append_json, write_json
from jev_bench.runner import (
    directory_for,
    make_chains,
    make_plan,
    pending_plan,
    reconcile,
    run_lock,
    validate_attempts,
)
from jev_bench.services import check_services
from jev_bench.stats import aggregate_cost

from tests.support import PROJECT, advisor_settings, config

NATIVE_ONLY = {
    "benchmark": {"name": "bfcl"},
    "conditions": {"native": {"executor": "qwen", "advice": "none"}},
}


class ConfigTests(unittest.TestCase):
    def test_approved_wifi_description_smokes_do_not_open_other_scenarios(self):
        value = load_config(PROJECT / "configs/toolsandbox_pa_tool_smoke.json")
        value["task_ids"]["scenarios"] = [
            "wifi_off_3_distraction_tools_tool_description_scrambled",
            "wifi_off_3_distraction_tools_arg_description_scrambled",
        ]
        validate(value)
        value["task_ids"]["scenarios"] = ["wifi_off"]
        with self.assertRaisesRegex(ValueError, "approved local scenarios"):
            validate(value)

    def test_main_templates_allow_standalone_star(self):
        for template in (
            "bfcl_mt",
            "tau2",
            "toolhop",
            "apibank",
            "toolsandbox",
            "stabletoolbench",
        ):
            with self.subTest(benchmark=template):
                value = load_config(PROJECT / f"configs/{template}_main_smoke.json")
                executor = value["conditions"]["star"]["executor"]
                value["conditions"] = {"star": value["conditions"]["star"]}
                value["executors"] = {executor: value["executors"][executor]}
                value["advisors"] = {}
                validate(value)
                task = next(iter(value["task_ids"].items()))
                plan = make_plan(value, {(task[0], task[1][0]): {}})
                self.assertEqual(len(plan), 1)
                self.assertEqual(plan[0]["condition"], "star")
                self.assertEqual(
                    list(make_chains(value, plan)),
                    [value["executors"][executor]["base_url"]],
                )

    def test_continuous_scores_require_canonical_numeric_values(self):
        for category, metric in (
            ("level_1_response", "rouge_l_f"),
            ("G1_instruction", "solvable_pass_rate"),
        ):
            planned = {
                "condition": "native",
                "category": category,
                "task_id": "fixture",
                "trial": 0,
                "seed": 1,
            }
            row = {
                **planned,
                "attempt": 1,
                "success": None,
                "official_scored": True,
                "failure_kind": None,
                "official_score": {"metric": metric, "value": 1.0},
                metric: 1.0,
            }
            validate_attempts([row], [planned])
            for change in (
                {"success": True},
                {metric: True},
                {metric: False},
                {metric: 0.5},
            ):
                with (
                    self.subTest(category=category, change=change),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    validate_attempts([{**row, **change}], [planned])
            bad = deepcopy(row)
            del bad[metric]
            with self.assertRaises(KeyError):
                validate_attempts([bad], [planned])

    def test_simulator_official_endpoint_and_disabled_thinking_are_required(self):
        value = load_config(PROJECT / "configs/tau2_native_smoke.json")
        for field, invalid in (
            ("base_url", "http://api.deepseek.com"),
            ("base_url", "https://api.batgpt.net/v1"),
            ("model", "deepseek-v4-flash"),
            ("thinking", "enabled"),
            ("thinking", False),
        ):
            bad = deepcopy(value)
            bad["user_simulator"][field] = invalid
            with (
                self.subTest(field=field, invalid=invalid),
                self.assertRaises(ValueError),
            ):
                validate(bad)
        for field in ("base_url", "thinking", "model"):
            bad = deepcopy(value)
            del bad["user_simulator"][field]
            with self.assertRaises(ValueError):
                validate(bad)

    def test_approved_sampling_is_fixed(self):
        for template in (
            "bfcl_mt",
            "tau2",
            "toolhop",
            "apibank",
            "toolsandbox",
            "stabletoolbench",
        ):
            for mode in ("main", "pa_tool"):
                baseline = load_config(
                    PROJECT / f"configs/{template}_{mode}_smoke.json"
                )
                for executor in baseline["executors"]:
                    for field, setting in (
                        ("temperature", 0),
                        ("top_p", 1),
                        ("top_k", 1),
                        ("min_p", 0.1),
                    ):
                        with self.subTest(
                            template=template, mode=mode, executor=executor, field=field
                        ):
                            value = deepcopy(baseline)
                            value["executors"][executor]["thinking"][field] = setting
                            with self.assertRaisesRegex(ValueError, "approved Qwen"):
                                validate(value)

    def test_required_fields_and_types(self):
        baseline = config()
        for field in baseline:
            broken = deepcopy(baseline)
            del broken[field]
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                validate(broken)
        for key, value in [
            ("trials", True),
            ("seed", None),
            ("output_root", "../outputs"),
        ]:
            broken = deepcopy(baseline)
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(broken)

    def test_single_gate_does_not_require_another_condition(self):
        value = config()
        value["advisors"] = {"selected": advisor_settings()}
        value["conditions"] = {
            "arbitrary_name": {
                "executor": "qwen4b",
                "advice": "gate",
                "advisor": "selected",
                "gate": {"rule": "top1_prob_lt", "threshold": 0.3, "loop_guard": True},
            }
        }
        self.assertEqual(validate(value), value)
        value["conditions"]["arbitrary_name"]["gate"]["threshold"] = 0.4
        with self.assertRaises(ValueError):
            validate(value)

    def test_full_plan_reproducible_and_grouped_by_endpoint(self):
        value = config()
        value["conditions"]["second"] = {"executor": "qwen4b", "advice": "neutral"}
        tasks = {
            (category, f"{category}_{i}"): {"id": f"{category}_{i}"}
            for category in CATEGORIES
            for i in range(200)
        }
        first = make_plan(value, tasks)
        self.assertEqual(first, make_plan(value, dict(reversed(list(tasks.items())))))
        self.assertEqual(len(first), 1600)
        self.assertEqual(len(make_chains(value, first)), 1)
        value["executors"]["other"] = {
            **value["executors"]["qwen4b"],
            "base_url": "http://127.0.0.1:8005/v1",
        }
        value["conditions"]["second"]["executor"] = "other"
        self.assertEqual(len(make_chains(value, first)), 2)

    def test_formal_subsets_rejected(self):
        value = config()
        value["output_root"] = "outputs"
        with self.assertRaises(ValueError):
            validate(value)
        value["task_ids"] = dict.fromkeys(CATEGORIES, "all")
        validate(value)
        for trials in (2, 3):
            value["trials"] = trials
            with self.assertRaisesRegex(ValueError, "one trial"):
                validate(value)
        value["output_root"] = "tmp"
        validate(
            value
        )  # Synthetic fixtures may still exercise repeated-trial statistics.

    def test_service_identity_and_context(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as model_path:
            for name in ("qwen3-4b", "star-4b"):
                with self.subTest(model=name):
                    value = config()
                    value["executors"]["qwen4b"].update(
                        model=name, model_path=model_path
                    )
                    model = {
                        "id": name,
                        "root": str(Path(model_path).resolve()),
                        "max_model_len": 131072,
                    }
                    with patch(
                        "jev_bench.services.json_request",
                        return_value={"data": [model]},
                    ):
                        self.assertEqual(
                            check_services(value)["executors"]["qwen4b"], model
                        )
                        model["id"] = "wrong-model"
                        with self.assertRaises(ValueError):
                            check_services(value)
                        model["id"] = name
                        model["max_model_len"] = 32768
                        with self.assertRaises(ValueError):
                            check_services(value)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.plan = [
            {
                "condition": "native",
                "task_id": "multi_turn_base_0",
                "category": "multi_turn_base",
                "trial": 0,
            }
        ]
        self.failure = {
            **self.plan[0],
            "attempt": 1,
            "success": None,
            "official_scored": False,
            "failure_kind": "service_error",
        }
        self.success = {
            **self.plan[0],
            "attempt": 2,
            "success": True,
            "official_scored": True,
            "failure_kind": None,
        }

    def test_only_service_failures_are_rescheduled(self):
        self.assertEqual(pending_plan(self.plan, [self.failure])[0]["attempt"], 2)
        self.assertEqual(pending_plan(self.plan, [self.failure, self.success]), [])
        counted_failure = {
            **self.failure,
            "success": False,
            "failure_kind": "executor_context_limit",
        }
        self.assertEqual(pending_plan(self.plan, [counted_failure]), [])
        with self.assertRaises(ValueError):
            validate_attempts([counted_failure, self.success], self.plan)

    def test_duplicates_wrong_category_and_unknown_failures(self):
        for rows in (
            [self.failure, self.failure],
            [{**self.failure, "category": "multi_turn_miss_func"}],
            [{**self.failure, "failure_kind": "unknown"}],
            [{**self.failure, "success": 0}],
            [{**self.failure, "success": True}],
            [{**self.failure, "success": False}],
        ):
            with self.assertRaises(ValueError):
                validate_attempts(rows, self.plan)
        fatal = {**self.failure, "failure_kind": "scoring_error"}
        validate_attempts([fatal], self.plan)
        with self.assertRaisesRegex(ValueError, "cannot be resumed"):
            pending_plan(self.plan, [fatal])

    def test_reconcile_finished_but_unindexed_result(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            append_json(Path(path) / "episodes.jsonl", self.failure)
            write_json(directory_for(path, self.success) / "episode.json", self.success)
            self.assertEqual(reconcile(path, self.plan), [self.failure, self.success])
            self.assertEqual(len(reconcile(path, self.plan)), 2)

    def test_incomplete_attempt_is_not_silently_rerun(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            directory_for(path, self.failure).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "diagnosis"):
                reconcile(path, self.plan)

    def test_second_coordinator_is_rejected(self):
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            run_lock(path),
            self.assertRaisesRegex(ValueError, "another coordinator"),
            run_lock(path),
        ):
            pass

    def test_explicit_resume_recovers_interruption_without_erasing_cost(self):
        @record_attempt
        def interrupted(config, item, task, answer, directory):
            with self.assertRaisesRegex(ValueError, "still active"):
                recover_interrupted(directory, item)
            append_json(
                directory / "diagnostics/executor.jsonl",
                {
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "total_tokens": 11,
                    },
                    "latency_seconds": 2,
                },
            )
            raise KeyboardInterrupt()

        item = {**self.plan[0], "attempt": 1}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            with self.assertRaises(KeyboardInterrupt):
                interrupted(NATIVE_ONLY, item, None, None, directory_for(path, item))
            rows = reconcile(path, self.plan, recover=True)
            self.assertEqual(rows[0]["failure_kind"], "interrupted")
            self.assertEqual(pending_plan(self.plan, rows)[0]["attempt"], 2)
            self.assertEqual(reconcile(path, self.plan), rows)
            cost = aggregate_cost(rows)
            self.assertEqual(cost["executor"]["total_tokens"]["observed_total"], 11)
            self.assertEqual(cost["attempts_with_incomplete_accounting"], 1)
            self.assertEqual(cost["duration_seconds"]["missing"], 1)

    def test_interrupted_scoring_never_resamples_model(self):
        @record_attempt
        def interrupted(config, item, task, answer, directory):
            mark_scoring(directory)
            raise KeyboardInterrupt()

        item = {**self.plan[0], "attempt": 1}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            with self.assertRaises(KeyboardInterrupt):
                interrupted(NATIVE_ONLY, item, None, None, directory_for(path, item))
            with self.assertRaisesRegex(ValueError, "without resampling"):
                reconcile(path, self.plan, recover=True)
            # Even a previously persisted soft-interrupt row cannot bypass the marker.
            row = {
                **item,
                "success": None,
                "official_scored": False,
                "failure_kind": "interrupted",
            }
            write_json(directory_for(path, item) / "episode.json", row)
            with self.assertRaisesRegex(ValueError, "without resampling"):
                reconcile(path, self.plan, recover=True)
            append_json(Path(path) / "episodes.jsonl", row)
            with self.assertRaisesRegex(ValueError, "without resampling"):
                reconcile(path, self.plan, recover=True)
