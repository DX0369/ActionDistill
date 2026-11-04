"""Bounded smoke subsets retain all category keys and formal coverage checks."""

import unittest
from copy import deepcopy

from jev_bench.config import load_config, validate

from tests.support import PROJECT


class SmokeSubsetTests(unittest.TestCase):
    def test_bfcl_and_apibank_single_category_subsets_are_only_for_tmp_smoke(self):
        for template in ("bfcl_mt", "apibank"):
            base = load_config(PROJECT / f"configs/{template}_main_smoke.json")
            selected = next(iter(base["task_ids"]))
            base["task_ids"] = {
                category: ids[:1] if category == selected else []
                for category, ids in base["task_ids"].items()
            }
            validate(base)
            for change in (
                "missing",
                "none",
                "all_empty",
                "formal",
                "repeated",
                "comparisons",
            ):
                value = deepcopy(base)
                if change == "missing":
                    del value["task_ids"][selected]
                elif change == "none":
                    value["task_ids"][selected] = None
                elif change == "all_empty":
                    value["task_ids"][selected] = []
                elif change == "formal":
                    value["output_root"] = "outputs"
                elif change == "comparisons":
                    value["comparisons"] = [["ours", "native"]]
                else:
                    value["trials"] = 2
                with (
                    self.subTest(template=template, change=change),
                    self.assertRaises(ValueError),
                ):
                    validate(value)

    def test_one_or_two_domains_require_explicit_empty_lists(self):
        base = load_config(PROJECT / "configs/tau2_pa_tool_smoke.json")
        for selected in (("airline",), ("airline", "telecom")):
            value = deepcopy(base)
            value["task_ids"] = {
                domain: ids if domain in selected else []
                for domain, ids in base["task_ids"].items()
            }
            validate(value)
        for change in ("missing", "none", "all_empty", "formal", "repeated"):
            value = deepcopy(base)
            value["task_ids"]["retail"] = []
            if change == "missing":
                del value["task_ids"]["retail"]
            elif change == "none":
                value["task_ids"]["retail"] = None
            elif change == "all_empty":
                value["task_ids"] = {domain: [] for domain in base["task_ids"]}
            elif change == "formal":
                value["output_root"] = "outputs"
            else:
                value["trials"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(value)
