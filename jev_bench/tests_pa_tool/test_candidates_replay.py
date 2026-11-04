"""Replay must reproduce old assets and leave changed parameter inputs unknown."""

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from jev_bench.io import append_json, write_json
from jev_bench.pa_tool_prepare import build_mapping
from tests.pa_fixtures import tool

from tests_pa_tool.test_reuse import event

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pa_replay", PROJECT / "scripts/analyze_pa_candidates.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = replay.frozen_prepare()

    def fixture(self, root, *, tool_changes=False, failure_slot=None):
        function = tool()
        function["function"]["parameters"]["properties"].pop("payload")
        catalog = {
            "version": 1,
            "benchmark": {"name": "bfcl"},
            "control_names": [],
            "bindings": {"task": "s"},
            "scopes": {"s": [function]},
            "executor": {"model": "qwen3-4b"},
        }
        old = self.old
        recorded = {}

        def generate(key, text):
            is_tool = text.startswith("Generate a tool")
            names = (
                (
                    ["new_tool"] * 16 + ["old_tool"] * 16
                    if tool_changes
                    else ["old_tool"] * 32
                )
                if is_tool
                else ["new_key"] * 32
            )
            reference = "old_tool" if is_tool else "new_key"
            events = []
            failures = []
            for slot, name in enumerate([*names, reference]):
                row = event(key, text, slot)
                row["name"] = name
                row["provider"]["request"]["temperature"] = 0 if slot == 32 else 0.4
                row["provider"]["response"]["choices"][0]["message"]["content"] = name
                if is_tool and slot == failure_slot:
                    error = old.NameGenerationError("length")
                    row["name"] = None
                    row["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "reason": error.reason,
                    }
                    row["provider"]["response"]["choices"][0]["finish_reason"] = (
                        "length"
                    )
                    failures.append({"slot": slot, "reason": "length"})
                events.append(row)
            recorded[key] = events
            if failures:
                raise old.NameGenerationError("length", failures=failures)
            return names, reference

        mapping = old.build_mapping(catalog, generate)
        write_json(root / "catalog.json", catalog)
        write_json(root / "mapping.json", mapping)
        write_json(
            root / "metadata/run_status.json",
            {
                "status": "completed",
                "protocol": old.PROTOCOL,
                "code": {
                    "trees": {"jev_bench/src": replay.OLD_TREE},
                    "dirty_paths": [],
                },
            },
        )
        ledger = root / "diagnostics/generation.jsonl"
        for rows in recorded.values():
            for row in rows:
                append_json(
                    ledger, {**row, "phase": "started", "name": None, "error": None}
                )
                append_json(ledger, row)
        return replay.read_baseline(root, old)

    def test_old_field_comparison_includes_fallback_and_stops_on_changed_mapping(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            data = self.fixture(root, failure_slot=2)
            self.assertTrue(data["baseline_exact"])
            self.assertEqual(len(data["mapping"]["generation_fallbacks"]), 1)
            mapping = deepcopy(data["mapping"])
            mapping["generation_fallbacks"] = {}
            write_json(root / "mapping.json", mapping)
            with self.assertRaisesRegex(ValueError, "field-for-field"):
                replay.read_baseline(root, self.old)

    def test_reference_is_old_slot32_and_missing_parameter_context_is_not_reused(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            data = self.fixture(Path(path), tool_changes=True)
            chosen = {
                k: replay.selection(c, range(16)) for k, c in data["components"].items()
            }
            rows, scopes, reached, unknown = replay.replay_scopes(data, chosen)
            self.assertEqual(rows[0]["selected"], "new_tool")
            self.assertEqual(rows[1]["status"], "unknown_context_changed")
            self.assertNotIn("selected", rows[1])
            self.assertIsNone(scopes[0]["complete_mapping_same"])
            self.assertEqual((len(reached), len(unknown)), (1, 1))
            tool_component = next(
                c
                for c in data["components"].values()
                if c["spec"]["component"] == "tool"
            )
            tool_component["events"][16]["name"] = "random_slot_not_reference"
            tool_component["events"][32]["name"] = None
            self.assertEqual(
                replay.selection(tool_component, range(16))["fallback"],
                "reference_failed",
            )

    def test_complete_context_replay_matches_production_mapping_and_costs_cache_once(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            data = self.fixture(Path(path))
            data["catalog"]["scopes"]["s2"] = deepcopy(data["catalog"]["scopes"]["s"])
            data["mapping"]["scopes"]["s2"] = deepcopy(data["mapping"]["scopes"]["s"])
            chosen = {
                k: replay.selection(c, range(16)) for k, c in data["components"].items()
            }
            rows, scopes, reached, unknown = replay.replay_scopes(data, chosen)
            mapping = build_mapping(
                data["catalog"],
                lambda k, _: (
                    [e["name"] for e in data["components"][k]["events"][:16]],
                    data["components"][k]["events"][32]["name"],
                ),
            )
            for row in rows:
                result = mapping["scopes"][row["scope"]][row["tool"]]
                expected = (
                    result["name"]
                    if row["component"] == "tool"
                    else result["parameters"][row["parameter"]]
                )
                self.assertEqual(row["selected"], expected)
            metrics = replay.metrics(data, chosen, rows, scopes, reached, unknown)
            self.assertEqual(metrics["fixed_old_context_cost"]["requests"], 34)
            self.assertEqual(metrics["reachable_recorded_cost"]["requests"], 34)
            self.assertEqual(metrics["scopes"]["complete_mapping_same"], 2)

    def test_minimum_and_duplicate_frequency_are_kept_in_subset_selection(self):
        component = {
            "events": [{"name": "first"}] * 16
            + [{"name": "second"}] * 16
            + [{"name": "second"}],
            "original_name": "original",
            "old_ranking": [{"name": "second"}],
        }
        chosen = replay.selection(component, [*range(8), *range(16, 24)])
        self.assertEqual(chosen["ranking"][0]["count"], 8)
        self.assertEqual(chosen["local"], "second")
        self.assertEqual(chosen["reference_slot"], 32)
        component["events"][:4] = [{"name": None}] * 4
        self.assertIsNone(replay.selection(component, range(16))["fallback"])
        component["events"][4] = {"name": None}
        self.assertEqual(
            replay.selection(component, range(16))["fallback"],
            "insufficient_valid_candidates",
        )

    def test_collision_exhaustion_is_a_known_failure_separate_from_unknown_context(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            data = self.fixture(Path(path))
            other = deepcopy(data["catalog"]["scopes"]["s"][0])
            other["function"]["name"] = "other"
            data["catalog"]["scopes"]["s"].append(other)
            spec = {"function": other["function"], "occurrence": 0, "component": "tool"}
            key = replay.current.identity(spec)
            tool_component = next(
                c
                for c in data["components"].values()
                if c["spec"]["component"] == "tool"
            )
            data["components"][key] = {**deepcopy(tool_component), "spec": spec}
            selected = {
                k: replay.selection(c, range(16)) for k, c in data["components"].items()
            }
            rows, scopes, reached, unknown = replay.replay_scopes(data, selected)
            self.assertEqual(scopes[0]["status"], "candidate_exhaustion")
            self.assertIs(scopes[0]["complete_mapping_same"], False)
            counts = replay.metrics(data, selected, rows, scopes, reached, unknown)
            self.assertEqual(counts["scopes"]["candidate_exhaustion_namespaces"], 1)
            self.assertEqual(counts["scopes"]["unknown_context"], 0)
            self.assertEqual(
                counts["tool"]["components_blocked_by_candidate_exhaustion"], 2
            )


if __name__ == "__main__":
    unittest.main()
