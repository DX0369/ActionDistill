"""The official hidden-description rule is separate from generation failure."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from jev_bench.io import write_json
from jev_bench.pa_tool import Projection, asset_identity, load_projection, task_key
from jev_bench.pa_tool_prepare import build_mapping, estimate, validate_catalog
from tests.pa_fixtures import tool

PROJECT = Path(__file__).resolve().parents[1]


def catalog():
    first, second = tool("visible_tool_17"), tool("other")
    first["function"]["description"] = ""
    first["function"]["parameters"]["properties"].pop("payload")
    second["function"]["parameters"] = {
        "type": "object",
        "properties": {"visible_arg_3": {"type": "string"}},
    }
    return {
        "version": 1,
        "benchmark": {"name": "toolsandbox"},
        "executor": {"model": "fixture", "model_path": "/fixture"},
        "control_names": [],
        "scopes": {"scope": [first, second]},
        "bindings": {
            task_key({"category": "scenarios", "task_id": "fixture"}): "scope"
        },
        "description_augmentations": {
            "scope": ["TOOL_DESCRIPTION_SCRAMBLED", "ARG_DESCRIPTION_SCRAMBLED"]
        },
    }


class HiddenDescriptionTests(unittest.TestCase):
    def test_preservation_skips_requests_reserves_names_and_keeps_other_components(
        self,
    ):
        value = catalog()
        value["scopes"]["second_scope"] = deepcopy(value["scopes"]["scope"])
        value["description_augmentations"]["second_scope"] = value[
            "description_augmentations"
        ]["scope"][:]
        original = deepcopy(value)
        prompts = []

        def generate(key, text):
            prompts.append(text)
            if text.startswith("Generate a tool"):
                return ["visible_tool_17"] * 15 + ["new_tool"], "visible_tool_17"
            return ["new_key"] * 16, "new_key"

        mapping = build_mapping(value, generate)
        rows = mapping["scopes"]["scope"]
        self.assertEqual(
            rows,
            [
                {"name": "visible_tool_17", "parameters": {"key": "new_key"}},
                {"name": "new_tool", "parameters": {"visible_arg_3": "visible_arg_3"}},
            ],
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn("Tool: visible_tool_17 - \n", prompts[1])
        self.assertEqual(value, original)
        self.assertEqual(mapping["generation_fallbacks"], {})
        self.assertEqual(len(mapping["description_preservations"]), 2)
        self.assertTrue(
            all(
                set(row) == {"component", "visible_name", "reason"}
                for row in mapping["description_preservations"].values()
            )
        )
        counts = estimate(value)
        self.assertEqual(counts["requests_upper_bound"], 4 * 17)
        self.assertEqual(counts["description_preservations"]["unique_components"], 2)
        self.assertEqual(
            counts["description_preservations"]["by_component"],
            {"tool": 1, "parameter": 1},
        )
        projected = Projection(value["scopes"]["scope"], rows).request(
            [], value["scopes"]["scope"]
        )[1]
        self.assertEqual(projected[0]["function"]["description"], "")
        self.assertNotIn(
            "description",
            projected[1]["function"]["parameters"]["properties"]["visible_arg_3"],
        )

    def test_only_exact_official_missing_forms_are_allowed(self):
        changes = [
            lambda c: c["description_augmentations"].update(scope=[]),
            lambda c: c["description_augmentations"].update(
                scope=["TOOL_DESCRIPTION_SCRAMBLED"]
            ),
            lambda c: c["scopes"]["scope"][0]["function"].update(description=None),
            lambda c: c["scopes"]["scope"][0]["function"].pop("description"),
            lambda c: c["scopes"]["scope"][1]["function"]["parameters"]["properties"][
                "visible_arg_3"
            ].update(description=None),
            lambda c: c["benchmark"].update(name="bfcl"),
            lambda c: c.pop("description_augmentations"),
        ]
        for change in changes:
            value = catalog()
            change(value)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                validate_catalog(value)

    def test_illegal_or_duplicate_retained_names_still_fail(self):
        for name in ("not-a-legal-name", "visible_tool_17"):
            value = catalog()
            value["scopes"]["scope"][1]["function"].update(name=name, description="")
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_mapping(
                    value, lambda *args: self.fail("must fail before generation")
                )

    def test_loading_rejects_renamed_preservation_or_missing_audit_records(self):
        value = catalog()
        mapping = build_mapping(value, lambda key, text: (["chosen"] * 16, "chosen"))
        item = {"condition": "pa", "category": "scenarios", "task_id": "fixture"}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "mapping.json"
            config = {
                "benchmark": value["benchmark"],
                "executors": {"qwen": value["executor"]},
                "conditions": {
                    "pa": {"executor": "qwen", "pa_tool_mapping": str(path)}
                },
            }
            write_json(path.parent / "catalog.json", value)
            write_json(
                path.parent / "metadata/run_status.json", {"status": "completed"}
            )
            write_json(path, mapping)
            load_projection(config, item, expected_identity=asset_identity(path))
            for change in (
                lambda m: m["scopes"]["scope"][0].update(name="changed"),
                lambda m: m["scopes"]["scope"][1]["parameters"].update(
                    visible_arg_3="changed"
                ),
                lambda m: m["description_preservations"].clear(),
            ):
                altered = deepcopy(mapping)
                change(altered)
                write_json(path, altered)
                with self.assertRaises(ValueError):
                    load_projection(
                        config, item, expected_identity=asset_identity(path)
                    )
