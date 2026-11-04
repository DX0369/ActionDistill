import json
import unittest
from copy import deepcopy
from pathlib import Path

from jev_bench.benchmarks.apibank import (
    DESCRIPTION_MARKER,
    FILES,
    visible_tools,
)
from jev_bench.config import APIBANK_CATEGORIES, load_config, validate

from tests.support import PROJECT


class ApiBankBoundaryTests(unittest.TestCase):
    def config(self):
        return load_config(PROJECT / "configs/apibank_cpu_preview.json")

    def test_cpu_configuration_and_required_paths(self):
        config = self.config()
        self.assertEqual(set(config["task_ids"]), set(APIBANK_CATEGORIES))
        self.assertNotIn("user_simulator", config)
        for field in ("mode", "data_root", "retriever_model_path"):
            bad = deepcopy(config)
            del bad["protocol"][field]
            with self.assertRaises(ValueError):
                validate(bad)
        for field, value in (
            ("mode", "live"),
            ("data_root", "relative/path"),
            ("retriever_model_path", None),
        ):
            bad = deepcopy(config)
            bad["protocol"][field] = value
            with self.assertRaises(ValueError):
                validate(bad)
        for update in ({"output_root": "outputs"}, {"user_simulator": None}):
            with self.assertRaises(ValueError):
                validate({**config, **update})

    def test_visible_candidate_extraction_on_all_published_inputs(self):
        root = Path(self.config()["protocol"]["data_root"])
        counts = {}
        for category, filename in FILES.items():
            rows = json.loads((root / filename).read_text())
            counts[category] = len(rows)
            for row in rows:
                before = deepcopy(row)
                tools = visible_tools(
                    int(category.split("_")[1]), row["instruction"], row["input"]
                )
                self.assertTrue(tools["names"])
                self.assertEqual(len(tools["names"]), len(set(tools["names"])))
                self.assertEqual(row, before)
        self.assertEqual(list(counts.values()), [399, 375, 135, 103, 245, 50])

    def test_current_history_only_and_no_icl_example_candidates(self):
        root = Path(self.config()["protocol"]["data_root"])
        rows = json.loads((root / "level-2-api.json").read_text())
        initial = visible_tools(2, rows[0]["instruction"], rows[0]["input"])
        self.assertEqual(initial["names"], ["ToolSearcher"])
        later = visible_tools(2, rows[1]["instruction"], rows[1]["input"])
        self.assertIn("QueryHistoryToday", later["names"])
        self.assertNotIn("QueryHistoryToday", initial["names"])
        base = json.loads((root / "level-3-batch-inf.json").read_text())[0]
        icl = json.loads((root / "level-3-batch-inf-icl.json").read_text())[0]
        self.assertEqual(
            visible_tools(3, base["instruction"], base["input"]),
            visible_tools(3, icl["instruction"], icl["input"]),
        )
        self.assertEqual(
            visible_tools(3, base["instruction"], base["input"])["names"],
            ["ToolSearcher"],
        )

    def test_conflicting_visible_descriptions_retain_both_sources(self):
        root = Path(self.config()["protocol"]["data_root"])
        row = json.loads((root / "level-2-response.json").read_text())[0]
        evidence = visible_tools(2, row["instruction"], row["input"])
        indices = evidence["multiple_definitions"]["QueryHistoryToday"]
        definitions = [evidence["occurrences"][index] for index in indices]
        self.assertEqual(
            {row["source"]["field"] for row in definitions}, {"instruction", "input"}
        )
        self.assertGreater(
            len({row["schema"]["description"] for row in definitions}), 1
        )
        self.assertTrue(all("required" not in row["schema"] for row in definitions))

    def test_malformed_visible_inputs_fail_explicitly(self):
        schema = {
            "name": "ToolSearcher",
            "description": "Search",
            "input_parameters": {},
            "output_parameters": {},
        }
        instruction = "Instructions" + DESCRIPTION_MARKER + json.dumps(schema)
        for bad in (
            "no descriptions",
            instruction + DESCRIPTION_MARKER,
            "Instructions" + DESCRIPTION_MARKER + "not JSON",
        ):
            with self.assertRaises(ValueError):
                visible_tools(1, bad, "")
        for field in schema:
            bad = {key: value for key, value in schema.items() if key != field}
            with self.assertRaises(ValueError):
                visible_tools(
                    2, "Instructions" + DESCRIPTION_MARKER + json.dumps(bad), ""
                )
        for result in ("None", "()", "[]", "{'unknown': 1}", "not a literal"):
            with self.assertRaises((TypeError, ValueError, SyntaxError)):
                visible_tools(
                    2,
                    instruction,
                    "API-Request: [ToolSearcher(keywords='x')]->" + result,
                )
        initial = json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Search",
                "parameters": {},
                "response": {},
            }
        )
        wrapper = {
            "api_name": "ToolSearcher",
            "input": {"keywords": "x"},
            "output": schema,
            "exception": None,
        }
        for update in (
            {"input": None},
            {"input": {}},
            {"input": {"keywords": False}},
            {"exception": False, "output": None},
        ):
            with self.assertRaises((TypeError, ValueError)):
                visible_tools(
                    3,
                    "",
                    initial
                    + "\nAPI-Request: [ToolSearcher(keywords='x')]->"
                    + repr({**wrapper, **update}),
                )
        failed = {**wrapper, "exception": "explicit tool failure", "output": None}
        self.assertEqual(
            visible_tools(
                3,
                "",
                initial
                + "\nAPI-Request: [ToolSearcher(keywords='x')]->"
                + repr(failed),
            )["names"],
            ["ToolSearcher"],
        )


if __name__ == "__main__":
    unittest.main()
