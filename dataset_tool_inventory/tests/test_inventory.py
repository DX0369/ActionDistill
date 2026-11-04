from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_tool_inventory.aggregate import InventoryWriter
from dataset_tool_inventory.common import iter_json_array
from dataset_tool_inventory.schema import DecisionEvent, ToolEvent

TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


class InventoryTests(unittest.TestCase):
    def test_json_array_stream_handles_small_chunks(self):
        TMP_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps([{"x": 1}, {"x": "two"}]), encoding="utf-8")
            self.assertEqual(list(iter_json_array(path, chunk_size=3)), [{"x": 1}, {"x": "two"}])

    def test_tool_identity_does_not_split_by_model(self):
        TMP_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as directory:
            writer = InventoryWriter(Path(directory) / "out", ["demo"])
            writer.consume(ToolEvent("demo", "t1", "declared", "read", model="a", schema={"type": "object"}))
            writer.consume(ToolEvent("demo", "t2", "called", "read", model="b", schema={"type": "object"}))
            writer.consume(DecisionEvent("demo", "t2", 1, ("read",), "TOOL", ("read",), True))
            summary = writer.finalize()
            self.assertEqual(summary["datasets"]["demo"]["declared_tool_union"], 1)
            self.assertEqual(summary["datasets"]["demo"]["called_tool_union"], 1)
            self.assertEqual(summary["datasets"]["demo"]["decision_state_count"], 1)


if __name__ == "__main__":
    unittest.main()

