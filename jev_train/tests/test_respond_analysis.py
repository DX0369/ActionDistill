import importlib.util
from pathlib import Path
import unittest
import torch
import tempfile
import json

spec = importlib.util.spec_from_file_location("analyze_respond", Path(__file__).resolve().parents[1]/"scripts/analyze_respond.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RespondAnalysisTests(unittest.TestCase):
    def test_bias_after_temperature_and_soft_nll(self):
        logits = torch.full((2,255), float("-inf"))
        logits[:,:3] = torch.tensor([[2.,0.,-8.],[0.,2.,-8.]])
        targets = torch.zeros((2,255))
        targets[0,1] = 1
        targets[1,:2] = .5
        data = {"logits":logits, "targets":targets}
        result = analysis.adjusted_metrics(data,2,1.1,torch.tensor([1,2]))
        # First row switches only if b is added after dividing by T.
        self.assertEqual(result["acc"],1.)
        self.assertEqual(result["respond_recall"],1.)
        self.assertTrue(torch.isfinite(torch.tensor(result["nll"])))
        expected = -(targets[:,:3]*torch.log_softmax(torch.tensor([[1.,1.1,-4.],[0.,1.,-2.9]]),1)).sum(1).mean()
        self.assertAlmostEqual(result["nll"],float(expected),places=6)

    def test_canonical_label_and_previous_type(self):
        self.assertTrue(analysis.gold_respond({"id":"a","options":["foo","RESPOND"],"gold":[1]}))
        with self.assertRaises(KeyError):
            analysis.gold_respond({"id":"a","options":["RESPOND"]})
        self.assertEqual(analysis.previous_kind([]),"NONE")
        self.assertEqual(analysis.previous_kind(["RESPOND"]),"RESPOND")
        self.assertEqual(analysis.previous_kind(["foo","bar"]),"TOOL_SET")
        with self.assertRaises(ValueError):
            analysis.previous_kind(["RESPOND","foo"])

    def test_streaming_boundary_rejects_invalid_gold(self):
        row = dict(id="a",dataset="d",model="m",task_group="g",outcome="success",candidate_level="S",
                   context="",question="q",options=["foo","RESPOND"],gold=[-1],n_prompt_tokens=2,
                   messages_dropped=0,truncation_counts={})
        with tempfile.TemporaryDirectory(dir=analysis.PROJECT/"tmp",prefix="respond_unit_") as directory:
            path=Path(directory)/"sample.jsonl"
            for invalid in ([-1],[True]):
                row["gold"]=invalid
                path.write_text(json.dumps(row)+"\n")
                with self.assertRaises(ValueError):
                    analysis.read_population(path,set())

    def test_zero_respond_dataset_is_retained(self):
        logits=torch.full((1,255),float("-inf"));logits[0,:2]=torch.tensor([1.,0.])
        target=torch.zeros((1,255));target[0,0]=1
        data=dict(id=["x"],dataset=["no_respond"],outcome=["success"],transition=["first"],
                  logits=logits,targets=target,tool_names=[["foo","RESPOND"]],nopts=[2])
        rows={"x":dict(dataset="no_respond",outcome="success",transition="first",id="x",options=["foo","RESPOND"],gold=[0])}
        result=analysis.miss_analysis(data,rows,1,torch.tensor([1]))
        self.assertEqual(result["slices"]["dataset"]["no_respond"]["gold_respond"],0)
        self.assertIsNone(result["slices"]["dataset"]["no_respond"]["recall"])


if __name__ == "__main__":
    unittest.main()
