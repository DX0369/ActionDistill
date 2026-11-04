"""Small boundaries for live advice; real val roundtrip is a separate smoke."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from jev_train.advisor import (
    JevAdvisor,
    JevRenderer,
    load_temperature,
    make_server,
    render_request,
)
from jev_train.render import prompt_item, render_event
from test_build import Tokenizer


class AdvisorTests(unittest.TestCase):
    def test_tokenizer_only_renderer_matches_existing_render_without_weights(self):
        tok = Tokenizer()
        tok.decode = lambda ids, **kwargs: ' '.join(map(str, ids))
        payload = {'messages': [{'role': 'user', 'content': 'Find the file.'}],
                   'tools': [{'name': 'find', 'description': 'Search'}], 'respond': True}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as directory:
            root = Path(directory)
            for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                (root/name).write_text('{}')
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=tok), \
                    patch('decider.model.DecisionModel') as model:
                renderer = JevRenderer(root)
                rendered = renderer.handle(payload, render_only=True)
                model.assert_not_called()
            legacy = object.__new__(JevAdvisor)
            legacy.model = SimpleNamespace(tok=tok)
            legacy.model_path, legacy.temperature_path, legacy.temperature = 'unused', 'unused', 1.
            expected = legacy.handle(payload, render_only=True)
            for key in ('rendered', 'prompt', 'prompt_token_ids', 'n_prompt_tokens', 'messages_dropped',
                        'truncation_counts', 'truncated'):
                self.assertEqual(rendered[key], expected[key])
            self.assertNotIn('model', rendered)
            self.assertNotIn('temperature', rendered)
            with self.assertRaisesRegex(ValueError, 'cannot predict'):
                renderer.handle(payload)
            # Exercise the HTTP route without opening a socket or starting a service.
            with patch('jev_train.advisor.HTTPServer') as factory:
                make_server(renderer, 0)
            handler = object.__new__(factory.call_args.args[1])
            handler.path, handler.send_json = '/advise', Mock()
            handler.do_POST()
            self.assertEqual(handler.send_json.call_args.args[0], 405)

    def test_shared_render_and_invalid_inputs(self):
        tok = Tokenizer()
        payload = {"messages": [{"role": "user", "content": "Find the file."}],
                   "tools": [{"name": "find", "description": "Search", "required": ["pattern"], "parameters":
                              {"properties": {"pattern": {"type": "string"}}}}], "respond": True}
        row, item, names = render_request(payload, tok)
        expected, reason = render_event(SimpleNamespace(
            metadata={"_history": payload["messages"], "_candidate_definitions": payload["tools"]},
            candidate_tools=("find",), action_kind="TOOL", action_names=("find",)),
            tok, length_margin=64)
        self.assertIsNone(reason)
        self.assertEqual(item["ids"], prompt_item(expected["context"], expected["question"],
                                                  expected["options"], expected["gold"], tok)["ids"])
        self.assertEqual(names, ["find", "RESPOND"])
        for invalid_fields in ({"tools": []}, {"respond": False}, {"messages": []},
                      {"tools": [{"name": "RESPOND"}]},
                      {"messages": [{"role": "assistant", "tool_calls": [
                          {"function": {"name": "find"}}]}]},
                      {"messages": [{"role": "user", "content": "x" * 40000}]}):
            with self.assertRaises((ValueError, KeyError)):
                render_request({**payload, **invalid_fields}, tok)

    def test_duplicate_schemas_are_positional_and_training_remains_strict(self):
        tok = Tokenizer()
        tools = [{"name": "dup", "description": "First schema", "parameters": {"properties": {"first": {"type": "string"}}}},
                 {"name": "dup", "description": "Second schema", "parameters": {"properties": {"second": {"type": "integer"}}}}]
        payload = {"messages": [{"role": "user", "content": "Help"}], "tools": tools, "respond": True}
        original = json.dumps(payload)
        row, item, names = render_request(payload, tok)
        self.assertEqual(names, ["dup", "dup", "RESPOND"])
        self.assertEqual(len(row["options"]), 3)
        self.assertIn("First schema", row["options"][0])
        self.assertIn("Second schema", row["options"][1])
        self.assertIn("first", row["options"][0])
        self.assertIn("second", row["options"][1])
        self.assertEqual(json.dumps(payload), original)
        event = SimpleNamespace(metadata={"_history": payload["messages"], "_candidate_definitions": tools},
                                candidate_tools=("dup", "dup"), action_kind="RESPOND", action_names=())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            render_event(event, tok)
        event.action_kind, event.action_names = "TOOL", ("dup",)
        with self.assertRaisesRegex(ValueError, "placeholder"):
            render_event(event, tok, allow_duplicate_candidates=True)
        with self.assertRaisesRegex(ValueError, "too_many_options"):
            render_request({**payload, "tools": [tools[0]] * 255}, tok)

    def test_duplicate_service_response_preserves_raw_argmax(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.tok = Tokenizer()
                self.tok.pad_token_id = 0
            def forward(self, batch):
                return torch.log(torch.tensor([[.30, .29, .40, .01]]))
        advisor = object.__new__(JevAdvisor)
        advisor.model, advisor.model_path, advisor.temperature_path, advisor.temperature = Model(), "cpu", "cpu", 1.
        result = advisor.handle({"messages": [{"role": "user", "content": "Help"}],
                                 "tools": [{"name": name} for name in ("dup", "dup", "other")], "respond": True})
        self.assertEqual(result["top1"], "other")
        self.assertEqual(advisor.describe()["duplicate_candidates"], "positional")
        self.assertEqual(result["duplicate_candidates"], "positional")
        self.assertEqual([row["name"] for row in result["probabilities"]], ["dup", "dup", "other", "RESPOND"])
        self.assertAlmostEqual(sum(row["probability"] for row in result["probabilities"]), 1, places=6)

    def test_temperature_and_calibrated_probabilities(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.tok = Tokenizer()
                self.tok.pad_token_id = 0

            def forward(self, batch):
                return torch.tensor([[0., 2.]])

        advisor = object.__new__(JevAdvisor)
        advisor.model = Model()
        advisor.model_path = "test"
        advisor.temperature_path = "test"
        advisor.temperature = 2.
        result = advisor.handle({"messages": [{"role": "user", "content": "Help"}],
                                 "tools": [{"name": "find"}], "respond": True})
        self.assertEqual(result["top1"], "RESPOND")
        self.assertAlmostEqual(result["probabilities"][1]["probability"],
                               float(torch.softmax(torch.tensor([0., 1.]), dim=0)[1]))
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as d:
            path = Path(d) / "temperature.json"
            for value in (0, -1, True, "1", float("nan")):
                path.write_text(json.dumps({"temperature": value}))
                with self.assertRaises(ValueError):
                    load_temperature(path)


if __name__ == "__main__":
    unittest.main()
