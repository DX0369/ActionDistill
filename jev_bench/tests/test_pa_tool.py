import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.client import ParticipantResponseError
from jev_bench.config import validate
from jev_bench.pa_tool import Projection, asset_identity, load_projection

from tests.pa_fixtures import freeze, result, rows_for, tool
from tests.support import PROJECT, config


class ProjectionTests(unittest.TestCase):
    def test_schema_nested_values_history_and_raw_evidence(self):
        tools = [tool()]
        original = deepcopy(tools)
        projection = Projection(tools, rows_for(tools))
        messages = [
            {"role": "system", "content": "lookup(key=...) is a policy example."}
        ]
        history, exposed = projection.request(messages, tools)
        schema = exposed[0]["function"]["parameters"]
        self.assertEqual(schema["required"], ["pa_arg_0"])
        self.assertEqual(
            schema["properties"]["pa_arg_1"]["properties"], {"key": {"type": "string"}}
        )
        self.assertEqual(history, messages)
        raw = result(
            [("pa_tool_0", '{"pa_arg_0":"lookup", "pa_arg_1":{"key":"pa_tool_0"}}')]
        )
        saved = deepcopy(raw.raw_response)
        restored = projection.response(raw)
        self.assertEqual(restored.message.tool_calls[0].function.name, "lookup")
        self.assertEqual(
            json.loads(restored.message.tool_calls[0].function.arguments),
            {"key": "lookup", "payload": {"key": "pa_tool_0"}},
        )
        canonical = restored.raw_response["choices"][0]["message"]
        messages += [
            canonical,
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "name": "lookup",
                "content": "lookup/key remain in business text",
            },
        ]
        projected, _ = projection.request(messages, tools)
        self.assertEqual(projected[1]["tool_calls"][0]["function"]["name"], "pa_tool_0")
        self.assertEqual(projected[2]["name"], "pa_tool_0")
        self.assertEqual(projected[2]["content"], messages[2]["content"])
        self.assertEqual(raw.raw_response, saved)
        self.assertEqual(tools, original)

    def test_hidden_candidates_parallel_occurrences_and_reused_ids(self):
        tools = [tool(), tool(), tool("later")]
        p = Projection(tools, rows_for(tools))
        messages, shown = p.request([], tools[:2])
        self.assertEqual(len(shown), 2)
        with self.assertRaises(ParticipantResponseError):
            p.response(result([("pa_tool_2", "{}")]))
        first = p.response(result([("pa_tool_1", "{}"), ("pa_tool_0", "{}")]))
        messages.append(first.raw_response["choices"][0]["message"])
        _, shown = p.request(messages, tools)
        self.assertEqual(len(shown), 3)
        second = p.response(result([("pa_tool_0", "{}")]))  # reused provider call-0
        messages.append(second.raw_response["choices"][0]["message"])
        projected, _ = p.request(messages, tools)
        self.assertEqual(
            [call["function"]["name"] for call in projected[0]["tool_calls"]],
            ["pa_tool_1", "pa_tool_0"],
        )
        self.assertEqual(projected[1]["tool_calls"][0]["function"]["name"], "pa_tool_0")

    def test_interface_violations_fail_but_official_argument_errors_survive(self):
        p = Projection(
            [tool()],
            [{"name": "pa_lookup", "parameters": {"key": "item", "payload": "data"}}],
        )
        p.request([], [tool()])
        for name, args in [
            ("lookup", "{}"),
            ("absent", "{}"),
            ("pa_lookup", '{"key":1}'),
        ]:
            with (
                self.subTest(name=name, args=args),
                self.assertRaises(ParticipantResponseError),
            ):
                p.response(result([(name, args)]), suffix=True)
        for arguments in ("{bad", "[]", '{"item":1,"unknown":2}'):
            restored = p.response(result([("pa_lookup", arguments)]))
            if arguments.startswith('{"'):
                self.assertEqual(
                    json.loads(restored.message.tool_calls[0].function.arguments),
                    {"key": 1, "unknown": 2},
                )
            else:
                self.assertEqual(
                    restored.message.tool_calls[0].function.arguments, arguments
                )

    def test_schema_references_and_history_collisions_fail_explicitly(self):
        sentinel = tool()
        sentinel["function"]["parameters"]["required"].append("")
        p = Projection([sentinel], rows_for([sentinel]))
        _, tools = p.request([], [sentinel])
        self.assertEqual(
            tools[0]["function"]["parameters"]["required"], ["pa_arg_0", ""]
        )
        for change in (
            {"const": {"key": "x"}},
            {"properties": {"key": {"$ref": "#/properties/key"}, "payload": {}}},
        ):
            invalid = tool()
            invalid["function"]["parameters"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                Projection([invalid], rows_for([invalid]))
        p = Projection([tool()], rows_for([tool()]))
        with self.assertRaises(ValueError):
            p.project_arguments({"key": "x", "pa_arg_0": "y"}, 0)

    def test_configuration_and_completed_asset_boundary(self):
        value = config()
        item = {
            "condition": "native",
            "category": "multi_turn_base",
            "task_id": "fixture",
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            p = freeze(value, item, [tool()], directory)
            validate(value)
            with self.assertRaisesRegex(ValueError, "frozen asset identity"):
                load_projection(value, item)
            self.assertIsInstance(
                load_projection(value, item, expected_identity=p.asset_identity),
                Projection,
            )
            mapping = Path(directory) / "mapping.json"
            content = json.loads(mapping.read_text())
            content["protocol"]["candidates"] = 32
            mapping.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_projection(value, item, expected_identity=asset_identity(mapping))
            del content["protocol"]
            mapping.write_text(json.dumps(content))
            with self.assertRaises(KeyError):
                load_projection(value, item, expected_identity=asset_identity(mapping))
            value["conditions"]["native"]["advice"] = "neutral"
            with self.assertRaises(ValueError):
                validate(value)

    def test_frozen_identity_rejects_legal_same_path_replacements(self):
        value = config()
        item = {
            "condition": "native",
            "category": "multi_turn_base",
            "task_id": "fixture",
        }
        self.assertIsNone(load_projection(value, item))
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            p = freeze(value, item, [tool()], directory)
            mapping = Path(directory) / "mapping.json"
            for filename in ("mapping.json", "catalog.json"):
                path = Path(directory) / filename
                original = path.read_bytes()
                changed = json.loads(original)
                if filename == "mapping.json":
                    changed["scopes"]["fixture"][0]["name"] = "other_alias"
                else:
                    changed["scopes"]["fixture"][0]["function"]["description"] = (
                        "A different public description"
                    )
                path.write_text(json.dumps(changed))
                with (
                    self.subTest(filename=filename),
                    self.assertRaisesRegex(ValueError, "frozen identity"),
                ):
                    load_projection(value, item, expected_identity=p.asset_identity)
                self.assertIsInstance(
                    load_projection(
                        value, item, expected_identity=asset_identity(mapping)
                    ),
                    Projection,
                )
                path.write_bytes(original)


class OfficialBoundaryTests(unittest.TestCase):
    def test_bfcl_four_loops_restore_scoring_and_withheld_visibility(self):
        from jev_bench.benchmarks.bfcl import load_tasks, run_episode, score_saved
        from jev_bench.pa_tool import task_key
        from jev_bench.pa_tool_catalog import build_catalog

        from tests.test_bfcl import ScriptedClient

        value = config()
        tasks, answers = load_tasks(value)
        catalog = build_catalog(value, value["conditions"]["native"]["executor"])
        for category, number in zip(value["task_ids"], [0, 0, 0, 100]):
            identity = category, f"{category}_{number}"
            item = {
                "category": category,
                "task_id": identity[1],
                "condition": "native",
                "trial": 0,
                "attempt": 1,
            }
            tools = catalog["scopes"][catalog["bindings"][task_key(item)]]
            with (
                self.subTest(category=category),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as temporary,
            ):
                root = Path(temporary)
                projection = freeze(value, item, tools, root / "mapping")
                by_name = {
                    tool["function"]["name"]: row
                    for tool, row in zip(tools, projection.rows)
                }
                actions = []
                for turn in answers[identity]:
                    calls = []
                    for text in turn:
                        expression = ast.parse(text, mode="eval")
                        row = by_name[expression.body.func.id]
                        expression.body.func.id = row["name"]
                        for keyword in expression.body.keywords:
                            keyword.arg = row["parameters"][keyword.arg]
                        calls.append(ast.unparse(expression))
                    actions.append(calls)
                clients = []

                def factory(settings, sink, actions=actions, clients=clients):
                    client = ScriptedClient(settings, sink, actions)
                    clients.append(client)
                    return client

                with redirect_stdout(io.StringIO()):
                    outcome = run_episode(
                        value,
                        item,
                        tasks[identity],
                        answers[identity],
                        root / "episode",
                        client_factory=factory,
                        pa_tool_identity=projection.asset_identity,
                    )
                self.assertTrue(outcome["official_scored"])
                self.assertTrue(outcome["success"])
                self.assertEqual(outcome["failure_kind"], None)
                for request in clients[0].requests:
                    self.assertTrue(
                        all(
                            t["function"]["name"].startswith("pa_tool_")
                            for t in request["tools"]
                        )
                    )
                    if category == "multi_turn_miss_func":
                        turn = sum(m["role"] == "user" for m in request["messages"]) - 1
                        self.assertEqual(
                            by_name["sort"]["name"]
                            in [t["function"]["name"] for t in request["tools"]],
                            turn >= 3,
                        )
                # Scoring recovery consumes canonical saved calls, without a mapping/model.
                (root / "episode/score.json").unlink()
                (root / "mapping/mapping.json").unlink()
                with (
                    patch(
                        "jev_bench.client.OpenAI",
                        side_effect=AssertionError("no inference during recovery"),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertTrue(
                        score_saved(
                            value,
                            item,
                            tasks[identity],
                            answers[identity],
                            root / "episode",
                        )["success"]
                    )

    def test_old_name_is_episode_failure_and_raw_response_is_preserved(self):
        from jev_bench.benchmarks.bfcl import load_tasks, run_episode
        from jev_bench.io import read_jsonl
        from jev_bench.pa_tool import task_key
        from jev_bench.pa_tool_catalog import build_catalog

        from tests.support import completion, sdk

        value = config()
        tasks, answers = load_tasks(value)
        identity = "multi_turn_base", "multi_turn_base_0"
        item = {
            "category": identity[0],
            "task_id": identity[1],
            "condition": "native",
            "trial": 0,
            "attempt": 1,
        }
        catalog = build_catalog(value, value["conditions"]["native"]["executor"])
        tools = catalog["scopes"][catalog["bindings"][task_key(item)]]
        call = {
            "id": "old",
            "type": "function",
            "function": {"name": tools[0]["function"]["name"], "arguments": "{}"},
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as temporary:
            root = Path(temporary)
            p = freeze(value, item, tools, root / "mapping")
            with (
                patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(lambda **_: completion(calls=[call])),
                ),
                redirect_stdout(io.StringIO()),
            ):
                outcome = run_episode(
                    value,
                    item,
                    tasks[identity],
                    answers[identity],
                    root / "episode",
                    pa_tool_identity=p.asset_identity,
                )
            self.assertEqual(outcome["failure_kind"], "executor_response_error")
            self.assertFalse(outcome["success"])
            raw = read_jsonl(root / "episode/diagnostics/executor.jsonl")
            self.assertEqual(
                raw[0]["response"]["choices"][0]["message"]["tool_calls"][0], call
            )
