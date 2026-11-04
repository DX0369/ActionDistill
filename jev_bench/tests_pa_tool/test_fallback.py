"""Original-name fallback preserves the algorithm and explicit failure boundary."""

import json
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.io import append_json
from jev_bench.pa_tool import Projection
from jev_bench.pa_tool_generation import CandidateGenerator, generation_cost, slots
from jev_bench.pa_tool_prepare import (
    NameGenerationError,
    build_mapping,
    generated_name,
    identity,
    resolve_names,
)
from tests.pa_fixtures import result, tool

PROJECT = Path(__file__).resolve().parents[1]


def catalog(tools):
    return {
        "version": 1,
        "benchmark": {"name": "bfcl"},
        "control_names": [],
        "scopes": {"scope": tools},
        "bindings": {"task": "scope"},
    }


class FallbackTests(unittest.TestCase):
    def test_minimum_valid_count_reference_failure_and_audit_are_distinct(self):
        function = tool()
        function["function"]["parameters"] = {"type": "object", "properties": {}}
        for failed, reference, expected_reason in (
            (4, "chosen", None),
            (5, "chosen", "insufficient_valid_candidates"),
            (0, None, "reference_failed"),
            (5, None, "reference_failed"),
        ):
            with self.subTest(failed=failed, reference=reference):
                mapping = build_mapping(
                    catalog([function]),
                    lambda *_, failed=failed, reference=reference: (
                        [None] * failed + ["chosen"] * (16 - failed),
                        reference,
                    ),
                )
                self.assertEqual(
                    mapping["scopes"]["scope"][0]["name"],
                    "chosen" if expected_reason is None else "lookup",
                )
                key, stats = next(iter(mapping["candidate_statistics"].items()))
                self.assertEqual(
                    stats,
                    {
                        "component": "tool",
                        "valid_candidates": 16 - failed,
                        "discarded_candidate_slots": list(range(failed)),
                        "reference_valid": reference is not None,
                    },
                )
                if expected_reason is None:
                    self.assertEqual(mapping["generation_fallbacks"], {})
                else:
                    self.assertEqual(
                        mapping["generation_fallbacks"][key]["reason"], expected_reason
                    )
                    self.assertEqual(
                        mapping["generation_fallbacks"][key]["failed_slots"],
                        list(range(failed)) + ([16] if reference is None else []),
                    )

    def test_failed_first_slot_does_not_stop_remaining_slots_and_is_not_resampled(self):
        for concurrency in (1, 4):
            requests = []
            seed0 = int(identity([20261015, "component", 0])[:15], 16)

            class Client:
                def __init__(self, settings, sink):
                    self.sink = sink

                def close(self):
                    pass

                def complete(self, requests=requests, seed0=seed0, **request):
                    requests.append(request)
                    value = result(content="chosen")
                    value.raw_response["model"] = "qwen3-4b"
                    if request["extra_body"]["seed"] == seed0:
                        value.finish_reason = "length"
                        value.raw_response["choices"][0]["finish_reason"] = "length"
                    self.sink(
                        "executor",
                        {
                            "request": request,
                            "response": value.raw_response,
                            "usage": value.usage,
                            "latency_seconds": 1,
                            "error": None,
                        },
                    )
                    return value

            with (
                self.subTest(concurrency=concurrency),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            ):
                ledger = Path(path) / "generation.jsonl"
                settings = {"base_url": "http://fake", "model": "qwen3-4b"}
                generator = CandidateGenerator(
                    settings, ledger, concurrency=concurrency, client_factory=Client
                )
                expected = ([None] + ["chosen"] * 15, "chosen")
                self.assertEqual(generator("component", "prompt"), expected)
                self.assertEqual(len(requests), 17)
                self.assertEqual(len(slots(ledger)), 17)
                self.assertEqual(generation_cost(ledger)["requests"], 17)
                resumed = CandidateGenerator(
                    settings,
                    ledger,
                    client_factory=lambda *_: self.fail("no resampling"),
                )
                self.assertEqual(resumed("component", "prompt"), expected)
                self.assertEqual(len(requests), 17)

    def test_tool_and_parameter_originals_are_reserved_and_round_trip(self):
        first, second = tool(), tool("other_tool")
        second["function"]["parameters"] = {"type": "object", "properties": {}}
        tools = [first, second]
        original = deepcopy(tools)
        calls = []
        outcomes = iter(
            [
                ([None] * 5 + ["unused"] * 11, "unused"),
                (["lookup"] * 15 + ["new_tool"], "lookup"),
                (["unused"] * 16, None),
                (["key"] * 15 + ["new_payload"], "key"),
            ]
        )

        def generate(key, text):
            calls.append(text)
            value = next(outcomes)
            return value

        mapping = build_mapping(catalog(tools), generate)
        rows = mapping["scopes"]["scope"]
        self.assertEqual(
            rows,
            [
                {
                    "name": "lookup",
                    "parameters": {"key": "key", "payload": "new_payload"},
                },
                {"name": "new_tool", "parameters": {}},
            ],
        )
        self.assertTrue(all("Tool: lookup - " in text for text in calls[2:]))
        self.assertEqual(
            {
                (r["component"], r["original_name"], r["reason"])
                for r in mapping["generation_fallbacks"].values()
            },
            {
                ("tool", "lookup", "insufficient_valid_candidates"),
                ("parameter", "key", "reference_failed"),
            },
        )
        projection = Projection(tools, rows)
        projection.request([], tools)
        restored = projection.response(
            result(
                [
                    ("lookup", '{"key":"x","new_payload":{"key":"nested"}}'),
                    ("new_tool", "{}"),
                ]
            )
        ).message.tool_calls
        self.assertEqual(
            [call.function.name for call in restored], ["lookup", "other_tool"]
        )
        self.assertEqual(
            json.loads(restored[0].function.arguments),
            {"key": "x", "payload": {"key": "nested"}},
        )
        self.assertEqual(tools, original)

    def test_duplicate_illegal_or_exhausted_original_names_still_stop(self):
        def failed(*_):
            return [None] * 16, None

        for tools in ([tool(), tool()], [tool("not-a-legal-name")]):
            with (
                self.subTest(tools=tools),
                self.assertRaisesRegex(ValueError, "original names"),
            ):
                build_mapping(catalog(tools), failed)
        with self.assertRaisesRegex(ValueError, "candidates exhausted"):
            resolve_names(
                {0: None, 1: [{"name": "lookup", "score": 15}]},
                original_names={0: "lookup", 1: "other"},
            )
        with self.assertRaisesRegex(ValueError, "original names"):
            resolve_names({0: None}, reserved=["Finish"], original_names={0: "Finish"})
        with self.assertRaises(KeyError):
            resolve_names({0: None}, original_names={})

    def test_fallback_only_accepts_known_model_output_failures(self):
        for reason, content, calls in (
            ("length", "partial_name", []),
            ("stop", None, []),
            ("stop", "two names", []),
            ("stop", "", []),
            ("stop", "name", [{"id": "unexpected"}]),
        ):
            with (
                self.subTest(reason=reason, content=content),
                self.assertRaises(NameGenerationError),
            ):
                generated_name(reason, content, calls)
        for reason in (None, "error", "content_filter", "tool_calls"):
            with self.subTest(reason=reason), self.assertRaises(ValueError) as caught:
                generated_name(reason, "name", [])
            self.assertNotIsInstance(caught.exception, NameGenerationError)
        for reason in ("stop", "length"):
            for content, calls in (
                (False, []),
                (123, []),
                ([], []),
                ({}, []),
                (None, {}),
            ):
                with (
                    self.subTest(reason=reason, content=content, calls=calls),
                    self.assertRaises(ValueError) as caught,
                ):
                    generated_name(reason, content, calls)
                self.assertNotIsInstance(caught.exception, NameGenerationError)
        self.assertEqual(generated_name("stop", "  name\n", []), "name")
        for failure in (ConnectionError("unavailable"), KeyError("missing field")):

            def generate(*_, failure=failure):
                raise failure

            with self.subTest(failure=failure), self.assertRaises(type(failure)):
                build_mapping(catalog([tool()]), generate)

    def test_successfully_generated_original_name_is_not_counted_as_fallback(self):
        function = tool()
        function["function"]["parameters"] = {"type": "object", "properties": {}}
        mapping = build_mapping(
            catalog([function]), lambda *_: (["lookup"] * 16, "lookup")
        )
        self.assertEqual(mapping["scopes"]["scope"][0]["name"], "lookup")
        self.assertEqual(mapping["generation_fallbacks"], {})

    def test_parallel_service_error_overrides_model_fallback_and_keeps_cost(self):
        barrier = threading.Barrier(2)
        saved_failure = threading.Event()
        seed0 = int(identity([20261015, "component", 0])[:15], 16)
        requested = []

        def append(path, event):
            append_json(path, event)
            if event["phase"] == "finished" and event["slot"] == 0:
                saved_failure.set()

        class Client:
            def __init__(self, settings, sink):
                self.sink = sink

            def close(self):
                pass

            def complete(client, **request):
                requested.append(request)
                if len(requested) <= 2:
                    barrier.wait(timeout=5)
                if request["extra_body"]["seed"] == seed0:
                    value = result(content=None)
                    value.finish_reason = "length"
                    value.raw_response["model"] = "qwen3-4b"
                    value.raw_response["choices"][0]["finish_reason"] = "length"
                    client.sink(
                        "executor",
                        {
                            "usage": value.usage,
                            "latency_seconds": 1,
                            "response": value.raw_response,
                            "error": None,
                        },
                    )
                    return value
                self.assertTrue(saved_failure.wait(timeout=5))
                raise ConnectionError("provider offline")

        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            generator = CandidateGenerator(
                {"base_url": "http://unused", "model": "qwen3-4b"},
                path,
                concurrency=2,
                client_factory=Client,
            )
            with (
                patch("jev_bench.pa_tool_generation.append_json", side_effect=append),
                self.assertRaisesRegex(ConnectionError, "provider offline"),
            ):
                generator("component", "prompt")
            # A verified name failure may free a worker before the fatal
            # service failure becomes visible; every started request is drained.
            self.assertIn(len(requested), (2, 3))
            saved = slots(path)
            self.assertEqual(
                saved["component", 0]["error"]["type"], "NameGenerationError"
            )
            self.assertEqual(saved["component", 1]["error"]["type"], "ConnectionError")
            cost = generation_cost(path)
            self.assertEqual(cost["requests"], len(requested))
            self.assertEqual(cost["total_tokens"]["missing"], len(requested) - 1)
            with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                CandidateGenerator(
                    {"base_url": "http://unused", "model": "qwen3-4b"}, path
                )


if __name__ == "__main__":
    unittest.main()
