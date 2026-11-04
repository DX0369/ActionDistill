"""CPU-only naming concurrency, ledger, and inherited-cost contracts."""

import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.io import append_json, read_jsonl
from jev_bench.pa_tool_generation import CandidateGenerator, generation_cost, slots
from jev_bench.pa_tool_prepare import PROTOCOL, identity
from tests.pa_fixtures import result

PROJECT = Path(__file__).resolve().parents[1]
COMPONENT = "component"
TEXT = "Original naming prompt."
SETTINGS = {"base_url": "http://primary"}


def seed(slot, key=COMPONENT):
    return int(identity([PROTOCOL["seed"], key, slot])[:15], 16)


def successful_event(slot):
    return {
        "component": COMPONENT,
        "slot": slot,
        "seed": seed(slot),
        "text": TEXT,
        "phase": "finished",
        "name": f"name_{slot}",
        "error": None,
        "provider": {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_seconds": 1,
        },
    }


def client_factory(handler, clients):
    class Client:
        def __init__(self, settings, sink):
            self.sink = sink
            self.calls = 0
            self.closed = False
            clients.append(self)

        def complete(self, **request):
            self.calls += 1
            return handler(self.sink, request)

        def close(self):
            self.closed = True

    return Client


class GenerationParallelTests(unittest.TestCase):
    def test_two_endpoints_route_original_slots_without_resampling(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            settings = (
                {"base_url": "http://primary", "model": "/same/model"},
                {"base_url": "http://secondary", "model": "/same/model"},
            )
            by_seed = {seed(slot): slot for slot in range(17)}
            barrier = threading.Barrier(16)
            lock = threading.Lock()
            requested = {}
            clients = []

            class Client:
                def __init__(client, configuration, sink):
                    client.settings = configuration
                    client.sink = sink
                    client.closed = False
                    clients.append(client)

                def close(client):
                    client.closed = True

                def complete(client, **request):
                    slot = by_seed[request["extra_body"]["seed"]]
                    self.assertEqual(client.settings, settings[slot % 2])
                    self.assertEqual(request["temperature"], 0 if slot == 16 else 0.4)
                    self.assertEqual(
                        request["messages"], [{"role": "user", "content": TEXT}]
                    )
                    with lock:
                        self.assertNotIn(slot, requested)
                        requested[slot] = client.settings["base_url"]
                    if slot < 16:
                        barrier.wait(timeout=5)
                    client.sink("executor", successful_event(slot)["provider"])
                    return result(content=f"name_{slot}")

            generator = CandidateGenerator(
                settings[0],
                path,
                concurrency=16,
                additional_settings=(settings[1],),
                client_factory=Client,
            )
            expected = ([f"name_{slot}" for slot in range(16)], "name_16")
            self.assertEqual(generator(COMPONENT, TEXT), expected)
            self.assertEqual(generator(COMPONENT, TEXT), expected)
            self.assertEqual(set(requested), set(range(17)))
            self.assertEqual(len(clients), 17)
            self.assertTrue(all(client.closed for client in clients))
            events = read_jsonl(path)
            self.assertEqual(len(events), 34)
            for event in events:
                self.assertEqual(
                    event["provider_base_url"], settings[event["slot"] % 2]["base_url"]
                )
                self.assertEqual(event["seed"], seed(event["slot"]))
            self.assertEqual(generation_cost(path, origin="new")["requests"], 17)
            self.assertEqual(
                generation_cost(path)["total_tokens"]["observed_total"], 17 * 15
            )
            missing = Path(directory) / "missing_base_url.jsonl"
            invalid = CandidateGenerator(
                {},
                missing,
                additional_settings=(settings[1],),
                client_factory=lambda *_: self.fail(
                    "missing base_url must not request"
                ),
            )
            with self.assertRaises(KeyError):
                invalid(COMPONENT, TEXT)
            self.assertFalse(missing.exists())

    def test_8_and_16_out_of_order_preserve_protocol_cost_and_main_thread_writes(self):
        for concurrency in (8, 16):
            with self.subTest(concurrency=concurrency):
                self._assert_out_of_order(concurrency)

    def _assert_out_of_order(self, concurrency):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            main_thread = threading.get_ident()
            condition = threading.Condition()
            barrier = threading.Barrier(concurrency)
            turn = concurrency - 1
            active = peak = 0
            completed = []
            clients = []
            persisted = set()
            requests = {}
            by_seed = {seed(slot): slot for slot in range(17)}

            def append(target, event):
                self.assertEqual(threading.get_ident(), main_thread)
                append_json(target, event)
                if event["phase"] == "started":
                    self.assertEqual(read_jsonl(target)[-1], event)
                    persisted.add(event["slot"])

            class Saved(dict):
                def __setitem__(saved_self, key, event):
                    self.assertEqual(threading.get_ident(), main_thread)
                    super().__setitem__(key, event)

            def complete(sink, request):
                nonlocal active, peak, turn
                slot = by_seed[request["extra_body"]["seed"]]
                self.assertIn(slot, persisted)
                with condition:
                    active += 1
                    peak = max(peak, active)
                    requests[slot] = request
                sink(
                    "executor",
                    {
                        "request": request,
                        "usage": {
                            "prompt_tokens": slot + 1,
                            "completion_tokens": 1,
                            "total_tokens": slot + 2,
                        },
                        "latency_seconds": slot + 1,
                    },
                )
                try:
                    if slot < concurrency:
                        barrier.wait(timeout=5)
                        with condition:
                            self.assertTrue(
                                condition.wait_for(lambda: turn == slot, timeout=5)
                            )
                            completed.append(slot)
                            turn -= 1
                            condition.notify_all()
                    else:
                        with condition:
                            completed.append(slot)
                    return result(content=f"  name_{slot}\n")
                finally:
                    with condition:
                        active -= 1

            generator = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=concurrency,
                client_factory=client_factory(complete, clients),
            )
            generator.saved = Saved(generator.saved)
            with patch("jev_bench.pa_tool_generation.append_json", side_effect=append):
                samples, reference = generator(COMPONENT, TEXT)
            generator.close()
            self.assertEqual(samples, [f"name_{slot}" for slot in range(16)])
            self.assertEqual(reference, "name_16")
            self.assertEqual(peak, concurrency)
            self.assertGreater(completed.index(0), completed.index(concurrency - 1))
            self.assertEqual(len(clients), 17)
            self.assertTrue(
                all(client.calls == 1 and client.closed for client in clients)
            )
            self.assertEqual(len(read_jsonl(path)), 34)
            for slot, request in requests.items():
                self.assertEqual(
                    request,
                    {
                        "messages": [{"role": "user", "content": TEXT}],
                        "tools": None,
                        "tool_choice": None,
                        "max_tokens": 8192,
                        "temperature": 0 if slot == 16 else 0.4,
                        "top_p": 1,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": True},
                            "top_k": -1,
                            "min_p": 0,
                            "seed": seed(slot),
                        },
                    },
                )
                event = generator.saved[COMPONENT, slot]
                self.assertEqual(event["provider"]["request"], request)
                self.assertEqual(event["provider"]["usage"]["total_tokens"], slot + 2)
            cost = generation_cost(path)
            self.assertEqual(cost["requests"], 17)
            self.assertEqual(cost["total_tokens"]["observed_total"], sum(range(2, 19)))
            resumed = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=concurrency,
                client_factory=lambda *_: self.fail("cached slots must not request"),
            )
            self.assertEqual(resumed(COMPONENT, TEXT), (samples, reference))
            self.assertEqual(generation_cost(path), cost)
            self.assertEqual(len(read_jsonl(path)), 34)

    def test_first_failure_stops_submission_and_drains_every_started_request(self):
        for concurrency in (8, 16):
            with self.subTest(concurrency=concurrency):
                self._assert_failure_drains(concurrency)

    def _assert_failure_drains(self, concurrency):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            barrier = threading.Barrier(concurrency)
            failure_saved = threading.Event()
            clients = []
            requested = []
            by_seed = {seed(slot): slot for slot in range(17)}

            def append(target, event):
                append_json(target, event)
                if event["phase"] == "finished" and event["slot"] == 0:
                    failure_saved.set()

            def complete(sink, request):
                slot = by_seed[request["extra_body"]["seed"]]
                requested.append(slot)
                barrier.wait(timeout=5)
                if slot == 0:
                    sink("executor", successful_event(slot)["provider"])
                    raise RuntimeError("first request failed")
                self.assertTrue(failure_saved.wait(timeout=5))
                if slot == 1:
                    # A second in-flight failure has no provider measurements.
                    raise RuntimeError("missing provider diagnostics")
                sink("executor", successful_event(slot)["provider"])
                return result(content=f"name_{slot}")

            generator = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=concurrency,
                client_factory=client_factory(complete, clients),
            )
            with (
                patch("jev_bench.pa_tool_generation.append_json", side_effect=append),
                self.assertRaisesRegex(RuntimeError, "first request failed"),
            ):
                generator(COMPONENT, TEXT)
            self.assertEqual(sorted(requested), list(range(concurrency)))
            events = slots(path)
            self.assertEqual(len(events), concurrency)
            self.assertTrue(
                all(event["phase"] == "finished" for event in events.values())
            )
            self.assertEqual(events[COMPONENT, 0]["error"]["type"], "RuntimeError")
            self.assertEqual(events[COMPONENT, 1]["error"]["type"], "RuntimeError")
            self.assertTrue(all(client.closed for client in clients))
            cost = generation_cost(path, origin="new")
            self.assertEqual(cost["requests"], concurrency)
            self.assertEqual(
                cost["total_tokens"],
                {
                    "observed_total": (concurrency - 1) * 15,
                    "known": concurrency - 1,
                    "missing": 1,
                },
            )
            self.assertEqual(cost["latency_seconds"]["missing"], 1)
            with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                generator(COMPONENT, TEXT)
            with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                CandidateGenerator(SETTINGS, path)
            self.assertEqual(len(read_jsonl(path)), concurrency * 2)
            self.assertEqual(generation_cost(path), cost)

    def test_main_thread_interruption_drains_requests_without_replacement(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            release = threading.Event()
            clients = []

            def complete(sink, request):
                self.assertTrue(release.wait(timeout=5))
                sink("executor", successful_event(0)["provider"])
                return result(content="name")

            def interrupted_wait(*args, **kwargs):
                release.set()
                raise KeyboardInterrupt("operator interrupted")

            generator = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=8,
                client_factory=client_factory(complete, clients),
            )
            with (
                patch(
                    "jev_bench.pa_tool_generation.wait", side_effect=interrupted_wait
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                generator(COMPONENT, TEXT)
            self.assertEqual(len(clients), 8)
            self.assertTrue(all(client.closed for client in clients))
            self.assertTrue(
                all(event["phase"] == "finished" for event in slots(path).values())
            )
            self.assertEqual(generation_cost(path)["requests"], 8)

    def test_partial_success_cache_skips_slots_and_rejects_changed_prompt_before_requests(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            for slot in (3, 16):
                event = successful_event(slot)
                append_json(path, {**event, "phase": "started", "name": None})
                append_json(path, event)
            clients = []
            requested = []
            by_seed = {seed(slot): slot for slot in range(17)}

            def complete(sink, request):
                slot = by_seed[request["extra_body"]["seed"]]
                requested.append(slot)
                sink("executor", successful_event(slot)["provider"])
                return result(content=f"name_{slot}")

            generator = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=8,
                client_factory=client_factory(complete, clients),
            )
            with self.assertRaisesRegex(ValueError, "resumed naming request differs"):
                generator(COMPONENT, "changed")
            self.assertEqual(clients, [])
            generator.concurrency = 16
            self.assertEqual(
                generator(COMPONENT, TEXT),
                ([f"name_{i}" for i in range(16)], "name_16"),
            )
            self.assertEqual(set(requested), set(range(17)) - {3, 16})
            self.assertEqual(generation_cost(path)["requests"], 17)

    def test_lazy_reuse_is_terminal_and_separates_historical_cost(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            sources = {
                slot: {
                    **successful_event(slot),
                    "source": {
                        "directory": "/historical",
                        "component": COMPONENT,
                        "slot": slot,
                    },
                }
                for slot in range(17)
            }
            original = deepcopy(sources)
            calls = []

            def reuse(key, text):
                calls.append((key, text))
                return sources

            generator = CandidateGenerator(
                SETTINGS,
                path,
                concurrency=16,
                reuse=reuse,
                client_factory=lambda *_: self.fail("reused slots must not request"),
            )
            expected = ([f"name_{i}" for i in range(16)], "name_16")
            self.assertEqual(generator(COMPONENT, TEXT), expected)
            self.assertEqual(generator(COMPONENT, TEXT), expected)
            self.assertEqual(calls, [(COMPONENT, TEXT)])
            self.assertEqual(sources, original)
            self.assertEqual(len(read_jsonl(path)), 17)
            self.assertEqual(generation_cost(path, origin="new")["requests"], 0)
            self.assertEqual(generation_cost(path, origin="reused")["requests"], 17)
            self.assertEqual(
                generation_cost(path)["total_tokens"]["observed_total"], 17 * 15
            )
            resumed = CandidateGenerator(
                SETTINGS, path, reuse=lambda *_: self.fail("cached")
            )
            self.assertEqual(resumed(COMPONENT, TEXT), expected)
            self.assertEqual(len(read_jsonl(path)), 17)
            event = read_jsonl(path)[0]
            append_json(path, event)
            with self.assertRaisesRegex(ValueError, "sampled twice"):
                slots(path)

    def test_new_cost_includes_uncertain_requests_and_reuse_validation_fails_explicitly(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            event = successful_event(0)
            append_json(
                path,
                {
                    **event,
                    "phase": "started",
                    "name": None,
                    "provider": {"usage": None, "latency_seconds": None},
                },
            )
            new = generation_cost(path, origin="new")
            self.assertEqual(new["requests"], 1)
            self.assertEqual(new["total_tokens"]["missing"], 1)
            self.assertEqual(generation_cost(path, origin="reused")["requests"], 0)
            with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                CandidateGenerator(SETTINGS, path)
            with self.assertRaises(ValueError):
                generation_cost(path, origin="finished")

        for field, value in (
            ("error", {"type": "Error"}),
            ("text", "changed"),
            ("seed", 0),
            ("source", None),
            ("source", {}),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory,
            ):
                path = Path(directory) / "generation.jsonl"
                source = {
                    **successful_event(0),
                    "source": {"directory": "/old"},
                    field: value,
                }
                generator = CandidateGenerator(
                    SETTINGS,
                    path,
                    reuse=lambda *_, source=source: {0: source},
                    client_factory=lambda *_: self.fail(
                        "invalid reuse must not request"
                    ),
                )
                with self.assertRaisesRegex(ValueError, "PA-Tool reused"):
                    generator(COMPONENT, TEXT)
                self.assertFalse(path.exists())

        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            generator = CandidateGenerator(
                SETTINGS,
                path,
                reuse=lambda *_: {0: successful_event(0)},
                client_factory=lambda *_: self.fail("missing source must not request"),
            )
            with self.assertRaises(KeyError):
                generator(COMPONENT, TEXT)
            self.assertFalse(path.exists())

    def test_invalid_concurrency_is_explicit(self):
        for concurrency in (0, -1, False, 1.5, "8", None):
            with (
                self.subTest(concurrency=concurrency),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                CandidateGenerator(
                    SETTINGS, PROJECT / "tmp/unused.jsonl", concurrency=concurrency
                )


if __name__ == "__main__":
    unittest.main()
