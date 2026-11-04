"""Explicit reuse validates real request evidence without contacting a service."""

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from jev_bench.io import append_json, write_json
from jev_bench.pa_tool_generation import CandidateGenerator, generation_cost, slots
from jev_bench.pa_tool_prepare import PROTOCOL, NameGenerationError, identity, prompt
from jev_bench.pa_tool_reuse import ReuseSource
from tests.pa_fixtures import result, tool

PROJECT = Path(__file__).resolve().parents[1]


def catalog():
    function = tool()
    function["function"]["parameters"] = {"type": "object", "properties": {}}
    return {
        "version": 1,
        "benchmark": {"name": "bfcl"},
        "control_names": [],
        "scopes": {"source_domain": [function]},
        "bindings": {"task": "source_domain"},
        "definition_sources": {},
        "executor": {
            "base_url": "http://127.0.0.1:1/v1",
            "model": "qwen3-4b",
            "model_path": "/data/wangsq/model/Qwen3-4B",
            "max_model_len": 131072,
            "timeout_seconds": 600,
        },
    }


def component(value):
    function = next(iter(value["scopes"].values()))[0]["function"]
    return (
        identity({"function": function, "occurrence": 0, "component": "tool"}),
        prompt("tool", function["description"]),
    )


def event(key, text, slot=0):
    seed = int(identity([PROTOCOL["seed"], key, slot])[:15], 16)
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return {
        "component": key,
        "slot": slot,
        "seed": seed,
        "text": text,
        "phase": "finished",
        "name": "old_name",
        "error": None,
        "provider": {
            "request": {
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": text}],
                "temperature": 0 if slot == 16 else 0.4,
                "max_tokens": 8192,
                "top_p": 1,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": -1,
                    "min_p": 0,
                    "seed": seed,
                },
            },
            "response": {
                "model": "qwen3-4b",
                "usage": usage,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "\nold_name",
                            "tool_calls": [],
                        },
                    }
                ],
            },
            "error": None,
            "usage": usage,
            "latency_seconds": 1,
        },
    }


def source(directory, value, events, *, protocol=None):
    write_json(directory / "catalog.json", value)
    write_json(
        directory / "metadata/run_status.json",
        {
            "status": "failed",
            "protocol": protocol if protocol is not None else PROTOCOL,
            "service": {
                "id": value["executor"]["model"],
                "root": value["executor"]["model_path"],
                "max_model_len": value["executor"]["max_model_len"],
            },
        },
    )
    path = directory / "diagnostics/generation.jsonl"
    for record in events:
        if record["phase"] != "reused":
            append_json(
                path,
                {
                    **record,
                    "phase": "started",
                    "name": None,
                    "error": None,
                    "provider": {"usage": None, "latency_seconds": None},
                },
            )
        if record["phase"] != "started":
            append_json(path, record)


class Client:
    calls: ClassVar[list] = []

    def __init__(self, settings, sink):
        self.sink = sink

    def close(self):
        pass

    def complete(self, **request):
        self.calls.append(request)
        value = result(content="new_name")
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


class ReuseTests(unittest.TestCase):
    def test_old_32_protocol_is_not_imported_into_pa16(self):
        value = catalog()
        key, text = component(value)
        original = self.root / "old32"
        source(
            original, value, [event(key, text)], protocol={**PROTOCOL, "candidates": 32}
        )
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            ReuseSource(original, value)

    def test_success_slot_with_missing_name_cannot_become_an_excluded_candidate(self):
        value = catalog()
        key, text = component(value)
        original = self.root / "invalid_success"
        record = event(key, text)
        record["name"] = None
        source(original, value, [record])
        with self.assertRaisesRegex(ValueError, "successful saved slot"):
            CandidateGenerator(
                value["executor"], original / "diagnostics/generation.jsonl"
            )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Client.calls = []

    def test_cross_domain_lazy_import_keeps_source_and_separates_cost(self):
        value = catalog()
        key, text = component(value)
        unrelated = event("unrelated_component", "unrelated prompt")
        unrelated["phase"] = "started"
        original = self.root / "old"
        source(
            original,
            value,
            [event(key, text, slot) for slot in range(17)] + [unrelated],
        )
        before = {
            str(path.relative_to(original)): path.read_bytes()
            for path in original.rglob("*")
            if path.is_file()
        }
        current = deepcopy(value)
        current["executor"]["base_url"] = "http://127.0.0.1:2/v1"
        current["scopes"] = {"other_domain": current["scopes"]["source_domain"]}
        current["bindings"] = {"different_task": "other_domain"}
        ledger = self.root / "new/generation.jsonl"
        generator = CandidateGenerator(
            current["executor"],
            ledger,
            client_factory=Client,
            reuse=ReuseSource(original, current),
        )
        self.addCleanup(generator.close)
        self.assertEqual(generator(key, text), (["old_name"] * 16, "old_name"))
        self.assertEqual(Client.calls, [])
        generator("unseen_component", "new prompt")
        self.assertEqual(len(Client.calls), 17)
        self.assertEqual(generation_cost(ledger)["requests"], 34)
        self.assertEqual(generation_cost(ledger, origin="new")["requests"], 17)
        self.assertEqual(generation_cost(ledger, origin="reused")["requests"], 17)
        self.assertEqual(
            generation_cost(ledger, origin="reused")["total_tokens"]["observed_total"],
            17 * 15,
        )
        self.assertEqual(slots(ledger)[key, 0]["phase"], "reused")
        self.assertEqual(slots(ledger)[key, 0]["source"]["directory"], str(original))
        self.assertNotIn(("unrelated_component", 0), slots(ledger))
        self.assertEqual(
            before,
            {
                str(path.relative_to(original)): path.read_bytes()
                for path in original.rglob("*")
                if path.is_file()
            },
        )

    def test_matching_failed_or_uncertain_slot_never_issues_a_new_request(self):
        value = catalog()
        key, text = component(value)
        for phase in ("started", "finished"):
            with self.subTest(phase=phase):
                saved = event(key, text)
                saved["phase"] = phase
                saved["error"] = {"type": "ValueError", "message": "bad response"}
                original = self.root / phase
                source(original, value, [saved])
                generator = CandidateGenerator(
                    value["executor"],
                    self.root / f"{phase}.jsonl",
                    client_factory=Client,
                    reuse=ReuseSource(original, value),
                )
                self.addCleanup(generator.close)
                with self.assertRaisesRegex(ValueError, "failed or uncertain"):
                    generator(key, text)
        self.assertEqual(Client.calls, [])

    def test_verified_model_failure_completes_cli_with_original_name_and_original_cost(
        self,
    ):
        value = catalog()
        key, text = component(value)
        events = [event(key, text, slot) for slot in range(17)]
        failed = events[16]
        error = NameGenerationError("length")
        failed["name"] = None
        failed["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "reason": error.reason,
        }
        failed["provider"]["response"]["choices"][0].update(finish_reason="length")
        failed["provider"]["response"]["choices"][0]["message"]["content"] = None
        original = self.root / "source"
        source(original, value, events, protocol=PROTOCOL)
        before = {
            str(path.relative_to(original)): path.read_bytes()
            for path in original.rglob("*")
            if path.is_file()
        }
        spec = importlib.util.spec_from_file_location(
            "pa_fallback_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        target = self.root / "target"
        args = SimpleNamespace(
            qwen_model_path=Path(value["executor"]["model_path"]),
            protocol=PROJECT / "configs/pa_tool_prepare.json",
            execute=True,
            resume=False,
            reuse_from=original,
            concurrency=2,
            additional_base_url=None,
            output_dir=target,
        )
        with (
            patch.object(cli, "check_generator", return_value={}),
            patch.object(
                cli, "source_identity", return_value={"trees": {}, "dirty_paths": []}
            ),
            patch.object(
                cli,
                "CandidateGenerator",
                side_effect=lambda settings, path, **kw: CandidateGenerator(
                    settings,
                    path,
                    client_factory=lambda *_: self.fail(
                        "no request after verified output failure"
                    ),
                    **kw,
                ),
            ),
        ):
            cli.generate(args, value)
        summary = json.loads((target / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(
            summary["naming_fallbacks"],
            {"unique_components": 1, "tools": 1, "parameters": 0},
        )
        self.assertEqual(summary["preparation_cost"]["requests"], 17)
        self.assertEqual(summary["new_preparation_cost"]["requests"], 0)
        self.assertEqual(summary["reused_preparation_cost"]["requests"], 17)
        mapping = json.loads((target / "mapping.json").read_text())
        self.assertEqual(
            summary["candidate_outcomes"]["tool"],
            {
                "unique_components": 1,
                "valid_candidates_histogram": {"16": 1},
                "excluded_candidate_slots": 0,
                "invalid_references": 1,
            },
        )
        self.assertEqual(mapping["scopes"]["source_domain"][0]["name"], "lookup")
        self.assertEqual(
            mapping["generation_fallbacks"][key]["failed_slots"],
            [16],
        )
        ledger = target / "diagnostics/generation.jsonl"
        saved = slots(ledger)[key, 16]
        self.assertEqual(saved["error"], failed["error"])
        self.assertIsNone(saved["name"])
        self.assertEqual(saved["provider"], failed["provider"])
        saved_cost = generation_cost(ledger)
        resumed = CandidateGenerator(
            value["executor"],
            ledger,
            client_factory=lambda *_: self.fail("no resample"),
        )
        self.assertEqual(resumed(key, text), (["old_name"] * 16, None))
        self.assertEqual(generation_cost(ledger), saved_cost)
        self.assertEqual(
            before,
            {
                str(path.relative_to(original)): path.read_bytes()
                for path in original.rglob("*")
                if path.is_file()
            },
        )

    def test_named_failure_requires_matching_response_and_current_protocol(self):
        value = catalog()
        key, text = component(value)
        failed = event(key, text)
        error = NameGenerationError("length")
        failed["name"] = None
        failed["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "reason": error.reason,
        }
        # A label alone cannot turn a successful provider response into fallback.
        original = self.root / "source"
        source(original, value, [failed], protocol=PROTOCOL)
        with self.assertRaises(ValueError):
            ReuseSource(original, value)(key, text)
        with self.assertRaises(ValueError):
            CandidateGenerator(
                value["executor"], original / "diagnostics/generation.jsonl"
            )
        failed["provider"]["response"]["choices"][0].update(finish_reason="length")
        failed["provider"]["response"]["choices"][0]["message"]["content"] = None
        for index, change in enumerate(
            (
                "service_error",
                "missing_reason",
                "wrong_reason",
                "missing_response",
                "wrong_model",
                "multiple_choices",
            )
        ):
            record = deepcopy(failed)
            if change == "service_error":
                record["provider"]["error"] = {
                    "type": "ConnectionError",
                    "message": "offline",
                }
            elif change == "missing_reason":
                del record["error"]["reason"]
            elif change == "wrong_reason":
                record["error"]["reason"] = "invalid_final_name"
            elif change == "missing_response":
                del record["provider"]["response"]
            elif change == "wrong_model":
                record["provider"]["response"]["model"] = "wrong-model"
            else:
                record["provider"]["response"]["choices"] *= 2
            directory = self.root / f"invalid_{index}"
            source(directory, value, [record], protocol=PROTOCOL)
            with self.subTest(change=change), self.assertRaises((KeyError, ValueError)):
                ReuseSource(directory, value)(key, text)
            with self.subTest(resume=change), self.assertRaises((KeyError, ValueError)):
                CandidateGenerator(
                    value["executor"], directory / "diagnostics/generation.jsonl"
                )
        old = self.root / "old_protocol"
        source(
            old,
            value,
            [failed],
            protocol={k: v for k, v in PROTOCOL.items() if k != "generation_failure"},
        )
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            ReuseSource(old, value)

    def test_parameter_alias_is_part_of_the_reuse_identity(self):
        value = catalog()
        function = tool()["function"]
        value["scopes"]["source_domain"][0]["function"] = function
        spec = {
            "function": function,
            "occurrence": 0,
            "component": "parameter",
            "parameter": "key",
            "tool_alias": "old_tool",
            "description_source": "description",
        }
        text = prompt(
            "parameter",
            "Record key",
            tool_name="old_tool",
            tool_description=function["description"],
        )
        key = identity(spec)
        original = self.root / "source"
        source(original, value, [event(key, text)])
        reuse = ReuseSource(original, value)
        self.assertEqual(len(reuse(key, text)), 1)
        self.assertEqual(reuse(identity({**spec, "tool_alias": "new_tool"}), text), {})
        with self.assertRaisesRegex(ValueError, "prompt or seed"):
            reuse(key, text.replace("old_tool", "new_tool"))

    def test_bad_request_response_and_missing_required_fields_fail_explicitly(self):
        value = catalog()
        key, text = component(value)
        mutations = {
            "temperature": lambda e: e["provider"]["request"].update(temperature=0.6),
            "tokens": lambda e: e["provider"]["request"].update(max_tokens=1024),
            "thinking": lambda e: e["provider"]["request"]["extra_body"][
                "chat_template_kwargs"
            ].update(enable_thinking=False),
            "sample_count": lambda e: e["provider"]["request"].update(n=1),
            "tools": lambda e: e["provider"]["request"].update(tools=[]),
            "tool_choice": lambda e: e["provider"]["request"].update(tool_choice=None),
            "request_model": lambda e: e["provider"]["request"].update(model="other"),
            "missing_sampling": lambda e: e["provider"]["request"].pop("top_p"),
            "missing_request": lambda e: e["provider"].pop("request"),
            "seed": lambda e: e.update(seed=0),
            "response_model": lambda e: e["provider"]["response"].update(model="other"),
            "finish": lambda e: e["provider"]["response"]["choices"][0].update(
                finish_reason="length"
            ),
            "name": lambda e: e.update(name="another_name"),
            "missing_usage": lambda e: e["provider"].pop("usage"),
            "provider_failure": lambda e: e["provider"].update(
                error={"message": "failed"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                saved = event(key, text)
                mutate(saved)
                original = self.root / label
                source(original, value, [saved])
                reuse = ReuseSource(original, value)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    reuse(key, text)

    def test_source_protocol_and_model_context_must_match(self):
        value = catalog()
        key, text = component(value)
        original = self.root / "source"
        source(
            original,
            value,
            [event(key, text)],
            protocol={**PROTOCOL, "max_tokens": 1024},
        )
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            ReuseSource(original, value)
        state = json.loads((original / "metadata/run_status.json").read_text())
        state["protocol"] = PROTOCOL
        state["service"]["max_model_len"] = 32768
        write_json(original / "metadata/run_status.json", state)
        with self.assertRaisesRegex(ValueError, "model identity"):
            ReuseSource(original, value)

    def test_reused_source_retains_its_origin_chain(self):
        value = catalog()
        key, text = component(value)
        original = self.root / "original"
        source(original, value, [event(key, text, slot) for slot in range(17)])
        reused = ReuseSource(original, value)(key, text)
        for record in reused.values():
            record["phase"] = "reused"
        previous = self.root / "previous"
        source(previous, value, list(reused.values()), protocol=PROTOCOL)
        imported = ReuseSource(previous, value)(key, text)[0]
        self.assertEqual(imported["source"]["directory"], str(previous))
        self.assertEqual(imported["source"]["origin"]["directory"], str(original))
        self.assertEqual(imported["provider"], reused[0]["provider"])
        ledger = self.root / "final/generation.jsonl"
        generator = CandidateGenerator(
            value["executor"],
            ledger,
            client_factory=Client,
            reuse=ReuseSource(previous, value),
        )
        self.addCleanup(generator.close)
        self.assertEqual(generator(key, text), (["old_name"] * 16, "old_name"))
        self.assertEqual(Client.calls, [])
        self.assertEqual(slots(ledger)[key, 0]["source"], imported["source"])
        self.assertEqual(generation_cost(ledger, origin="new")["requests"], 0)

    def test_cli_new_directory_records_reuse_and_cannot_be_resume(self):
        module_spec = importlib.util.spec_from_file_location(
            "pa_reuse_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        value = catalog()
        key, text = component(value)
        original = self.root / "source"
        source(original, value, [event(key, text, slot) for slot in range(17)])
        target = self.root / "target"
        args = SimpleNamespace(
            qwen_model_path=Path(value["executor"]["model_path"]),
            protocol=PROJECT / "configs/pa_tool_prepare.json",
            execute=True,
            resume=False,
            reuse_from=original,
            concurrency=2,
            additional_base_url="http://127.0.0.1:2/v1",
            output_dir=target,
        )
        with (
            patch.object(cli, "check_generator", return_value={}) as health_checks,
            patch.object(
                cli, "source_identity", return_value={"trees": {}, "dirty_paths": []}
            ),
            patch.object(
                cli,
                "CandidateGenerator",
                side_effect=lambda settings, path, **kwargs: CandidateGenerator(
                    settings, path, client_factory=Client, **kwargs
                ),
            ),
        ):
            cli.generate(args, value)
        summary = json.loads((target / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["preparation_cost"]["requests"], 17)
        self.assertEqual(summary["new_preparation_cost"]["requests"], 0)
        self.assertEqual(summary["reused_preparation_cost"]["requests"], 17)
        self.assertEqual(summary["reuse_source"], str(original))
        self.assertEqual(health_checks.call_count, 2)
        state = json.loads((target / "metadata/run_status.json").read_text())
        self.assertEqual(
            state["additional_services"],
            [
                {
                    "base_url": args.additional_base_url,
                    "health": {},
                }
            ],
        )
        self.assertEqual(Client.calls, [])
        args.resume = True
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            cli.generate(args, value)

    def test_cli_rejects_additional_endpoint_with_a_different_model(self):
        module_spec = importlib.util.spec_from_file_location(
            "pa_additional_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        value = catalog()
        target = self.root / "target"
        args = SimpleNamespace(
            qwen_model_path=Path(value["executor"]["model_path"]),
            protocol=PROJECT / "configs/pa_tool_prepare.json",
            execute=True,
            resume=False,
            reuse_from=None,
            concurrency=2,
            output_dir=target,
            additional_base_url="http://127.0.0.1:2/v1",
        )
        original_check = cli.check_generator

        def health(settings):
            if settings["base_url"] == value["executor"]["base_url"]:
                return {"id": "qwen3-4b"}
            return original_check(settings)

        with (
            patch.object(cli, "check_generator", side_effect=health),
            patch.object(cli, "bypass_local_proxy"),
            patch.object(
                cli,
                "json_request",
                return_value={
                    "data": [{"id": "another-model"}],
                },
            ) as request,
            patch.object(
                cli, "source_identity", return_value={"trees": {}, "dirty_paths": []}
            ),
            patch.object(
                cli,
                "CandidateGenerator",
                side_effect=AssertionError("must not generate"),
            ),
            self.assertRaisesRegex(ValueError, "model identity differs"),
        ):
            cli.generate(args, value)
        request.assert_called_once_with(
            args.additional_base_url + "/models", value["executor"]["timeout_seconds"]
        )
        self.assertFalse(target.exists())

    def test_cli_resume_rejects_a_changed_additional_endpoint(self):
        module_spec = importlib.util.spec_from_file_location(
            "pa_additional_resume_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        value = catalog()
        target = self.root / "target"
        write_json(target / "catalog.json", value)
        state = {
            "status": "interrupted",
            "protocol": PROTOCOL,
            "service": {},
            "code": {"trees": {}, "dirty_paths": []},
            "additional_services": [{"base_url": "http://old", "health": {}}],
        }
        write_json(target / "metadata/run_status.json", state)
        args = SimpleNamespace(
            qwen_model_path=Path(value["executor"]["model_path"]),
            protocol=PROJECT / "configs/pa_tool_prepare.json",
            execute=True,
            resume=True,
            reuse_from=None,
            concurrency=2,
            output_dir=target,
            additional_base_url="http://new",
        )
        with (
            patch.object(cli, "check_generator", return_value={}),
            patch.object(
                cli, "source_identity", return_value={"trees": {}, "dirty_paths": []}
            ),
            patch.object(
                cli, "CandidateGenerator", side_effect=AssertionError("must not resume")
            ),
            self.assertRaisesRegex(ValueError, "same committed"),
        ):
            cli.generate(args, value)
        self.assertEqual(
            json.loads((target / "metadata/run_status.json").read_text()), state
        )
