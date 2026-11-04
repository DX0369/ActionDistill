"""Run in llm-efficiency: the installed RapidFuzz implements edit distance."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_bench.io import append_json, write_json
from jev_bench.pa_tool_generation import CandidateGenerator, generation_cost
from jev_bench.pa_tool_prepare import (
    PROTOCOL,
    build_mapping,
    prompt,
    rank_names,
    resolve_names,
)
from tests.pa_fixtures import result, tool

PROJECT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def test_fresh_prepare_previews_each_catalog_environment_without_model_calls(self):
        spec = importlib.util.spec_from_file_location(
            "pa_fresh_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        value = {
            "version": 1,
            "benchmark": {"name": "bfcl"},
            "control_names": [],
            "scopes": {"s": [tool()]},
            "bindings": {"x": "s"},
            "executor": {
                "model": "qwen3-4b",
                "model_path": "/data/wangsq/model/Qwen3-4B",
            },
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            target = Path(path) / "fresh"
            conda_envs = Path(path) / "relocated envs"
            model_path = Path(path) / "models/Qwen3-4B"
            value["executor"]["model_path"] = str(model_path)
            for name, environment in (
                ("bfcl_mt", "trust"),
                ("tau2", "llm-efficiency"),
                ("toolhop", "jev-toolhop"),
                ("apibank", "trust"),
            ):
                config = PROJECT / f"configs/{name}_main_smoke.json"
                catalogs = []
                python = conda_envs / environment / "bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
                python.chmod(0o700)

                def extract(
                    command,
                    *,
                    check,
                    env,
                    environment=environment,
                    config=config,
                    catalogs=catalogs,
                ):
                    self.assertEqual(
                        command[:2],
                        [str(conda_envs / environment / "bin/python"), "-B"],
                    )
                    self.assertTrue(check)
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                    self.assertEqual(command[3:6], ["catalog", "--config", str(config)])
                    catalog = Path(command[command.index("--output") + 1])
                    self.assertTrue(catalog.is_relative_to(PROJECT / "tmp"))
                    catalog.write_text(json.dumps(value))
                    catalogs.append(catalog)

                with (
                    self.subTest(benchmark=name),
                    patch(
                        "sys.argv",
                        [
                            "prepare_pa_tool.py",
                            "prepare",
                            "--config",
                            str(config),
                            "--catalog-python",
                            str(python),
                            "--qwen-model-path",
                            str(model_path),
                            "--output-dir",
                            str(target),
                        ],
                    ),
                    patch.object(cli, "subprocess") as process,
                    patch.object(cli, "check_generator") as service,
                    patch.object(cli, "CandidateGenerator") as generator,
                    patch.object(cli, "ReuseSource") as reuse,
                    redirect_stdout(io.StringIO()),
                ):
                    process.run.side_effect = extract
                    cli.main()
                    process.run.assert_called_once()
                    service.assert_not_called()
                    generator.assert_not_called()
                    reuse.assert_not_called()
                self.assertFalse(target.exists())
                self.assertTrue(catalogs)
                self.assertTrue(all(not item.exists() for item in catalogs))

    def test_fresh_prepare_requires_output_to_execute_and_rejects_historical_reuse(
        self,
    ):
        spec = importlib.util.spec_from_file_location(
            "pa_fresh_boundary_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        config = PROJECT / "configs/bfcl_mt_main_smoke.json"
        base = [
            "prepare_pa_tool.py",
            "prepare",
            "--config",
            str(config),
            "--catalog-python",
            "/deployment/envs/runner/bin/python",
            "--qwen-model-path",
            "/deployment/Qwen3-4B",
        ]
        with patch.object(cli, "prepare_from_config") as prepare:
            for option in ("--catalog-python", "--qwen-model-path"):
                missing = base.copy()
                index = missing.index(option)
                del missing[index : index + 2]
                with (
                    self.subTest(missing=option),
                    patch("sys.argv", missing),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    cli.main()
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()
            for flags in (["--execute"], ["--reuse-from", "old"], ["--resume"]):
                with (
                    self.subTest(flags=flags),
                    patch("sys.argv", base + flags),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    cli.main()
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()
            with patch("sys.argv", base + ["--execute", "--output-dir", "fresh"]):
                cli.main()
            args = prepare.call_args.args[0]
            self.assertTrue(args.execute)
            self.assertFalse(args.resume)
            self.assertIsNone(args.reuse_from)
            self.assertEqual(args.protocol, PROJECT / "configs/pa_tool_prepare.json")

    def test_prepare_rejects_unusable_catalog_interpreter_before_extraction(self):
        spec = importlib.util.spec_from_file_location(
            "pa_interpreter_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            root = Path(directory)
            non_executable = root / "non_executable/trust/bin/python"
            non_executable.parent.mkdir(parents=True)
            non_executable.touch(mode=0o600)
            with (
                patch.object(cli, "subprocess") as process,
                patch.object(cli, "generate") as generate,
            ):
                for prefix in (
                    Path(""),
                    Path("envs"),
                    root / "missing",
                    root / "non_executable",
                ):
                    with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                        cli.prepare_from_config(
                            SimpleNamespace(
                                config=PROJECT / "configs/bfcl_mt_main_smoke.json",
                                catalog_python=prefix/'trust/bin/python',
                            )
                        )
                    process.run.assert_not_called()
                    generate.assert_not_called()

    def test_relocated_naming_model_is_explicit_and_identity_remains_strict(self):
        spec = importlib.util.spec_from_file_location(
            "pa_model_path_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            root = Path(directory)
            model_path = root / "relocated model"
            model_path.mkdir()
            settings = {
                "model": "qwen3-4b",
                "model_path": str(model_path),
                "base_url": "http://127.0.0.1:18000/v1",
                "max_model_len": 131072,
                "timeout_seconds": 10,
            }
            catalog = {
                "version": 1,
                "benchmark": {"name": "bfcl"},
                "control_names": [],
                "scopes": {"s": [tool()]},
                "bindings": {"x": "s"},
                "executor": settings,
            }
            catalog_path = root / "catalog.json"
            write_json(catalog_path, catalog)
            base = ["prepare_pa_tool.py", "generate", "--catalog", str(catalog_path)]
            with (
                patch.object(cli, "check_generator") as health,
                patch.object(cli, "CandidateGenerator") as generator,
                redirect_stdout(io.StringIO()),
            ):
                with (
                    patch("sys.argv", base),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    cli.main()
                for declared, model in (
                    (str(model_path), "qwen3-4b"),
                    (str(root / "other-model"), "qwen3-4b"),
                    ("relative-model", "qwen3-4b"),
                    (str(model_path), "actionopd-4b"),
                ):
                    catalog["executor"]["model"] = model
                    write_json(catalog_path, catalog)
                    with (
                        self.subTest(declared=declared, model=model),
                        patch("sys.argv", base + ["--qwen-model-path", declared]),
                    ):
                        if declared == str(model_path) and model == "qwen3-4b":
                            cli.main()
                        else:
                            with self.assertRaises(ValueError):
                                cli.main()
                    health.assert_not_called()
                    generator.assert_not_called()
            settings["model"] = "qwen3-4b"
            expected = {
                "id": "qwen3-4b",
                "root": str(model_path),
                "max_model_len": 131072,
            }
            with patch.object(
                cli, "json_request", return_value={"data": [expected]}
            ) as request:
                self.assertEqual(cli.check_generator(settings), expected)
                request.assert_called_once_with(settings["base_url"] + "/models", 10)
                for difference in (
                    {"id": "actionopd-4b"},
                    {"root": str(root / "other-model")},
                    {"max_model_len": 8192},
                ):
                    request.return_value = {"data": [{**expected, **difference}]}
                    with (
                        self.subTest(difference=difference),
                        self.assertRaises(ValueError),
                    ):
                        cli.check_generator(settings)

    def test_stable_empty_description_is_explicit_counted_context_only_input(self):
        from jev_bench.pa_tool_prepare import estimate

        function = tool()
        function["function"]["description"] = "A tool context."
        function["function"]["parameters"]["properties"]["key"]["description"] = ""
        catalog = {
            "version": 1,
            "benchmark": {"name": "stabletoolbench"},
            "control_names": ["Finish"],
            "scopes": {"s": [function]},
            "bindings": {"x": "s"},
        }
        prompts = []

        def generate(key, text):
            prompts.append(text)
            name = f"candidate_{len(prompts)}"
            return [name] * 16, name

        build_mapping(catalog, generate)
        self.assertIn("Description:\n\nExample:", prompts[1])
        self.assertIn("Tool: candidate_1 - A tool context.", prompts[1])
        self.assertNotIn("key", prompts[1])
        self.assertEqual(estimate(catalog)["empty_parameter_descriptions"], 1)
        del function["function"]["parameters"]["properties"]["key"]["description"]
        with self.assertRaises(ValueError):
            estimate(catalog)

    def test_density_weights_duplicates_excludes_self_and_uses_character_distance(self):
        samples = ["apple"] * 7 + ["apply"] * 5 + ["zebra"] * 4
        rows = rank_names(samples, "apply")
        self.assertEqual(
            [(r["name"], r["count"], r["score"]) for r in rows],
            [
                ("apply", 5, Fraction(11, 15)),
                ("apple", 7, Fraction(11, 15)),
                ("zebra", 4, Fraction(3, 15)),
            ],
        )
        self.assertEqual(resolve_names({0: rows}), {0: "apply"})

    def test_equal_density_and_distance_use_first_sample_slot(self):
        rows = rank_names(["zzzzz"] * 8 + ["aaaaa"] * 8, "mmmmm")
        self.assertEqual([row["name"] for row in rows], ["zzzzz", "aaaaa"])
        self.assertEqual([row["first_sample_slot"] for row in rows], [0, 8])
        self.assertEqual(resolve_names({0: rows}), {0: "zzzzz"})

    def test_density_normalizes_valid_count_for_collision_and_preserves_holes(self):
        full = rank_names(["shared"] * 10 + ["xxxxxxxx"] * 6, "shared")
        partial = rank_names([None] * 4 + ["shared"] * 8 + ["yyyyyyyy"] * 4, "shared")
        self.assertEqual(full[0]["score"], Fraction(9, 15))
        self.assertEqual(partial[0]["score"], Fraction(7, 11))
        self.assertEqual(partial[0]["first_sample_slot"], 4)
        self.assertEqual(
            resolve_names({0: full, 1: partial}), {1: "shared", 0: "xxxxxxxx"}
        )

    def test_ranking_rejects_insufficient_candidates_invalid_names_or_reference(self):
        for samples, reference in (
            ([None] * 5 + ["valid"] * 11, "valid"),
            (["valid"] * 15, "valid"),
            (["valid"] * 15 + ["two words"], "valid"),
            (["valid"] * 16, None),
        ):
            with (
                self.subTest(samples=samples, reference=reference),
                self.assertRaises(ValueError),
            ):
                rank_names(samples, reference)

    def test_collisions_preserve_component_order_and_exhaustion_still_stops(self):
        row = lambda name, score, distance=0: {
            "name": name,
            "score": score,
            "reference_distance": distance,
        }
        ranks = {
            0: [row("same", 30), row("other", 2)],
            1: [row("same", 20), row("next", 3)],
        }
        saved = deepcopy(ranks)
        self.assertEqual(resolve_names(ranks), {0: "same", 1: "next"})
        self.assertEqual(ranks, saved)
        for first, second in ((9, 2), ("z_property", "a_property")):
            self.assertEqual(
                resolve_names(
                    {
                        first: [row("same", 10, 5), row("other", 2)],
                        second: [row("same", 10, 0), row("next", 3)],
                    }
                ),
                {first: "same", second: "next"},
            )
        for ranks in (
            {0: [row("x", 1)], 1: [row("x", 1)]},
            {0: [row("Finish", 1)]},
        ):
            with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                resolve_names(ranks, reserved=["Finish"])

    def test_parameter_collision_uses_original_property_order(self):
        function = tool()
        properties = function["function"]["parameters"]["properties"]
        function["function"]["parameters"]["properties"] = {
            "payload": properties["payload"],
            "key": properties["key"],
        }
        catalog = {
            "version": 1,
            "benchmark": {"name": "bfcl"},
            "control_names": [],
            "scopes": {"s": [function]},
            "bindings": {"x": "s"},
        }

        def generate(key, text):
            if text.startswith("Generate a tool"):
                return ["selected_tool"] * 16, "selected_tool"
            return ["shared"] * 15 + ["unused"], "shared"

        mapping = build_mapping(catalog, generate)
        self.assertEqual(
            mapping["scopes"]["s"][0]["parameters"],
            {"payload": "shared", "key": "unused"},
        )

    def test_parameter_context_is_selected_alias_and_control_is_unchanged(self):
        finish = {
            "type": "function",
            "function": {
                "name": "Finish",
                "parameters": {"properties": {"return_type": {"type": "string"}}},
            },
        }
        catalog = {
            "version": 1,
            "benchmark": {"name": "stabletoolbench"},
            "control_names": ["Finish"],
            "scopes": {"s": [tool(), finish]},
            "bindings": {"x": "s"},
        }
        prompts = []

        def generate(key, text):
            prompts.append(text)
            name = "find_record" if len(prompts) == 1 else f"argument_{len(prompts)}"
            return [name] * 16, name

        mapping = build_mapping(catalog, generate)
        self.assertIn("Tool: find_record - Look up", prompts[1])
        self.assertNotIn("Tool: lookup", prompts[1])
        self.assertEqual(
            mapping["scopes"]["s"][1],
            {"name": "Finish", "parameters": {"return_type": "return_type"}},
        )
        self.assertEqual(len(prompts), 3)
        self.assertTrue(
            prompt("tool", "A description").endswith(
                "Generate only the name without additional explanation."
            )
        )

    def test_generated_finish_suffix_cannot_capture_control_call(self):
        from jev_bench.pa_tool import Projection
        from tests.pa_fixtures import result

        business = tool()
        business["function"]["parameters"] = {"properties": {}}
        finish = {
            "type": "function",
            "function": {
                "name": "Finish",
                "parameters": {
                    "properties": {
                        "return_type": {"type": "string"},
                        "final_answer": {"type": "string"},
                    }
                },
            },
        }
        tools = [business, finish]
        catalog = {
            "version": 1,
            "benchmark": {"name": "stabletoolbench"},
            "control_names": ["Finish"],
            "scopes": {"s": tools},
            "bindings": {"fixture": "s"},
        }
        mapping = build_mapping(
            catalog, lambda key, text: (["lookupFinish"] * 16, "lookupFinish")
        )
        p = Projection(tools, mapping["scopes"]["s"])
        p.request([], tools)
        for arguments in (
            ' { "return_type": "give_answer", "final_answer": "done" } ',
            '{"return_type": "give_answer", "final_answer": "unterminated',
        ):
            raw = result([("Finish", arguments)])
            restored = p.response(raw, suffix=True).message.tool_calls[0].function
            self.assertEqual(restored.name, "Finish")
            self.assertEqual(restored.arguments, arguments)
            self.assertEqual(raw.message.tool_calls[0].function.name, "Finish")
        self.assertEqual(p.restore_call("upFinish", {}, suffix=True)[0], "lookup")

    def test_17_distinct_seeds_resume_and_cost_are_not_duplicated(self):
        requests = []

        class Client:
            def __init__(self, settings, sink):
                self.sink = sink

            def close(self):
                pass

            def complete(self, **request):
                requests.append(request)
                value = result(content="  candidate_name\n")
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

        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            generator = CandidateGenerator(
                {"base_url": "http://primary"}, path, client_factory=Client
            )
            samples, reference = generator("component", "text")
            generator.close()
            self.assertEqual(samples, ["candidate_name"] * 16)
            self.assertEqual(reference, "candidate_name")
            self.assertEqual(len({r["extra_body"]["seed"] for r in requests}), 17)
            self.assertEqual([r["temperature"] for r in requests], [0.4] * 16 + [0])
            self.assertTrue(
                all(
                    r["extra_body"]["chat_template_kwargs"]["enable_thinking"]
                    for r in requests
                )
            )
            cost = generation_cost(path)
            resumed = CandidateGenerator(
                {"base_url": "http://primary"}, path, client_factory=Client
            )
            resumed("component", "text")
            resumed.close()
            self.assertEqual(len(requests), 17)
            self.assertEqual(generation_cost(path), cost)
            self.assertEqual(cost["total_tokens"]["observed_total"], 17 * 15)

    def test_incomplete_or_invalid_requests_never_silently_resample(self):
        for reason, content in (
            ("length", "good_name"),
            ("stop", "use this name"),
            ("stop", None),
        ):

            class Client:
                def __init__(self, settings, sink):
                    self.sink = sink

                def close(self):
                    pass

                def complete(self, content=content, reason=reason, **request):
                    value = result(content=content)
                    value.finish_reason = reason
                    value.raw_response["model"] = "qwen3-4b"
                    value.raw_response["choices"][0]["finish_reason"] = reason
                    self.sink(
                        "executor",
                        {
                            "usage": value.usage,
                            "latency_seconds": 1,
                            "response": value.raw_response,
                            "error": None,
                        },
                    )
                    return value

            with (
                self.subTest(reason=reason, content=content),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory,
            ):
                path = Path(directory) / "generation.jsonl"
                generator = CandidateGenerator(
                    {"base_url": "http://primary", "model": "qwen3-4b"},
                    path,
                    client_factory=Client,
                )
                self.assertEqual(generator("key", "text"), ([None] * 16, None))
                generator.close()
                saved_cost = generation_cost(path)
                resumed = CandidateGenerator(
                    {"base_url": "http://primary", "model": "qwen3-4b"},
                    path,
                    client_factory=lambda *_: self.fail(
                        "failed component must not resample"
                    ),
                )
                self.assertEqual(resumed("key", "text"), ([None] * 16, None))
                self.assertEqual(generation_cost(path), saved_cost)
                self.assertEqual(generation_cost(path)["requests"], 17)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            path = Path(directory) / "generation.jsonl"
            append_json(
                path,
                {
                    "phase": "started",
                    "component": "x",
                    "slot": 0,
                    "provider": {"usage": None, "latency_seconds": None},
                },
            )
            with self.assertRaises(ValueError):
                CandidateGenerator(
                    {"base_url": "http://primary"},
                    path,
                    client_factory=lambda *_: self.fail("must not construct a client"),
                )
            self.assertEqual(generation_cost(path)["total_tokens"]["missing"], 1)

    def test_protocol_file_matches_frozen_settings(self):
        import json

        from jev_bench.pa_tool_prepare import check_protocol

        check_protocol(
            json.loads((PROJECT / "configs/pa_tool_prepare.json").read_text())
        )
        with self.assertRaises(ValueError):
            check_protocol({**PROTOCOL, "candidates": 15})
        with self.assertRaises(ValueError):
            check_protocol(
                {k: v for k, v in PROTOCOL.items() if k != "generation_failure"}
            )

    def test_array_description_exception_is_narrow_and_recorded(self):
        from jev_bench.pa_tool_prepare import parameter_description

        array = {
            "type": "array",
            "items": {"type": "string", "description": "Original array wording"},
        }
        self.assertEqual(
            parameter_description(array, allow_toolhop=True),
            ("Original array wording", "items.description"),
        )
        for schema, allowed in (
            (array, False),
            ({**array, "description": ""}, True),
            ({**array, "type": "object"}, True),
        ):
            with (
                self.subTest(schema=schema, allowed=allowed),
                self.assertRaises(ValueError),
            ):
                parameter_description(schema, allow_toolhop=allowed)
        function = tool()
        function["function"]["parameters"]["properties"] = {"items": array}
        function["function"]["parameters"]["required"] = ["items"]
        catalog = {
            "version": 1,
            "benchmark": {"name": "toolhop"},
            "control_names": [],
            "scopes": {"s": [function]},
            "bindings": {"x": "s"},
        }
        mapping = build_mapping(catalog, lambda key, text: (["chosen"] * 16, "chosen"))
        self.assertEqual(
            mapping["parameter_description_sources"],
            {"s": [{"items": "items.description"}]},
        )
        self.assertNotIn(
            "description", function["function"]["parameters"]["properties"]["items"]
        )

    def test_toolhop_descendants_preserve_text_order_schema_and_source_paths(self):
        from jev_bench.pa_tool_prepare import parameter_description

        schema = {
            "type": "object",
            "properties": {
                "z_secret_name": {"type": "number", "description": "  Minimum.  "},
                "a_secret_name": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nested_secret": {
                                "type": "string",
                                "description": "Maximum.",
                            }
                        },
                    },
                },
            },
        }
        original = deepcopy(schema)
        source = {
            "rule": "descendant_descriptions",
            "paths": [
                ["properties", "z_secret_name", "description"],
                [
                    "properties",
                    "a_secret_name",
                    "items",
                    "properties",
                    "nested_secret",
                    "description",
                ],
            ],
        }
        self.assertEqual(
            parameter_description(schema, allow_toolhop=True),
            ("  Minimum.  \nMaximum.", source),
        )
        array = {"type": "array", "items": schema}
        self.assertEqual(
            parameter_description(array, allow_toolhop=True)[1]["paths"],
            [["items", *path] for path in source["paths"]],
        )
        for empty in ("", "   "):
            with self.subTest(items_description=empty):
                array["items"] = {**schema, "description": empty}
                self.assertEqual(
                    parameter_description(array, allow_toolhop=True)[0],
                    "  Minimum.  \nMaximum.",
                )
        array["items"] = {**schema, "description": None}
        with self.assertRaisesRegex(ValueError, "description must be a string"):
            parameter_description(array, allow_toolhop=True)
        function = tool()
        function["function"]["parameters"]["properties"] = {
            "private_original_name": schema
        }
        function["function"]["parameters"]["required"] = ["private_original_name"]
        catalog = {
            "version": 1,
            "benchmark": {"name": "toolhop"},
            "control_names": [],
            "scopes": {"s": [function]},
            "bindings": {"x": "s"},
        }
        prompts = []

        def generate(key, text):
            prompts.append(text)
            return ["chosen"] * 16, "chosen"

        mapping = build_mapping(catalog, generate)
        self.assertEqual(
            mapping["parameter_description_sources"],
            {"s": [{"private_original_name": source}]},
        )
        self.assertIn("Description:\n  Minimum.  \nMaximum.\n", prompts[1])
        for name in (
            "private_original_name",
            "z_secret_name",
            "a_secret_name",
            "nested_secret",
        ):
            self.assertNotIn(name, prompts[1])
        self.assertEqual(schema, original)

    def test_toolhop_descendants_do_not_override_explicit_descriptions_or_other_benchmarks(
        self,
    ):
        from jev_bench.pa_tool_prepare import parameter_description

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string", "description": "Child."}},
        }
        self.assertEqual(
            parameter_description(
                {**schema, "description": "Parent."}, allow_toolhop=True
            ),
            ("Parent.", "description"),
        )
        for invalid, allowed in (
            (schema, False),
            ({**schema, "description": ""}, True),
            ({**schema, "description": None}, True),
            ({"type": "object", "properties": {"x": {"type": "string"}}}, True),
        ):
            with (
                self.subTest(schema=invalid, allowed=allowed),
                self.assertRaises(ValueError),
            ):
                parameter_description(invalid, allow_toolhop=allowed)

    def test_cli_preview_is_offline_and_failure_status_survives_bad_cost_ledger(self):
        module_spec = importlib.util.spec_from_file_location(
            "pa_prepare_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        catalog = {
            "version": 1,
            "benchmark": {"name": "bfcl"},
            "control_names": [],
            "scopes": {"s": [tool()]},
            "bindings": {"x": "s"},
            "executor": {
                "model": "qwen3-4b",
                "model_path": "/data/wangsq/model/Qwen3-4B",
            },
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            target = Path(directory) / "prepare"
            args = SimpleNamespace(
                qwen_model_path=Path(catalog["executor"]["model_path"]),
                protocol=PROJECT / "configs/pa_tool_prepare.json",
                execute=False,
                resume=False,
                reuse_from=None,
                concurrency=1,
                additional_base_url=None,
                output_dir=target,
            )
            with patch.object(
                cli,
                "check_generator",
                side_effect=AssertionError("preview must be offline"),
            ):
                cli.generate(args, catalog)
            self.assertFalse(target.exists())
            args.execute = True
            with (
                patch.object(cli, "check_generator", return_value={}),
                patch.object(
                    cli,
                    "source_identity",
                    return_value={"trees": {}, "dirty_paths": []},
                ),
                patch.object(
                    cli, "CandidateGenerator", side_effect=ValueError("broken ledger")
                ),
                patch.object(
                    cli, "generation_cost", side_effect=ValueError("broken ledger")
                ),
                self.assertRaisesRegex(ValueError, "broken ledger"),
            ):
                cli.generate(args, catalog)
            state = json.loads((target / "metadata/run_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error"]["message"], "broken ledger")
            self.assertEqual(state["finalization_error"]["message"], "broken ledger")
            self.assertFalse((target / "summary.json").exists())

    def test_resume_rejects_candidates_generated_by_uncommitted_code(self):
        module_spec = importlib.util.spec_from_file_location(
            "pa_prepare_resume_cli", PROJECT / "scripts/prepare_pa_tool.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        catalog = {
            "version": 1,
            "benchmark": {"name": "bfcl"},
            "control_names": [],
            "scopes": {"s": [tool()]},
            "bindings": {"x": "s"},
            "executor": {
                "model": "qwen3-4b",
                "model_path": "/data/wangsq/model/Qwen3-4B",
            },
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            target = Path(directory)
            args = SimpleNamespace(
                qwen_model_path=Path(catalog["executor"]["model_path"]),
                protocol=PROJECT / "configs/pa_tool_prepare.json",
                execute=True,
                resume=True,
                reuse_from=None,
                concurrency=1,
                additional_base_url=None,
                output_dir=target,
            )
            write_json(target / "catalog.json", catalog)
            write_json(
                target / "metadata/run_status.json",
                {
                    "status": "interrupted",
                    "protocol": PROTOCOL,
                    "service": {},
                    "code": {"trees": {}, "dirty_paths": ["local implementation"]},
                },
            )
            with (
                patch.object(cli, "check_generator", return_value={}),
                patch.object(
                    cli,
                    "source_identity",
                    return_value={"trees": {}, "dirty_paths": []},
                ),
                patch.object(
                    cli,
                    "CandidateGenerator",
                    side_effect=AssertionError("must not resume"),
                ),
                self.assertRaisesRegex(ValueError, "same committed"),
            ):
                cli.generate(args, catalog)
