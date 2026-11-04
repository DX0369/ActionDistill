import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from jev_bench.benchmarks.stabletoolbench import (
    Archive,
    CacheMiss,
    load_tasks,
    validate_api,
)
from jev_bench.benchmarks.stabletoolbench_scoring import (
    fixture_evaluator,
    score_fixture,
)
from jev_bench.config import PROJECT_ROOT, STABLETOOLBENCH_CATEGORIES, load_config


def response(arguments):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments=json.dumps(arguments))
                        )
                    ]
                )
            )
        ]
    )


def example(
    *,
    finish=True,
    final_answer='{"return_type":"give_answer","final_answer":"fixture"}',
):
    return {
        "query": "List breweries.",
        "available_tools": [{"name": "breweries"}, {"name": "Finish"}],
        "answer": {
            "method": "cpu_fixture",
            "total_steps": 1,
            "final_answer": final_answer,
            "answer_details": [
                {
                    "role": "tool",
                    "message": {
                        "name": "Finish" if finish else "breweries",
                        "arguments": "{}",
                        "response": "[]",
                    },
                    "next": [],
                }
            ],
        },
    }


class StableToolBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            PROJECT_ROOT / "configs/stabletoolbench_cpu_preview.json"
        )
        cls.config["task_ids"] = {
            category: "all" for category in STABLETOOLBENCH_CATEGORIES
        }
        cls.tasks, cls.answers = load_tasks(cls.config)

    def setUp(self):
        for target in ("socket.socket.connect", "socket.getaddrinfo"):
            blocker = patch(
                target, side_effect=AssertionError("CPU fixture attempted network")
            )
            blocker.start()
            self.addCleanup(blocker.stop)

    def test_published_id_coverage_and_no_relevant_api_leakage(self):
        self.assertEqual(
            [
                sum(key[0] == cat for key in self.tasks)
                for cat in STABLETOOLBENCH_CATEGORIES
            ],
            [163, 153, 158, 106, 124, 61],
        )
        self.assertEqual(len({key[1] for key in self.tasks}), 765)
        self.assertEqual(
            sum(len(task["api_list"]) for task in self.tasks.values()), 4070
        )
        for key, task in self.tasks.items():
            self.assertEqual(set(task), {"query_id", "query", "api_list"})
            self.assertEqual(set(self.answers[key]), {"relevant APIs"})
        # Published colliding candidates remain present, not silently deduplicated.
        task = self.tasks[("G1_instruction", "27847")]
        self.assertGreater(
            len(task["api_list"]),
            len(self.answers[("G1_instruction", "27847")]["relevant APIs"]),
        )

    def test_archive_paths_cover_candidates_without_extracting_or_writing(self):
        with Archive(self.config) as archive:
            paths = set()
            for task in self.tasks.values():
                for api in task["api_list"]:
                    category = archive.names.standardize_category(api["category_name"])
                    tool = archive.names.standardize(api["tool_name"])
                    path = f"tools/{category}/{tool}.json"
                    self.assertGreater(archive.zip.getinfo(path).file_size, 0)
                    paths.add(path)
            self.assertEqual(len(paths), 814)
            self.assertEqual(archive.zip.mode, "r")

    def test_real_catalog_and_exact_cached_response(self):
        with Archive(self.config) as archive:
            task = self.tasks[("G1_instruction", "25658")]
            selected = [
                api for api in task["api_list"] if api["api_name"] == "Breweries"
            ]
            self.assertEqual(len(selected), 1)
            self.assertTrue(archive.api_definitions(selected[0]))
            kwargs = {
                "category": "Data",
                "tool_name": "open_brewery_db",
                "api_name": "breweries",
            }
            for arguments in ({}, "{}", ""):
                self.assertEqual(
                    archive.lookup(**kwargs, tool_input=arguments),
                    {"error": "", "response": []},
                )
            with self.assertRaises(CacheMiss):
                archive.lookup(**kwargs, tool_input={"missing_fixture_key": 1})
            with self.assertRaises(CacheMiss):
                archive.lookup(**{**kwargs, "api_name": "missing_api"}, tool_input={})
            self.assertEqual(
                archive.lookup(**kwargs, tool_input="not JSON"),
                {"error": "Tool input parse error...\n", "response": ""},
            )
            self.assertEqual(archive.cache_events[-1]["type"], "tool_input_parse_error")

    def test_cache_key_preserves_input_order(self):
        # Synthetic ZIP reader isolates the official Python-repr key convention.
        with Archive(self.config) as archive:
            stored = {str({"a": 1, "b": 2}): {"response": "ordered"}}
            with (
                patch.object(
                    archive.zip,
                    "NameToInfo",
                    {"tool_response_cache/Data/fixture_for_Data/fixture.json": None},
                ),
                patch.object(
                    archive.zip, "read", return_value=json.dumps(stored).encode()
                ),
            ):
                kwargs = {
                    "category": "Data",
                    "tool_name": "fixture",
                    "api_name": "fixture",
                }
                self.assertEqual(
                    archive.lookup(**kwargs, tool_input={"a": 1, "b": 2}),
                    {"response": "ordered"},
                )
                with self.assertRaises(CacheMiss):
                    archive.lookup(**kwargs, tool_input={"b": 2, "a": 1})

    def test_official_wrapper_whitelist_order_prompt_dispatch_and_parse_observation(
        self,
    ):
        definition = {
            "name": "Echo",
            "description": "Echo",
            "required_parameters": [],
            "optional_parameters": [],
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path:
            zipped = Path(path) / "catalog.zip"
            with ZipFile(zipped, "w") as target:
                target.writestr(
                    "tools/A/shared.json",
                    json.dumps(
                        {
                            "tool_name": "Shared",
                            "tool_description": "First description",
                            "api_list": [definition],
                        }
                    ),
                )
                target.writestr(
                    "tools/B/other.json",
                    json.dumps(
                        {
                            "tool_name": "Shared",
                            "tool_description": None,
                            "api_list": [definition],
                        }
                    ),
                )
                target.writestr(
                    "tools/A/spare.json",
                    json.dumps(
                        {
                            "tool_name": "Spare",
                            "tool_description": "",
                            "api_list": [definition],
                        }
                    ),
                )
            value = deepcopy(self.config)
            value["protocol"]["cache_archive"] = str(zipped)
            task = {
                "query": "Echo",
                "api_list": [
                    {"category_name": "A", "tool_name": "Shared", "api_name": "Echo"},
                    {"category_name": "A", "tool_name": "Shared", "api_name": "Absent"},
                ],
            }
            ordered = ["tools/A/shared.json", "tools/B/other.json"]
            payloads = []

            def transport(payload):
                payloads.append(payload)
                return {"error": "", "response": "x" * 200}

            with (
                Archive(value) as archive,
                patch.object(archive.zip, "read", wraps=archive.zip.read) as reads,
            ):
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    archive.environment(
                        task,
                        catalog_order=[*ordered, "tools/A/spare.json"],
                        transport=transport,
                        max_observation_length=80,
                        observ_compress_method="truncate",
                        sleep=lambda seconds: None,
                    )
                with self.assertRaises(CacheMiss):
                    archive.lookup(
                        category="A",
                        tool_name="shared",
                        api_name="echo",
                        tool_input="{}",
                    )
                env = archive.environment(
                    task,
                    catalog_order=ordered,
                    transport=transport,
                    max_observation_length=80,
                    observ_compress_method="truncate",
                    sleep=lambda seconds: None,
                )
                self.assertIn("1.other: None", env.task_description)
                self.assertEqual(env.functions[0]["function"]["name"], "echo_for_other")
                self.assertEqual(env.functions[-1]["function"]["name"], "Finish")
                self.assertEqual(
                    env.candidate_diagnostics,
                    [
                        {
                            "index": 1,
                            "category_name": "A",
                            "tool_name": "Shared",
                            "api_name": "Absent",
                            "reason": "catalog_api_not_found",
                        }
                    ],
                )
                self.assertEqual(
                    sum(
                        call.args[0] == "tools/B/other.json"
                        for call in reads.call_args_list
                    ),
                    1,
                )
                observation, status = env.step(action_name="other", action_input="{}")
                self.assertEqual(status, 0)
                self.assertEqual(len(observation), 83)
                self.assertEqual(payloads[-1]["api_name"], "echo")
                self.assertEqual(payloads[-1]["tool_name"], "other")
                self.assertEqual(
                    env.step(action_name="absent", action_input="{}")[1], 1
                )
                self.assertEqual(
                    env.step(
                        action_name="Finish",
                        action_input='{"return_type":"give_answer","final_answer":"done"}',
                    )[1],
                    3,
                )
                self.assertEqual(env.check_success(), 1)
                reverse = archive.environment(
                    task,
                    catalog_order=ordered[::-1],
                    transport=transport,
                    max_observation_length=1000,
                    observ_compress_method="truncate",
                    sleep=lambda seconds: None,
                )
                self.assertIn("1.shared: First description", reverse.task_description)
                self.assertEqual(
                    reverse.functions[0]["function"]["name"], "echo_for_shared"
                )
                self.assertEqual(
                    sum(
                        call.args[0] == "tools/B/other.json"
                        for call in reads.call_args_list
                    ),
                    2,
                )

                def cache(payload):
                    return archive.lookup(
                        **{
                            name: payload[name]
                            for name in (
                                "category",
                                "tool_name",
                                "api_name",
                                "tool_input",
                            )
                        }
                    )

                parsed = archive.environment(
                    task,
                    catalog_order=ordered,
                    transport=cache,
                    max_observation_length=1000,
                    observ_compress_method="truncate",
                    sleep=lambda seconds: None,
                )
                self.assertIn("1.other: None", parsed.task_description)
                self.assertEqual(
                    parsed.functions[0]["function"]["name"], "echo_for_other"
                )
                self.assertEqual(
                    sum(
                        call.args[0] == "tools/B/other.json"
                        for call in reads.call_args_list
                    ),
                    2,
                )
                with redirect_stdout(io.StringIO()):
                    observation, status = parsed.step(
                        action_name="echo_for_other", action_input="not JSON"
                    )
                self.assertEqual(status, 0)
                self.assertEqual(
                    json.loads(observation),
                    {"error": "Tool input parse error...\n", "response": ""},
                )

    def test_official_wrapper_keeps_alias_collision_and_omission_diagnostics(self):
        with Archive(self.config) as archive, redirect_stdout(io.StringIO()):
            for identity in (("G1_instruction", "27847"), ("G1_category", "21453")):
                task = self.tasks[identity]
                members = list(
                    dict.fromkeys(
                        f"tools/{api['category_name']}/{archive.names.standardize(api['tool_name'])}.json"
                        for api in task["api_list"]
                    )
                )
                # This fixture explicitly fixes the two-level directory order.
                members.sort()
                requests = []

                def transport(payload, requests=requests):
                    requests.append(payload)
                    return {"error": "", "response": "fixture"}

                env = archive.environment(
                    task,
                    catalog_order=members,
                    transport=transport,
                    max_observation_length=1000,
                    observ_compress_method="truncate",
                    sleep=lambda seconds: None,
                )
                if identity[1] == "27847":
                    alias = env.functions[0]["function"]["name"]
                    self.assertEqual(alias, env.functions[1]["function"]["name"])
                    env.step(action_name=alias, action_input="{}")
                    self.assertEqual(requests[-1]["category"], env.cate_names[0])
                    self.assertEqual(requests[-1]["tool_name"], env.tool_names[0])
                    self.assertEqual(
                        requests[-1]["api_name"], env.api_name_reflect[alias]
                    )
                else:
                    self.assertIn(
                        3, [row["index"] for row in env.candidate_diagnostics]
                    )
                    self.assertTrue(
                        all(
                            row["reason"] == "catalog_api_not_found"
                            for row in env.candidate_diagnostics
                        )
                    )

    def test_catalog_first_match_omission_and_converter_boundary(self):
        api = next(
            api
            for api in self.tasks[("G1_instruction", "25658")]["api_list"]
            if api["api_name"] == "Breweries"
        )
        definition = {
            "name": "Breweries",
            "description": "",
            "required_parameters": [],
            "optional_parameters": [],
        }
        with Archive(self.config) as archive:
            document = {
                "tool_name": "fixture",
                "api_list": [
                    {"name": "Unrelated"},
                    definition,
                    {"name": "Breweries"},
                ],
            }
            with patch.object(
                archive.zip, "read", return_value=json.dumps(document).encode()
            ):
                matched = archive.api_definitions(api)
                self.assertEqual(len(matched), 1)
                self.assertEqual(matched[0]["api_description"], "")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        archive.api_definitions({**api, "api_name": "missing"}), []
                    )
            bad = {**matched[0], "api_description": None}
            with self.assertRaises(AttributeError):
                archive.openai_function(bad, "fixture")
            bad = {
                **matched[0],
                "required_parameters": [
                    {"name": "x", "type": "STRING", "description": ""}
                ],
            }
            with self.assertRaises(KeyError):
                archive.openai_function(bad, "fixture")

    def test_real_missing_null_description_alias_and_official_schema_quirks(self):
        with Archive(self.config) as archive, redirect_stdout(io.StringIO()):
            task = self.tasks[("G1_category", "21453")]
            self.assertEqual(archive.api_definitions(task["api_list"][3]), [])
            task = self.tasks[("G1_instruction", "40436")]
            api = task["api_list"][0]
            self.assertIsNone(
                archive.tool_document(api["category_name"], api["tool_name"])[
                    "tool_description"
                ]
            )
            aliases = []
            for api in self.tasks[("G1_instruction", "27847")]["api_list"][:2]:
                (definition,) = archive.api_definitions(api)
                converted, category, pure = archive.openai_function(
                    definition, archive.names.standardize(api["tool_name"])
                )
                aliases.append(converted["function"]["name"])
                self.assertEqual(category, api["category_name"])
                self.assertTrue(pure)
            self.assertEqual(aliases[0], aliases[1])
            self.assertEqual(len(aliases[0]), 64)
            definition = {
                **definition,
                "required_parameters": [],
                "optional_parameters": [
                    {
                        "name": "only_optional",
                        "type": "STRING",
                        "description": "x",
                        "default": "",
                    }
                ],
            }
            converted, _, _ = archive.openai_function(definition, "fixture")
            self.assertEqual(
                converted["function"]["parameters"],
                {
                    "type": "object",
                    "properties": {},
                    "required": [""],
                    "optional": [""],
                },
            )
            for value in (False, 0, [], ""):
                definition["required_parameters"] = [
                    {"name": "x", "type": "STRING", "description": "", "default": value}
                ]
                converted, _, _ = archive.openai_function(definition, "fixture")
                properties = converted["function"]["parameters"]["properties"]
                self.assertIn("only_optional", properties)
                if value == "":
                    self.assertNotIn("example_value", properties["x"])
                else:
                    self.assertEqual(properties["x"]["example_value"], value)

    def test_official_sopr_prompt_status_and_finish_gate(self):
        for status in ("Solved", "Unsolved", "Unsure"):
            requests = []

            def judge(status=status, requests=requests, **kwargs):
                requests.append(kwargs)
                return response({"answer_status": status, "reason": "scripted fixture"})

            original = example()
            before = deepcopy(original)
            state = random.getstate()
            with fixture_evaluator(self.config, judge) as evaluator:
                result = score_fixture(self.config, original, evaluator)
            self.assertEqual(result["answer_status"], status)
            self.assertEqual(original, before)
            self.assertEqual(random.getstate(), state)
            self.assertEqual(len(requests), 2 if status == "Unsure" else 1)
            self.assertEqual(
                requests[0]["tool_choice"]["function"]["name"], "check_answer_status"
            )
            if status == "Unsure":
                self.assertEqual(
                    requests[1]["tool_choice"]["function"]["name"],
                    "parse_answer_status",
                )

    def test_no_finish_empty_and_give_up_do_not_call_judge(self):
        def forbidden(**kwargs):
            raise AssertionError("judge must not be called")

        with fixture_evaluator(self.config, forbidden) as evaluator:
            for row in (
                example(finish=False),
                example(final_answer=""),
                example(final_answer="give_up_and_restart"),
            ):
                self.assertEqual(
                    score_fixture(self.config, row, evaluator)["answer_status"],
                    "Unsolved",
                )

    def test_invalid_judge_and_converted_fields_fail(self):
        for reply in (
            {"answer_status": "not_a_status", "reason": "x"},
            {"reason": "x"},
            {"answer_status": "Solved"},
        ):
            with (
                fixture_evaluator(
                    self.config, lambda reply=reply, **kwargs: response(reply)
                ) as evaluator,
                self.assertRaises((ValueError, KeyError)),
            ):
                score_fixture(self.config, example(), evaluator)
        for field in ("method", "total_steps", "final_answer", "answer_details"):
            row = example()
            del row["answer"][field]
            with self.assertRaises(ValueError):
                score_fixture(self.config, row, None)

    def test_parameter_defaults_are_preserved_and_missing_is_rejected(self):
        api = deepcopy(self.tasks[("G1_instruction", "25658")]["api_list"][0])
        for value in (False, 0, "", []):
            api["required_parameters"] = [
                {"name": "x", "type": "STRING", "description": "", "default": value}
            ]
            before = deepcopy(api)
            validate_api(api)
            self.assertEqual(api, before)
        del api["required_parameters"][0]["default"]
        with self.assertRaises(ValueError):
            validate_api(api)
        for field in ("category_name", "tool_name", "api_name", "required_parameters"):
            bad = {key: value for key, value in api.items() if key != field}
            with self.assertRaises(ValueError):
                validate_api(bad)

    def test_incomplete_tool_nodes_are_not_scored_as_missing_finish(self):
        for field in ("name", "arguments", "response"):
            row = example()
            del row["answer"]["answer_details"][0]["message"][field]
            with self.assertRaises(ValueError):
                score_fixture(self.config, row, None)
        row = example()
        del row["answer"]["answer_details"][0]["next"]
        with self.assertRaises(ValueError):
            score_fixture(self.config, row, None)


if __name__ == "__main__":
    unittest.main()
