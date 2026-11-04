import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.apibank import visible_tools
from jev_bench.benchmarks.apibank_runtime import advisor_tools
from jev_bench.client import ParticipantResponseError
from jev_bench.pa_tool import Projection
from jev_bench.pa_tool_text import call_text, interface_tools, project_task

from tests.pa_fixtures import freeze, rows_for
from tests.support import PROJECT, completion, sdk


class TextProjectionTests(unittest.TestCase):
    def test_search_metadata_variants_share_alias_without_changing_public_text(self):
        initial = {
            "name": "GetUserToken",
            "description": "Initial token lookup description.",
            "input_parameters": {
                "username": {"type": "str", "description": "User name"}
            },
            "output_parameters": {"token": {"type": "str", "description": "Token"}},
        }
        discovered = {
            **deepcopy(initial),
            "description": "Retrieved token lookup description.",
            "desc_for_search": "get user token Retrieved token lookup description.",
        }
        searcher = {
            "name": "ToolSearcher",
            "description": "Find a tool.",
            "input_parameters": {
                "keywords": {"type": "str", "description": "Search terms"}
            },
            "output_parameters": {},
        }
        task = {
            "level": 2,
            "instruction": "Policy\nAPI descriptions:\n\n"
            + json.dumps(initial)
            + "\n"
            + json.dumps(searcher)
            + "\n",
            "input": "API-Request: [ToolSearcher(keywords='token')]->"
            + repr(discovered)
            + "\nGenerate response:",
        }
        task["visible_tools"] = visible_tools(2, task["instruction"], task["input"])
        original = deepcopy(task)
        tools, positions, sources = interface_tools(task)
        self.assertEqual(positions, [0, 1, 0])
        self.assertEqual(tools[0]["function"]["description"], initial["description"])
        self.assertEqual(sources[0]["source"]["field"], "instruction")
        shown = project_task(task, Projection(tools, rows_for(tools)))
        self.assertIn(discovered["description"], shown["input"])
        self.assertIn(repr(discovered["desc_for_search"]), shown["input"])
        self.assertEqual(task, original)
        missing_initial = deepcopy(task)
        first = missing_initial["visible_tools"]["occurrences"][0]
        first["schema"]["description"] = discovered["description"]
        first["source"]["field"] = "input"
        with self.assertRaisesRegex(ValueError, "one instruction definition"):
            interface_tools(missing_initial)
        for field in ("input_parameters", "output_parameters"):
            changed = deepcopy(task)
            changed["visible_tools"]["occurrences"][-1]["schema"][field]["extra"] = {
                "type": "str",
                "description": "Different interface",
            }
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "conflicting"),
            ):
                interface_tools(changed)

    def test_api_interface_failure_survives_group_scoring_without_fake_prediction(self):
        from jev_bench.benchmarks.apibank import load_tasks, run_episode
        from jev_bench.benchmarks.apibank_group_scoring import (
            GroupScoringError,
            _worker_score,
            score_groups,
        )
        from jev_bench.config import load_config
        from jev_bench.io import append_json, read_jsonl, write_json
        from jev_bench.runner import directory_for, make_plan, pending_plan
        from jev_bench.stats import summarize

        config = load_config(PROJECT / "configs/apibank_main_smoke.json")
        config["conditions"] = {"native": config["conditions"]["native"]}
        tasks, answers = load_tasks(config)
        plan = make_plan(config, tasks)
        failed = next(key for key in tasks if key[0] == "level_2_response")
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            root = Path(directory)
            for planned in plan:
                item = {**planned, "attempt": 1}
                identity = item["category"], item["task_id"]
                task = tasks[identity]
                tools, _, _ = interface_tools(task)
                p = freeze(config, item, tools, root / "mapping")
                project_task(task, p)
                search = (
                    p.rows[p.originals["ToolSearcher"][0]]
                    if identity == failed
                    else None
                )
                prediction = (
                    f"[{search['name']}({search['parameters']['keywords']}='first', wrapper({search['parameters']['keywords']}='second'))]"
                    if identity == failed
                    else call_text(
                        answers[identity]["published_target"], p, restore=False
                    )
                )
                with patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(
                        lambda prediction=prediction, **_: completion(prediction)
                    ),
                ):
                    row = run_episode(
                        config,
                        item,
                        task,
                        None,
                        directory_for(root, item),
                        pa_tool_identity=p.asset_identity,
                    )
                append_json(root / "episodes.jsonl", row)
                if identity == failed:
                    records = read_jsonl(
                        directory_for(root, item) / "diagnostics/executor.jsonl"
                    )
                    self.assertEqual(
                        records[0]["response"]["choices"][0]["message"]["content"],
                        prediction,
                    )
                    self.assertFalse(
                        (directory_for(root, item) / "trajectory.json").exists()
                    )
            (root / "mapping/mapping.json").unlink()
            with (
                patch(
                    "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                    side_effect=RuntimeError("score interruption"),
                ),
                self.assertRaises(GroupScoringError) as failed_group,
            ):
                score_groups(config, plan, tasks, root, code={}, recovery=False)
            write_json(
                root / "metadata/run_status.json",
                {
                    "status": "failed",
                    "phase": "scoring",
                    "config": config,
                    "plan": plan,
                    "error": {
                        "message": "score interruption",
                        "scoring_attempts": failed_group.exception.scoring_attempts,
                        "scoring_groups": failed_group.exception.scoring_groups,
                    },
                },
            )
            spec = importlib.util.spec_from_file_location(
                "pa_score_recovery", PROJECT / "scripts/summarize.py"
            )
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
            with patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                wraps=_worker_score,
            ) as worker:
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            str(PROJECT / "scripts/summarize.py"),
                            "--run-dir",
                            str(root),
                            "--rescore",
                        ],
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    cli.main()
                self.assertNotIn(failed, worker.call_args.args[1])
                self.assertEqual(worker.call_args.kwargs["terminal_failures"], {failed})
            rows = read_jsonl(root / "episodes.jsonl")
            failure = next(
                row for row in rows if (row["category"], row["task_id"]) == failed
            )
            self.assertEqual(failure["failure_kind"], "executor_response_error")
            self.assertFalse(failure["official_scored"])
            self.assertFalse(failure["success"])
            self.assertEqual(pending_plan(plan, rows), [])
            summary = summarize(rows, plan, [], benchmark="apibank")
            self.assertTrue(summary["execution_coverage_complete"])
            self.assertFalse(summary["coverage_complete"])
            self.assertEqual(summary["conditions"]["native"]["protocol_failures"], 1)
            self.assertEqual(
                summary["conditions"]["native"]["coverage_by_category"][failed[0]][
                    "missing_scores"
                ],
                1,
            )
            state = json.loads((root / "metadata/run_status.json").read_text())
            self.assertEqual(state["status"], "partial")
            state.update(
                status="failed",
                phase="summarizing",
                error={"message": "summary write interrupted"},
            )
            write_json(root / "metadata/run_status.json", state)
            with (
                patch.object(
                    sys,
                    "argv",
                    [str(PROJECT / "scripts/summarize.py"), "--run-dir", str(root)],
                ),
                patch(
                    "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                    side_effect=AssertionError("summary recovery must not score"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                cli.main()
            self.assertEqual(
                json.loads((root / "metadata/run_status.json").read_text())["status"],
                "partial",
            )
            with patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=AssertionError("do not rescore"),
            ):
                self.assertEqual(
                    score_groups(config, plan, tasks, root, code={}, recovery=True), 0
                )

    def setUp(self):
        from jev_bench.benchmarks.apibank import configure
        from jev_bench.config import load_config

        configure(load_config(PROJECT / "configs/apibank_main_smoke.json"), None)

    def test_discovery_and_unicode_spans_preserve_descriptions_values_and_outputs(self):
        initial = {
            "name": "ToolSearcher",
            "description": "搜索 Name(keyword=...).",
            "input_parameters": {
                "keywords": {"type": "str", "description": "Search term"}
            },
            "output_parameters": {},
        }
        discovered = {
            "name": "Lookup",
            "description": "Lookup 名称",
            "input_parameters": {"key": {"type": "str", "description": "key 字段"}},
            "output_parameters": {
                "key": {"type": "str", "description": "key is unchanged"}
            },
        }
        task = {
            "level": 2,
            "instruction": "Policy: ToolSearcher is mentioned literally.\nAPI descriptions:\n\n"
            + json.dumps(initial, ensure_ascii=False)
            + "\n",
            "input": "User: Lookup(key='姓名')\nAPI-Request: [ToolSearcher(keywords='Lookup')]->"
            + repr(discovered)
            + "\nAPI-Request: [Lookup(key='key')]->{'key': 'Lookup'}\nGenerate response:",
        }
        task["visible_tools"] = visible_tools(
            task["level"], task["instruction"], task["input"]
        )
        original = deepcopy(task)
        tools = advisor_tools(task)
        p = Projection(tools, rows_for(tools))
        shown = project_task(task, p)
        self.assertEqual(task, original)
        self.assertIn(
            "Policy: ToolSearcher is mentioned literally.", shown["instruction"]
        )
        self.assertIn("搜索 Name(keyword=...).", shown["instruction"])
        self.assertIn("User: Lookup(key='姓名')", shown["input"])
        self.assertIn("[pa_tool_0(pa_arg_0='Lookup')]", shown["input"])
        self.assertIn("[pa_tool_1(pa_arg_0='key')]->{'key': 'Lookup'}", shown["input"])
        self.assertIn("'output_parameters': {'key':", shown["input"])
        self.assertIn("Lookup 名称", shown["input"])
        prediction = "说明 key [pa_tool_1(pa_arg_0='Lookup(key=原文)')] 保持原文"
        self.assertEqual(
            call_text(prediction, p, restore=True),
            "说明 key [Lookup(key='Lookup(key=原文)')] 保持原文",
        )
        with self.assertRaises(ParticipantResponseError):
            call_text("[Lookup(key='x')]", p, restore=True)
        with self.assertRaises(ParticipantResponseError):
            call_text("[pa_tool_1(key='x')]", p, restore=True)
        with self.assertRaises(ParticipantResponseError):
            call_text("[pa_tool_1(wrapper(key='x'))]", p, restore=True)

    def test_text_mapping_preserves_official_duplicate_key_semantics_both_ways(self):
        from api_call_extraction import parse_api_call

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Lookup",
                    "parameters": {"properties": {"key": {"type": "string"}}},
                },
            }
        ]
        p = Projection(tools, [{"name": "pa_lookup", "parameters": {"key": "item"}}])
        p.request([], tools)
        for restore in (True, False):
            source, key, target, mapped = (
                ("pa_lookup", "item", "Lookup", "key")
                if restore
                else ("Lookup", "key", "pa_lookup", "item")
            )
            error = ParticipantResponseError if restore else ValueError
            for last in ("first", "second"):
                with (
                    self.subTest(restore=restore, nested_value=last),
                    self.assertRaisesRegex(error, "official parsed call"),
                ):
                    call_text(
                        f"[{source}({key}='first', wrapper({key}='{last}'))]",
                        p,
                        restore=restore,
                    )
            text = f"正文 [{source}({key}='first', {key}='second')] 原文"
            translated = call_text(text, p, restore=restore)
            self.assertEqual(
                translated, f"正文 [{target}({mapped}='first', {mapped}='second')] 原文"
            )
            self.assertEqual(parse_api_call(translated), (target, {mapped: "second"}))
            text = f"正文 [{source}({key}='中文 {key}=literal')] 原文"
            self.assertEqual(
                call_text(text, p, restore=restore),
                f"正文 [{target}({mapped}='中文 {key}=literal')] 原文",
            )

    def test_level3_wrapper_only_changes_discovered_definition(self):
        initial = {
            "apiCode": "ToolSearcher",
            "description": "Search",
            "parameters": {"keywords": {"type": "str", "description": "Search terms"}},
            "response": {},
        }
        found = {
            "name": "Find",
            "description": "Find",
            "input_parameters": {},
            "output_parameters": {},
        }
        wrapper = {
            "api_name": "ToolSearcher",
            "input": {"keywords": "Find"},
            "output": [found],
            "exception": None,
        }
        task = {
            "level": 3,
            "instruction": "Unchanged",
            "input": json.dumps(initial)
            + "\nUser: Hi\nAPI-Request: [ToolSearcher(keywords='Find')]->"
            + repr(wrapper)
            + "\nGenerate API request:",
        }
        task["visible_tools"] = visible_tools(3, task["instruction"], task["input"])
        tools = advisor_tools(task)
        p = Projection(tools, rows_for(tools))
        shown = project_task(task, p)
        self.assertIn('"apiCode": "pa_tool_0"', shown["input"])
        self.assertIn(
            "'api_name': 'ToolSearcher', 'input': {'keywords': 'Find'}", shown["input"]
        )
        self.assertIn("'name': \"pa_tool_1\"", shown["input"])

    def test_published_points_share_description_variants_and_save_canonical_predictions(
        self,
    ):
        from jev_bench.benchmarks.apibank import load_tasks, run_episode
        from jev_bench.config import load_config

        config = load_config(PROJECT / "configs/apibank_main_smoke.json")
        tasks, answers = load_tasks(config)
        saw_variant = False
        for (category, task_id), task in tasks.items():
            tools, positions, sources = interface_tools(task)
            if task["visible_tools"]["multiple_definitions"]:
                saw_variant = True
                name = next(iter(task["visible_tools"]["multiple_definitions"]))
                indices = [
                    i
                    for i, row in enumerate(task["visible_tools"]["occurrences"])
                    if row["name"] == name
                ]
                self.assertEqual(len({positions[i] for i in indices}), 1)
                self.assertEqual(
                    next(s for s in sources if s["name"] == name)["source"]["field"],
                    "instruction",
                )
                changed = deepcopy(task)
                occurrence = changed["visible_tools"]["occurrences"][indices[-1]]
                occurrence["schema"]["input_parameters"]["extra"] = {
                    "type": "str",
                    "description": "conflict",
                }
                with self.assertRaises(ValueError):
                    interface_tools(changed)
            item = {
                "condition": "native",
                "category": category,
                "task_id": task_id,
                "trial": 0,
                "attempt": 1,
                "seed": 20261015,
            }
            with (
                self.subTest(category=category),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory,
            ):
                root = Path(directory)
                p = freeze(config, item, tools, root / "mapping")
                project_task(task, p)
                target = answers[category, task_id]["published_target"]
                alias = call_text(target, p, restore=False)
                with patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(lambda alias=alias, **_: completion(alias)),
                ):
                    outcome = run_episode(
                        config,
                        item,
                        task,
                        {"hidden": "never pass"},
                        root / "episode",
                        pa_tool_identity=p.asset_identity,
                    )
                self.assertEqual(outcome["failure_kind"], "pending_scoring")
                trajectory = json.loads((root / "episode/trajectory.json").read_text())
                self.assertEqual(trajectory["prediction"], target)
        self.assertTrue(saw_variant)
