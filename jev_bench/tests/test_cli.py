import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.bfcl import costs
from jev_bench.config import CATEGORIES, load_config
from jev_bench.io import append_json, write_json
from jev_bench.runner import directory_for, run_lock

from tests.support import PROJECT, config

spec = importlib.util.spec_from_file_location(
    "jev_bench_run", PROJECT / "scripts/run.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def test_formal_toolsandbox_key_required_only_for_live_cli(self):
        value = load_config(PROJECT / "configs/toolsandbox_main_smoke.json")
        value["conditions"] = {"native": value["conditions"]["native"]}
        value["advisors"] = {}
        value["protocol"]["mode"] = "formal"
        value["output_root"] = "outputs"
        value["task_ids"] = {"scenarios": "all"}
        tasks = {
            ("scenarios", f"fixture_{family}_{variant}"): {
                "family_id": str(family),
                "variant": str(variant),
                "categories": [],
            }
            for family in range(129)
            for variant in range(8)
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            root = Path(directory)
            with (
                patch.object(cli, "PROJECT_ROOT", root),
                patch.object(cli, "load_config", return_value=value),
                patch.object(cli, "configure"),
                patch.object(cli, "load_tasks", return_value=(tasks, {})),
                patch.dict(os.environ, {}, clear=True),
                patch("jev_bench.services.json_request") as request,
                patch("jev_bench.services.make_advisor") as advisor,
                patch.object(cli, "execute") as execute,
                redirect_stdout(io.StringIO()),
            ):
                for flags in ([], ["--check-services"], ["--execute"]):
                    with (
                        self.subTest(flags=flags),
                        patch(
                            "sys.argv",
                            [
                                "run.py",
                                "--config",
                                "fixture.json",
                                "--run-id",
                                "missing-key",
                                *flags,
                            ],
                        ),
                    ):
                        if flags:
                            with self.assertRaisesRegex(ValueError, "RAPID_API_KEY"):
                                cli.main()
                        else:
                            self.assertEqual(cli.main(), 0)
                    request.assert_not_called()
                    advisor.assert_not_called()
                    execute.assert_not_called()
                    self.assertFalse((root / "outputs").exists())

    def test_pa_assets_are_saved_and_resume_rejects_same_path_replacements(self):
        from copy import deepcopy

        from tests.pa_fixtures import freeze, tool

        value = config()
        value["conditions"] = {"native": value["conditions"]["native"]}
        value["comparisons"] = []
        task_key = ("multi_turn_base", "fixture")
        tasks = {task_key: {}}
        item = {"condition": "native", "category": task_key[0], "task_id": task_key[1]}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            root = Path(directory)
            p = freeze(value, item, [tool()], root / "mapping")
            original_config = deepcopy(value)
            status_path = root / "tmp/bound/metadata/run_status.json"
            argv = [
                "run.py",
                "--config",
                "fixture.json",
                "--run-id",
                "bound",
                "--execute",
            ]

            def checked_execute(config, plan, tasks, answers, run_dir, state):
                self.assertEqual(state["pa_tool_assets"], {"native": p.asset_identity})
                self.assertEqual(
                    json.loads(status_path.read_text())["pa_tool_assets"],
                    state["pa_tool_assets"],
                )
                return []

            with (
                patch.object(cli, "PROJECT_ROOT", root),
                patch.object(cli, "load_config", return_value=value),
                patch.object(cli, "configure"),
                patch.object(cli, "load_tasks", return_value=(tasks, {})),
                patch.object(cli, "check_services", return_value={}),
                patch.object(
                    cli, "code_version", return_value={"trees": {}, "dirty_paths": []}
                ),
                patch.object(cli, "execute", side_effect=checked_execute) as execute,
                patch.object(
                    cli, "summarize", return_value={"coverage_complete": True}
                ),
                redirect_stdout(io.StringIO()),
            ):
                with patch("sys.argv", argv):
                    self.assertEqual(cli.main(), 0)
                state = json.loads(status_path.read_text())
                state["status"] = "interrupted"
                write_json(status_path, state)
                for filename in ("mapping.json", "catalog.json"):
                    path = root / "mapping" / filename
                    original = path.read_bytes()
                    changed = json.loads(original)
                    if filename == "mapping.json":
                        changed["scopes"]["fixture"][0]["name"] = "other_name"
                    else:
                        changed["scopes"]["fixture"][0]["function"]["description"] = (
                            "Changed description"
                        )
                    path.write_text(json.dumps(changed))
                    execute.reset_mock()
                    with (
                        self.subTest(filename=filename),
                        patch("sys.argv", argv + ["--resume"]),
                        self.assertRaisesRegex(ValueError, "resume PA-Tool assets"),
                    ):
                        cli.main()
                    execute.assert_not_called()
                    self.assertEqual(json.loads(status_path.read_text()), state)
                    path.write_bytes(original)
                with patch("sys.argv", argv + ["--resume"]):
                    self.assertEqual(cli.main(), 0)
                self.assertEqual(value, original_config)

    def test_formal_rescore_rejects_dirty_runtime_before_recovery(self):
        spec = importlib.util.spec_from_file_location(
            "dirty_rescore", PROJECT / "scripts/summarize.py"
        )
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        value = config()
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            run_dir = root / "outputs/recovery"
            write_json(
                run_dir / "metadata/run_status.json",
                {"config": value, "plan": [], "status": "interrupted"},
            )
            with (
                patch.object(summary, "PROJECT_ROOT", root),
                patch.object(summary, "configure"),
                patch.object(
                    summary,
                    "source_identity",
                    return_value={"dirty_paths": ["M jev_bench/src/code.py"]},
                ),
                patch.object(summary, "recover_scores") as recover,
                patch(
                    "sys.argv", ["summarize.py", "--run-dir", str(run_dir), "--rescore"]
                ),
                self.assertRaisesRegex(ValueError, "committed runtime code"),
            ):
                summary.main()
            recover.assert_not_called()
            self.assertFalse((run_dir / "metrics.json").exists())

    def test_score_recovery_leaves_dead_inference_and_unstarted_episodes_resumable(
        self,
    ):
        spec = importlib.util.spec_from_file_location(
            "partial_rescore", PROJECT / "scripts/summarize.py"
        )
        cli_summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_summary)
        value = config()
        plan = [
            {
                "condition": "native",
                "category": category,
                "task_id": f"{category}_0",
                "trial": 0,
                "seed": 1,
            }
            for category in CATEGORIES
        ]
        row = {
            **plan[0],
            "attempt": 1,
            "success": True,
            "official_scored": True,
            "failure_kind": None,
            "cost": costs(None, None, advisor_type=None),
            "duration_seconds": 0,
            "decision_steps": 1,
            "tool_call_count": 0,
            "response_anomalies": [],
            "scoring_recovery": {"previous_failure_kind": "scoring_error"},
        }
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            run_dir = root / "tmp/recovery"
            status = run_dir / "metadata/run_status.json"
            write_json(
                status,
                {
                    "config": value,
                    "plan": plan,
                    "status": "failed",
                    "error": {
                        "type": "InterruptedScoring",
                        "scoring_attempts": [{**plan[0], "attempt": 1}],
                    },
                },
            )
            append_json(run_dir / "episodes.jsonl", row)
            killed = {**plan[1], "attempt": 1}
            write_json(
                directory_for(run_dir, killed) / "diagnostics/attempt.json",
                {
                    "item": killed,
                    "benchmark": "bfcl",
                    "advisor_type": None,
                    "phase": "inference",
                },
            )
            with (
                patch.object(cli_summary, "PROJECT_ROOT", root),
                patch.object(cli_summary, "configure"),
                patch.object(cli_summary, "load_tasks", return_value=({}, {})),
                patch.object(cli_summary, "recover_scores", return_value=0),
                patch.object(
                    cli_summary,
                    "source_identity",
                    return_value={"commit": "fixture", "dirty_paths": []},
                ),
                patch(
                    "sys.argv", ["summarize.py", "--run-dir", str(run_dir), "--rescore"]
                ),
                redirect_stdout(io.StringIO()),
            ):
                cli_summary.main()
            state = json.loads(status.read_text())
            self.assertEqual(state["status"], "partial")
            self.assertEqual(state["phase"], "awaiting_resume")
            self.assertEqual(state["attempts"], 2)
            self.assertNotIn("error", state)
            self.assertEqual(state["scoring_recovery"]["code"]["commit"], "fixture")
            self.assertEqual(
                state["scoring_recovery"]["previous_errors"][0]["type"],
                "InterruptedScoring",
            )

    def test_batch_fatal_without_scoring_evidence_cannot_be_cleared(self):
        spec = importlib.util.spec_from_file_location(
            "fatal_rescore", PROJECT / "scripts/summarize.py"
        )
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            run_dir = root / "tmp/recovery"
            status = run_dir / "metadata/run_status.json"
            state = {
                "config": config(),
                "plan": [],
                "status": "failed",
                "error": {"type": "BrokenProcessPool", "message": "initializer failed"},
            }
            write_json(status, state)
            for rescore in ([], ["--rescore"]):
                with (
                    patch.object(summary, "PROJECT_ROOT", root),
                    patch.object(summary, "recover_scores") as recover,
                    patch(
                        "sys.argv",
                        ["summarize.py", "--run-dir", str(run_dir), *rescore],
                    ),
                    self.assertRaisesRegex(ValueError, "scoring-only failure evidence"),
                ):
                    summary.main()
                recover.assert_not_called()
                self.assertEqual(json.loads(status.read_text()), state)
                self.assertFalse((run_dir / "metrics.json").exists())

    def test_incomplete_scoring_recovery_cannot_clear_original_fatal(self):
        spec = importlib.util.spec_from_file_location(
            "incomplete_rescore", PROJECT / "scripts/summarize.py"
        )
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        plan = [
            {
                "condition": "native",
                "category": CATEGORIES[0],
                "task_id": f"{CATEGORIES[0]}_{index}",
                "trial": 0,
            }
            for index in range(2)
        ]
        for repaired_first in (False, True):
            with (
                self.subTest(repaired_first=repaired_first),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            ):
                root = Path(path)
                run_dir = root / "tmp/recovery"
                state = {
                    "config": config(),
                    "plan": plan,
                    "status": "failed",
                    "phase": "episodes",
                    "error": {
                        "type": "InterruptedScoring",
                        "scoring_attempts": [{**plan[0], "attempt": 1}],
                    },
                }
                status = run_dir / "metadata/run_status.json"
                write_json(status, state)
                for index, item in enumerate(plan):
                    repaired = index == 0 and repaired_first
                    row = {
                        **item,
                        "attempt": 1,
                        "success": True if repaired else None,
                        "official_scored": repaired,
                        "failure_kind": None
                        if repaired
                        else "scoring_error"
                        if index == 0
                        else "unexpected_error",
                        "cost": costs(None, None, advisor_type=None),
                        "duration_seconds": 0,
                        "decision_steps": 1,
                        "tool_call_count": 0,
                        "response_anomalies": [],
                    }
                    if repaired:
                        row["scoring_recovery"] = {
                            "previous_failure_kind": "scoring_error"
                        }
                    append_json(run_dir / "episodes.jsonl", row)
                with (
                    patch.object(summary, "PROJECT_ROOT", root),
                    patch.object(summary, "configure"),
                    patch.object(summary, "load_tasks", return_value=({}, {})),
                    patch.object(summary, "recover_scores", return_value=0),
                    patch.object(
                        summary, "source_identity", return_value={"dirty_paths": []}
                    ),
                    patch(
                        "sys.argv",
                        ["summarize.py", "--run-dir", str(run_dir), "--rescore"],
                    ),
                    self.assertRaises(ValueError),
                ):
                    summary.main()
                after = json.loads(status.read_text())
                self.assertEqual(
                    (after["status"], after["error"]), (state["status"], state["error"])
                )
                self.assertFalse((run_dir / "metrics.json").exists())

    def test_code_identity_ignores_other_projects_but_tracks_advisor(self):
        value = config()
        value["conditions"]["native"]["advisor"] = "fixture"
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            for name in (
                "jev_bench/src",
                "jev_bench/scripts",
                "jev_bench/configs",
                "jev_train/src/jev_train",
                "jev_train/scripts",
                "dataset_tool_inventory/src",
                "decider",
                "other",
            ):
                (root / name).mkdir(parents=True)
                (root / name / "code.py").write_text("version = 1\n")
            (root / "jev_train/scripts/serve_jev.py").write_text("version = 1\n")
            selected = root / "jev_bench/configs/selected.json"
            selected.write_text("{}\n")
            (root / "jev_bench/README.md").write_text("Documentation\n")

            def git(*args):
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "-c",
                        "user.name=CPU fixture",
                        "-c",
                        "user.email=cpu@example.invalid",
                        *args,
                    ],
                    stderr=subprocess.DEVNULL,
                )

            git("init")
            git("add", ".")
            git("commit", "-m", "fixture")
            # The decider checkout is independently versioned in production.
            with (
                patch.object(cli, "PROJECT_ROOT", root / "jev_bench"),
                patch.object(cli, "WORKSPACE_ROOT", root),
            ):
                (root / "third_party").mkdir()
                (root / "third_party/decider").symlink_to(
                    root, target_is_directory=True
                )
                original = cli.code_version(value, selected)
                (root / "other/code.py").write_text("version = 2\n")
                (root / "jev_bench/README.md").write_text("Updated documentation\n")
                (root / "jev_bench/configs/unrelated.json").write_text("{}\n")
                git(
                    "add",
                    "other",
                    "jev_bench/README.md",
                    "jev_bench/configs/unrelated.json",
                )
                git("commit", "-m", "unrelated")
                unrelated = cli.code_version(value, selected)
                self.assertNotEqual(original["commit"], unrelated["commit"])
                self.assertEqual(original["trees"], unrelated["trees"])
                self.assertFalse(unrelated["dirty_paths"])
                (root / "decider/__pycache__").mkdir()
                (root / "decider/__pycache__/code.pyc").write_bytes(b"cache")
                self.assertFalse(cli.code_version(value, selected)["dirty_paths"])
                (root / "decider/new.py").write_text("value = 1\n")
                self.assertTrue(cli.code_version(value, selected)["dirty_paths"])
                (root / "decider/new.py").unlink()
                (root / "jev_train/src/jev_train/code.py").write_text("version = 2\n")
                self.assertTrue(cli.code_version(value, selected)["dirty_paths"])
                git("add", "jev_train")
                git("commit", "-m", "advisor")
                self.assertNotEqual(
                    original["trees"], cli.code_version(value, selected)["trees"]
                )
                selected.write_text('{"changed": true}\n')
                self.assertTrue(cli.code_version(value, selected)["dirty_paths"])

    def test_standalone_summary_cannot_overwrite_active_run(self):
        summary_spec = importlib.util.spec_from_file_location(
            "jev_bench_summary", PROJECT / "scripts/summarize.py"
        )
        summary_cli = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary_cli)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            run_dir = root / "tmp/active"
            run_dir.mkdir(parents=True)
            with (
                run_lock(run_dir),
                patch.object(summary_cli, "PROJECT_ROOT", root),
                patch("sys.argv", ["summarize.py", "--run-dir", str(run_dir)]),
                self.assertRaisesRegex(ValueError, "another coordinator"),
            ):
                summary_cli.main()
            self.assertFalse((run_dir / "metrics.json").exists())

    def test_completed_requires_written_metrics(self):
        value = config()
        tasks = {
            (category, f"{category}_0"): {"id": f"{category}_0"}
            for category in CATEGORIES
        }

        def fake_execute(config, plan, tasks, answers, run_dir, state):
            state.update(status="running", phase="episodes_complete")
            write_json(run_dir / "metadata/run_status.json", state)
            rows = [
                {
                    **item,
                    "attempt": 1,
                    "success": True,
                    "official_scored": True,
                    "failure_kind": None,
                    "cost": costs(None, None, advisor_type=None),
                    "duration_seconds": 0,
                    "decision_steps": 1,
                    "tool_call_count": 0,
                    "response_anomalies": [],
                }
                for item in plan
            ]
            for row in rows:
                append_json(run_dir / "episodes.jsonl", row)
            return rows

        for fail_metrics in (False, True):
            with (
                self.subTest(fail_metrics=fail_metrics),
                tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory,
            ):
                root = Path(directory)
                run_dir = root / "tmp/cli_test"
                status_path = run_dir / "metadata/run_status.json"

                def checked_write(
                    path, data, status_path=status_path, fail_metrics=fail_metrics
                ):
                    if path.name == "metrics.json":
                        self.assertEqual(
                            json.loads(status_path.read_text())["status"], "running"
                        )
                        if fail_metrics:
                            raise OSError("cannot persist metrics")
                    write_json(path, data)

                with (
                    patch.object(cli, "PROJECT_ROOT", root),
                    patch.object(cli, "load_config", return_value=value),
                    patch.object(cli, "configure"),
                    patch.object(cli, "load_tasks", return_value=(tasks, {})),
                    patch.object(cli, "check_services", return_value={}),
                    patch.object(
                        cli,
                        "code_version",
                        return_value={"commit": "test", "dirty_paths": []},
                    ),
                    patch.object(cli, "execute", side_effect=fake_execute),
                    patch.object(cli, "write_json", side_effect=checked_write),
                    patch(
                        "sys.argv",
                        [
                            "run.py",
                            "--config",
                            "test.json",
                            "--execute",
                            "--run-id",
                            "cli_test",
                        ],
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    if fail_metrics:
                        with self.assertRaisesRegex(OSError, "cannot persist metrics"):
                            cli.main()
                    else:
                        self.assertEqual(cli.main(), 0)

                state = json.loads(status_path.read_text())
                self.assertEqual(
                    state["status"], "failed" if fail_metrics else "completed"
                )
                self.assertTrue(state["ended_at"].endswith("+08:00"))
                self.assertEqual((run_dir / "metrics.json").exists(), not fail_metrics)
                if fail_metrics:
                    spec = importlib.util.spec_from_file_location(
                        "summary_recovery", PROJECT / "scripts/summarize.py"
                    )
                    summary_cli = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(summary_cli)
                    with (
                        patch.object(summary_cli, "PROJECT_ROOT", root),
                        patch.object(summary_cli, "configure"),
                        patch("sys.argv", ["summarize.py", "--run-dir", str(run_dir)]),
                        redirect_stdout(io.StringIO()),
                    ):
                        summary_cli.main()
                    self.assertEqual(
                        json.loads(status_path.read_text())["status"], "completed"
                    )
                    self.assertEqual(
                        json.loads(status_path.read_text())["attempts"], len(tasks)
                    )

    def test_summary_failure_cannot_be_cleared_with_partial_coverage(self):
        spec = importlib.util.spec_from_file_location(
            "partial_summary", PROJECT / "scripts/summarize.py"
        )
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            root = Path(path)
            run_dir = root / "tmp/recovery"
            state = {
                "config": config(),
                "status": "failed",
                "phase": "summarizing",
                "error": {"type": "OSError", "message": "write failed"},
                "plan": [
                    {
                        "condition": "native",
                        "category": CATEGORIES[0],
                        "task_id": CATEGORIES[0] + "_0",
                        "trial": 0,
                    }
                ],
            }
            write_json(run_dir / "metadata/run_status.json", state)
            with (
                patch.object(summary, "PROJECT_ROOT", root),
                patch.object(summary, "configure"),
                patch("sys.argv", ["summarize.py", "--run-dir", str(run_dir)]),
                self.assertRaisesRegex(ValueError, "complete scored coverage"),
            ):
                summary.main()
            self.assertEqual(
                json.loads((run_dir / "metadata/run_status.json").read_text()), state
            )
            self.assertFalse((run_dir / "metrics.json").exists())
