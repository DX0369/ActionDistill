"""Published text generation and ordered score recovery, without model requests."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks import apibank
from jev_bench.benchmarks.apibank_group_scoring import (
    GroupScoringError,
    _worker_score,
    score_groups,
)
from jev_bench.benchmarks.apibank_runtime import (
    actor_messages,
    advisor_messages,
    advisor_tools,
    recent_calls,
)
from jev_bench.config import load_config
from jev_bench.io import append_json, read_jsonl, write_json
from jev_bench.runner import directory_for, make_plan, pending_plan, validate_attempts
from jev_bench.scoring import recover_scores
from jev_bench.services import check_services
from jev_bench.stats import summarize

from tests.support import PROJECT, completion, sdk


class ApiBankRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT / "configs/apibank_main_smoke.json")
        self.config["conditions"] = {"native": self.config["conditions"]["native"]}
        apibank.configure(self.config, None)
        self.tasks, self.answers = apibank.load_tasks(self.config)
        self.plan = make_plan(self.config, self.tasks)
        temporary = tempfile.TemporaryDirectory(
            dir=PROJECT / "tmp", prefix="apibank_runtime_"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def generate(self, planned):
        item = {**planned, "attempt": 1}
        identity = item["category"], item["task_id"]
        output = self.answers[identity]["published_target"]
        requests = []

        def create(**request):
            requests.append(deepcopy(request))
            return completion(output)

        with patch("jev_bench.client.OpenAI", return_value=sdk(create)):
            row = apibank.run_episode(
                self.config,
                item,
                self.tasks[identity],
                {"not_visible": "secret-gold"},
                directory_for(self.root, item),
            )
        append_json(self.root / "episodes.jsonl", row)
        self.assertEqual(requests[0]["messages"], actor_messages(self.tasks[identity]))
        self.assertNotIn("secret-gold", str(requests))
        self.assertEqual(row["failure_kind"], "pending_scoring")
        return row

    def result(self):
        return {
            "points": [
                {
                    "category": key[0],
                    "task_id": key[1],
                    "source_id": task["source_id"],
                    **(
                        {"correct": True}
                        if task["kind"] == "api"
                        else {"rouge_l_f": 0.5}
                    ),
                }
                for key, task in self.tasks.items()
            ],
            "categories": {},
            "protocol_events": {},
            "level_3_sample_accuracy": {},
        }

    def test_visible_conversion_keeps_material_and_occurrences(self):
        for task in self.tasks.values():
            original = deepcopy(task)
            tools = advisor_tools(task)
            self.assertEqual(len(tools), len(task["visible_tools"]["occurrences"]))
            self.assertIn(task["instruction"], advisor_messages(task)[1]["content"])
            self.assertIn(task["input"], advisor_messages(task)[1]["content"])
            self.assertEqual(actor_messages(task)[0]["content"], task["instruction"])
            self.assertEqual(actor_messages(task)[1]["content"], task["input"])
            recent_calls(task)
            self.assertEqual(task, original)
            for tool in tools:
                self.assertNotIn("required", tool["function"]["parameters"])
        task = deepcopy(next(iter(self.tasks.values())))
        del task["instruction"]
        with self.assertRaises(KeyError):
            actor_messages(task)
        task = deepcopy(next(iter(self.tasks.values())))
        task["input"] = "Unexpected framing\nGenerate API request:"
        with self.assertRaises(ValueError):
            recent_calls(task)

    def test_deepseek_uses_actor_text_and_official_name_extraction(self):
        from tests.test_remote_advisors import deepseek_response

        self.config = load_config(PROJECT / "configs/apibank_main_smoke.json")
        planned = next(item for item in self.plan if item["category"].endswith("_api"))
        task = self.tasks[(planned["category"], planned["task_id"])]
        name = advisor_tools(task)[0]["function"]["name"]
        cases = [
            ("stop", f"Before [{name}(key='x')] after", (), name),
            ("stop", f"[{name}(garbage)] [Other(x=1)]", (), name),
            ("stop", "A normal reply", (), "RESPOND"),
            ("stop", f"[{name}(key='x')", (), "RESPOND"),
            ("stop", "[NotInCandidates()]", (), None),
            ("stop", "[RESPOND()]", (), None),
            ("stop", "", (), None),
            ("length", f"[{name}()]", (), None),
            ("tool_calls", None, (name,), None),
        ]
        for index, (reason, content, calls, expected) in enumerate(cases):
            requests = []

            def create(
                requests=requests,
                reason=reason,
                content=content,
                calls=calls,
                **request,
            ):
                requests.append(deepcopy(request))
                return (
                    deepseek_response(reason, content, calls)
                    if request["model"] == "deepseek-flash"
                    else completion("Done")
                )

            item = {**planned, "condition": "deepseek_advisor", "attempt": 1}
            directory = self.root / str(index)
            with (
                self.subTest(reason=reason, content=content),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "cpu-key"}),
                patch("jev_bench.client.OpenAI", return_value=sdk(create)),
            ):
                row = apibank.run_episode(
                    self.config, item, task, {"not_visible": "secret-gold"}, directory
                )
            self.assertEqual(row["failure_kind"], "pending_scoring")
            self.assertEqual(requests[0]["messages"], actor_messages(task))
            self.assertNotIn("tools", requests[0])
            self.assertNotIn("tool_choice", requests[0])
            self.assertNotIn("secret-gold", str(requests))
            record = read_jsonl(directory / "diagnostics/advisor.jsonl")[0]
            if expected is None:
                self.assertEqual(record["fallback"], "advisor_format_error_neutral")
                self.assertIsNone(record["prediction"])
            else:
                self.assertEqual(record["prediction"]["top1"], expected)
                self.assertIsNone(record["fallback"])
            ledger = read_jsonl(directory / "diagnostics/advisor_requests.jsonl")
            self.assertEqual(ledger[0]["output_protocol"], "apibank_text")
            self.assertEqual(row["cost"]["advisor"]["provider_requests"], 1)
            self.assertEqual(
                row["cost"]["advisor"]["completion_tokens"]["observed_total"], 5
            )

    def test_partial_generation_then_group_score_and_idempotent_recovery(self):
        self.generate(self.plan[0])
        with patch(
            "jev_bench.benchmarks.apibank_group_scoring._worker_score"
        ) as worker:
            self.assertEqual(
                score_groups(
                    self.config,
                    self.plan,
                    self.tasks,
                    self.root,
                    code={},
                    recovery=False,
                ),
                0,
            )
            worker.assert_not_called()
        self.assertEqual(
            len(pending_plan(self.plan, read_jsonl(self.root / "episodes.jsonl"))),
            len(self.plan) - 1,
        )
        for item in self.plan[1:]:
            self.generate(item)
        with patch(
            "jev_bench.benchmarks.apibank_group_scoring._worker_score",
            return_value=self.result(),
        ) as worker:
            self.assertEqual(
                score_groups(
                    self.config,
                    self.plan,
                    self.tasks,
                    self.root,
                    code={},
                    recovery=False,
                ),
                6,
            )
            self.assertEqual(worker.call_count, 1)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=AssertionError("must not rescore"),
            ),
        ):
            self.assertEqual(
                recover_scores(
                    self.config, self.plan, self.tasks, self.answers, self.root, code={}
                ),
                0,
            )
        rows = read_jsonl(self.root / "episodes.jsonl")
        validate_attempts(rows, self.plan)
        metrics = summarize(rows, self.plan, [], benchmark="apibank")
        self.assertTrue(metrics["coverage_complete"])
        self.assertEqual(
            metrics["conditions"]["native"]["scores"]["level_1_response"]["mean"], 0.5
        )

    def test_score_failure_and_crash_after_canonical_score_recover_without_generation(
        self,
    ):
        before = [self.generate(item) for item in self.plan]
        with (
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._persist_member",
                side_effect=OSError("interrupted write"),
            ),
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                return_value=self.result(),
            ),
            self.assertRaises(GroupScoringError),
        ):
            score_groups(
                self.config, self.plan, self.tasks, self.root, code={}, recovery=False
            )
        with (
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=AssertionError("must reuse group score"),
            ),
            patch(
                "jev_bench.client.OpenAI", side_effect=AssertionError("no regeneration")
            ),
        ):
            self.assertEqual(
                recover_scores(
                    self.config, self.plan, self.tasks, self.answers, self.root, code={}
                ),
                6,
            )
        after = read_jsonl(self.root / "episodes.jsonl")
        self.assertEqual([r["cost"] for r in before], [r["cost"] for r in after])
        self.assertTrue(all(r["attempt"] == 1 for r in after))

    def test_worker_failure_preserves_scoring_evidence(self):
        for item in self.plan:
            self.generate(item)
        with (
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=ValueError("bad scorer"),
            ),
            self.assertRaises(GroupScoringError) as raised,
        ):
            score_groups(
                self.config, self.plan, self.tasks, self.root, code={}, recovery=False
            )
        self.assertEqual(len(raised.exception.scoring_attempts), 6)
        self.assertEqual(len(raised.exception.scoring_groups), 1)
        rows = read_jsonl(self.root / "episodes.jsonl")
        self.assertTrue(all(r["failure_kind"] == "scoring_error" for r in rows))
        self.assertTrue(
            all(
                json.loads(
                    (directory_for(self.root, r) / "trajectory.json").read_text()
                )["partial"]
                is False
                for r in rows
            )
        )

    def test_partial_group_scoring_interrupt_can_resume_remaining_inference(self):
        row = self.generate(self.plan[0])
        row["failure_kind"] = "scoring_error"
        write_json(directory_for(self.root, row) / "episode.json", row)
        (self.root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
        status = {
            "status": "failed",
            "phase": "inference",
            "config": self.config,
            "plan": self.plan,
            "error": {
                "type": "RuntimeError",
                "scoring_attempts": [{**self.plan[0], "attempt": 1}],
            },
        }
        write_json(self.root / "metadata/run_status.json", status)
        spec = importlib.util.spec_from_file_location(
            "apibank_rescore_cli", PROJECT / "scripts/summarize.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (
            patch(
                "sys.argv", ["summarize.py", "--run-dir", str(self.root), "--rescore"]
            ),
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=AssertionError("partial group"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            module.main()
        rows = read_jsonl(self.root / "episodes.jsonl")
        self.assertEqual(rows[0]["failure_kind"], "pending_scoring")
        self.assertEqual(len(pending_plan(self.plan, rows)), len(self.plan) - 1)
        self.assertEqual(
            json.loads((self.root / "metadata/run_status.json").read_text())["status"],
            "partial",
        )

    def test_diagnostic_write_failure_keeps_group_failure_manifest(self):
        for item in self.plan:
            self.generate(item)
        with (
            patch(
                "jev_bench.benchmarks.apibank_group_scoring._worker_score",
                side_effect=ValueError("scorer"),
            ),
            patch(
                "jev_bench.benchmarks.apibank_group_scoring.append_json",
                side_effect=OSError("diagnostic write failed"),
            ),
            self.assertRaises(GroupScoringError) as raised,
        ):
            score_groups(
                self.config, self.plan, self.tasks, self.root, code={}, recovery=False
            )
        self.assertEqual(len(raised.exception.scoring_groups[0]["members"]), 6)
        self.assertIn("diagnostic write failed", str(raised.exception))

    def test_scoring_interpreter_must_be_explicit_absolute_and_executable(self):
        non_executable = self.root / "python"
        non_executable.touch(mode=0o600)
        for value in (
            None,
            "",
            "  ",
            "python",
            str(self.root / "missing"),
            str(self.root),
            str(non_executable),
        ):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {}, clear=True),
                patch("jev_bench.services.json_request") as request,
                patch(
                    "jev_bench.benchmarks.apibank_group_scoring.subprocess.run"
                ) as worker,
            ):
                if value is not None:
                    os.environ["JEV_APIBANK_SCORING_PYTHON"] = value
                with self.assertRaises(ValueError):
                    check_services(self.config)
                with self.assertRaises(ValueError):
                    _worker_score(self.config, {})
                request.assert_not_called()
                worker.assert_not_called()

    def test_preflight_and_scoring_use_the_same_relocated_interpreter(self):
        python = self.root / "relocated env/bin/python"
        python.parent.mkdir(parents=True)
        python.touch(mode=0o700)
        model_path = self.root / "model"
        model_path.mkdir()
        executor = self.config["executors"]["qwen3_4b"]
        executor["model_path"] = str(model_path)
        model = {
            "id": executor["model"],
            "root": str(model_path),
            "max_model_len": executor["max_model_len"],
        }
        with (
            patch.dict(os.environ, {"JEV_APIBANK_SCORING_PYTHON": str(python)}),
            patch("jev_bench.services.json_request", return_value={"data": [model]}),
            patch(
                "jev_bench.benchmarks.apibank_group_scoring.subprocess.run"
            ) as worker,
        ):
            self.assertEqual(
                check_services(self.config)["executors"]["qwen3_4b"], model
            )
            worker.return_value.stdout = "{}"
            self.assertEqual(_worker_score(self.config, {}), {})
            command = worker.call_args.args[0]
            self.assertEqual(
                command,
                [str(python), "-B", str(PROJECT / "scripts/apibank_score_worker.py")],
            )
            self.assertEqual(worker.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_real_old_environment_worker_scores_saved_published_text(self):
        predictions = {key: self.answers[key]["published_target"] for key in self.tasks}
        result = _worker_score(self.config, predictions)
        self.assertEqual(len(result["points"]), len(self.tasks))
        self.assertFalse(result["level_3_sample_accuracy"]["complete"])
        self.assertIsNone(result["level_3_sample_accuracy"]["denominator"])


if __name__ == "__main__":
    unittest.main()
