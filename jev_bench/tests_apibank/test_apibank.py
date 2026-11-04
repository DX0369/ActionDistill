import json
import os
import shutil
import socket
import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.apibank import configure, load_tasks, point_id
from jev_bench.benchmarks.apibank_scoring import (
    make_manager,
    official_runtime,
    score_api,
    score_batch,
    score_response,
)
from jev_bench.config import APIBANK_CATEGORIES, PROJECT_ROOT, load_config


def config():
    value = load_config(PROJECT_ROOT / "configs/apibank_cpu_preview.json")
    configure(value, PROJECT_ROOT / "tmp")
    return value


def target(name, inputs, output, *, exception=None):
    return {
        "role": "API",
        "api_name": name,
        "result": {
            "api_name": name,
            "input": inputs,
            "output": output,
            "exception": exception,
        },
    }


class ApiBankTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(prefix="apibank_test_", dir=PROJECT_ROOT / "tmp")
        )
        self.connect = patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("CPU fixture attempted a connection"),
        )
        self.resolve = patch(
            "socket.getaddrinfo",
            side_effect=AssertionError("CPU fixture attempted DNS"),
        )
        self.connect.start()
        self.resolve.start()

    def tearDown(self):
        self.connect.stop()
        self.resolve.stop()
        failed = any(
            test is self
            for test, _ in self._outcome.result.errors + self._outcome.result.failures
        )
        if failed:
            print(f"CPU fixture diagnostics retained: {self.directory}")
        else:
            shutil.rmtree(self.directory)

    def test_all_source_ids_and_published_text_are_preserved(self):
        value = deepcopy(self.config)
        value["task_ids"] = dict.fromkeys(APIBANK_CATEGORIES, "all")
        tasks, answers = load_tasks(value)
        self.assertEqual(tasks.keys(), answers.keys())
        self.assertEqual(
            [
                sum(key[0] == category for key in tasks)
                for category in APIBANK_CATEGORIES
            ],
            [399, 375, 135, 103, 245, 50],
        )
        self.assertEqual(len(tasks), 1307)
        for key, task in tasks.items():
            self.assertNotIn("ground_truth", task)
            self.assertNotIn("published_target", task)
            self.assertNotIn("apis", task)
            self.assertEqual(
                answers[key]["ground_truth"]["role"],
                "API" if task["kind"] == "api" else "AI",
            )
        data = Path(value["protocol"]["data_root"])
        for row in json.loads((data / "level-2-api.json").read_text()):
            loaded = tasks["level_2_api", point_id(row["file"], row["id"])]
            self.assertEqual(
                (loaded["instruction"], loaded["input"]),
                (row["instruction"], row["input"]),
            )
        # The official split can place API targets at odd IDs: never infer role by parity.
        self.assertTrue(
            any(
                task["source_id"]["id"] % 2 == 1
                for (category, _), task in tasks.items()
                if category == "level_2_api"
            )
        )
        preview, _ = load_tasks(self.config)
        self.assertEqual(len(preview), 6)

    def test_invalid_publication_and_icl_mappings_are_rejected(self):
        source = Path(self.config["protocol"]["data_root"])
        copied = self.directory / "test-data"
        shutil.copytree(source, copied)
        value = deepcopy(self.config)
        value["protocol"]["data_root"] = str(copied)
        cases = [
            ("level-1-api.json", lambda rows: [rows[0], *rows]),
            (
                "level-1-api.json",
                lambda rows: [
                    {k: v for k, v in rows[0].items() if k != "expected_output"},
                    *rows[1:],
                ],
            ),
            ("level-1-api.json", lambda rows: [{**rows[0], "id": True}, *rows[1:]]),
            (
                "level-1-api.json",
                lambda rows: [{**rows[0], "file": "missing.jsonl"}, *rows[1:]],
            ),
            (
                "level-3-batch-inf-response.json",
                lambda rows: [{**rows[0], "api_id": 0}, *rows[1:]],
            ),
            (
                "level-3-batch-inf-icl.json",
                lambda rows: [{**rows[0], "output": "changed"}, *rows[1:]],
            ),
        ]
        for filename, corrupt in cases:
            with self.subTest(filename=filename, corrupt=corrupt):
                text = (source / filename).read_text()
                (copied / filename).write_text(json.dumps(corrupt(json.loads(text))))
                try:
                    with self.assertRaises((ValueError, KeyError)):
                        load_tasks(value)
                finally:
                    (copied / filename).write_text(text)

    def test_official_l12_calculation_and_database_lookup(self):
        fixtures = [
            (
                "[Calculator(formula='(1 + 2) * 3')]",
                target("Calculator", {"formula": "(1 + 2) * 3"}, 9.0),
            ),
            (
                "[QueryStock(stock_code='AAPL', date='2022-01-01')]",
                target(
                    "QueryStock", {"stock_code": "AAPL", "date": "2022-01-01"}, 180.5
                ),
            ),
        ]
        with official_runtime(self.config, 1):
            for prediction, gold in fixtures:
                with self.subTest(prediction=prediction):
                    original = deepcopy(gold)
                    manager = make_manager(1)
                    self.assertTrue(score_api(manager, 1, gold, prediction)["correct"])
                    self.assertEqual(gold, original)
            calculator = fixtures[0][1]
            self.assertFalse(
                score_api(
                    make_manager(2), 2, calculator, "[Calculator(formula='1 + 2')]"
                )["correct"]
            )
            mismatch = score_api(
                make_manager(2),
                2,
                calculator,
                "[QueryStock(stock_code='AAPL', date='2022-01-01')]",
            )
            self.assertEqual(mismatch["reason"], "API_NAME_MISMATCH")

    def test_l12_predictions_match_official_evaluator_or_exceptions(self):
        with official_runtime(self.config, 1):
            from evaluator_by_json import Evaluator, Sample

            gold = target("Calculator", {"formula": "1+2"}, 3.0)
            oracle = Evaluator([Sample([], {"Calculator"}, gold)])
            for prediction in (
                "[Calculator(formula='1+2')]",
                "[Calculator(formula='1+3')]",
                "[QueryStock(stock_code='AAPL', date='2022-01-01')]",
            ):
                with self.subTest(prediction=prediction):
                    expected, _ = oracle.evaluate(0, prediction)
                    actual = score_api(make_manager(1), 1, gold, prediction)
                    self.assertEqual(actual["correct"], expected)
            for formula, reason in (
                ("1+3", "OUTPUT_MISMATCH"),
                ("2+1", "INPUT_MISMATCH"),
                ("1/0", "HAS_EXCEPTION"),
            ):
                prediction = f"[Calculator(formula='{formula}')]"
                expected, result = oracle.evaluate(0, prediction)
                actual = score_api(make_manager(1), 1, gold, prediction)
                self.assertEqual(actual["correct"], expected)
                self.assertEqual(actual["result"], result)
                self.assertEqual(actual["reason"], reason)
            # The official JSON driver maps these two exceptions at its outer loop.
            for prediction, error, message, reason in (
                (
                    "[Calculator()]",
                    TypeError,
                    "required positional argument",
                    "MISS_INPUT_ARGUMENT",
                ),
                (
                    "[Calculator(extra='x')]",
                    AssertionError,
                    "invalid parameter name",
                    "INVALID_INPUT_PARAMETER",
                ),
            ):
                with self.subTest(prediction=prediction):
                    with self.assertRaisesRegex(error, message):
                        oracle.evaluate(0, prediction)
                    self.assertEqual(
                        score_api(make_manager(1), 1, gold, prediction)["reason"],
                        reason,
                    )

    def test_batch_manager_lifecycle_order_and_official_sample_denominator(self):
        tasks, answers, predictions = {}, {}, {}
        for level in (1, 2, 3):
            count = 245 if level == 3 else 2
            for index in range(count):
                identity = (f"level_{level}_api", str(index))
                tasks[identity] = {
                    "level": level,
                    "kind": "api",
                    "source_id": {"sample_id": index % 50},
                }
                answers[identity] = {"ground_truth": {"index": index}}
                predictions[identity] = (
                    "bad" if level == 3 and index in (0, 50, 1) else "ok"
                )
        managers, observed = [], []

        def manager(level):
            token = object()
            managers.append((level, token))
            return token

        def score(manager, level, gold, prediction):
            observed.append((level, manager, gold["index"]))
            if level == 1 and gold["index"] == 0:
                return {
                    "correct": False,
                    "reason": "INVALID_PARAMETER_TYPE",
                    "result": None,
                    "protocol_events": {"apibank_parameter_type_error": 1},
                }
            return {"correct": prediction == "ok", "reason": "fixture", "result": None}

        with (
            patch(
                "jev_bench.benchmarks.apibank.load_tasks", return_value=(tasks, answers)
            ),
            patch(
                "jev_bench.benchmarks.apibank_scoring.official_runtime",
                side_effect=lambda *args: nullcontext(),
            ),
            patch(
                "jev_bench.benchmarks.apibank_scoring.make_manager", side_effect=manager
            ),
            patch("jev_bench.benchmarks.apibank_scoring.score_api", side_effect=score),
        ):
            first = score_batch(self.config, predictions)
            self.assertEqual(
                first["protocol_events"], {"apibank_parameter_type_error": 1}
            )
            first_l3 = managers[-1][1]
            self.assertEqual([level for level, _ in managers], [1, 1, 2, 2, 3])
            self.assertEqual(
                [i for level, _, i in observed if level == 3], list(range(245))
            )
            self.assertTrue(
                all(token is first_l3 for level, token, _ in observed if level == 3)
            )
            self.assertEqual(
                first["level_3_sample_accuracy"],
                {
                    "complete": True,
                    "denominator": 50,
                    "accuracy": 48 / 50,
                    "failed_sample_ids": [0, 1],
                },
            )
            second = score_batch(self.config, predictions)
            self.assertEqual(second["protocol_events"], first["protocol_events"])
            self.assertIsNot(first_l3, managers[-1][1])
            missing = predictions.copy()
            missing.pop(next(iter(missing)))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                score_batch(self.config, missing)
            # A partial selected corpus cannot masquerade as the official /50 metric.
            key = ("level_3_api", "244")
            tasks.pop(key)
            answers.pop(key)
            predictions.pop(key)
            partial = score_batch(self.config, predictions)
            self.assertIsNone(partial["level_3_sample_accuracy"]["accuracy"])

    def test_official_l3_correct_incorrect_and_explicit_errors(self):
        gold = target("TaxCalculator", {"salary": 5000.0}, {"salary_after_tax": 3500.0})
        with official_runtime(self.config, 3):
            self.assertTrue(
                score_api(make_manager(3), 3, gold, "[TaxCalculator(salary='5000')]")[
                    "correct"
                ]
            )
            self.assertFalse(
                score_api(make_manager(3), 3, gold, "[TaxCalculator(salary='3000')]")[
                    "correct"
                ]
            )
            for prediction, error, message in (
                ("not a call", ValueError, "no API call"),
                ("[TaxCalculator()]", TypeError, "salary"),
                (
                    "[TaxCalculator(salary='bad')]",
                    AssertionError,
                    "invalid parameter type",
                ),
                (
                    "[TaxCalculator(salary='5000', extra='x')]",
                    AssertionError,
                    "invalid parameter name",
                ),
                ("[MissingTool(salary='5000')]", Exception, "invalid tool name"),
                ("[Calculator(formula='1 + 2')]", AssertionError, "API name"),
            ):
                with self.subTest(prediction=prediction):
                    self.assertFalse(
                        score_api(make_manager(3), 3, gold, prediction)["correct"]
                    )

    def test_get_today_null_is_preserved_and_missing_is_not_null(self):
        from apis.get_today import GetToday

        gold = {"role": "API", "api_name": "GetToday", "result": GetToday().call()}
        self.assertIsNone(gold["result"]["input"])
        with official_runtime(self.config, 1):
            self.assertTrue(
                score_api(make_manager(1), 1, gold, "[GetToday()]")["correct"]
            )
            for value in ([], 0, ""):
                invalid = deepcopy(gold)
                invalid["result"]["input"] = value
                with self.assertRaises(TypeError):
                    score_api(None, 1, invalid, "[GetToday()]")
            del gold["result"]["input"]
            with self.assertRaises(KeyError):
                score_api(None, 1, gold, "[GetToday()]")

    def test_published_json_error_map_and_valid_exception_target(self):
        gold = target("Calculator", {"formula": "1+2"}, 3.0)
        with official_runtime(self.config, 1):
            for prediction, reason in (
                ("no call", "NO_API_CALL"),
                ("[Calculator()]", "MISS_INPUT_ARGUMENT"),
                ("[Calculator(formula='1+2',extra='x')]", "INVALID_INPUT_PARAMETER"),
            ):
                self.assertEqual(
                    score_api(make_manager(1), 1, gold, prediction)["reason"], reason
                )
            exception_gold = target(
                "Calculator", {"formula": "bad"}, None, exception="invalid formula"
            )
            self.assertTrue(
                score_api(
                    make_manager(1), 1, exception_gold, "[Calculator(formula='bad')]"
                )["correct"]
            )
            from evaluator_by_json import Evaluator, Sample

            type_gold = target("BookHotel", {}, None)
            prediction = "[BookHotel(room_count='bad')]"
            with self.assertRaisesRegex(AssertionError, "invalid parameter type"):
                Evaluator([Sample([], {"BookHotel"}, type_gold)]).evaluate(
                    0, prediction
                )
            for level in (1, 2):
                result = score_api(make_manager(level), level, type_gold, prediction)
                self.assertFalse(result["correct"])
                self.assertEqual(result["reason"], "INVALID_PARAMETER_TYPE")
                self.assertEqual(
                    result["protocol_events"], {"apibank_parameter_type_error": 1}
                )
                manager = make_manager(level)
                checker = manager.init_tool("Calculator")
                with (
                    patch.object(
                        checker,
                        "check_api_call_correctness",
                        side_effect=AssertionError("invalid parameter type."),
                    ),
                    self.assertRaises(AssertionError),
                ):
                    score_api(manager, level, gold, "[Calculator(formula='1+2')]")
            for level in (1, 3):
                with patch.object(
                    make_manager(1), "api_call", side_effect=RuntimeError("fixture")
                ) as call:
                    manager = type("Manager", (), {"api_call": call})()
                    with self.assertRaisesRegex(RuntimeError, "fixture"):
                        score_api(manager, level, gold, "[Calculator(formula='1+2')]")

    def test_mutating_checker_cannot_change_score_targets(self):
        from lv3_apis.add_meeting import AddMeeting

        gold = target(
            "AddMeeting",
            {
                "meeting_topic": "test",
                "start_time": "2023-01-01 10:00:00",
                "end_time": "2023-01-01 11:00:00",
                "location": "office",
                "attendees": ["a"],
            },
            "success",
        )
        # Use the actual checker with its required result fields; the manager is
        # scripted here to isolate its documented input mutation from tool state.
        original = deepcopy(gold)
        result = deepcopy(gold["result"])
        manager = type(
            "ScriptedManager",
            (),
            {
                "api_call": lambda self, name, **kwargs: result,
                "init_tool": lambda self, name: AddMeeting(),
            },
        )()
        try:
            self.assertTrue(score_api(manager, 3, gold, "[AddMeeting()]")["correct"])
        finally:
            self.assertEqual(gold, original)
            self.assertEqual(result, original["result"])

    def test_response_rouge_is_separate_from_api_accuracy(self):
        from evaluator_by_json import calculate_rouge_l_score as l12_rouge
        from lv3_evaluator import calculate_rouge_l_score as l3_rouge

        for level, official in ((1, l12_rouge), (2, l12_rouge), (3, l3_rouge)):
            scored = score_response(level, "the cat sat on the mat", "the cat sat")
            self.assertEqual(
                scored,
                {
                    "rouge_l_f": round(
                        official("the cat sat on the mat", "the cat sat"), 4
                    )
                    if level < 3
                    else official("the cat sat on the mat", "the cat sat")
                },
            )
            self.assertNotIn("correct", scored)
        self.assertEqual(score_response(3, "reference text", ""), {"rouge_l_f": 0})
        self.assertEqual(score_response(1, "reference text", ""), {"rouge_l_f": 0})
        self.assertEqual(
            score_response(3, "the cat sat on the mat", "AI: the cat sat User:"),
            {"rouge_l_f": l3_rouge("the cat sat on the mat", "the cat sat")},
        )

    def test_local_retriever_and_runtime_restoration(self):
        from apis import tool_search
        from sentence_transformers import SentenceTransformer

        original_constructor = tool_search.SentenceTransformer
        original_directory = Path.cwd()
        previous_environment = {
            key: os.environ.get(key)
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "CUDA_VISIBLE_DEVICES",
            )
        }
        calls = []

        def observed(*args, **kwargs):
            calls.append((args, kwargs))
            return SentenceTransformer(*args, **kwargs)

        with (
            patch("sentence_transformers.SentenceTransformer", side_effect=observed),
            official_runtime(self.config, 2),
        ):
            manager = make_manager(2)
            result = manager.api_call(
                "ToolSearcher", keywords="calculator arithmetic mathematical expression"
            )
            self.assertIsNone(result["exception"])
            self.assertEqual(result["output"]["name"], "Calculator")
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0],
                ((self.config["protocol"]["retriever_model_path"],), {"device": "cpu"}),
            )
        self.assertIs(tool_search.SentenceTransformer, original_constructor)
        self.assertEqual(Path.cwd(), original_directory)
        self.assertEqual(
            {key: os.environ.get(key) for key in previous_environment},
            previous_environment,
        )

    def test_runtime_restores_state_after_failure(self):
        from apis import tool_search

        previous_directory = Path.cwd()
        previous_constructor = tool_search.SentenceTransformer
        with (
            self.assertRaisesRegex(RuntimeError, "fixture"),
            official_runtime(self.config, 1),
        ):
            raise RuntimeError("fixture")
        self.assertIs(tool_search.SentenceTransformer, previous_constructor)
        self.assertEqual(Path.cwd(), previous_directory)

    def test_required_targets_and_native_result_schema(self):
        with official_runtime(self.config, 1):
            manager = make_manager(1)
            username = next(iter(manager.init_databases["Account"]))
            token = manager.init_databases["Account"][username]["token"]
            gold = {
                "role": "API",
                "api_name": "CheckToken",
                "result": {
                    "input": {"token": token},
                    "output": {"username": username},
                    "exception": None,
                },
            }
            prediction = f"[CheckToken(token='{token}')]"
            checked = score_api(manager, 1, gold, prediction)
            self.assertTrue(checked["correct"])
            self.assertNotIn("api_name", checked["result"])
            for field in ("role", "api_name", "result"):
                invalid = {key: value for key, value in gold.items() if key != field}
                with self.assertRaises(KeyError):
                    score_api(manager, 1, invalid, prediction)
            for field in ("input", "output", "exception"):
                invalid = deepcopy(gold)
                del invalid["result"][field]
                with self.assertRaises(KeyError):
                    score_api(manager, 1, invalid, "[Calculator(formula='1 + 2')]")

    def test_absent_proxy_has_scoped_translate_compatibility(self):
        with patch.dict(os.environ):
            os.environ.pop("ALL_PROXY", None)
            with official_runtime(self.config, 1):
                self.assertEqual(os.environ["ALL_PROXY"], "")
            self.assertNotIn("ALL_PROXY", os.environ)
            os.environ["ALL_PROXY"] = "http://proxy.fixture.invalid:80"
            with official_runtime(self.config, 1):
                self.assertEqual(
                    os.environ["ALL_PROXY"], "http://proxy.fixture.invalid:80"
                )
            self.assertEqual(os.environ["ALL_PROXY"], "http://proxy.fixture.invalid:80")


if __name__ == "__main__":
    unittest.main()
