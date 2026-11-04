import json
import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from jev_bench.advice import (
    NEUTRAL_MESSAGE,
    PREFER_TOOL,
    AdvisorServiceError,
    HTTPResponseError,
    JevClient,
    advise,
    choose_hint,
)
from jev_bench.client import (
    CompletionResponseError,
    ContextLimitError,
    QwenClient,
    SimulatorClient,
    is_context_limit_error,
)
from openai import BadRequestError, InternalServerError, OpenAI

from tests.support import advisor_settings, completion, config, prediction, sdk

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
HISTORY = [
    {"role": "system", "content": "Follow the user."},
    {"role": "user", "content": "Hello"},
]


def spawned_advisor_error():
    raise AdvisorServiceError(
        "invalid advice", {"status": "advisor_error", "raw_response": {"top1": "wrong"}}
    )


def spawned_sdk_error(status):
    from jev_bench.benchmarks import run_episode

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status,
            json={"error": {"message": "backend failure", "type": "server_error"}},
        )
    )
    with OpenAI(
        api_key="cpu",
        base_url="http://127.0.0.1:1/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    ) as client:

        def fail(*args):
            return client.chat.completions.create(
                model="fixture", messages=[{"role": "user", "content": "fixture"}]
            )

        with patch(
            "jev_bench.benchmarks.adapter",
            return_value=SimpleNamespace(run_episode=fail),
        ):
            run_episode({})


class AdviceTests(unittest.TestCase):
    def test_official_simulator_disables_thinking_without_effort(self):
        from jev_bench.config import PROJECT_ROOT, load_config

        settings = load_config(PROJECT_ROOT / "configs/tau2_native_smoke.json")[
            "user_simulator"
        ]
        requests, events = [], []

        def create(**request):
            requests.append(request)
            return completion()

        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "cpu-key"}),
            patch("jev_bench.client.OpenAI", return_value=sdk(create)) as factory,
        ):
            client = SimulatorClient(
                settings, lambda kind, record: events.append(record)
            )
        self.assertEqual(
            factory.call_args.kwargs["base_url"], "https://api.deepseek.com"
        )
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        for tools in (None, TOOLS):
            client.complete(messages=HISTORY, tools=tools)
        for request in requests:
            self.assertEqual(request["model"], "deepseek-flash")
            self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertEqual(request["temperature"], 0)
            self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("tools", requests[0])
        self.assertEqual(requests[1]["tool_choice"], "auto")
        self.assertNotIn("cpu-key", json.dumps(events))
        client.close()

    def test_health_requires_duplicate_candidate_capability(self):
        client = JevClient(advisor_settings())
        healthy = {**prediction(), "status": "ready"}
        with patch("jev_bench.advice.json_request", return_value=healthy):
            self.assertEqual(client.health()["duplicate_candidates"], "positional")
        for value in (None, False, True, 1, "summed"):
            bad = {**healthy, "duplicate_candidates": value}
            with (
                patch("jev_bench.advice.json_request", return_value=bad),
                self.assertRaises(ValueError),
            ):
                client.health()
        del healthy["duplicate_candidates"]
        with (
            patch("jev_bench.advice.json_request", return_value=healthy),
            self.assertRaises(KeyError),
        ):
            client.health()

    def test_duplicate_positions_are_validated_then_summed_by_name(self):
        client = JevClient(advisor_settings())
        names = ("dup", "other", "dup", "RESPOND")
        tools = [
            {
                "type": "function",
                "function": {
                    **TOOLS[0]["function"],
                    "name": name,
                    "description": str(index),
                },
            }
            for index, name in enumerate(names[:-1])
        ]
        wire = prediction(names, (0.26, 0.28, 0.25, 0.21))
        original = deepcopy(wire)
        with patch("jev_bench.advice.json_request", return_value=wire) as request:
            actual = client.predict(HISTORY, tools)
        self.assertEqual(wire, original)
        self.assertEqual(
            request.call_args.args[2]["tools"], [tool["function"] for tool in tools]
        )
        self.assertEqual(actual["raw_probabilities"], wire["probabilities"])
        self.assertEqual(actual["raw_top1"], "other")
        self.assertEqual(actual["top1"], "dup")
        self.assertEqual(actual["aggregation"], "sum_by_tool_name")
        self.assertEqual(
            [row["name"] for row in actual["probabilities"]],
            ["dup", "other", "RESPOND"],
        )
        self.assertAlmostEqual(actual["probabilities"][0]["probability"], 0.51)
        policy = {
            "advice": "gate",
            "gate": {"rule": "top1_prob_lt", "threshold": 0.3, "loop_guard": True},
        }
        hint, gate = choose_hint(policy, actual, [])
        self.assertEqual(hint, PREFER_TOOL.format(name="dup"))
        self.assertEqual(gate["triggered"], "none")
        self.assertEqual(
            choose_hint(policy, actual, [{"name": "dup", "arguments": {}}] * 2)[1][
                "triggered"
            ],
            "loop",
        )
        for damaged in (
            {**wire, "top1": "dup"},
            {**wire, "probabilities": list(reversed(wire["probabilities"]))},
        ):
            with (
                patch("jev_bench.advice.json_request", return_value=damaged),
                self.assertRaises(ValueError),
            ):
                client.predict(HISTORY, tools)

    def test_aggregation_changes_respond_bypass_and_preserves_first_name_ties(self):
        client = JevClient(advisor_settings())
        for names, values, expected in (
            (("dup", "dup", "RESPOND"), (0.30, 0.29, 0.41), "dup"),
            (("b", "a", "b", "a", "RESPOND"), (0.2, 0.3, 0.3, 0.2, 0), "b"),
        ):
            tools = [
                {"type": "function", "function": {**TOOLS[0]["function"], "name": name}}
                for name in names[:-1]
            ]
            with patch(
                "jev_bench.advice.json_request", return_value=prediction(names, values)
            ):
                result = client.predict(HISTORY, tools)
            self.assertEqual(result["top1"], expected)
            if expected == "dup":
                self.assertEqual(result["raw_top1"], "RESPOND")
                hint, gate = choose_hint(
                    {
                        "advice": "gate",
                        "gate": {
                            "rule": "respond_prob_ge",
                            "threshold": 0.1,
                            "loop_guard": True,
                        },
                    },
                    result,
                    [],
                )
                self.assertEqual(hint, NEUTRAL_MESSAGE)
                self.assertEqual(gate["triggered"], "confidence")

    def test_exact_overlength_fallback_keeps_errors_cost_and_null_match(self):
        from jev_bench.cost import costs
        from jev_bench.stats import aggregate_cost

        client = JevClient(advisor_settings())
        body = {"error": {"type": "ValueError", "message": "over_length_after_drop"}}
        records = []
        for mode in ("prefer", "gate"):
            with patch(
                "jev_bench.advice.json_request",
                side_effect=HTTPResponseError(422, body),
            ) as request:
                record = advise({"advice": mode}, client, HISTORY, TOOLS, [])
            request.assert_called_once()
            self.assertEqual(record["status"], "neutral_fallback")
            self.assertEqual(record["fallback"], "over_length_after_drop_neutral")
            self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
            self.assertEqual(record["http_error"], {"status_code": 422, "body": body})
            for field in ("prediction", "gate", "first_action_matches"):
                self.assertIsNone(record[field])
            records.append(record)
        with patch("jev_bench.advice.json_request", return_value=prediction()):
            records.append(advise({"advice": "prefer"}, client, HISTORY, TOOLS, []))
        cost = costs(
            None,
            SimpleNamespace(advisor_records=records),
            advisor_type="jev_local",
        )
        self.assertEqual(cost["advisor"]["requests"], 3)
        self.assertEqual(cost["advisor"]["over_length_after_drop_neutral"], 2)
        self.assertEqual(cost["advisor"]["prompt_tokens"]["missing"], 2)
        combined = aggregate_cost(
            [
                {
                    "cost": cost,
                    "response_anomalies": [],
                    "duration_seconds": 1,
                    "decision_steps": 3,
                    "tool_call_count": 0,
                }
            ]
            * 2
        )
        self.assertEqual(combined["advisor"]["over_length_after_drop_neutral"], 4)
        for status, detail in (
            (503, body),
            (400, body),
            (
                422,
                {"error": {"type": "TypeError", "message": "over_length_after_drop"}},
            ),
            (
                422,
                {
                    "error": {
                        "type": "ValueError",
                        "message": "prefix over_length_after_drop",
                    }
                },
            ),
            (422, "over_length_after_drop"),
        ):
            with (
                patch(
                    "jev_bench.advice.json_request",
                    side_effect=HTTPResponseError(status, detail),
                ),
                self.assertRaises(AdvisorServiceError),
            ):
                advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])

    def test_real_sdk_fatal_errors_survive_spawn(self):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for status in (401, 500, 502, 503, 504):
                expected = (
                    "AuthenticationError" if status == 401 else "InternalServerError"
                )
                with (
                    self.subTest(status=status),
                    self.assertRaisesRegex(RuntimeError, expected),
                ):
                    pool.submit(spawned_sdk_error, status).result(timeout=30)
                self.assertEqual(pool.submit(pow, 2, 3).result(timeout=30), 8)

    def test_advisor_error_survives_spawn_without_breaking_pool(self):
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            with self.assertRaises(AdvisorServiceError) as error:
                pool.submit(spawned_advisor_error).result(timeout=30)
            self.assertEqual(str(error.exception), "invalid advice")
            self.assertEqual(
                error.exception.record,
                {"status": "advisor_error", "raw_response": {"top1": "wrong"}},
            )
            self.assertEqual(pool.submit(pow, 2, 3).result(timeout=30), 8)

    def test_invalid_response_is_preserved_and_incomplete_http_is_explicit(self):
        client = JevClient(advisor_settings())
        damaged = {**prediction(), "top1": "wrong"}
        with (
            patch("jev_bench.advice.json_request", return_value=damaged),
        ):
            record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(record["raw_response"], damaged)
        self.assertIsNone(record["prediction"])
        self.assertEqual(record["fallback"], "advisor_format_error_neutral")
        from jev_bench.cost import local_advisor_cost

        accounting = local_advisor_cost([record])
        self.assertEqual(
            accounting["prompt_tokens"],
            {"observed_total": 30, "known": 1, "missing": 0},
        )
        self.assertEqual(accounting["model_latency_seconds"]["observed_total"], 0.01)
        with (
            patch(
                "jev_bench.advice.json_request", side_effect=IncompleteRead(b"partial")
            ),
            self.assertRaises(AdvisorServiceError),
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])

    def test_reserved_input_fails_before_request(self):
        tools = deepcopy(TOOLS)
        tools[0]["function"]["name"] = "RESPOND"
        client = JevClient(advisor_settings())
        with (
            patch("jev_bench.advice.json_request") as request,
            self.assertRaises(AdvisorServiceError),
        ):
            advise({"advice": "prefer"}, client, HISTORY, tools, [])
        request.assert_not_called()

    def test_truncation_flag_and_sparse_counter_contract(self):
        client = JevClient(advisor_settings())
        valid = {**prediction(), "messages_dropped": 3, "truncated": True}
        with patch("jev_bench.advice.json_request", return_value=valid):
            self.assertTrue(client.predict(HISTORY, TOOLS)["truncated"])

    def test_damaged_metadata_cannot_become_neutral_advice(self):
        client = JevClient(advisor_settings())
        missing = object()
        cases = [
            ("n_prompt_tokens", 32705),
            ("n_prompt_tokens", True),
            ("latency_seconds", float("nan")),
            ("latency_seconds", -0.01),
            ("truncated", 0),
            ("messages_dropped", 3),
            ("truncation_counts", {}),
        ] + [(field, missing) for field in (
            "n_prompt_tokens", "truncated", "messages_dropped", "truncation_counts", "latency_seconds",
        )]
        for field, value in cases:
            # An invalid action must not mask missing or damaged metadata.
            for top1 in ("echo", "wrong"):
                damaged = {**prediction(), "top1": top1}
                if value is missing:
                    del damaged[field]
                else:
                    damaged[field] = value
                with self.subTest(field=field, value=value, top1=top1):
                    with (
                        patch("jev_bench.advice.json_request", return_value=damaged),
                        self.assertRaises(AdvisorServiceError) as caught,
                    ):
                        advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
                    record = caught.exception.record
                    self.assertEqual(record["status"], "advisor_error")
                    self.assertIs(record["raw_response"], damaged)
                    for key in ("fallback", "hint", "prediction", "validated_metadata", "usage"):
                        self.assertIsNone(record[key])

    def test_required_response_fields(self):
        client = JevClient(advisor_settings())
        base = prediction()
        # Metadata omissions are covered through advise(), including no fallback.
        for field in base.keys() - {
            "n_prompt_tokens",
            "truncated",
            "messages_dropped",
            "truncation_counts",
            "latency_seconds",
        }:
            damaged = deepcopy(base)
            del damaged[field]
            with (
                self.subTest(field=field),
                patch("jev_bench.advice.json_request", return_value=damaged),
                self.assertRaises((ValueError, KeyError, TypeError)),
            ):
                client.predict(HISTORY, TOOLS)

    def test_invalid_fields_and_identity(self):
        client = JevClient(advisor_settings())
        for field, value in [
            ("model", "/wrong"),
            ("temperature", 1.4),
            ("temperature_file", "/wrong"),
            ("top1", "RESPOND"),
        ]:
            result = {**prediction(), field: value}
            with (
                self.subTest(field=field),
                patch("jev_bench.advice.json_request", return_value=result),
                self.assertRaises(ValueError),
            ):
                client.predict(HISTORY, TOOLS)
        for value in [True, float("nan"), float("inf"), -0.1]:
            result = prediction()
            result["probabilities"][0]["probability"] = value
            with (
                patch("jev_bench.advice.json_request", return_value=result),
                self.assertRaises(ValueError),
            ):
                client.predict(HISTORY, TOOLS)

    def test_http_failure_retains_status_and_structured_cause(self):
        client = JevClient(advisor_settings())
        body = {"error": {"type": "ValueError", "message": "too_many_options"}}
        for status, expected in (
            (422, "advisor_error"),
            (429, "service_error"),
            (503, "service_error"),
        ):
            with (
                patch(
                    "jev_bench.advice.json_request",
                    side_effect=HTTPResponseError(status, body),
                ),
                self.assertRaises(AdvisorServiceError) as caught,
            ):
                advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            self.assertEqual(caught.exception.record["status"], expected)
            self.assertEqual(
                caught.exception.record["http_error"],
                {"status_code": status, "body": body},
            )

    def test_gate_boundaries_and_respond_bypass(self):
        for p, expected in [(0.1, "confidence"), (0.09999, "none")]:
            policy = {
                "advice": "gate",
                "gate": {
                    "rule": "respond_prob_ge",
                    "threshold": 0.1,
                    "loop_guard": True,
                },
            }
            hint, details = choose_hint(policy, prediction(values=(1 - p, p)), [])
            self.assertEqual(details["triggered"], expected)
            self.assertEqual(hint == NEUTRAL_MESSAGE, expected != "none")
        for top, expected in [(0.3, "none"), (0.29999, "confidence")]:
            policy["gate"] = {
                "rule": "top1_prob_lt",
                "threshold": 0.3,
                "loop_guard": True,
            }
            pred = prediction(
                ("echo", "b", "c", "RESPOND"), (top, 0.28, 0.27, 0.45 - top)
            )
            self.assertEqual(choose_hint(policy, pred, [])[1]["triggered"], expected)
        self.assertNotEqual(
            choose_hint(policy, prediction(values=(0.1, 0.9)), [])[0], NEUTRAL_MESSAGE
        )

    def test_loop_json_semantics_and_precedence(self):
        policy = {
            "advice": "gate",
            "gate": {"rule": "respond_prob_ge", "threshold": 0.1, "loop_guard": True},
        }
        calls = [
            {"name": "echo", "arguments": {"a": 1, "b": 2}},
            {"name": "echo", "arguments": {"b": 2, "a": 1}},
        ]
        self.assertEqual(
            choose_hint(policy, prediction(), calls)[1]["triggered"], "loop"
        )
        calls[1]["arguments"]["a"] = 1.0
        self.assertEqual(
            choose_hint(policy, prediction(), calls)[1]["triggered"], "confidence"
        )


class ClientTests(unittest.TestCase):
    def test_missing_message_is_a_model_response_error(self):
        raw = {"choices": [{"message": None}], "usage": None}
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=None)], model_dump=lambda **kwargs: raw
        )
        with patch(
            "jev_bench.client.OpenAI", return_value=sdk(lambda **kwargs: response)
        ):
            client = QwenClient(config()["executors"]["qwen4b"], lambda *args: None)
        with self.assertRaises(CompletionResponseError):
            client.complete(
                messages=HISTORY,
                tools=None,
                tool_choice=None,
                max_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                extra_body={},
            )
        self.assertEqual(client.records[0]["response"], raw)

    def test_sdk_request_preserves_sampling_and_optional_tool_fields(self):
        requests = []
        mock_sdk = sdk(lambda **kwargs: requests.append(deepcopy(kwargs)) or completion())
        for model in ("qwen3-4b", "star-4b"):
            settings = {**config()["executors"]["qwen4b"], "model": model}
            with patch("jev_bench.client.OpenAI", return_value=mock_sdk):
                client = QwenClient(settings, lambda *args: None)
            for tools in (TOOLS, None):
                with self.subTest(model=model, tools=tools is not None):
                    request = {
                        "messages": HISTORY,
                        "tools": tools,
                        "tool_choice": "auto" if tools else None,
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "max_tokens": 8192,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": True},
                            "top_k": 20,
                            "min_p": 0,
                        },
                    }
                    expected = {"model": model, **request}
                    if tools is None:
                        del expected["tools"]
                        del expected["tool_choice"]
                    client.complete(**request)
                    self.assertEqual(requests[-1], expected)
                    self.assertNotIn("stream", requests[-1])

    def test_context_error_requires_exact_evidence(self):
        # vLLM 0.19.1 renderers/params.py: text precheck and token check.
        # The character bound here is a fixture, not a tokenizer measurement.
        prefix = "This model's maximum context length is 131072 tokens. However, you requested 8192 output tokens and your prompt contains "
        suffix = "Please reduce the length of the input prompt or the number of requested output tokens."
        token_message = (
            prefix
            + "at least 122881 input tokens, for a total of at least 131073 tokens. "
            + suffix
            + " (parameter=input_tokens, value=122881)"
        )
        text_message = (
            prefix
            + "491521 characters (more than 491520 characters, which is the upper bound for 122880 input tokens). "
            + suffix
            + " (parameter=input_text, value=491521)"
        )
        cases = (
            (400, "input_tokens", token_message, ContextLimitError),
            (400, "input_text", text_message, ContextLimitError),
            (400, "max_tokens", "max_tokens must be at least 1", BadRequestError),
            (400, "input_text", "invalid chat template", BadRequestError),
            (500, "input_tokens", token_message, InternalServerError),
        )
        settings = config()["executors"]["qwen4b"]
        for status, param, message, error_type in cases:
            with self.subTest(status=status, param=param, message=message):
                body = {
                    "type": "BadRequestError",
                    "param": param,
                    "message": message,
                    "code": status,
                }
                transport = httpx.MockTransport(
                    lambda request, status=status, body=body: httpx.Response(
                        status, json={"error": body}
                    )
                )
                with OpenAI(
                    api_key="test-only",
                    base_url=settings["base_url"],
                    max_retries=0,
                    http_client=httpx.Client(transport=transport),
                ) as sdk_client:
                    with patch("jev_bench.client.OpenAI", return_value=sdk_client):
                        client = QwenClient(settings, lambda *args: None)
                    with self.assertRaises(error_type) as caught:
                        client.complete(
                            messages=HISTORY,
                            tools=None,
                            tool_choice=None,
                            temperature=0.6,
                            top_p=0.95,
                            max_tokens=8192,
                            extra_body={},
                        )
                if error_type is ContextLimitError:
                    self.assertEqual(caught.exception.diagnostics["error_body"], body)
                else:
                    self.assertFalse(is_context_limit_error(caught.exception))
                self.assertEqual(len(client.records), 1)
                self.assertIsNone(client.records[0]["usage"])
                self.assertEqual(
                    client.records[0]["error"]["type"],
                    "BadRequestError" if status == 400 else "InternalServerError",
                )
