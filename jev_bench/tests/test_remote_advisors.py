"""CPU contract tests: no credentials, external requests, models or GPU services."""

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

import httpx
from jev_bench.advice import (
    DATASET_INVENTORY_SRC,
    NEUTRAL_MESSAGE,
    POSITIONAL_TOOL_NAMES_MESSAGE,
    PREFER_RESPOND,
    PREFER_TOOL,
    AdvisorServiceError,
    HTTPResponseError,
    ProviderHTTPError,
    TypeSafeClient,
    advise,
    choose_hint,
    json_request,
    make_advisor,
)
from jev_bench.attempt import recover_interrupted
from jev_bench.config import validate
from jev_bench.cost import costs
from jev_bench.io import append_json, read_jsonl, write_json
from jev_bench.runner import directory_for, run_lock
from jev_bench.scoring import recover_scores
from jev_bench.services import check_services
from jev_bench.stats import aggregate_cost
from openai import OpenAI
from openai.types.completion_usage import CompletionTokensDetails

from tests.support import PROJECT, advisor_settings, completion, config, prediction, sdk
from tests.test_advice_client import HISTORY, TOOLS


def settings(kind):
    return {
        "type": kind,
        "timeout_seconds": 120,
        **(
            {
                "base_url": "https://api.typesafe.ai/v1",
                "model": "jev-1.14.0",
                "api_key_env": "TYPESAFE_API_KEY",
                "render_advisor": "local",
            }
            if kind == "typesafe"
            else {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
                "max_tokens": 16384,
                "reasoning_effort": "high",
            }
        ),
    }


def rendered():
    result = prediction()
    result.pop("probabilities")
    result.pop("top1")
    result["rendered"] = {
        "context": "[user] Hello",
        "question": "Select the next action.",
        "options": ["echo: Echo text (params: text)", "RESPOND: reply without a tool"],
    }
    return result


def typesafe_response():
    return {
        "model": "jev-1.14.0",
        "answers": {
            "next_action": {
                "type": "choice",
                "choice": "echo",
                "confidence": 0.8,
                "probabilities": {"RESPOND": 0.2, "echo": 0.8},
            }
        },
        "usage": {"input_tokens": 50, "output_tokens": 2},
    }


def deepseek_response(reason="stop", content="Done", names=()):
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for i, name in enumerate(names)
    ]
    response = completion(content, calls)
    response.model = "deepseek-v4.1-flash-fixture"
    response.choices[0].finish_reason = reason
    response.usage.completion_tokens_details = CompletionTokensDetails(
        reasoning_tokens=3
    )
    return response


@contextmanager
def http_fixture(response, status=200):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                {
                    "headers": dict(self.headers),
                    "path": self.path,
                    "body": json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    ),
                }
            )
            body = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class RemoteAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            "os.environ",
            {
                "TYPESAFE_API_KEY": "cpu-typesafe-secret",
                "DEEPSEEK_API_KEY": "cpu-deepseek-secret",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def client(self, kind, sink=lambda *_: None):
        value = settings(kind)
        return make_advisor(value, sink, advisors={"local": advisor_settings()})

    def test_duplicate_candidates_are_positional_then_aggregated(self):
        tools = deepcopy(TOOLS * 2)
        tools[1]["function"]["description"] = "A different visible declaration."
        original = deepcopy(tools)
        render = rendered()
        render["rendered"]["options"].insert(1, "echo: second declaration")
        response = typesafe_response()
        response["answers"]["next_action"].update(
            choice="candidate_0001",
            confidence=0.4,
            probabilities={
                "candidate_0000": 0.3,
                "candidate_0001": 0.4,
                "RESPOND": 0.3,
            },
        )
        client = self.client("typesafe")
        with patch(
            "jev_bench.advice.json_request", side_effect=[render, response]
        ) as request:
            prediction = client.predict(HISTORY, tools)
        self.assertEqual(
            list(request.call_args.args[2]["questions"]["next_action"]["criteria"]),
            ["candidate_0000", "candidate_0001", "RESPOND"],
        )
        self.assertEqual(prediction["top1"], "echo")
        self.assertAlmostEqual(prediction["probabilities"][0]["probability"], 0.7)
        self.assertEqual(len(prediction["raw_probabilities"]), 3)
        self.assertEqual(tools, original)

        requests = []

        def create(**request):
            requests.append(request)
            return deepseek_response("tool_calls", None, ("candidate_0001",))

        with patch("jev_bench.client.OpenAI", return_value=sdk(create)):
            client = self.client("deepseek")
            self.addCleanup(client.close)
            prediction = client.predict(HISTORY, tools)
        self.assertEqual(prediction["top1"], "echo")
        self.assertEqual(prediction["raw_top1"], "candidate_0001")
        self.assertEqual(
            [t["function"]["name"] for t in requests[0]["tools"]],
            ["candidate_0000", "candidate_0001"],
        )
        self.assertEqual(
            requests[0]["tools"][1]["function"]["parameters"],
            tools[1]["function"]["parameters"],
        )
        self.assertEqual(requests[0]["messages"][:-1], HISTORY)
        self.assertEqual(
            requests[0]["messages"][-1],
            {"role": "system", "content": POSITIONAL_TOOL_NAMES_MESSAGE},
        )
        self.assertEqual(requests[0]["tool_choice"], "auto")
        self.assertEqual(tools, original)

    def test_deepseek_alias_note_preserves_history_and_strict_response_names(self):
        tools = deepcopy(TOOLS * 4)
        for tool, name in zip(
            tools,
            [
                "personal_info_extractor",
                "digit_extractor",
                "digit_extractor",
                "concatenate_values",
            ],
            strict=True,
        ):
            tool["function"]["name"] = name
        history = HISTORY + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_digit",
                        "type": "function",
                        "function": {"name": "digit_extractor", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_digit", "content": "9"},
        ]
        original = deepcopy(history)
        expected = deepcopy(history)
        expected[-2]["reasoning_content"] = ""
        for name in ("personal_info_extractor", "concatenate_values", "RESPOND"):
            reply = (
                deepseek_response()
                if name == "RESPOND"
                else deepseek_response("tool_calls", None, (name,))
            )
            with (
                self.subTest(name=name),
                patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(lambda reply=reply, **_: reply),
                ),
            ):
                client = self.client("deepseek")
                self.addCleanup(client.close)
                record = advise({"advice": "prefer"}, client, history, tools, [])
            self.assertEqual(history, original)
            self.assertEqual(
                client.records[0]["request"]["messages"],
                expected
                + [{"role": "system", "content": POSITIONAL_TOOL_NAMES_MESSAGE}],
            )
            if name == "RESPOND":
                self.assertEqual(record["prediction"]["top1"], "RESPOND")
                self.assertEqual(record["hint"], PREFER_RESPOND)
                self.assertIsNone(record["fallback"])
            else:
                self.assertEqual(record["fallback"], "advisor_format_error_neutral")
                self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
                self.assertIsNone(record["prediction"])
                self.assertEqual(
                    client.last_response["choices"][0]["message"]["tool_calls"][0][
                        "function"
                    ]["name"],
                    name,
                )

    def test_textual_api_call_is_neutral_and_never_respond(self):
        with patch(
            "jev_bench.client.OpenAI",
            return_value=sdk(
                lambda **_: deepseek_response(
                    "stop", '[ToolSearcher(keywords="weather")]', ()
                )
            ),
        ):
            client = self.client("deepseek")
            self.addCleanup(client.close)
            record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(record["fallback"], "advisor_format_error_neutral")
        self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
        for field in ("prediction", "gate", "first_action_matches"):
            self.assertIsNone(record[field])
        cost = costs(
            None, None, advisor_type="deepseek", advisor_requests=client.records
        )["advisor"]
        self.assertEqual(cost["advisor_format_error_neutral"], 1)
        self.assertEqual(cost["provider_requests"], 1)
        self.assertEqual(cost["completion_tokens"]["observed_total"], 5)

    def test_config_requires_explicit_complete_types(self):
        value = config()
        value["advisors"] = {
            "local": advisor_settings(),
            **{x: settings(x) for x in ("typesafe", "deepseek")},
        }
        self.assertEqual(validate(value), value)
        for name, item in value["advisors"].items():
            for field in item:
                broken = deepcopy(value)
                del broken["advisors"][name][field]
                with (
                    self.subTest(name=name, missing=field),
                    self.assertRaises((ValueError, KeyError)),
                ):
                    validate(broken)
        for field, invalid in (
            ("model", "jev-latest"),
            ("render_advisor", "absent"),
            ("render_advisor", "deepseek"),
            ("base_url", "http://api.typesafe.ai/v1"),
        ):
            broken = deepcopy(value)
            broken["advisors"]["typesafe"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(broken)
        value["conditions"]["native"].update(
            advice="gate",
            advisor="deepseek",
            gate={"rule": "respond_prob_ge", "threshold": 0.1, "loop_guard": True},
        )
        with self.assertRaisesRegex(ValueError, "no candidate probabilities"):
            validate(value)

    def test_missing_keys_fail_before_requests(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("jev_bench.advice.json_request") as request,
        ):
            for kind in ("typesafe", "deepseek"):
                with (
                    self.subTest(kind=kind),
                    self.assertRaisesRegex(ValueError, "missing or empty"),
                ):
                    self.client(kind)
            request.assert_not_called()

    def test_typesafe_preserves_rendered_options_and_order(self):
        client = self.client("typesafe")
        row = rendered()
        with patch(
            "jev_bench.advice.json_request", side_effect=[row, typesafe_response()]
        ) as request:
            result = client.predict(HISTORY, TOOLS)
        body = request.call_args_list[1].args[2]
        self.assertEqual(body["state"], row["rendered"]["context"])
        question = body["questions"]["next_action"]
        self.assertEqual(question["instructions"], row["rendered"]["question"])
        self.assertEqual(list(question["criteria"]), ["echo", "RESPOND"])
        self.assertEqual(
            list(question["criteria"].values()), row["rendered"]["options"]
        )
        self.assertEqual(
            request.call_args.kwargs["headers"],
            {"Authorization": "Bearer cpu-typesafe-secret"},
        )
        self.assertEqual(
            [p["name"] for p in result["probabilities"]], ["echo", "RESPOND"]
        )
        self.assertEqual(
            client.last_usage,
            {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
        )
        self.assertEqual(client.records[0]["local_prompt_tokens"], 30)
        self.assertIs(client.records[0]["provider_called"], True)
        with (
            patch("jev_bench.advice.json_request") as request,
            self.assertRaisesRegex(ValueError, "reserved"),
        ):
            reserved = deepcopy(TOOLS)
            reserved[0]["function"]["name"] = "RESPOND"
            client.predict(HISTORY, reserved)
        request.assert_not_called()
        self.assertIs(client.records[1]["provider_called"], False)
        cost = costs(
            None, None, advisor_type="typesafe", advisor_requests=client.records
        )["advisor"]
        self.assertEqual(cost["requests"], 2)
        self.assertEqual(cost["provider_requests"], 1)
        self.assertEqual(cost["remote_input_tokens"]["missing"], 0)

    def test_typesafe_missing_usage_is_missing_after_provider_call(self):
        response = typesafe_response()
        del response["usage"]
        client = self.client("typesafe")
        with (
            patch("jev_bench.advice.json_request", side_effect=[rendered(), response]),
            self.assertRaises(AdvisorServiceError),
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertIs(client.records[0]["provider_called"], True)
        cost = costs(
            None, None, advisor_type="typesafe", advisor_requests=client.records
        )["advisor"]
        self.assertEqual(cost["requests"], 1)
        self.assertEqual(cost["provider_requests"], 1)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "remote_input_tokens",
            "remote_output_tokens",
        ):
            self.assertEqual(
                cost[field], {"observed_total": 0, "known": 0, "missing": 1}
            )
        # TypeSafe produces no reasoning; the field is omitted, not missing.
        self.assertNotIn("reasoning_tokens", cost)
        self.assertEqual(cost["model_latency_seconds"]["known"], 1)

    def test_provider_call_flag_is_required_in_remote_ledger(self):
        client = self.client("typesafe")
        with patch(
            "jev_bench.advice.json_request",
            side_effect=[rendered(), typesafe_response()],
        ):
            client.predict(HISTORY, TOOLS)
        del client.records[0]["provider_called"]
        with self.assertRaisesRegex(KeyError, "provider_called"):
            costs(None, None, advisor_type="typesafe", advisor_requests=client.records)

    def test_typesafe_rejects_bad_payload_without_losing_known_usage(self):
        mutations = [
            lambda r: r.update(model="jev-latest"),
            lambda r: r["answers"]["next_action"].update(choice="absent"),
            lambda r: r["answers"]["next_action"].update(choice="RESPOND"),
            lambda r: r["answers"]["next_action"].update(probabilities={"echo": 1}),
            lambda r: r["answers"]["next_action"]["probabilities"].update(echo=0.1),
            lambda r: r["answers"]["next_action"]["probabilities"].update(echo=True),
            lambda r: r["answers"]["next_action"].pop("confidence"),
        ]
        for index, change in enumerate(mutations):
            response = typesafe_response()
            change(response)
            client = self.client("typesafe")
            with (
                self.subTest(change=change),
                patch(
                    "jev_bench.advice.json_request", side_effect=[rendered(), response]
                ),
            ):
                if index == 0:
                    with self.assertRaises(AdvisorServiceError):
                        advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
                else:
                    record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
                    self.assertEqual(record["fallback"], "advisor_format_error_neutral")
                    self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
                    self.assertIsNone(record["prediction"])
            self.assertEqual(client.last_usage["total_tokens"], 52)
        for key in ("input_tokens", "output_tokens"):
            response = typesafe_response()
            del response["usage"][key]
            with (
                patch(
                    "jev_bench.advice.json_request", side_effect=[rendered(), response]
                ),
                self.assertRaises(AdvisorServiceError),
            ):
                advise(
                    {"advice": "prefer"}, self.client("typesafe"), HISTORY, TOOLS, []
                )

    def test_typesafe_retains_service_choice_on_tie(self):
        response = typesafe_response()
        response["answers"]["next_action"].update(
            choice="RESPOND", probabilities={"echo": 0.5, "RESPOND": 0.5}
        )
        with patch("jev_bench.advice.json_request", side_effect=[rendered(), response]):
            result = self.client("typesafe").predict(HISTORY, TOOLS)
        self.assertEqual(result["top1"], "RESPOND")

    def test_only_renderer_exact_overlength_uses_neutral(self):
        body = {"error": {"type": "ValueError", "message": "over_length_after_drop"}}
        client = self.client("typesafe")
        with patch(
            "jev_bench.advice.json_request", side_effect=HTTPResponseError(422, body)
        ) as request:
            record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
        self.assertIsNone(record["prediction"])
        self.assertIs(client.records[0]["provider_called"], False)
        cost = costs(
            None,
            SimpleNamespace(advisor_records=[record]),
            advisor_type="typesafe",
            advisor_requests=client.records,
        )["advisor"]
        self.assertEqual(cost["requests"], 1)
        self.assertEqual(cost["provider_requests"], 0)
        self.assertEqual(cost["over_length_after_drop_neutral"], 1)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "remote_input_tokens",
            "remote_output_tokens",
            "model_latency_seconds",
        ):
            self.assertEqual(
                cost[field], {"observed_total": 0, "known": 0, "missing": 0}
            )
        with (
            patch(
                "jev_bench.advice.json_request",
                side_effect=[rendered(), HTTPResponseError(422, body)],
            ),
            self.assertRaises(AdvisorServiceError) as caught,
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(caught.exception.record["status"], "advisor_error")
        self.assertIs(client.records[1]["provider_called"], True)

    def test_remote_http_error_retains_reported_usage(self):
        body = {
            "error": {"message": "fixture"},
            "usage": {"input_tokens": 17, "output_tokens": 0},
        }
        client = self.client("typesafe")
        with (
            patch(
                "jev_bench.advice.json_request",
                side_effect=[rendered(), HTTPResponseError(429, body)],
            ),
            self.assertRaises(AdvisorServiceError) as caught,
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(caught.exception.record["status"], "service_error")
        self.assertIs(client.records[0]["provider_called"], True)
        cost = costs(
            None, None, advisor_type="typesafe", advisor_requests=client.records
        )["advisor"]
        self.assertEqual(cost["requests"], 1)
        self.assertEqual(cost["provider_requests"], 1)
        for field, value in (
            ("prompt_tokens", 17),
            ("completion_tokens", 0),
            ("total_tokens", 17),
            ("remote_input_tokens", 17),
            ("remote_output_tokens", 0),
        ):
            self.assertEqual(
                cost[field], {"observed_total": value, "known": 1, "missing": 0}
            )
        self.assertEqual(cost["model_latency_seconds"]["known"], 1)

    def test_remote_http_status_classification(self):
        for status in (400, 401, 403, 422, 429, 503, 529):
            client = self.client("typesafe")
            with (
                self.subTest(status=status),
                patch(
                    "jev_bench.advice.json_request",
                    side_effect=[
                        rendered(),
                        ProviderHTTPError(status, {"error": "fixture"}),
                    ],
                ),
                self.assertRaises(AdvisorServiceError) as caught,
            ):
                advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            expected = (
                "service_error" if status == 429 or status >= 500 else "advisor_error"
            )
            self.assertEqual(caught.exception.record["status"], expected)

    def test_bad_error_usage_keeps_http_classification_and_redacts_keys(self):
        key = "cpu-typesafe-secret"
        body = {
            "error": {"message": "Invalid key: " + key},
            "usage": {"input_tokens": 10},
        }
        client = self.client("typesafe")
        with (
            patch(
                "jev_bench.advice.json_request",
                side_effect=[rendered(), ProviderHTTPError(429, body)],
            ),
            self.assertRaises(AdvisorServiceError) as caught,
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(caught.exception.record["status"], "service_error")
        self.assertIsNone(client.last_usage)
        self.assertIn("usage_error", client.records[0])
        self.assertNotIn(key, str(caught.exception))
        self.assertNotIn(key, json.dumps([client.records, caught.exception.record]))
        with (
            patch.dict("os.environ", {"TYPESAFE_API_KEY": "cpu-key\nInvalid: header"}),
            self.assertRaisesRegex(ValueError, "invalid HTTP credential"),
        ):
            self.client("typesafe")

    def test_transport_error_redacts_reason_as_well_as_args(self):
        key = "cpu-typesafe-secret"
        client = self.client("typesafe")
        with (
            patch(
                "jev_bench.advice.json_request",
                side_effect=[rendered(), URLError("invalid bearer " + key)],
            ),
            self.assertRaises(AdvisorServiceError) as caught,
        ):
            advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(caught.exception.record["status"], "service_error")
        self.assertNotIn(key, str(caught.exception))
        self.assertNotIn(key, json.dumps([client.records, caught.exception.record]))

    def test_deepseek_error_redaction_and_malformed_reasoning_usage(self):
        key = "cpu-deepseek-secret"
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                401, json={"error": {"message": "invalid key " + key}}
            )
        )
        with OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
            max_retries=0,
            http_client=httpx.Client(transport=transport),
        ) as endpoint:
            with patch("jev_bench.client.OpenAI", return_value=endpoint):
                client = self.client("deepseek")
            with self.assertRaises(AdvisorServiceError) as caught:
                advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            self.assertNotIn(key, str(caught.exception))
            self.assertNotIn(key, json.dumps([client.records, caught.exception.record]))
        response = deepseek_response()
        raw = response.model_dump(mode="json")
        raw["usage"]["completion_tokens_details"] = []
        with patch(
            "jev_bench.client.OpenAI",
            return_value=sdk(lambda response=response, **_: response),
        ):
            client = self.client("deepseek")
            with (
                patch.object(response, "model_dump", return_value=raw),
                self.assertRaises(AdvisorServiceError) as caught,
            ):
                advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            self.assertEqual(caught.exception.record["status"], "advisor_error")
            self.assertEqual(client.last_usage["total_tokens"], 15)
            client.close()

    def test_deepseek_clients_insert_inventory_path_once(self):
        with (
            patch.object(
                sys, "path", [p for p in sys.path if p != DATASET_INVENTORY_SRC]
            ),
            patch(
                "jev_bench.client.OpenAI",
                return_value=sdk(lambda **_: deepseek_response()),
            ),
        ):
            for _ in range(3):
                client = self.client("deepseek")
                client.close()
                self.assertEqual(sys.path.count(DATASET_INVENTORY_SRC), 1)

    def test_deepseek_request_excludes_private_reasoning_and_temperature(self):
        requests = []
        history = HISTORY + [
            {
                "role": "assistant",
                "content": "<think>secret chain</think>Visible text",
                "reasoning": "private reasoning",
                "reasoning_content": "other secret",
            }
        ]
        original = deepcopy(history)

        def create(**request):
            requests.append(request)
            return deepseek_response("tool_calls", None, ("echo", "echo"))

        with patch("jev_bench.client.OpenAI", return_value=sdk(create)):
            client = self.client("deepseek")
            self.addCleanup(client.close)
            result = client.predict(history, TOOLS)
        self.assertEqual(history, original)
        request = requests[0]
        self.assertNotIn("temperature", request)
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["tools"], TOOLS)
        self.assertEqual(
            request["messages"][-1],
            {"role": "assistant", "content": "Visible text", "reasoning_content": ""},
        )
        self.assertEqual(result["top1"], "echo")
        self.assertIsNone(result["probabilities"])
        self.assertEqual(
            choose_hint({"advice": "prefer"}, result, [])[0],
            PREFER_TOOL.format(name="echo"),
        )
        self.assertEqual(client.records[0]["reasoning_tokens"], 3)
        self.assertIs(client.records[0]["provider_called"], True)

    def test_deepseek_external_tool_history_uses_empty_reasoning_without_mutation(self):
        history = HISTORY + [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "executor private chain",
                "tool_calls": [
                    {
                        "id": "call_echo",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_echo", "content": "ok"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Thanks."},
        ]
        original = deepcopy(history)
        requests = []

        def create(**request):
            requests.append(request)
            return deepseek_response()

        with patch("jev_bench.client.OpenAI", return_value=sdk(create)):
            client = self.client("deepseek")
            self.addCleanup(client.close)
            client.predict(history, TOOLS)
            client.predict(HISTORY, TOOLS)
        self.assertEqual(history, original)
        expected = deepcopy(history)
        for message in expected:
            if message["role"] == "assistant":
                message["reasoning_content"] = ""
        self.assertEqual(requests[0]["messages"], expected)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            client.records[0]["history_compatibility"],
            {
                "rule": "external_assistant_empty_reasoning",
                "assistant_messages": 2,
            },
        )
        self.assertEqual(
            client.records[1]["history_compatibility"],
            {
                "rule": "external_assistant_empty_reasoning",
                "assistant_messages": 0,
            },
        )

    def test_accounting_and_identity_errors_do_not_become_neutral(self):
        for damage in ("usage", "identity", "empty_choice_identity"):
            response = deepseek_response()
            if damage == "usage":
                response.usage.prompt_tokens = None
            elif damage == "identity":
                response.model = ""
            else:
                response.choices = []
                response.model = ""
            with (
                self.subTest(damage=damage),
                patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(lambda response=response, **_: response),
                ),
            ):
                client = self.client("deepseek")
                try:
                    with self.assertRaises(AdvisorServiceError) as raised:
                        advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
                    self.assertIsNone(raised.exception.record["fallback"])
                    self.assertEqual(client.records[0]["provider_called"], True)
                finally:
                    client.close()

    def test_deepseek_completion_boundaries_and_missing_usage(self):
        cases = [
            ("stop", "Done", (), "RESPOND"),
            ("tool_calls", None, ("echo",), "echo"),
            ("length", "Done", (), None),
            ("content_filter", "Done", (), None),
            ("stop", "", (), None),
            ("stop", "Done", ("echo",), None),
            ("tool_calls", None, ("absent",), None),
            ("tool_calls", None, (), None),
        ]
        for reason, content, names, expected in cases:
            with (
                self.subTest(reason=reason, names=names),
                patch(
                    "jev_bench.client.OpenAI",
                    return_value=sdk(
                        lambda reason=reason, content=content, names=names, **_: (
                            deepseek_response(reason, content, names)
                        )
                    ),
                ),
            ):
                client = self.client("deepseek")
                try:
                    if expected is None:
                        record = advise(
                            {"advice": "prefer"}, client, HISTORY, TOOLS, []
                        )
                        self.assertEqual(
                            record["fallback"], "advisor_format_error_neutral"
                        )
                        self.assertEqual(record["hint"], NEUTRAL_MESSAGE)
                        self.assertIsNone(record["prediction"])
                    else:
                        record = advise(
                            {"advice": "prefer"}, client, HISTORY, TOOLS, []
                        )
                        self.assertEqual(record["prediction"]["top1"], expected)
                    self.assertEqual(client.last_usage["completion_tokens"], 5)
                finally:
                    client.close()
        response = deepseek_response()
        response.usage = None
        with patch(
            "jev_bench.client.OpenAI",
            return_value=sdk(lambda response=response, **_: response),
        ):
            client = self.client("deepseek")
            record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            self.assertEqual(record["hint"], PREFER_RESPOND)
            data = costs(
                None,
                SimpleNamespace(advisor_records=[record]),
                advisor_type="deepseek",
                advisor_requests=client.records,
            )
            self.assertIs(client.records[0]["provider_called"], True)
            self.assertEqual(data["advisor"]["provider_requests"], 1)
            self.assertEqual(
                data["advisor"]["completion_tokens"],
                {"observed_total": 0, "known": 0, "missing": 1},
            )
            client.close()

    def test_deepseek_sdk_errors_are_advisor_errors_without_retries(self):
        for status in (400, 401, 429, 503):
            seen = []

            def respond(request, seen=seen, status=status):
                seen.append(request)
                return httpx.Response(
                    status,
                    json={"error": {"message": "fixture", "type": "request_error"}},
                )

            with OpenAI(
                api_key="cpu",
                base_url="https://api.deepseek.com",
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            ) as endpoint:
                with patch("jev_bench.client.OpenAI", return_value=endpoint):
                    client = self.client("deepseek")
                with (
                    self.subTest(status=status),
                    self.assertRaises(AdvisorServiceError) as caught,
                ):
                    advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
                self.assertEqual(len(seen), 1)
                self.assertEqual(
                    caught.exception.record["status"],
                    "service_error" if status >= 429 else "advisor_error",
                )
                self.assertIsNone(client.last_usage)
                self.assertEqual(client.last_response["error"]["message"], "fixture")

    def test_http_headers_proxy_and_secrets_are_not_persisted(self):
        with (
            tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
            http_fixture(typesafe_response()) as (url, received),
        ):
            sink = lambda kind, value: append_json(
                Path(path) / (kind + ".jsonl"), value
            )
            value = settings("typesafe")
            value["base_url"] = url
            client = TypeSafeClient(value, sink, advisor_settings())
            with (
                patch.object(client.renderer, "render", return_value=rendered()),
                patch.dict(
                    "os.environ",
                    {
                        "HTTP_PROXY": "http://127.0.0.1:1",
                        "http_proxy": "http://127.0.0.1:1",
                    },
                ),
            ):
                record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
            sink("advisor", record)
            self.assertEqual(
                received[0]["headers"]["Authorization"], "Bearer cpu-typesafe-secret"
            )
            self.assertEqual(received[0]["path"], "/systemone")
            self.assertEqual(len(read_jsonl(Path(path) / "advisor_requests.jsonl")), 1)
            self.assertEqual(len(read_jsonl(Path(path) / "advisor.jsonl")), 1)
            for file in Path(path).glob("*.jsonl"):
                self.assertNotIn("cpu-typesafe-secret", file.read_text())
        with patch("jev_bench.advice.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value = io.StringIO(
                "{}"
            )
            json_request("https://api.typesafe.ai/v1/models", 1)
        opener.assert_called_once_with()

    def test_remote_usage_survives_failed_prediction_and_orphan_request(self):
        events = []
        response = typesafe_response()
        response["answers"]["next_action"]["choice"] = "unknown"
        client = self.client(
            "typesafe", lambda kind, value: events.append((kind, value))
        )
        with (
            patch("jev_bench.advice.json_request", side_effect=[rendered(), response]),
        ):
            record = advise({"advice": "prefer"}, client, HISTORY, TOOLS, [])
        self.assertEqual(record["fallback"], "advisor_format_error_neutral")
        normal = costs(
            None,
            SimpleNamespace(advisor_records=[record]),
            advisor_type="typesafe",
            advisor_requests=client.records,
        )
        orphan = costs(
            None, None, advisor_type="typesafe", advisor_requests=[events[0][1]]
        )
        for data in (normal, orphan):
            self.assertEqual(data["advisor"]["requests"], 1)
            self.assertEqual(data["advisor"]["provider_requests"], 1)
            self.assertEqual(data["advisor"]["total_tokens"]["observed_total"], 52)
            self.assertEqual(
                data["advisor"]["local_prompt_tokens"]["observed_total"], 30
            )
        rows = [
            {
                "cost": data,
                "response_anomalies": [],
                "duration_seconds": 1,
                "decision_steps": 1,
                "tool_call_count": 0,
            }
            for data in (normal, orphan)
        ]
        self.assertEqual(
            aggregate_cost(rows)["advisor"]["total_tokens"]["observed_total"], 104
        )
        self.assertEqual(aggregate_cost(rows)["advisor"]["provider_requests"], 2)

    def test_real_recovery_paths_use_remote_ledger_once(self):
        response = deepseek_response()
        with patch(
            "jev_bench.client.OpenAI",
            return_value=sdk(lambda response=response, **_: response),
        ):
            deepseek = self.client("deepseek")
            deepseek_decision = advise(
                {"advice": "prefer"}, deepseek, HISTORY, TOOLS, []
            )
            deepseek.close()
        typesafe = self.client("typesafe")
        body = {"error": {"type": "ValueError", "message": "over_length_after_drop"}}
        with patch(
            "jev_bench.advice.json_request", side_effect=HTTPResponseError(422, body)
        ):
            advise({"advice": "prefer"}, typesafe, HISTORY, TOOLS, [])
        for client, decision in ((deepseek, deepseek_decision), (typesafe, None)):
            for phase in ("inference", "scoring"):
                with (
                    self.subTest(client=type(client).__name__, phase=phase),
                    tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path,
                ):
                    root = Path(path)
                    planned = {
                        "condition": "native",
                        "category": "multi_turn_base",
                        "task_id": "cpu",
                        "trial": 0,
                        "seed": 1,
                    }
                    item = {**planned, "attempt": 1}
                    directory = directory_for(root, item)
                    write_json(
                        directory / "diagnostics/attempt.json",
                        {
                            "item": item,
                            "advisor_type": client.records[0]["advisor_type"],
                            "benchmark": "bfcl",
                            "phase": phase,
                        },
                    )
                    append_json(
                        directory / "diagnostics/advisor_requests.jsonl",
                        client.records[0],
                    )
                    if decision is not None:
                        append_json(directory / "diagnostics/advisor.jsonl", decision)
                    if phase == "inference":
                        row = recover_interrupted(directory, item)
                        self.assertEqual(recover_interrupted(directory, item), row)
                    else:
                        write_json(directory / "trajectory.json", {"result": "fixture"})
                        inputs = {(planned["category"], planned["task_id"]): {}}
                        with (
                            run_lock(root),
                            patch(
                                "jev_bench.scoring.adapter",
                                return_value=SimpleNamespace(
                                    score_saved=lambda *args: {
                                        "success": True,
                                        "tool_call_count": 0,
                                    }
                                ),
                            ),
                        ):
                            self.assertEqual(
                                recover_scores(
                                    config(), [planned], inputs, inputs, root, code={}
                                ),
                                1,
                            )
                            self.assertEqual(
                                recover_scores(
                                    config(), [planned], inputs, inputs, root, code={}
                                ),
                                0,
                            )
                        row = json.loads((directory / "episode.json").read_text())
                    cost = row["cost"]["advisor"]
                    self.assertEqual(cost["requests"], 1)
                    self.assertEqual(
                        cost,
                        costs(
                            None,
                            None,
                            advisor_type=client.records[0]["advisor_type"],
                            advisor_requests=client.records,
                        )["advisor"],
                    )
                    if decision is not None:
                        self.assertEqual(cost["provider_requests"], 1)
                        self.assertEqual(cost["completion_tokens"]["observed_total"], 5)
                        self.assertEqual(cost["reasoning_tokens"]["observed_total"], 3)
                        for field in (
                            "truncated",
                            "messages_dropped",
                            "local_prompt_tokens",
                            "render_latency_seconds",
                        ):
                            self.assertNotIn(field, cost)
                    else:
                        self.assertEqual(cost["provider_requests"], 0)
                        self.assertEqual(cost["remote_input_tokens"]["missing"], 0)
                        self.assertEqual(cost["over_length_after_drop_neutral"], 1)

    def test_read_only_preflight_never_calls_inference(self):
        value = config()
        value["advisors"] = {
            "local": advisor_settings(),
            "typesafe": settings("typesafe"),
            "deepseek": settings("deepseek"),
        }
        value["conditions"] = {
            name: {"executor": "qwen4b", "advice": "prefer", "advisor": name}
            for name in ("typesafe", "deepseek")
        }
        model = {
            "id": "qwen3-4b",
            "root": "/data/wangsq/model/Qwen3-4B",
            "max_model_len": 131072,
        }
        with (
            patch("jev_bench.services.json_request", return_value={"data": [model]}),
            patch(
                "jev_bench.advice.json_request",
                side_effect=[
                    {**prediction(), "status": "ready"},
                    {"data": [{"id": "deepseek-flash"}]},
                ],
            ) as request,
        ):
            observed = check_services(value)
        self.assertFalse(observed["advisors"]["typesafe"]["remote_checked"])
        self.assertFalse(observed["advisors"]["deepseek"]["inference_checked"])
        self.assertTrue(
            all(
                call.args[0].endswith(("/health", "/models"))
                for call in request.call_args_list
            )
        )

    def test_main_smokes_are_six_rows_three_executors_and_not_formal(self):
        from jev_bench.config import load_config
        from jev_bench.runner import make_chains

        for prefix in ("bfcl_mt", "tau2"):
            value = load_config(PROJECT / "configs" / f"{prefix}_main_smoke.json")
            self.assertEqual(value["output_root"], "tmp")
            self.assertEqual(value["trials"], 1)
            self.assertEqual(value["comparisons"], [])
            self.assertEqual(
                set(value["conditions"]),
                {"native", "official_jev", "ours", "actionopd", "deepseek_advisor", "star"},
            )
            self.assertEqual(value["conditions"]["actionopd"]["advice"], "none")
            self.assertEqual(
                len(
                    make_chains(
                        value, [{"condition": name} for name in value["conditions"]]
                    )
                ),
                3,
            )

    def test_cost_fields_follow_the_configured_advisor_type(self):
        from jev_bench.advice import remote_record

        def ledger(kind):
            record = remote_record(kind)
            record.update(
                provider_called=True,
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                remote_input_tokens=4,
                remote_output_tokens=2,
                model_latency_seconds=0.5,
                latency_seconds=0.6,
            )
            return record

        native = costs(None, None, advisor_type=None)["advisor"]
        self.assertEqual(native, {"requests": 0})
        local = costs(
            None, SimpleNamespace(advisor_records=[]), advisor_type="jev_local"
        )["advisor"]
        self.assertNotIn("provider_requests", local)
        self.assertNotIn("remote_input_tokens", local)
        self.assertIn("truncated", local)
        deepseek = costs(
            None, None, advisor_type="deepseek", advisor_requests=[ledger("deepseek")]
        )["advisor"]
        for field in (
            "truncated",
            "messages_dropped",
            "local_prompt_tokens",
            "render_latency_seconds",
            "over_length_after_drop_neutral",
        ):
            self.assertNotIn(field, deepseek)
        self.assertEqual(deepseek["reasoning_tokens"]["missing"], 1)
        typesafe = costs(
            None, None, advisor_type="typesafe", advisor_requests=[ledger("typesafe")]
        )["advisor"]
        self.assertNotIn("reasoning_tokens", typesafe)
        self.assertEqual(typesafe["provider_requests"], 1)

        # Configuration, records and ledger must agree; nothing is inferred.
        with self.assertRaisesRegex(ValueError, "without an advisor"):
            costs(None, SimpleNamespace(advisor_records=[{}]), advisor_type=None)
        with self.assertRaisesRegex(ValueError, "no remote request ledger"):
            costs(
                None,
                None,
                advisor_type="jev_local",
                advisor_requests=[ledger("typesafe")],
            )
        with self.assertRaisesRegex(ValueError, "configured advisor type"):
            costs(
                None,
                None,
                advisor_type="typesafe",
                advisor_requests=[ledger("deepseek")],
            )
        with self.assertRaises(TypeError):
            costs(None, None)

    def test_aggregate_cost_keeps_one_advisor_field_set(self):
        def row(advisor_type, requests=()):
            return {
                "cost": costs(
                    None,
                    None,
                    advisor_type=advisor_type,
                    advisor_requests=requests,
                ),
                "response_anomalies": [],
                "duration_seconds": 1,
                "decision_steps": 1,
                "tool_call_count": 0,
            }

        native = aggregate_cost([row(None), row(None)])["advisor"]
        self.assertEqual(native, {"requests": 0})
        self.assertEqual(aggregate_cost([])["advisor"], {"requests": 0})
        deepseek = aggregate_cost([row("deepseek")])["advisor"]
        self.assertEqual(deepseek["provider_requests"], 0)
        self.assertNotIn("truncated_request_rate", deepseek)
        with self.assertRaisesRegex(ValueError, "different advisor cost fields"):
            aggregate_cost([row(None), row("deepseek")])

    def test_attempt_marker_records_the_advisor_type(self):
        from jev_bench.attempt import record_attempt

        @record_attempt
        def episode(config, item, task, answer, directory):
            return json.loads((directory / "diagnostics/attempt.json").read_text())

        value = config()
        value["advisors"] = {"remote": settings("deepseek")}
        value["conditions"]["native"] = {
            "executor": next(iter(value["executors"])),
            "advice": "prefer",
            "advisor": "remote",
        }
        item = {"condition": "native", "attempt": 1}
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as path:
            marker = episode(value, item, None, None, Path(path) / "episode")
        self.assertEqual(marker["advisor_type"], "deepseek")
        self.assertEqual(marker["item"], item)
