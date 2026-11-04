"""Currency backend migration through the unchanged official tool."""

import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from jev_bench.benchmarks.toolsandbox import configure
from jev_bench.config import load_config
from tests import PROJECT

CONFIG = load_config(PROJECT / "configs/toolsandbox_cpu_preview.json")
configure(CONFIG, None)

from jev_bench.benchmarks.toolsandbox_currency import CURRENCY_HOST, CURRENCY_URL
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    new_context,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.tools import rapid_api_search_tools as tools
from tool_sandbox.tools.setting import set_wifi_status


class CurrencyTests(unittest.TestCase):
    def setUp(self):
        self.context = ExecutionContext()
        scope = new_context(self.context)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        environment = patch.dict(os.environ, {"RAPID_API_KEY": "cpu-test-key"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_new_endpoint_preserves_inputs_and_accepts_zero(self):
        for value in (0, 0.89446):
            amount = 0 if value == 0 else 1
            with (
                self.subTest(value=value),
                patch.object(tools.requests, "get") as request,
            ):
                request.return_value.json.return_value = {
                    "success": True,
                    "result": value,
                }
                result = tools.convert_currency(amount, "usd", "eur")
                self.assertEqual(result, float(value))
                self.assertIs(type(result), float)
                request.assert_called_once_with(
                    url=CURRENCY_URL,
                    headers={
                        "X-RapidAPI-Host": CURRENCY_HOST,
                        "X-RapidAPI-Key": "cpu-test-key",
                    },
                    params={"from": "USD", "to": "EUR", "amount": amount},
                )

    def test_required_response_fields_fail_explicitly(self):
        for body in ({"result": 1}, {"success": True}):
            with (
                self.subTest(body=body),
                patch.object(tools.requests, "get") as request,
            ):
                request.return_value.json.return_value = body
                with self.assertRaises(KeyError):
                    tools.convert_currency(1, "USD", "EUR")
                self.assertEqual(request.call_count, 1)

    def test_provider_failure_and_non_numeric_results_are_not_conversion_values(self):
        invalid = [
            {"success": success, "result": 1} for success in (False, None, 1, "true")
        ] + [
            {"success": True, "result": value}
            for value in (None, True, "0.89446", {}, float("nan"), float("inf"))
        ]
        for body in invalid:
            with (
                self.subTest(body=body),
                patch.object(tools.requests, "get") as request,
            ):
                request.return_value.json.return_value = body
                with self.assertRaises((RuntimeError, ValueError)):
                    tools.convert_currency(1, "USD", "EUR")
                self.assertEqual(request.call_count, 1)

    def test_network_failure_propagates_without_retry_or_old_service_fallback(self):
        failure = tools.requests.Timeout("CPU timeout")
        with patch.object(tools.requests, "get", side_effect=failure) as request:
            with self.assertRaises(tools.requests.Timeout) as raised:
                tools.convert_currency(1, "USD", "EUR")
            self.assertIs(raised.exception, failure)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.kwargs["url"], CURRENCY_URL)

    def test_other_rapidapi_tools_keep_their_endpoint_and_response(self):
        body = {"results": [{"address": "CPU address"}]}
        before = deepcopy(body)
        with patch.object(tools.requests, "get") as request:
            request.return_value.json.return_value = body
            self.assertEqual(tools.search_lat_lon(37.3346, -122.009), "CPU address")
        self.assertEqual(body, before)
        self.assertEqual(
            request.call_args.kwargs["url"],
            "https://trueway-geocoding.p.rapidapi.com/ReverseGeocode",
        )
        self.assertEqual(
            request.call_args.kwargs["headers"]["X-RapidAPI-Host"],
            "trueway-geocoding.p.rapidapi.com",
        )

    def test_official_currency_wifi_and_key_checks_still_precede_requests(self):
        with patch.object(tools.requests, "get") as request:
            with self.assertRaises(ValueError):
                tools.convert_currency(1, "$", "EUR")
            set_wifi_status(False)
            with self.assertRaises(ConnectionError):
                tools.convert_currency(1, "USD", "EUR")
            set_wifi_status(True)
            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaises(PermissionError),
            ):
                tools.convert_currency(1, "USD", "EUR")
            request.assert_not_called()

    def test_reconfigure_preserves_cached_tool_schema_and_official_trace(self):
        tool = self.context.name_to_tool["convert_currency"]
        schema = convert_to_openai_tools({"convert_currency": tool})
        configure(CONFIG, None)
        configure(CONFIG, None)
        self.assertIs(tools.convert_currency, tool)
        self.assertEqual(convert_to_openai_tools({"convert_currency": tool}), schema)
        self.context.add_to_database(
            DatabaseNamespace.SANDBOX,
            [
                {
                    "sender": RoleType.AGENT,
                    "recipient": RoleType.EXECUTION_ENVIRONMENT,
                    "content": "convert_currency(amount=1, from_currency_code='USD', to_currency_code='EUR')",
                }
            ],
        )
        self.context.trace_tool = True
        with patch.object(tools.requests, "get") as request:
            request.return_value.json.return_value = {
                "success": True,
                "result": 0.89446,
            }
            self.assertEqual(tool(1, "USD", "EUR"), 0.89446)
            self.assertEqual(request.call_count, 1)
        rows = self.context.get_database(DatabaseNamespace.SANDBOX).to_dicts()
        trace = [json.loads(value) for value in rows[-1]["tool_trace"]]
        self.assertEqual(
            trace,
            [
                {
                    "tool_name": "convert_currency",
                    "result": 0.89446,
                    "arguments": {
                        "amount": 1,
                        "from_currency_code": "USD",
                        "to_currency_code": "EUR",
                    },
                }
            ],
        )
