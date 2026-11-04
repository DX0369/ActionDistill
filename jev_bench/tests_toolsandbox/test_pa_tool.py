import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.toolsandbox import run_episode
from jev_bench.config import PROJECT_ROOT, load_config
from tests.pa_fixtures import alias_message, freeze

from tests_toolsandbox import test_toolsandbox as fixtures
from tests_toolsandbox.support import Endpoint


class PaToolSandboxTests(unittest.TestCase):
    def test_official_description_hidden_variants_keep_visible_catalog(self):
        from jev_bench.pa_tool_catalog import build_catalog
        from jev_bench.pa_tool_prepare import description_preservations, estimate
        from tool_sandbox.scenarios import named_scenarios

        scenarios = named_scenarios(fixtures.ToolBackend.DEFAULT)
        names = [
            "remove_contact_with_id_3_distraction_tools_" + suffix
            for suffix in ("tool_description_scrambled", "arg_description_scrambled")
        ]
        tasks = {("scenarios", name): fixtures.task(scenarios[name], name) for name in names}
        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        with patch("jev_bench.pa_tool_catalog.load_tasks", return_value=(tasks, {})):
            catalog = build_catalog(config, config["conditions"]["native"]["executor"])
        records = description_preservations(catalog)
        self.assertEqual({row["reason"] for row in records.values()}, {
            "tool_description_scrambled", "arg_description_scrambled"
        })
        for scope, tools in catalog["scopes"].items():
            if catalog["description_augmentations"][scope] == ["TOOL_DESCRIPTION_SCRAMBLED"]:
                self.assertTrue(all(tool["function"]["description"] == "" for tool in tools))
            else:
                self.assertTrue(any(
                    "description" not in parameter
                    for tool in tools
                    for parameter in tool["function"]["parameters"]["properties"].values()
                ))
        counts = estimate(catalog)
        self.assertLess(counts["requests_upper_bound"], 33 * counts["components_upper_bound"])

    def test_interface_failure_is_terminal_and_has_no_invented_similarity(self):
        from jev_bench.runner import pending_plan
        from jev_bench.stats import summarize
        from tool_sandbox.common.execution_context import new_context
        from tool_sandbox.common.tool_conversion import convert_to_openai_tools
        from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

        from tests_toolsandbox.support import call, response

        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        scenario = fixtures.named_multiple_user_turn_scenarios(
            fixtures.ToolBackend.DEFAULT
        )[fixtures.SEND]
        with new_context(scenario.starting_context):
            tools = convert_to_openai_tools(OpenAIAPIAgent.get_available_tools())
        planned = {
            "condition": "native",
            "category": "scenarios",
            "task_id": fixtures.SEND,
            "trial": 0,
            "seed": 20261015,
        }
        item = {**planned, "attempt": 1}
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory:
            root = Path(directory)
            p = freeze(config, item, tools, root / "mapping")
            actor, user = (
                Endpoint(
                    [
                        response(
                            calls=[call("set_cellular_service_status", {"on": True})]
                        )
                    ]
                ),
                Endpoint([]),
            )
            row = run_episode(
                config,
                item,
                fixtures.task(scenario, fixtures.SEND),
                None,
                root / "episode",
                client_factory=actor.client,
                user_client_factory=lambda settings, sink: user.client(
                    settings, sink, user=True
                ),
                pa_tool_identity=p.asset_identity,
            )
            self.assertEqual(row["failure_kind"], "executor_response_error")
            self.assertIsNone(row["similarity"])
            self.assertFalse(row["official_scored"])
            self.assertFalse((root / "episode/score.json").exists())
            self.assertEqual(pending_plan([planned], [row]), [])
            summary = summarize([row], [planned], [], benchmark="toolsandbox")
            self.assertTrue(summary["execution_coverage_complete"])
            self.assertFalse(summary["coverage_complete"])
            self.assertEqual(summary["conditions"]["native"]["protocol_failures"], 1)

    def test_official_multiturn_route_keeps_user_and_scoring_canonical(self):
        from tool_sandbox.common.execution_context import new_context
        from tool_sandbox.common.tool_conversion import convert_to_openai_tools
        from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        scenario = fixtures.named_multiple_user_turn_scenarios(
            fixtures.ToolBackend.DEFAULT
        )[fixtures.SEND]
        with new_context(scenario.starting_context):
            tools = convert_to_openai_tools(OpenAIAPIAgent.get_available_tools())
        item = {
            "condition": "native",
            "category": "scenarios",
            "task_id": fixtures.SEND,
            "trial": 0,
            "seed": 20261015,
            "attempt": 1,
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory:
            root = Path(directory)
            p = freeze(config, item, tools, root / "mapping")
            agent_steps, user_steps = fixtures.send_script(True)
            projected = []
            for step in agent_steps:
                projected.append(
                    (
                        lambda request, step=step: alias_message(
                            step(request), tools, p.rows
                        )
                    )
                    if callable(step)
                    else alias_message(step, tools, p.rows)
                )
            actor, user = Endpoint(projected), Endpoint(user_steps)
            with (
                patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("no network"),
                ),
                patch("httpx.Client.send", side_effect=AssertionError("no network")),
            ):
                outcome = run_episode(
                    config,
                    item,
                    fixtures.task(scenario, fixtures.SEND),
                    None,
                    root / "episode",
                    client_factory=actor.client,
                    user_client_factory=lambda settings, sink: user.client(
                        settings, sink, user=True
                    ),
                    pa_tool_identity=p.asset_identity,
                )
            self.assertTrue(outcome["official_scored"])
            self.assertEqual(outcome["similarity"], 1)
            self.assertEqual(outcome["tool_call_count"], 3)
            self.assertFalse(actor.steps)
            self.assertFalse(user.steps)
            self.assertTrue(
                all(
                    t["function"]["name"].startswith("pa_tool_")
                    for r in actor.requests
                    for t in r.get("tools", [])
                )
            )
            self.assertTrue(
                all(
                    not t["function"]["name"].startswith("pa_tool_")
                    for r in user.requests
                    for t in r.get("tools", [])
                )
            )
