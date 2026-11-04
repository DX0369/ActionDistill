import ast
import json
import random
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.advice import (
    NEUTRAL_MESSAGE,
    PREFER_RESPOND,
    PREFER_TOOL,
    AdvisorServiceError,
)
from jev_bench.benchmarks.toolsandbox import (
    catalog,
    configure,
    evaluate_scenario,
    load_tasks,
)
from jev_bench.benchmarks.toolsandbox_scores import describe
from jev_bench.client import ParticipantResponseError
from jev_bench.config import PROJECT_ROOT, load_config
from jev_bench.io import append_json, read_jsonl

from tests_toolsandbox.support import Advisor, Endpoint, call, response

CONFIG = load_config(PROJECT_ROOT / "configs/toolsandbox_cpu_preview.json")
configure(CONFIG, PROJECT_ROOT / "tmp")

from jev_bench.benchmarks.toolsandbox_participants import ConfiguredUser, GuidedAgent
from openai import NOT_GIVEN
from tool_sandbox.common.execution_context import (
    ScenarioCategories,
    get_current_context,
    new_context,
)
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios.insufficient_information_scenarios import (
    named_insufficient_information_scenarios,
)
from tool_sandbox.scenarios.multiple_user_turn_scenarios import (
    named_multiple_user_turn_scenarios,
)

SEND = "send_message_with_contact_content_cellular_off_multiple_user_turn"
REMOVE = "remove_contact_by_phone_no_search_contacts_insufficient_information"
CONFIRM = "Your message to Fredrik Thordendal has been sent saying: How's the new album coming along"


def task(scenario, name):
    return {
        "scenario": scenario,
        "family_id": name,
        "variant": "original",
        "categories": [str(c) for c in scenario.categories],
    }


def send_script(solve):
    def send_using_observation(request):
        observation = next(
            m["content"] for m in reversed(request["messages"]) if m["role"] == "tool"
        )
        contacts = ast.literal_eval(observation)
        return response(
            calls=[
                call(
                    "send_message_with_phone_number",
                    {
                        "phone_number": contacts[0]["phone_number"],
                        "content": "How's the new album coming along",
                    },
                    "send_message",
                )
            ]
        )

    steps = [
        response(
            content="Who should I send it to, and what should the message say?",
            reasoning="Ask for the missing recipient and content.",
        ),
        response(
            calls=[
                call("set_cellular_service_status", {"on": True}, "enable_cellular")
            ],
            reasoning="Enable the cellular service.",
        ),
        response(
            calls=[
                call(
                    "search_contacts", {"name": "Fredrik Thordendal"}, "search_contact"
                )
            ]
        ),
    ]
    if solve:
        steps.append(send_using_observation)
    steps.append(response(content=CONFIRM))
    user = [
        response(
            content="Send a message to Fredrik Thordendal saying: How's the new album coming along"
        ),
        response(calls=[call("end_conversation", {}, "user_stop")]),
    ]
    return steps, user


class ToolSandboxTests(unittest.TestCase):
    def test_live_wrapper_and_scoring_recovery_preserve_continuous_outcome(self):
        from jev_bench.benchmarks.toolsandbox import run_episode
        from jev_bench.runner import directory_for, pending_plan, validate_attempts
        from jev_bench.scoring import recover_scores
        from tool_sandbox.common.evaluation import Evaluation

        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        item = {
            "condition": "native",
            "category": "scenarios",
            "task_id": SEND,
            "trial": 0,
            "attempt": 1,
            "seed": 20261015,
        }
        selected = task(self.send, SEND)
        for erase_row in (False, True):
            root = self.directory / f"recovery_{erase_row}"
            directory = directory_for(root, item)
            actor_steps, user_steps = send_script(True)
            actor, user = Endpoint(actor_steps), Endpoint(user_steps)
            with (
                patch.object(
                    Evaluation, "evaluate", side_effect=RuntimeError("score fixture")
                ),
                self.assertRaisesRegex(RuntimeError, "score fixture"),
            ):
                run_episode(
                    config,
                    item,
                    selected,
                    None,
                    directory,
                    client_factory=actor.client,
                    user_client_factory=lambda settings, sink, user=user: user.client(
                        settings, sink, user=True
                    ),
                )
            row_path = directory / "episode.json"
            row = json.loads(row_path.read_text())
            self.assertEqual(row["failure_kind"], "scoring_error")
            original_cost = row["cost"]
            self.assertEqual(
                json.loads((directory / "diagnostics/attempt.json").read_text())[
                    "phase"
                ],
                "scoring",
            )
            if erase_row:
                row_path.unlink()
            else:
                append_json(root / "episodes.jsonl", row)
            count = recover_scores(
                config,
                [item],
                {("scenarios", SEND): selected},
                {("scenarios", SEND): None},
                root,
                code={"fixture": True},
            )
            self.assertEqual(count, 1)
            recovered = read_jsonl(root / "episodes.jsonl")[0]
            self.assertIsNone(recovered["success"])
            self.assertEqual(
                recovered["official_score"], {"metric": "similarity", "value": 1}
            )
            self.assertEqual(recovered["family_id"], SEND)
            self.assertEqual(recovered["termination_reason"], "user_end_conversation")
            if not erase_row:
                self.assertEqual(recovered["cost"], original_cost)
            validate_attempts([recovered], [item])
            self.assertEqual(pending_plan([item], [recovered]), [])
            self.assertFalse(actor.steps)
            self.assertFalse(user.steps)
            self.assertEqual(
                recover_scores(
                    config,
                    [item],
                    {("scenarios", SEND): selected},
                    {("scenarios", SEND): None},
                    root,
                    code={"fixture": True},
                ),
                0,
            )

    def test_live_wrapper_records_only_executor_tool_calls(self):
        from jev_bench.benchmarks.toolsandbox import run_episode

        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        actor_steps, user_steps = send_script(True)
        actor, user = Endpoint(actor_steps), Endpoint(user_steps)
        item = {
            "condition": "native",
            "category": "scenarios",
            "task_id": SEND,
            "trial": 0,
            "attempt": 1,
            "seed": 20261015,
        }
        row = run_episode(
            config,
            item,
            task(self.send, SEND),
            None,
            self.directory / "live",
            client_factory=actor.client,
            user_client_factory=lambda settings, sink: user.client(
                settings, sink, user=True
            ),
        )
        self.assertTrue(row["official_scored"])
        self.assertIsNone(row["success"])
        self.assertEqual(row["tool_call_count"], 3)
        self.assertEqual(row["cost"]["executor"]["requests"], len(actor_steps))
        self.assertEqual(row["cost"]["user_simulator"]["requests"], len(user_steps))

    @classmethod
    def setUpClass(cls):
        cls.send = named_multiple_user_turn_scenarios(ToolBackend.DEFAULT)[SEND]
        cls.remove = named_insufficient_information_scenarios(ToolBackend.DEFAULT)[
            REMOVE
        ]

    def setUp(self):
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(prefix="toolsandbox_test_", dir=PROJECT_ROOT / "tmp")
        )
        # An integration fixture must never spend an external request or API key.
        self.network = patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("CPU fixture attempted network"),
        )
        self.httpx = patch(
            "httpx.Client.send",
            side_effect=AssertionError("CPU fixture attempted network"),
        )
        self.network.start()
        self.httpx.start()

    def tearDown(self):
        self.network.stop()
        self.httpx.stop()
        failed = any(
            test is self
            for test, _ in self._outcome.result.errors + self._outcome.result.failures
        )
        if failed:
            print(f"CPU fixture diagnostics retained: {self.directory}")
        else:
            shutil.rmtree(self.directory)

    def agent(self, steps, *, policy=None, advisor=None):
        endpoint = Endpoint(steps)
        settings = CONFIG["executors"]["qwen4b"]
        sink = lambda kind, value: append_json(
            self.directory / "diagnostics" / f"{kind}.jsonl", value
        )
        agent = GuidedAgent(
            client=endpoint.client(settings, sink),
            settings=settings,
            policy=policy or {"advice": "none"},
            advisor_client=advisor,
            event_sink=sink,
        )
        return agent, endpoint

    def play(self, scenario, name, steps, user_steps, *, policy=None, advisor=None):
        agent, endpoint = self.agent(steps, policy=policy, advisor=advisor)
        user_endpoint = Endpoint(user_steps)
        user = ConfiguredUser(
            client=user_endpoint.client(
                {
                    "model": "cpu-user",
                    "base_url": "http://127.0.0.1:1/v1",
                    "timeout_seconds": 1,
                },
                lambda kind, value: append_json(
                    self.directory / "diagnostics" / f"{kind}.jsonl", value
                ),
                user=True,
            )
        )
        before = get_current_context()
        try:
            row = evaluate_scenario(
                task(scenario, name),
                name,
                self.directory / "attempt",
                agent=agent,
                user=user,
            )
        finally:
            self.assertIs(get_current_context(), before)
            agent.client.close()
            user.client.close()
        self.assertFalse(endpoint.steps)
        self.assertFalse(user_endpoint.steps)
        return row, agent, endpoint, user_endpoint

    def test_multi_turn_real_tools_full_score_and_request_visibility(self):
        advisor = Advisor()
        row, agent, endpoint, user = self.play(
            self.send,
            SEND,
            *send_script(True),
            policy={"advice": "prefer"},
            advisor=advisor,
        )
        self.assertEqual(
            (
                row["similarity"],
                row["milestone_similarity"],
                row["minefield_similarity"],
            ),
            (1, 1, 0),
        )
        self.assertNotIn("success", row)
        self.assertEqual(row["termination_reason"], "user_end_conversation")
        self.assertEqual(row["decision_steps"], 5)
        self.assertEqual(row["cost"]["executor"]["requests"], 5)
        self.assertEqual(row["cost"]["advisor"]["requests"], 5)
        self.assertEqual(row["cost"]["user_simulator"]["requests"], 2)
        self.assertEqual(
            [r["first_action_matches"] for r in agent.advisor_records],
            [True, False, False, False, True],
        )
        for request, advised in zip(endpoint.requests, advisor.requests, strict=True):
            self.assertEqual(
                request["messages"][-1], {"role": "system", "content": PREFER_RESPOND}
            )
            self.assertEqual(
                sum(m.get("content") == PREFER_RESPOND for m in request["messages"]), 1
            )
            self.assertEqual(request["messages"][:-1], advised["messages"])
            self.assertEqual(request["tools"], advised["tools"])
            self.assertEqual(request["tool_choice"], "auto")
            self.assertEqual(request["max_tokens"], 8192)
            self.assertNotIn(
                "end_conversation", [t["function"]["name"] for t in request["tools"]]
            )
        self.assertTrue(
            any(
                m.get("reasoning") == "Ask for the missing recipient and content."
                for m in endpoint.requests[1]["messages"]
            )
        )
        self.assertTrue(
            any(
                m.get("reasoning") == "Enable the cellular service."
                for m in endpoint.requests[2]["messages"]
            )
        )
        for request in user.requests:
            self.assertNotIn(PREFER_RESPOND, json.dumps(request))
            self.assertFalse(any("reasoning" in m for m in request["messages"]))
        official = self.directory / "attempt/official/trajectories" / SEND
        context = json.loads((official / "execution_context.json").read_text())
        self.assertIn("_dbs", context)
        self.assertNotIn(PREFER_RESPOND, json.dumps(context))
        score = json.loads((self.directory / "attempt/score.json").read_text())
        self.assertEqual(len(score["milestone_mapping"]), 4)

    def test_omitted_action_receives_partial_official_score(self):
        row, _, _, _ = self.play(self.send, SEND, *send_script(False))
        self.assertEqual(row["similarity"], 0.75)
        self.assertTrue(row["official_scored"])

    def test_parallel_calls_keep_official_execution_and_reasoning(self):
        steps, user = send_script(True)
        original_ids = ["chatcmpl-tool-a-b", "ts_" + b"chatcmpl-tool-a-b".hex()]
        for message, call_id in zip(steps[1:3], original_ids, strict=True):
            message["tool_calls"][0]["id"] = call_id
        user[-1]["tool_calls"][0]["id"] = "chatcmpl-tool-user-stop"
        steps[1:3] = [
            response(
                calls=steps[1]["tool_calls"] + steps[2]["tool_calls"],
                reasoning="Enable cellular and look up the contact independently.",
            )
        ]
        row, _, endpoint, _ = self.play(self.send, SEND, steps, user)
        self.assertEqual(row["similarity"], 1)
        self.assertEqual(row["decision_steps"], 4)
        parallel = [
            m
            for m in endpoint.requests[2]["messages"]
            if len(m.get("tool_calls", [])) == 2
        ]
        self.assertEqual(len(parallel), 1)
        self.assertEqual(
            parallel[0]["reasoning"],
            "Enable cellular and look up the contact independently.",
        )
        routed_ids = [c["id"] for c in parallel[0]["tool_calls"]]
        self.assertEqual(routed_ids, ["ts_" + x.encode().hex() for x in original_ids])
        tool_results = [
            m["tool_call_id"]
            for m in endpoint.requests[2]["messages"]
            if m["role"] == "tool"
        ]
        self.assertEqual(tool_results, routed_ids)
        self.assertEqual(
            row["response_normalizations"],
            {
                "executor": {"tool_call_id_to_python_identifier": 2},
                "user_simulator": {"tool_call_id_to_python_identifier": 1},
            },
        )
        records = read_jsonl(self.directory / "diagnostics/executor.jsonl")
        self.assertEqual(
            [
                c["id"]
                for c in records[1]["response"]["choices"][0]["message"]["tool_calls"]
            ],
            original_ids,
        )

    def test_message_limit_still_receives_official_continuous_score(self):
        scenario = deepcopy(self.remove)
        scenario.max_messages = 1
        row, _, _, user = self.play(
            scenario, REMOVE, [response(content="Please clarify.")], []
        )
        self.assertTrue(row["official_scored"])
        self.assertEqual(row["termination_reason"], "message_limit")
        self.assertEqual(row["similarity"], 1)
        self.assertFalse(user.requests)

    def test_empty_tool_calls_are_counted_without_changing_raw_responses(self):
        steps, user_steps = send_script(True)
        for message in [*steps, *user_steps]:
            if not callable(message) and message["tool_calls"] is None:
                message["tool_calls"] = []
        row, _, _, _ = self.play(self.send, SEND, steps, user_steps)
        self.assertEqual(row["similarity"], 1)
        self.assertEqual(
            row["response_normalizations"],
            {
                "executor": {"empty_tool_calls_to_none": 2},
                "user_simulator": {"empty_tool_calls_to_none": 1},
            },
        )
        for participant in ("executor", "user_simulator"):
            for record in read_jsonl(
                self.directory / "diagnostics" / f"{participant}.jsonl"
            ):
                raw = record["response"]["choices"][0]["message"]
                if isinstance(raw["content"], str):
                    self.assertEqual(raw["tool_calls"], [])

    def test_minefield_avoidance_scores_one(self):
        row, _, _, _ = self.play(
            self.remove,
            REMOVE,
            [
                response(
                    content="I cannot identify the contact with the available tools."
                )
            ],
            [response(calls=[call("end_conversation", {})])],
        )
        self.assertEqual(
            (
                row["milestone_similarity"],
                row["minefield_similarity"],
                row["similarity"],
            ),
            (1, 0, 1),
        )

    def test_minefield_trigger_scores_zero_even_when_tool_reports_error(self):
        row, _, endpoint, _ = self.play(
            self.remove,
            REMOVE,
            [
                response(
                    calls=[
                        call(
                            "remove_contact",
                            {"person_id": "00000000-0000-0000-0000-000000000000"},
                        )
                    ]
                ),
                response(
                    content="I cannot identify the contact with the available tools."
                ),
            ],
            [response(calls=[call("end_conversation", {})])],
        )
        self.assertEqual(
            (
                row["milestone_similarity"],
                row["minefield_similarity"],
                row["similarity"],
            ),
            (1, 1, 0),
        )
        self.assertTrue(row["official_scored"])
        self.assertIn("NoDataError", json.dumps(endpoint.requests[1]))

    def test_exceptions_preserve_evidence_and_never_become_zero(self):
        with self.assertRaisesRegex(RuntimeError, "scripted failure"):
            self.play(self.remove, REMOVE, [RuntimeError("scripted failure")], [])
        row = json.loads((self.directory / "attempt/episode.json").read_text())
        self.assertFalse(row["official_scored"])
        self.assertIsNone(row["similarity"])
        self.assertFalse((self.directory / "attempt/score.json").exists())
        self.assertTrue((self.directory / "attempt/diagnostics/failure.json").exists())
        self.assertTrue(
            (
                self.directory
                / "attempt/official/trajectories"
                / REMOVE
                / "execution_context.json"
            ).exists()
        )
        self.assertEqual(row["cost"]["executor"]["requests"], 1)

    def test_invalid_simulator_reply_leaves_score_missing(self):
        with self.assertRaises(ParticipantResponseError):
            self.play(
                self.remove, REMOVE, [response(content="Please clarify.")], [response()]
            )
        row = json.loads((self.directory / "attempt/episode.json").read_text())
        self.assertIsNone(row["similarity"])
        self.assertEqual(row["cost"]["user_simulator"]["requests"], 1)

    def test_agent_respond_is_not_episode_termination(self):
        # The user starts a new request after RESPOND; only its end tool terminates.
        row, _, _, user = self.play(
            self.remove,
            REMOVE,
            [
                response(content="Please clarify."),
                response(content="I cannot identify this contact."),
            ],
            [
                response(content="Please do not remove anyone."),
                response(calls=[call("end_conversation", {})]),
            ],
        )
        self.assertEqual(row["decision_steps"], 2)
        self.assertEqual(len(user.requests), 2)

    def test_empty_text_and_calls_follow_official_roles(self):
        cases = [
            ("null", None, " \n\t", {"executor": {}, "user_simulator": {}}),
            ("empty", [], "", {"executor": {"empty_tool_calls_to_none": 2},
                                "user_simulator": {"empty_tool_calls_to_none": 1}}),
        ]
        for name, calls, user_text, normalizations in cases:
            with self.subTest(name=name), patch.object(self, "directory", self.directory / name):
                self.directory.mkdir()
                row, _, executor, user = self.play(
                    self.remove,
                    REMOVE,
                    [response(content="", calls=calls), response(content=" \n\t", calls=calls)],
                    [response(content=user_text, calls=calls), response(calls=[call("end_conversation", {})])],
                )
                self.assertTrue(row["official_scored"])
                self.assertEqual(row["termination_reason"], "user_end_conversation")
                self.assertTrue(any(m["content"] == "" for m in user.requests[0]["messages"]))
                self.assertTrue(any(m["content"] == user_text for m in executor.requests[1]["messages"]))
                self.assertEqual(row["response_normalizations"], normalizations)

    def test_reused_participant_or_client_rejected_before_output(self):
        _, used, _, _ = self.play(
            self.remove,
            REMOVE,
            [response(content="Cannot identify the contact.")],
            [response(calls=[call("end_conversation", {})])],
        )
        fresh, _ = self.agent([])
        fresh.client = used.client
        for index, agent in enumerate((used, fresh)):
            output = self.directory / f"reused_{index}"
            with self.assertRaisesRegex(ValueError, "fresh participants"):
                evaluate_scenario(
                    task(self.remove, REMOVE), REMOVE, output, agent=agent, user=None
                )
            self.assertFalse(output.exists())

    def test_scrambled_candidates_use_official_agent_facing_names(self):
        scenario = deepcopy(self.remove)
        scenario.starting_context.tool_augmentation_list = [
            ScenarioCategories.TOOL_NAME_SCRAMBLED
        ]
        advisor = Advisor()

        def use_scrambled(request):
            with new_context(scenario.starting_context):
                available = scenario.starting_context.get_available_tools(
                    scrambling_allowed=True
                )
                name = next(
                    name
                    for name in available
                    if scenario.starting_context.get_execution_facing_tool_name(name)
                    == "remove_contact"
                )
            self.assertNotEqual(name, "remove_contact")
            self.assertIn(name, [t["function"]["name"] for t in request["tools"]])
            return response(
                calls=[
                    call(name, {"person_id": "00000000-0000-0000-0000-000000000000"})
                ]
            )

        row, _, endpoint, _ = self.play(
            scenario,
            REMOVE,
            [use_scrambled, response(content="I cannot identify this contact.")],
            [response(calls=[call("end_conversation", {})])],
            policy={"advice": "prefer"},
            advisor=advisor,
        )
        self.assertEqual(row["minefield_similarity"], 1)
        self.assertEqual(endpoint.requests[0]["tools"], advisor.requests[0]["tools"])

    def test_loop_guard_uses_current_user_turn_and_parallel_call_order(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        calls = [call("read", {"x": 1}, "one"), call("read", {"x": 1}, "two")]
        base = [
            {"role": "user", "content": "Help"},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ]
        for reset in (False, True):
            advisor = Advisor(
                top1="read", probabilities={"read": 0.99, "RESPOND": 0.01}
            )
            agent, endpoint = self.agent(
                [response(content="Done")],
                policy={
                    "advice": "gate",
                    "gate": {
                        "rule": "respond_prob_ge",
                        "threshold": 0.1,
                        "loop_guard": True,
                    },
                },
                advisor=advisor,
            )
            history = base + (
                [{"role": "user", "content": "New turn"}] if reset else []
            )
            agent.model_inference(history, tools)
            self.assertEqual(
                agent.advisor_records[0]["gate"]["triggered"],
                "none" if reset else "loop",
            )
            self.assertEqual(
                endpoint.requests[0]["messages"][-1]["content"],
                PREFER_TOOL.format(name="read") if reset else NEUTRAL_MESSAGE,
            )
            self.assertEqual(history[0], base[0])

    def test_confidence_gates_and_neutral_are_request_local(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"t{i}",
                    "description": "tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(4)
        ]
        history = [{"role": "user", "content": "Help"}]
        for rule, threshold in (("respond_prob_ge", 0.1), ("top1_prob_lt", 0.3)):
            advisor = Advisor(
                top1="t0",
                probabilities={
                    "t0": 0.25,
                    "t1": 0.2,
                    "t2": 0.2,
                    "t3": 0.2,
                    "RESPOND": 0.15,
                },
            )
            agent, endpoint = self.agent(
                [response(content="Done")],
                policy={
                    "advice": "gate",
                    "gate": {"rule": rule, "threshold": threshold, "loop_guard": True},
                },
                advisor=advisor,
            )
            agent.model_inference(history, tools)
            self.assertEqual(
                endpoint.requests[0]["messages"][-1]["content"], NEUTRAL_MESSAGE
            )
            self.assertTrue(
                all(agent.advisor_records[0]["gate"]["confidence_by_rule"].values())
            )
        agent, endpoint = self.agent(
            [response(content="Done")], policy={"advice": "neutral"}
        )
        agent.model_inference(history, NOT_GIVEN)
        self.assertEqual(
            endpoint.requests[0]["messages"][-1]["content"], NEUTRAL_MESSAGE
        )
        self.assertEqual(history, [{"role": "user", "content": "Help"}])

    def test_advisor_error_never_silently_calls_executor(self):
        advisor = Advisor()
        agent, endpoint = self.agent(
            [response(content="Done")], policy={"advice": "prefer"}, advisor=advisor
        )
        with (
            patch.object(advisor, "predict", side_effect=OSError("advisor down")),
            self.assertRaises(AdvisorServiceError),
        ):
            agent.model_inference([{"role": "user", "content": "Help"}], NOT_GIVEN)
        self.assertFalse(endpoint.requests)
        self.assertEqual(agent.advisor_records[0]["status"], "service_error")

    def test_overlength_fallback_routes_text_and_keeps_null_match(self):
        from jev_bench.advice import HTTPResponseError

        advisor = Advisor()
        with patch.object(
            advisor,
            "predict",
            side_effect=HTTPResponseError(
                422,
                {"error": {"type": "ValueError", "message": "over_length_after_drop"}},
            ),
        ):
            row, agent, executor, _ = self.play(
                self.remove,
                REMOVE,
                [response(content="Please clarify.")],
                [response(calls=[call("end_conversation", {})])],
                policy={"advice": "prefer"},
                advisor=advisor,
            )
        self.assertTrue(row["official_scored"])
        self.assertEqual(row["cost"]["advisor"]["over_length_after_drop_neutral"], 1)
        self.assertIsNone(agent.advisor_records[0]["prediction"])
        self.assertIsNone(agent.advisor_records[0]["first_action_matches"])
        self.assertEqual(
            executor.requests[0]["messages"][-1]["content"], NEUTRAL_MESSAGE
        )

    def test_malformed_responses_are_explicit(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        invalid = [
            response(),
            response(calls=[]),
            response(calls=[call("missing", {})]),
            response(calls=[call("read", [])]),
        ]
        for value in invalid:
            with self.subTest(value=value):
                agent, _ = self.agent([value])
                with self.assertRaises(ParticipantResponseError):
                    agent.model_inference([{"role": "user", "content": "Help"}], tools)

    def test_reasoning_alignment_rejects_changed_history(self):
        agent, _ = self.agent([response(content="Answer", reasoning="Prior reasoning")])
        history = [{"role": "user", "content": "Help"}]
        agent.model_inference(history, NOT_GIVEN)
        with self.assertRaisesRegex(ValueError, "sidecar"):
            agent.model_inference(
                history + [{"role": "assistant", "content": "Changed"}], NOT_GIVEN
            )


class CatalogAndScoresTests(unittest.TestCase):
    def test_formal_summary_keeps_families_and_official_overlapping_categories(self):
        from jev_bench.benchmarks.bfcl import costs
        from jev_bench.stats import summarize

        plan, rows = [], []
        for family in ("first", "second"):
            for variant in range(8):
                labels = [str(ScenarioCategories.THREE_DISTRACTION_TOOLS)]
                if variant:
                    labels.append(str(ScenarioCategories.TOOL_NAME_SCRAMBLED))
                for condition in ("ours", "native"):
                    item = {
                        "condition": condition,
                        "category": "scenarios",
                        "task_id": f"{family}_{variant}",
                        "trial": 0,
                        "family_id": family,
                        "variant": str(variant),
                        "categories": labels,
                    }
                    value = float(condition == "ours" and family == "first")
                    plan.append(item)
                    rows.append(
                        {
                            **item,
                            "attempt": 1,
                            "official_scored": True,
                            "success": None,
                            "failure_kind": None,
                            "similarity": value,
                            "milestone_similarity": value,
                            "minefield_similarity": 0.0,
                            "turn_count": 3,
                            "official_score": {"metric": "similarity", "value": value},
                            "cost": costs(None, None, advisor_type=None),
                            "duration_seconds": 1.0,
                            "decision_steps": 1,
                            "tool_call_count": 0,
                            "response_anomalies": [],
                        }
                    )
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="toolsandbox", formal=True
        )
        self.assertTrue(result["coverage_complete"])
        pair = result["comparisons"][0]
        self.assertEqual((pair["clusters"], pair["points"]), (2, 16))
        self.assertEqual(pair["difference"], 0.5)
        self.assertEqual(pair["bootstrap_95_percentile"], [0.0, 1.0])
        official = result["conditions"]["ours"]["official_categories"]
        self.assertEqual(
            official[str(ScenarioCategories.THREE_DISTRACTION_TOOLS)]["count"], 2
        )
        self.assertEqual(official["ALL_CATEGORIES"]["count"], 16)
        rows[0].update(
            official_scored=False, success=False, failure_kind="executor_response_error"
        )
        result = summarize(
            rows, plan, [["ours", "native"]], benchmark="toolsandbox", formal=True
        )
        self.assertIsNone(result["conditions"]["ours"]["official_categories"])
        self.assertIsNone(result["comparisons"][0]["difference"])

    def test_canonical_catalog_families_and_rng_isolation(self):
        before = random.getstate()
        first = catalog(42)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(first), 1032)
        self.assertEqual(len({item["family_id"] for item in first.values()}), 129)
        self.assertEqual(first[SEND]["variant"], "original")
        self.assertEqual(
            first[SEND + "_3_distraction_tools_tool_name_scrambled"]["family_id"], SEND
        )
        second = catalog(42)
        self.assertEqual(random.getstate(), before)
        for name in first:
            self.assertEqual(
                first[name]["scenario"].starting_context.tool_allow_list,
                second[name]["scenario"].starting_context.tool_allow_list,
            )
        with patch("jev_bench.benchmarks.toolsandbox.catalog", return_value=first):
            selected, answers = load_tasks(CONFIG)
            self.assertEqual({name for _, name in selected}, {SEND, REMOVE})
            self.assertEqual(selected.keys(), answers.keys())
            invalid = deepcopy(CONFIG)
            invalid["task_ids"]["scenarios"] = ["not-a-scenario"]
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_tasks(invalid)

    def test_official_continuous_summary_and_paired_direction(self):
        labels = [
            str(ScenarioCategories.THREE_DISTRACTION_TOOLS),
            str(ScenarioCategories.TOOL_NAME_SCRAMBLED),
        ]
        rows = [
            {
                "condition": condition,
                "task_id": name,
                "trial": 0,
                "official_scored": True,
                "similarity": value,
                "milestone_similarity": value,
                "minefield_similarity": 0,
                "turn_count": 3,
                "categories": labels if name == "scrambled" else labels[:1],
                "family_id": "base",
                "variant": name,
            }
            for condition, values in (("a", (0, 0.75)), ("b", (0.5, 1)))
            for name, value in zip(("original", "scrambled"), values, strict=True)
        ]
        plan = [
            {key: row[key] for key in ("condition", "task_id", "trial")} for row in rows
        ]
        summary = describe(rows, plan, [("b", "a"), ("a", "b")])
        self.assertEqual(
            summary["conditions"]["a"]["ALL_CATEGORIES"]["similarity"], 0.375
        )
        self.assertEqual(summary["conditions"]["a"][labels[0]]["count"], 1)
        self.assertEqual(summary["comparisons"][0]["mean_similarity_difference"], 0.375)
        self.assertEqual(
            summary["comparisons"][1]["mean_similarity_difference"], -0.375
        )
        self.assertEqual(summary["inference"], "not_configured")
        for bad in (
            rows[:-1],
            rows + [rows[0]],
            [{**rows[0], "official_scored": False}, *rows[1:]],
        ):
            with self.assertRaises(ValueError):
                describe(bad, plan, [("b", "a")])
        with self.assertRaisesRegex(ValueError, "matched"):
            describe(rows, plan, [("missing", "a")])
        uneven = rows + [
            {**row, "trial": 1} for row in rows if row["task_id"] == "original"
        ]
        uneven_plan = [
            {key: row[key] for key in ("condition", "task_id", "trial")}
            for row in uneven
        ]
        with self.assertRaisesRegex(ValueError, "trial sets"):
            describe(uneven, uneven_plan, [("b", "a")])
        for categories in ([labels[0], labels[0]], ["invented_category"]):
            with self.assertRaisesRegex(ValueError, "official labels"):
                describe([{**rows[0], "categories": categories}, *rows[1:]], plan, [])
        with self.assertRaisesRegex(ValueError, "metadata"):
            describe([{**rows[0], "family_id": "other"}, *rows[1:]], plan, [])
        with self.assertRaisesRegex(ValueError, "integer"):
            describe([{**rows[0], "trial": False}, *rows[1:]], plan, [])


if __name__ == "__main__":
    unittest.main()
