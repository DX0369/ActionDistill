import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from jev_bench.advice import NEUTRAL_MESSAGE, PREFER_TOOL, AdvisorServiceError
from jev_bench.benchmarks.tau2 import (
    BASE_COUNTS,
    configure,
    load_tasks,
    run_episode,
    score_saved,
)
from jev_bench.client import QwenClient, SimulatorClient
from jev_bench.config import PROJECT_ROOT, TAU2_DOMAINS, load_config, validate
from jev_bench.runner import directory_for, make_plan, pending_plan, validate_attempts
from jev_bench.stats import paired, summarize
from loguru import logger
from openai import AuthenticationError
from openai.types.chat import ChatCompletion

logger.disable("tau2")
(PROJECT_ROOT / "tmp").mkdir(exist_ok=True)


def config():
    value = load_config(PROJECT_ROOT / "configs/tau2_native_smoke.json")
    configure(value, PROJECT_ROOT / "tmp")
    return value


def call(name, arguments, call_id="cpu_call"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def response(*, content=None, tool_calls=None, reasoning=None):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "reasoning": reasoning,
    }


class ScriptedEndpoint:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        if not self.messages:
            raise RuntimeError("scripted endpoint exhausted")
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return ChatCompletion.model_validate(
            {
                "id": "cpu",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    def factory(self, client_type):
        def make(settings, sink):
            sdk = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)),
                close=lambda: None,
            )
            with (
                patch("jev_bench.client.OpenAI", return_value=sdk),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "cpu-fixture"}),
            ):
                return client_type(settings, sink)

        return make


def task_and_scripts(domain, *, solve=True):
    from tau2.data_model.tasks import (
        Action,
        EnvAssertion,
        EnvFunctionCall,
        EvaluationCriteria,
        InitialState,
        RewardType,
        Task,
        UserScenario,
    )

    if domain == "retail":
        name, arguments = (
            "cancel_pending_order",
            {"order_id": "#W5918442", "reason": "no longer needed"},
        )
    elif domain == "airline":
        name, arguments = "cancel_reservation", {"reservation_id": "4WQ150"}
    else:
        task = Task(
            id="cpu-payment",
            user_scenario=UserScenario(
                instructions="PRIVATE_SCENARIO: pay bill B1002."
            ),
            initial_state=InitialState(
                initialization_actions=[
                    EnvFunctionCall(
                        env_type="user",
                        func_name="set_user_info",
                        arguments={
                            "name": "John Smith",
                            "phone_number": "555-123-2001",
                        },
                    )
                ]
            ),
            evaluation_criteria=EvaluationCriteria(
                env_assertions=[
                    EnvAssertion(
                        env_type="assistant",
                        func_name="assert_no_overdue_bill",
                        arguments={"overdue_bill_id": "B1002"},
                    )
                ],
                reward_basis=[RewardType.ENV_ASSERTION],
            ),
        )
        agent = [
            response(
                tool_calls=[
                    call(
                        "send_payment_request",
                        {"customer_id": "C1001", "bill_id": "B1002"},
                    )
                ]
            ),
            response(content="Please pay using the request on your device."),
            response(
                tool_calls=[call("get_bills_for_customer", {"customer_id": "C1001"})]
            ),
            response(content="I checked the bill."),
        ]
        user = [
            response(content="I want to pay bill B1002."),
            response(tool_calls=[call("check_payment_request", {}, "private-check")]),
        ]
        if solve:
            user.append(response(tool_calls=[call("make_payment", {}, "private-pay")]))
        user.extend(
            [response(content="Please check my bill."), response(content="###STOP###")]
        )
        return task, agent, user

    task = Task(
        id=f"cpu-{domain}",
        user_scenario=UserScenario(
            instructions="PRIVATE_SCENARIO: cancel my booking or order."
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="cancel",
                    requestor="assistant",
                    name=name,
                    arguments=arguments,
                )
            ],
            reward_basis=[RewardType.DB],
        ),
    )
    agent = [response(content="Please confirm cancellation.")]
    if solve:
        agent.append(
            response(
                tool_calls=[call(name, arguments)], reasoning="The user confirmed."
            )
        )
    agent.append(response(content="Your request is complete."))
    user = [
        response(content="Please cancel it."),
        response(content="I confirm."),
        response(content="###STOP###"),
    ]
    return task, agent, user


class RecordedAdvisor:
    def __init__(self, *, respond_probability=0.15):
        self.requests = []
        self.last_response = None
        self.last_usage = None
        self.records = []
        self.respond_probability = respond_probability

    def close(self):
        pass

    def predict(self, messages, tools):
        self.requests.append(deepcopy({"messages": messages, "tools": tools}))
        names = [tool["function"]["name"] for tool in tools]
        return {
            "top1": names[0],
            "probabilities": [
                {
                    "name": name,
                    "probability": 1 - self.respond_probability if i == 0 else 0.0,
                }
                for i, name in enumerate(names)
            ]
            + [{"name": "RESPOND", "probability": self.respond_probability}],
            "model": "cpu-model",
            "temperature": 1.3,
            "n_prompt_tokens": 100,
            "latency_seconds": 0.01,
            "truncated": False,
            "messages_dropped": 0,
        }


class OfficialLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = config()

    def run_script(
        self,
        domain,
        root,
        *,
        solve=True,
        user_override=None,
        advice="none",
        advisor=None,
    ):
        value = deepcopy(self.settings)
        value["conditions"]["native"]["advice"] = advice
        if advisor is not None:
            value["advisors"]["fixture"] = {"type": "jev_local"}
            value["conditions"]["native"]["advisor"] = "fixture"
        if advice == "gate":
            value["conditions"]["native"]["gate"] = {
                "rule": "respond_prob_ge",
                "threshold": 0.1,
                "loop_guard": True,
            }
        task, agent_messages, user_messages = task_and_scripts(domain, solve=solve)
        if user_override is not None:
            user_messages = user_override
        executor, user = (
            ScriptedEndpoint(agent_messages),
            ScriptedEndpoint(user_messages),
        )
        item = {
            "condition": "native",
            "category": domain,
            "task_id": task.id,
            "trial": 0,
            "seed": 20261015,
            "attempt": 1,
        }
        row = run_episode(
            value,
            item,
            task,
            None,
            root,
            client_factory=executor.factory(QwenClient),
            user_client_factory=user.factory(SimulatorClient),
            advisor_factory=lambda settings, sink, **kwargs: advisor,
        )
        return row, executor, user

    def test_three_real_environments_and_official_success_failure(self):
        for domain in TAU2_DOMAINS:
            for solve in (True, False):
                with (
                    self.subTest(domain=domain, solve=solve),
                    tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path,
                ):
                    row, executor, user = self.run_script(
                        domain, Path(path) / "episode", solve=solve
                    )
                    self.assertEqual(row["success"], solve)
                    self.assertTrue(row["official_scored"])
                    self.assertEqual(row["reward"], float(solve))
                    self.assertEqual(row["termination_reason"], "user_stop")
                    original_score = json.loads(
                        (Path(path) / "episode/score.json").read_text()
                    )
                    (Path(path) / "episode/score.json").unlink()
                    task, _, _ = task_and_scripts(domain, solve=solve)
                    with (
                        patch(
                            "socket.socket.connect",
                            side_effect=AssertionError("scoring attempted network"),
                        ),
                        patch(
                            "socket.getaddrinfo",
                            side_effect=AssertionError("scoring attempted DNS"),
                        ),
                    ):
                        recovered = score_saved(
                            self.settings, row, task, None, Path(path) / "episode"
                        )
                    self.assertEqual(recovered["success"], row["success"])
                    self.assertEqual(
                        json.loads((Path(path) / "episode/score.json").read_text()),
                        original_score,
                    )
                    self.assertFalse(executor.messages)
                    self.assertFalse(user.messages)
                    self.assertEqual(
                        row["cost"]["user_simulator"]["requests"], len(user.requests)
                    )
                    self.assertEqual(
                        row["cost"]["user_simulator"]["total_tokens"]["observed_total"],
                        15 * len(user.requests),
                    )
                    visible = json.dumps(executor.requests)
                    self.assertNotIn("PRIVATE_SCENARIO", visible)
                    if domain == "telecom":
                        self.assertNotIn("private-pay", visible)
                        self.assertNotIn("private-check", visible)
                        self.assertEqual(row["user_tool_call_count"], 2 if solve else 1)
                        self.assertTrue(
                            any(
                                "tool" == m["role"]
                                and ("Paid" if solve else "Awaiting Payment")
                                in m["content"]
                                for r in executor.requests
                                for m in r["messages"]
                            )
                        )
                        trajectory = json.loads(
                            (Path(path) / "episode/trajectory.json").read_text()
                        )
                        self.assertIn("private-check", json.dumps(trajectory))

    def test_user_error_remains_unresolved_with_observed_cost(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path:
            row, _, user = self.run_script(
                "retail", Path(path) / "episode", user_override=[response(content="")]
            )
            self.assertIsNone(row["success"])
            self.assertFalse(row["official_scored"])
            self.assertEqual(row["failure_kind"], "user_simulator_error")
            self.assertEqual(row["cost"]["user_simulator"]["requests"], 1)
            self.assertEqual(len(user.requests), 1)
            self.assertEqual(pending_plan([row], [row])[0]["attempt"], 2)

    def test_user_undefined_tool_and_api_failure_are_unresolved(self):
        invalid_call = response(tool_calls=[call("not_a_user_tool", {})])
        auth = AuthenticationError(
            "expired credentials",
            response=httpx.Response(
                401,
                request=httpx.Request(
                    "POST", "http://example.invalid/v1/chat/completions"
                ),
            ),
            body={},
        )
        for message in (invalid_call, auth):
            with (
                self.subTest(message=type(message).__name__),
                tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path,
            ):
                row, _, user = self.run_script(
                    "retail", Path(path) / "episode", user_override=[message]
                )
                self.assertEqual(row["failure_kind"], "user_simulator_error")
                self.assertIsNone(row["success"])
                self.assertEqual(len(user.requests), 1)
                self.assertFalse((Path(path) / "episode/score.json").exists())
                self.assertEqual(pending_plan([row], [row])[0]["attempt"], 2)

    def test_telecom_advice_is_request_local_and_cannot_see_user_tools(self):
        for mode in ("neutral", "prefer", "gate"):
            advisor = RecordedAdvisor() if mode != "neutral" else None
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path,
            ):
                row, executor, _ = self.run_script(
                    "telecom", Path(path) / "episode", advice=mode, advisor=advisor
                )
                self.assertTrue(row["success"])
                for index, request in enumerate(executor.requests):
                    hint = request["messages"][-1]
                    self.assertEqual(hint["role"], "system")
                    expected = (
                        NEUTRAL_MESSAGE
                        if mode != "prefer"
                        else PREFER_TOOL.format(
                            name=request["tools"][0]["function"]["name"]
                        )
                    )
                    self.assertEqual(hint["content"], expected)
                    self.assertEqual(
                        sum(
                            message["role"] == "system"
                            for message in request["messages"]
                        ),
                        2,
                    )
                    if advisor is not None:
                        self.assertEqual(
                            advisor.requests[index]["messages"],
                            request["messages"][:-1],
                        )
                        self.assertEqual(
                            advisor.requests[index]["tools"], request["tools"]
                        )
                if advisor is not None:
                    visible = json.dumps(advisor.requests)
                    for private in (
                        "PRIVATE_SCENARIO",
                        "private-check",
                        "private-pay",
                        '"name": "make_payment"',
                    ):
                        self.assertNotIn(private, visible)
                    self.assertEqual(
                        row["cost"]["advisor"]["requests"], len(executor.requests)
                    )

    def test_failed_advisor_never_silently_runs_executor(self):
        advisor = RecordedAdvisor()
        with (
            patch.object(
                advisor, "predict", side_effect=ValueError("bad service response")
            ),
            tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path,
        ):
            with self.assertRaises(AdvisorServiceError):
                self.run_script(
                    "telecom", Path(path) / "episode", advice="prefer", advisor=advisor
                )
            row = json.loads((Path(path) / "episode/episode.json").read_text())
            self.assertIsNone(row["success"])
            self.assertEqual(row["failure_kind"], "advisor_error")
            self.assertEqual(row["cost"]["executor"]["requests"], 0)

    def test_overlength_fallback_preserves_official_loop_and_null_match(self):
        from jev_bench.advice import HTTPResponseError
        from jev_bench.io import read_jsonl

        advisor = RecordedAdvisor()
        with (
            patch.object(
                advisor,
                "predict",
                side_effect=HTTPResponseError(
                    422,
                    {
                        "error": {
                            "type": "ValueError",
                            "message": "over_length_after_drop",
                        }
                    },
                ),
            ),
            tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path,
        ):
            root = Path(path) / "episode"
            row, executor, _ = self.run_script(
                "retail", root, advice="prefer", advisor=advisor
            )
            records = read_jsonl(root / "diagnostics/advisor.jsonl")
        self.assertTrue(row["success"])
        self.assertEqual(
            row["cost"]["advisor"]["over_length_after_drop_neutral"],
            len(executor.requests),
        )
        self.assertEqual(len(records), len(executor.requests))
        for record, request in zip(records, executor.requests, strict=True):
            self.assertIsNone(record["prediction"])
            self.assertIsNone(record["first_action_matches"])
            self.assertEqual(request["messages"][-1]["content"], NEUTRAL_MESSAGE)

    def test_simulator_reasoning_survives_tools_and_later_turns_privately(self):
        _, _, messages = task_and_scripts("telecom")
        for index, message in enumerate(messages):
            del message["reasoning"]
            message["reasoning_content"] = (
                "" if index == 2 else f"PRIVATE_REASONING_{index}"
            )
        advisor = RecordedAdvisor()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as path:
            row, executor, user = self.run_script(
                "telecom",
                Path(path) / "episode",
                user_override=messages,
                advice="prefer",
                advisor=advisor,
            )
        self.assertTrue(row["success"])
        for request_index, request in enumerate(user.requests):
            own_messages = [m for m in request["messages"] if m["role"] == "assistant"]
            self.assertEqual(
                [m["reasoning_content"] for m in own_messages],
                [m["reasoning_content"] for m in messages[:request_index]],
            )
            self.assertTrue(all("reasoning" not in m for m in own_messages))
            self.assertNotIn("seed", request)
        for requests in (executor.requests, advisor.requests):
            self.assertNotIn("PRIVATE_REASONING_", json.dumps(requests))

    def test_reasoning_fields_do_not_alias_and_bad_types_fail(self):
        from jev_bench.benchmarks.tau2_participants import parse_message, to_openai
        from jev_bench.client import CompletionResult, ParticipantResponseError

        for value in ("", "private", None):
            raw = SimpleNamespace(
                content="hello", tool_calls=None, reasoning_content=value
            )
            result = CompletionResult(raw, None, 0, "stop", {})
            outgoing = to_openai(parse_message(result))
            self.assertEqual(outgoing["reasoning_content"], value)
            self.assertNotIn("reasoning", outgoing)
        for value in ([], 0):
            raw = SimpleNamespace(
                content="hello", tool_calls=None, reasoning_content=value
            )
            with self.assertRaises(ParticipantResponseError):
                parse_message(CompletionResult(raw, None, 0, "stop", {}), user=True)

    def test_malformed_user_usage_is_a_resumable_response_error(self):
        from jev_bench.benchmarks.tau2_participants import (
            DeepSeekUser,
            UserSimulatorError,
        )
        from tau2.data_model.message import AssistantMessage

        event = []
        completion = SimpleNamespace(
            model_dump=lambda **kwargs: {"usage": {"prompt_tokens": 10}}, choices=[]
        )
        sdk = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: completion)
            ),
            close=lambda: None,
        )
        with (
            patch("jev_bench.client.OpenAI", return_value=sdk),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}),
        ):
            client = SimulatorClient(
                self.settings["user_simulator"],
                lambda kind, record: event.append(record),
            )
        user = DeepSeekUser(client=client, tools=None, instructions="test")
        with self.assertRaises(UserSimulatorError):
            user.generate_next_message(
                AssistantMessage(role="assistant", content="Hello"),
                user.get_init_state(),
            )
        self.assertEqual(len(event), 1)
        self.assertIsNone(event[0]["usage"])
        self.assertEqual(event[0]["response"]["usage"], {"prompt_tokens": 10})


class LoadingAndStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = config()

    def test_base_278_and_overlapping_canonical_ids(self):
        value = deepcopy(self.settings)
        value["task_ids"] = dict.fromkeys(TAU2_DOMAINS, "all")
        tasks, answers = load_tasks(value)
        self.assertEqual(len(tasks), 278)
        self.assertEqual(set(tasks), set(answers))
        for domain, count in BASE_COUNTS.items():
            self.assertEqual(sum(category == domain for category, _ in tasks), count)
        self.assertIn(("retail", "0"), tasks)
        self.assertIn(("airline", "0"), tasks)
        plan = make_plan(value, tasks)
        self.assertEqual(len(plan), 278)
        self.assertEqual(plan, make_plan(value, dict(reversed(list(tasks.items())))))
        folders = {
            directory_for(Path("unused"), {**item, "attempt": 1}) for item in plan
        }
        self.assertEqual(len(folders), 278)
        self.assertTrue(all(len(path.parts[-3]) == 64 for path in folders))

    def test_preflight_rejects_missing_key_before_any_service_request(self):
        from jev_bench.services import check_services

        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}),
            patch("jev_bench.services.json_request") as request,
            self.assertRaisesRegex(ValueError, "key environment variable"),
        ):
            check_services(self.settings)
        request.assert_not_called()

    def test_required_protocol_fields_and_domain_membership(self):
        for section in ("protocol", "user_simulator"):
            for field in self.settings[section]:
                value = deepcopy(self.settings)
                del value[section][field]
                with (
                    self.subTest(section=section, field=field),
                    self.assertRaises((KeyError, ValueError)),
                ):
                    validate(value)
        value = deepcopy(self.settings)
        value["task_ids"]["retail"] = ["not-a-task"]
        with self.assertRaisesRegex(ValueError, "unknown base"):
            load_tasks(value)
        with (
            patch.dict("os.environ", {"TAU2_DATA_DIR": str(PROJECT_ROOT / "tmp")}),
            self.assertRaisesRegex(ValueError, "pinned checkout"),
        ):
            configure(self.settings, PROJECT_ROOT / "tmp")

    def test_task_weighting_and_official_pass_power(self):
        from jev_bench.cost import costs

        plan, rows = [], []
        for domain, size in zip(TAU2_DOMAINS, (1, 1, 2)):
            for i in range(size):
                for trial in range(3):
                    for condition in ("A", "B"):
                        item = {
                            "condition": condition,
                            "category": domain,
                            "task_id": str(i),
                            "trial": trial,
                        }
                        success = (condition == "A") == (domain != "telecom")
                        plan.append(item)
                        rows.append(
                            {
                                **item,
                                "attempt": 1,
                                "success": success,
                                "official_scored": True,
                                "failure_kind": None
                                if success
                                else "model_task_failure",
                                "duration_seconds": 0,
                                "decision_steps": 0,
                                "tool_call_count": 0,
                                "cost": costs(None, None, advisor_type=None),
                                "response_anomalies": [],
                            }
                        )
        result = summarize(rows, plan, [["A", "B"]], benchmark="tau2")
        self.assertEqual(result["conditions"]["A"]["official_success_rate"], 0.5)
        self.assertEqual(
            result["conditions"]["A"]["pass_power"], {"1": 0.5, "2": 0.5, "3": 0.5}
        )
        pair = result["paired_comparisons"][0]
        self.assertEqual(pair["difference"], 0)
        self.assertEqual(pair["p_exact_task_permutation"], 1)
        self.assertEqual(pair["bootstrap_95_percentile"], [0, 0])
        self.assertEqual(
            pair,
            paired(validate_attempts(rows, plan), plan, "A", "B", benchmark="tau2"),
        )
        incomplete = summarize(rows[:-1], plan, [["A", "B"]], benchmark="tau2")
        self.assertFalse(incomplete["coverage_complete"])
        self.assertIsNone(incomplete["conditions"]["A"]["pass_power"])
        # One task with 2/3 successes separates pass^k from pass@k.
        row = next(
            r for r in rows if r["category"] == "retail" and r["condition"] == "A"
        )
        row.update(success=False, failure_kind="model_task_failure")
        powers = summarize(rows, plan, [], benchmark="tau2")["conditions"]["A"][
            "pass_power"
        ]
        self.assertAlmostEqual(powers["1"], 5 / 12)
        self.assertAlmostEqual(powers["2"], 1 / 3)
        self.assertEqual(powers["3"], 1 / 4)
