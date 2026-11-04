import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from func_timeout import FunctionTimedOut
from jev_bench.benchmarks.toolhop import load_tasks, run_fixture, visible_tools
from jev_bench.config import PROJECT_ROOT, WORKSPACE_ROOT, load_config


def call(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": name,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


class ToolHopTests(unittest.TestCase):
    def test_missing_dependency_fails_before_actor_or_official_tool_error(self):
        target = {
            "functions": [
                "def fixture():\n import unavailable_toolhop_fixture_dependency\n return 1"
            ],
            "answer": "1",
        }
        actor = unittest.mock.Mock(side_effect=AssertionError("must not run actor"))
        with self.assertRaisesRegex(
            ModuleNotFoundError, "unavailable_toolhop_fixture_dependency"
        ):
            run_fixture(
                self.config,
                self.tasks[("tasks", "0")],
                target,
                actor,
                scenario="Free",
                max_turns=9,
            )
        actor.assert_not_called()

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "configs/toolhop_cpu_preview.json")
        cls.config["task_ids"] = {"tasks": "all"}
        cls.tasks, cls.answers = load_tasks(cls.config)

    def setUp(self):
        for target in ("socket.socket.connect", "socket.getaddrinfo"):
            blocker = patch(
                target, side_effect=AssertionError("CPU fixture attempted network")
            )
            blocker.start()
            self.addCleanup(blocker.stop)

    def test_installed_dependencies_execute_released_tools(self):
        cases = [
            (
                "28",
                "date_calculator",
                {"start_date": "2024-01-31", "amount": 1},
                "2024-02-29",
            ),
            (
                "68",
                "tv_show_cast_query",
                {"show_title": "Death in Paradise", "output_format": "xml"},
                "<root>",
            ),
            (
                "179",
                "day_of_week_calculator",
                {"date": "2024-01-01", "localization": "en"},
                "Monday",
            ),
            (
                "156",
                "date_calculator",
                {"start_date": "2024-01-01", "days_difference": 1},
                "2024-01-02",
            ),
            (
                "308",
                "extract_year",
                {"date_string": "2024-01-01", "output_format": "roman"},
                "MMXXIV",
            ),
            ("590", "prime_factorizer", {"number": 12}, "2"),
        ]
        for identity, name, arguments, expected in cases:
            observations = []

            def actor(
                messages,
                tools,
                name=name,
                arguments=arguments,
                observations=observations,
            ):
                feedback = [m for m in messages if m["role"] == "tool"]
                if not feedback:
                    return call(name, arguments)
                observations.append(feedback[-1]["content"])
                return {"role": "assistant", "content": "<answer>fixture</answer>"}

            with self.subTest(identity=identity):
                self.fixture(actor, identity=identity)
                self.assertEqual(len(observations), 1)
                self.assertIn(expected, observations[0])
                self.assertNotIn("error", observations[0].lower())

    def test_complete_published_coverage_and_input_privacy(self):
        self.assertEqual({key[1] for key in self.tasks}, {str(n) for n in range(995)})
        self.assertEqual(sum(len(task["tools"]) for task in self.tasks.values()), 3912)
        duplicates = 0
        for key, task in self.tasks.items():
            self.assertEqual(set(task), {"id", "question", "tools"})
            self.assertEqual(set(self.answers[key]), {"answer", "functions"})
            names = [tool["name"] for tool in task["tools"]]
            duplicates += len(names) != len(set(names))
        self.assertEqual(duplicates, 60)

    def fixture(self, actor, *, scenario="Free", max_turns=3, identity="0"):
        return run_fixture(
            self.config,
            self.tasks[("tasks", identity)],
            self.answers[("tasks", identity)],
            actor,
            scenario=scenario,
            max_turns=max_turns,
        )

    def test_real_multihop_execution_and_official_score(self):
        before = deepcopy((self.tasks[("tasks", "0")], self.answers[("tasks", "0")]))
        for scenario in ("Free", "Mandatory"):
            observed = []

            def actor(messages, tools, observed=observed):
                observed.append(deepcopy(messages))
                self.assertTrue(
                    all(set(tool) == {"type", "function"} for tool in tools)
                )
                self.assertNotIn("functions", messages[0])
                feedback = [
                    message for message in messages if message["role"] == "tool"
                ]
                if not feedback:
                    return call("extract_first_name", {"full_name": "Thomas Mawson"})
                if len(feedback) == 1:
                    return call(
                        "count_letters",
                        {
                            "input": json.loads(feedback[-1]["content"]),
                            "ignore_position": ["first", "last"],
                        },
                    )
                return {
                    "role": "assistant",
                    "content": f"<answer>{json.loads(feedback[-1]['content'])}</answer>",
                }

            result = self.fixture(actor, scenario=scenario)
            self.assertEqual(result["answer_correct"], 1)
            self.assertEqual(result["turn"], 2)
            self.assertEqual(
                [m["content"] for m in result["messages"] if m["role"] == "tool"],
                ['"Thomas"', "4"],
            )
            self.assertEqual(len(observed), 3)
        self.assertEqual(
            before, (self.tasks[("tasks", "0")], self.answers[("tasks", "0")])
        )

    def test_tool_errors_and_request_limit_keep_official_behavior(self):
        for arguments, expected in (
            ({"input": 42}, '"error": "Input must be a string."'),
            ({}, "an error occured when call count_letters:"),
        ):
            result = self.fixture(
                lambda messages, tools, arguments=arguments: call(
                    "count_letters", arguments
                ),
                max_turns=1,
            )
            self.assertEqual(result["messages"][-1]["role"], "tool")
            self.assertIn(expected, result["messages"][-1]["content"])
            self.assertEqual(result["answer_correct"], 0)

    def test_timeout_preserves_batch_order_and_hides_exception_source(self):
        observed = []

        def actor(messages, tools):
            observed.append(deepcopy(messages))
            if len(observed) > 1:
                return {"role": "assistant", "content": "<answer>4</answer>"}
            response = call("count_letters", {"input": "four"})
            response["tool_calls"] += call("count_letters", {"input": "slow"})[
                "tool_calls"
            ]
            response["tool_calls"] += call(
                "extract_first_name", {"full_name": "Thomas Mawson"}
            )["tool_calls"]
            for index, tool_call in enumerate(response["tool_calls"]):
                tool_call["id"] = f"call_{index}"
            return response

        def timer(timeout, function, args=(), kwargs=None):
            self.assertEqual(timeout, 10)
            if args[2].get("input") == "slow":
                raise FunctionTimedOut(
                    timedOutAfter=timeout,
                    timedOutFunction=function,
                    timedOutArgs=("SECRET_SOURCE",),
                )
            return function(*args, **(kwargs or {}))

        with patch("func_timeout.dafunc.func_timeout", side_effect=timer) as boundary:
            result = self.fixture(actor)
        self.assertEqual(boundary.call_count, 3)
        self.assertEqual(result["answer_correct"], 1)
        feedback = [message for message in observed[-1] if message["role"] == "tool"]
        self.assertEqual(
            [message["tool_call_id"] for message in feedback],
            ["call_0", "call_1", "call_2"],
        )
        self.assertEqual(
            [message["content"] for message in feedback],
            ["4", '{"error":"tool_execution_timeout"}', '"Thomas"'],
        )
        self.assertEqual(result["tool_error_counts"], {"tool_execution_timeout": 1})
        self.assertEqual(
            result["tool_error_events"],
            [
                {
                    "event": "tool_execution_timeout",
                    "tool_call_id": "call_1",
                    "tool_name": "count_letters",
                    "timeout_seconds": 10,
                }
            ],
        )
        self.assertNotIn("SECRET_SOURCE", json.dumps((result, observed)))

    def test_repeated_timeouts_are_counted_and_reset_per_episode(self):
        with patch(
            "func_timeout.dafunc.func_timeout",
            side_effect=FunctionTimedOut("private timeout"),
        ):
            result = self.fixture(
                lambda messages, tools: call("count_letters", {"input": "slow"}),
                max_turns=2,
            )
        self.assertEqual(result["tool_error_counts"], {"tool_execution_timeout": 2})
        self.assertEqual(len(result["tool_error_events"]), 2)
        for scenario in ("Direct", "Free"):
            fresh = self.fixture(
                lambda messages, tools: {
                    "role": "assistant",
                    "content": "<answer>4</answer>",
                },
                scenario=scenario,
            )
            self.assertEqual(fresh["tool_error_counts"], {"tool_execution_timeout": 0})
            self.assertEqual(fresh["tool_error_events"], [])

    def test_timeout_extension_does_not_swallow_process_interrupts(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with (
                self.subTest(exception=exception),
                patch("func_timeout.dafunc.func_timeout", side_effect=exception),
                self.assertRaises(exception),
            ):
                self.fixture(
                    lambda messages, tools: call("count_letters", {"input": "four"})
                )

    def test_official_last_tool_answer_match_is_not_final_answer_em(self):
        def actor(messages, tools):
            if messages[-1]["role"] != "tool":
                return call("count_letters", {"input": "four"})
            return {"role": "assistant", "content": "<answer>definitely wrong</answer>"}

        result = self.fixture(actor)
        self.assertEqual(result["messages"][-2]["content"], "4")
        self.assertEqual(result["answer_correct"], 1)

    def test_respond_stops_without_forcing_tools_and_preserves_duplicates(self):
        for scenario in ("Direct", "Free", "Mandatory"):
            result = self.fixture(
                lambda messages, tools: {
                    "role": "assistant",
                    "content": "<answer>4</answer>",
                },
                scenario=scenario,
            )
            self.assertEqual(result["answer_correct"], 1)
            self.assertEqual(len(result["messages"]), 2)
        captured = []
        self.fixture(
            lambda messages, tools: (
                captured.extend(tools) or {"role": "assistant", "content": "x"}
            ),
            identity="44",
        )
        self.assertEqual(
            [tool["function"] for tool in captured],
            self.tasks[("tasks", "44")]["tools"],
        )

    def test_invalid_actor_or_explicit_failure_is_never_a_scored_zero(self):
        for message in (
            None,
            {},
            {"role": "assistant"},
            {"role": "assistant", "content": False},
        ):
            with self.assertRaises(ValueError):
                self.fixture(lambda messages, tools, message=message: message)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):

            def fail(messages, tools):
                raise RuntimeError("fixture failure")

            self.fixture(fail)

    def test_explicit_null_tool_calls_is_an_official_final_reply(self):
        result = self.fixture(
            lambda messages, tools: {
                "role": "assistant",
                "content": "<answer>4</answer>",
                "tool_calls": None,
            }
        )
        self.assertEqual(result["answer_correct"], 1)
        self.assertIsNone(result["messages"][-1]["tool_calls"])

    def test_required_fields_invalid_values_and_unknown_ids_fail(self):
        tool = self.tasks[("tasks", "0")]["tools"][0]
        for field in ("name", "description", "parameters"):
            bad = deepcopy(tool)
            del bad[field]
            with self.assertRaises(ValueError):
                visible_tools({"hidden": bad})
        for field, value in (("name", ""), ("description", None), ("parameters", {})):
            bad = {**tool, field: value}
            with self.assertRaises((ValueError, TypeError)):
                visible_tools({"hidden": bad})
        config = deepcopy(self.config)
        config["task_ids"] = {"tasks": ["not_an_id"]}
        with self.assertRaisesRegex(ValueError, "unknown"):
            load_tasks(config)
        path = WORKSPACE_ROOT / self.config["benchmark"]["path"] / "data/ToolHop.json"
        row = json.loads(path.read_text())[0]
        for field in ("id", "question", "tools", "functions", "answer"):
            bad = {key: value for key, value in row.items() if key != field}
            with (
                patch("jev_bench.benchmarks.toolhop.json.loads", return_value=[bad]),
                self.assertRaises(KeyError),
            ):
                load_tasks(self.config)


if __name__ == "__main__":
    unittest.main()
