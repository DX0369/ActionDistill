import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.tau2 import run_episode, score_saved
from jev_bench.client import QwenClient, SimulatorClient
from jev_bench.config import PROJECT_ROOT
from tests.pa_fixtures import alias_message, freeze

from tests_tau2.test_tau2 import ScriptedEndpoint, config, task_and_scripts


class PaToolTau2Tests(unittest.TestCase):
    def test_three_domains_keep_simulated_user_and_official_scoring_canonical(self):
        value = config()
        from tau2.registry import registry

        for domain in ("retail", "airline", "telecom"):
            with (
                self.subTest(domain=domain),
                tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as directory,
            ):
                root = Path(directory)
                task, agent_messages, user_messages = task_and_scripts(domain)
                tools = [
                    tool.openai_schema
                    for tool in registry.get_env_constructor(domain)().get_tools()
                ]
                item = {
                    "condition": "native",
                    "category": domain,
                    "task_id": task.id,
                    "trial": 0,
                    "seed": 20261015,
                    "attempt": 1,
                }
                p = freeze(value, item, tools, root / "mapping")
                agent = ScriptedEndpoint(
                    [alias_message(m, tools, p.rows) for m in agent_messages]
                )
                user = ScriptedEndpoint(user_messages)
                outcome = run_episode(
                    value,
                    item,
                    task,
                    None,
                    root / "episode",
                    client_factory=agent.factory(QwenClient),
                    user_client_factory=user.factory(SimulatorClient),
                    pa_tool_identity=p.asset_identity,
                )
                self.assertTrue(outcome["official_scored"])
                self.assertTrue(outcome["success"])
                self.assertTrue(
                    all(
                        t["function"]["name"].startswith("pa_tool_")
                        for r in agent.requests
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
                (root / "episode/score.json").unlink()
                with patch(
                    "jev_bench.client.OpenAI",
                    side_effect=AssertionError("no inference during recovery"),
                ):
                    self.assertTrue(
                        score_saved(value, item, task, None, root / "episode")[
                            "success"
                        ]
                    )
