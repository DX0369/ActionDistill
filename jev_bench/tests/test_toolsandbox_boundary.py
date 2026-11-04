import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from jev_bench.benchmarks.toolsandbox import run_episode
from jev_bench.benchmarks.toolsandbox_scores import validate_score
from jev_bench.config import load_config, validate
from jev_bench.services import check_services

from tests.support import PROJECT


class ToolSandboxBoundaryTests(unittest.TestCase):
    def config(self):
        return load_config(PROJECT / "configs/toolsandbox_cpu_preview.json")

    def test_cpu_configuration_is_explicit(self):
        value = self.config()
        self.assertIsNone(value["user_simulator"])
        self.assertEqual(value["protocol"], {"mode": "cpu_only", "scenario_seed": 42})
        for field, replacement in (
            ("output_root", "outputs"),
            ("user_simulator", {}),
            ("protocol", {"mode": "live", "scenario_seed": 42}),
            ("protocol", {"mode": "cpu_only", "scenario_seed": True}),
        ):
            with self.subTest(field=field, replacement=replacement):
                invalid = deepcopy(value)
                invalid[field] = replacement
                with self.assertRaises(ValueError):
                    validate(invalid)
        for field in ("protocol", "user_simulator"):
            invalid = deepcopy(value)
            del invalid[field]
            with self.assertRaises(KeyError):
                validate(invalid)

    def test_runtime_rejects_cpu_preview(self):
        with self.assertRaisesRegex(ValueError, "CPU preview"):
            run_episode.__wrapped__(self.config(), {}, {}, None, PROJECT / "tmp")

    def test_formal_requires_rapidapi_key_before_service_requests(self):
        value = load_config(PROJECT / "configs/toolsandbox_main_smoke.json")
        value["protocol"]["mode"] = "formal"
        value["output_root"] = "outputs"
        value["task_ids"] = {"scenarios": "all"}
        validate(value)
        for key in (None, "", " \t"):
            environment = {} if key is None else {"RAPID_API_KEY": key}
            with (
                self.subTest(key=key),
                patch.dict(os.environ, environment, clear=True),
                patch("jev_bench.services.json_request") as request,
                patch("jev_bench.services.make_advisor") as advisor,
            ):
                with self.assertRaisesRegex(ValueError, "RAPID_API_KEY"):
                    check_services(value)
                request.assert_not_called()
                advisor.assert_not_called()

    def test_rapidapi_requirement_is_only_for_formal_toolsandbox(self):
        for mode, key in (("smoke", None), ("formal", "cpu-test-key")):
            value = load_config(PROJECT / "configs/toolsandbox_main_smoke.json")
            value["conditions"] = {"native": value["conditions"]["native"]}
            value["advisors"] = {}
            value["protocol"]["mode"] = mode
            if mode == "formal":
                value["output_root"] = "outputs"
                value["task_ids"] = {"scenarios": "all"}
            validate(value)
            executor = value["conditions"]["native"]["executor"]
            settings = value["executors"][executor]
            model = {
                "id": settings["model"],
                "root": str(Path(settings["model_path"]).resolve()),
                "max_model_len": settings["max_model_len"],
            }
            environment = {value["user_simulator"]["api_key_env"]: "cpu-simulator"}
            if key is not None:
                environment["RAPID_API_KEY"] = key
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "jev_bench.services.json_request", return_value={"data": [model]}
                ),
            ):
                observed = check_services(value)
            self.assertEqual(observed["executors"][executor], model)
            self.assertNotIn("cpu-test-key", str(observed))

    def test_continuous_score_never_becomes_boolean(self):
        for milestone, minefield in ((0, 0), (0.75, 0), (1, 0), (1, 0.01)):
            value = {
                "similarity": milestone if minefield == 0 else 0,
                "milestone_similarity": milestone,
                "minefield_similarity": minefield,
                "turn_count": 4,
            }
            validate_score(value)
        base = {
            "similarity": 0.75,
            "milestone_similarity": 0.75,
            "minefield_similarity": 0,
            "turn_count": 4,
        }
        for field in base:
            invalid = deepcopy(base)
            del invalid[field]
            with self.assertRaises(KeyError):
                validate_score(invalid)
        for value in (None, True, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_score({**base, "similarity": value})
        with self.assertRaisesRegex(ValueError, "official"):
            validate_score({**base, "similarity": 1})


if __name__ == "__main__":
    unittest.main()
