import io
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import patch

from jev_bench.benchmarks import run_episode
from jev_bench.config import load_config, validate
from jev_bench.services import check_services
from jev_bench.stats import paired, summarize

from tests.support import PROJECT
from tests.test_cli import cli


class CpuAdapterBoundaryTests(unittest.TestCase):
    def test_protocol_fields_and_output_boundaries(self):
        for name in ("toolhop", "stabletoolbench"):
            config = load_config(PROJECT / f"configs/{name}_cpu_preview.json")
            for field in config["protocol"]:
                bad = deepcopy(config)
                del bad["protocol"][field]
                with self.assertRaises(ValueError):
                    validate(bad)
            for change in ({"output_root": "outputs"}, {"user_simulator": None}):
                with self.assertRaises(ValueError):
                    validate({**config, **change})
            bad = deepcopy(config)
            bad["protocol"]["mode"] = "live"
            with self.assertRaises(ValueError):
                validate(bad)
        bad = deepcopy(config)
        bad["protocol"]["cache_archive"] = "relative.zip"
        with self.assertRaises(ValueError):
            validate(bad)

    def test_live_cli_stops_before_any_service_or_runtime(self):
        for name in ("toolhop", "stabletoolbench", "toolsandbox", "apibank"):
            path = PROJECT / f"configs/{name}_cpu_preview.json"
            for extra in (["--check-services"], ["--execute", "--run-id", "forbidden"]):
                with (
                    self.subTest(benchmark=name, arguments=extra),
                    patch("sys.argv", ["run.py", "--config", str(path), *extra]),
                    patch.object(cli, "configure") as configure,
                    patch.object(cli, "check_services") as services,
                    patch.object(cli, "execute") as execute,
                    redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(ValueError, "CPU-only"),
                ):
                    cli.main()
                configure.assert_not_called()
                services.assert_not_called()
                execute.assert_not_called()

    def test_cpu_service_and_runtime_boundaries(self):
        for name in ("toolhop", "stabletoolbench"):
            config = load_config(PROJECT / f"configs/{name}_cpu_preview.json")
            with patch("jev_bench.services.json_request") as request:
                with self.assertRaisesRegex(ValueError, "CPU-only"):
                    check_services(config)
                request.assert_not_called()
            if name != "toolhop":
                with self.assertRaisesRegex(ValueError, "CPU preview"):
                    run_episode(config)

    def test_toolhop_smoke_is_explicit_and_never_formal(self):
        config = load_config(PROJECT / "configs/toolhop_advisor_smoke.json")
        self.assertEqual(
            config["protocol"], {"mode": "smoke", "scenario": "Free", "max_turns": 9}
        )
        for field in ("scenario", "max_turns"):
            bad = deepcopy(config)
            del bad["protocol"][field]
            with self.assertRaises(ValueError):
                validate(bad)
        for change in ({"output_root": "outputs"}, {"comparisons": [["G1", "native"]]}):
            with self.assertRaises(ValueError):
                validate({**config, **change})
        for field, value in (("scenario", "Mandatory"), ("max_turns", 10)):
            bad = deepcopy(config)
            bad["protocol"][field] = value
            with self.assertRaises(ValueError):
                validate(bad)

    def test_toolhop_summary_does_not_use_boolean_comparisons(self):
        result = summarize([], [], [], benchmark="toolhop")
        self.assertEqual(result["purpose"], "engineering_smoke_only")
        self.assertNotIn("comparisons", result)
        with self.assertRaisesRegex(ValueError, "not approved"):
            summarize([], [], [["a", "b"]], benchmark="toolhop")
        with self.assertRaisesRegex(ValueError, "boolean"):
            paired({}, [], "a", "b", benchmark="toolhop")


if __name__ == "__main__":
    unittest.main()
