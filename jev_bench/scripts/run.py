#!/usr/bin/env python
"""Preview by default. Model requests require --execute; service checks are explicit."""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jev_bench.benchmarks import configure, load_tasks
from jev_bench.config import (
    EXTENDED_BENCHMARKS,
    WORKSPACE_ROOT,
    git_dirty,
    identifier,
    load_config,
    source_identity,
)
from jev_bench.io import now, read_jsonl, write_json
from jev_bench.runner import execute, make_chains, make_plan, run_lock
from jev_bench.services import check_services
from jev_bench.stats import summarize


def code_version(config, config_path):
    repository = PROJECT_ROOT.parent
    selected = str(Path(config_path).resolve().relative_to(repository.resolve()))
    paths = ["jev_bench/src", "jev_bench/scripts", selected]
    if any("advisor" in policy for policy in config["conditions"].values()):
        paths.extend(
            [
                "jev_train/scripts/serve_jev.py",
                "jev_train/src/jev_train",
                "dataset_tool_inventory/src",
            ]
        )
    identity = source_identity(repository, paths)
    trees, dirty = identity["trees"], identity["dirty_paths"]
    if any("advisor" in policy for policy in config["conditions"].values()):
        decider = WORKSPACE_ROOT / "third_party/decider"
        trees["third_party/decider/decider"] = subprocess.check_output(
            ["git", "-C", str(decider), "rev-parse", "HEAD:decider"], text=True
        ).strip()
        dirty.extend(
            "third_party/decider: " + entry for entry in git_dirty(decider, ["decider"])
        )
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--check-services", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.execute and args.run_id is None:
        parser.error("--execute requires --run-id")
    if args.run_id is not None:
        identifier(args.run_id)
    config = load_config(args.config)
    if (
        config["benchmark"]["name"] in EXTENDED_BENCHMARKS
        and config["protocol"]["mode"] == "cpu_only"
        and (args.execute or args.check_services)
    ):
        raise ValueError(
            f"{config['benchmark']['name']} is CPU-only: live inputs, failure handling, "
            "scope and statistics must be approved before service checks or execution"
        )
    (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="preview_", dir=PROJECT_ROOT / "tmp"
    ) as runtime:
        configure(config, runtime)
        tasks, answers = load_tasks(config)
        plan = make_plan(config, tasks)
        print(
            json.dumps(
                {
                    "decision_points"
                    if config["benchmark"]["name"] == "apibank"
                    else "episodes": len(plan),
                    "tasks": len(tasks),
                    "trials": config["trials"],
                    "conditions": list(config["conditions"]),
                    "categories": {
                        category: sum(
                            task_category == category for task_category, _ in tasks
                        )
                        for category in config["task_ids"]
                    },
                    "chains": {
                        url: len(items)
                        for url, items in make_chains(config, plan).items()
                    },
                    "execute": args.execute,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        pa_tool_assets = {}
        if args.check_services or args.execute:
            from jev_bench.pa_tool import asset_identity, load_projection

            pa_tool_assets = {
                condition: asset_identity(policy["pa_tool_mapping"])
                for condition, policy in config["conditions"].items()
                if "pa_tool_mapping" in policy
            }
            for item in plan:
                if item["condition"] in pa_tool_assets:
                    load_projection(
                        config,
                        item,
                        expected_identity=pa_tool_assets[item["condition"]],
                    )
        services = (
            check_services(config) if args.check_services or args.execute else None
        )
        if not args.execute:
            if services is not None:
                print(json.dumps(services, ensure_ascii=False))
            return 0
        code = code_version(config, args.config)
        if config["output_root"] == "outputs" and code["dirty_paths"]:
            raise ValueError(
                "formal execution requires committed jev_bench code and configuration"
            )
        run_dir = PROJECT_ROOT / config["output_root"] / args.run_id
        status_path = run_dir / "metadata/run_status.json"
        if not args.resume:
            run_dir.mkdir(parents=True, exist_ok=False)
        elif not status_path.exists():
            raise ValueError("resume requires an existing run status")
        with run_lock(run_dir) as stream:
            if args.resume:
                state = json.loads(status_path.read_text())
                if (
                    code["dirty_paths"]
                    or state["config"] != config
                    or state["plan"] != plan
                    or state["code"]["trees"] != code["trees"]
                ):
                    raise ValueError(
                        "resume requires the original committed code, config and plan"
                    )
                if state["status"] not in ("partial", "running", "interrupted"):
                    raise ValueError(
                        "only partial/interrupted-running batches can resume; diagnose failed batches"
                    )
                if state["services"] != services:
                    raise ValueError(
                        "resume service identities differ from the original run"
                    )
                if pa_tool_assets and state["pa_tool_assets"] != pa_tool_assets:
                    raise ValueError(
                        "resume PA-Tool assets differ from the original run"
                    )
            else:
                state = {
                    "status": "running",
                    "started_at": now(),
                    "config": config,
                    "plan": plan,
                    "code": code,
                    "services": services,
                    "attempts": 0,
                }
                if pa_tool_assets:
                    state["pa_tool_assets"] = pa_tool_assets
                write_json(status_path, state)
            stream.write(shlex.join([sys.executable, *sys.argv]) + "\n")
            stream.flush()
            rows = execute(config, plan, tasks, answers, run_dir, state)
            try:
                if config["benchmark"]["name"] == "apibank":
                    from jev_bench.benchmarks.apibank_group_scoring import score_groups

                    state["phase"] = "scoring"
                    write_json(status_path, state)
                    score_groups(
                        config, plan, tasks, run_dir, code=code, recovery=False
                    )
                    rows = read_jsonl(run_dir / "episodes.jsonl")
                state["phase"] = "summarizing"
                metrics = summarize(
                    rows,
                    plan,
                    config["comparisons"],
                    benchmark=config["benchmark"]["name"],
                    formal=config.get("protocol", {}).get("mode") == "formal",
                )
                write_json(run_dir / "metrics.json", metrics)
                state.update(
                    status="completed" if metrics["coverage_complete"] else "partial",
                    phase="finished",
                )
                if metrics["coverage_complete"]:
                    state.pop("error", None)
            except BaseException as exc:
                state.update(
                    status="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                if (
                    config["benchmark"]["name"] == "apibank"
                    and state["phase"] == "scoring"
                ):
                    from jev_bench.benchmarks.apibank_group_scoring import (
                        GroupScoringError,
                    )

                    if isinstance(exc, GroupScoringError):
                        state["error"].update(
                            scoring_attempts=exc.scoring_attempts,
                            scoring_groups=exc.scoring_groups,
                        )
                raise
            finally:
                state["ended_at"] = now()
                write_json(status_path, state)
        return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
