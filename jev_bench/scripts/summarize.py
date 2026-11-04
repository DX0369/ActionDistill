#!/usr/bin/env python
"""Summarize one run's canonical episode index without contacting any model."""

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jev_bench.benchmarks import configure, load_tasks
from jev_bench.config import check_checkout, source_identity, validate
from jev_bench.io import now, write_json
from jev_bench.runner import key, pending_plan, reconcile, resolved, run_lock
from jev_bench.scoring import recover_scores
from jev_bench.stats import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="recover scoring from saved trajectories; never regenerate actor output (StableToolBench may resume its judge)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    if run_dir.parent not in (PROJECT_ROOT / "tmp", PROJECT_ROOT / "outputs"):
        raise ValueError("summarize only runs belonging to jev_bench")
    with run_lock(run_dir) as command:
        state = json.loads((run_dir / "metadata/run_status.json").read_text())
        original_error = state.get("error")
        scoring_attempts = None
        scoring_groups = None
        summary_recovery = (
            state["status"] == "failed"
            and state.get("phase") == "summarizing"
            and not args.rescore
        )
        if state["status"] == "failed" and not summary_recovery:
            scoring_attempts = (original_error or {}).get("scoring_attempts")
            scoring_groups = (original_error or {}).get("scoring_groups")
            group_failure = (
                state["config"]["benchmark"]["name"] == "apibank"
                and state.get("phase") == "scoring"
                and isinstance(scoring_groups, list)
                and bool(scoring_groups)
            )
            if (
                not args.rescore
                or (not group_failure and not scoring_attempts)
                or not isinstance(scoring_attempts, list)
            ):
                raise ValueError(
                    "failed batch lacks scoring-only failure evidence; diagnose without clearing its error"
                )
        check_checkout(validate(state["config"]))
        (PROJECT_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="rescore_", dir=PROJECT_ROOT / "tmp"
        ) as runtime:
            configure(state["config"], runtime)
            if args.rescore:
                code = source_identity(
                    PROJECT_ROOT.parent, ["jev_bench/src", "jev_bench/scripts"]
                )
                if run_dir.parent == PROJECT_ROOT / "outputs" and code["dirty_paths"]:
                    raise ValueError(
                        "formal score recovery requires committed runtime code"
                    )
                state["scoring_recovery"] = {
                    **state.get("scoring_recovery", {}),
                    "at": now(),
                    "code": code,
                }
                write_json(run_dir / "metadata/run_status.json", state)
                command.write(shlex.join([sys.executable, *sys.argv]) + "\n")
                command.flush()
                tasks, answers = load_tasks(state["config"])
                recover_scores(
                    state["config"], state["plan"], tasks, answers, run_dir, code=code
                )
        rows = reconcile(run_dir, state["plan"], recover=args.rescore)
        if scoring_groups:
            for group in scoring_groups:
                score_path = (
                    run_dir
                    / "scores/apibank"
                    / group["condition"]
                    / f"trial_{group['trial']}.json"
                )
                saved = json.loads(score_path.read_text())
                if saved["members"] != group["members"]:
                    raise ValueError(
                        "original scoring group differs from recovered attempts"
                    )
                for item in group["members"]:
                    matches = [
                        row
                        for row in rows
                        if key(row) == key(item) and row["attempt"] == item["attempt"]
                    ]
                    if (
                        len(matches) == 1
                        and matches[0]["failure_kind"] == "executor_response_error"
                        and matches[0]["success"] is False
                        and matches[0]["official_scored"] is False
                    ):
                        points = [
                            point
                            for point in saved["score"]["points"]
                            if point["category"] == item["category"]
                            and point["task_id"] == item["task_id"]
                        ]
                        if (
                            len(points) != 1
                            or points[0]["failure_kind"] != "executor_response_error"
                            or points[0]["official_scored"] is not False
                        ):
                            raise ValueError(
                                "terminal failure differs from its canonical group result"
                            )
                        continue
                    if (
                        len(matches) != 1
                        or not matches[0]["official_scored"]
                        or matches[0]["official_score_source"]
                        != str(score_path.relative_to(run_dir))
                    ):
                        raise ValueError(
                            "original API-Bank group has not been completely score-recovered"
                        )
        elif scoring_attempts is not None:
            for item in scoring_attempts:
                matches = [
                    row
                    for row in rows
                    if key(row) == key(item) and row["attempt"] == item["attempt"]
                ]
                if (
                    state["config"]["benchmark"]["name"] == "apibank"
                    and len(matches) == 1
                    and matches[0]["failure_kind"] == "pending_scoring"
                    and matches[0]
                    .get("scoring_recovery", {})
                    .get("previous_failure_kind")
                    == "scoring_error"
                ):
                    # API-Bank's validated saved prediction is waiting for the
                    # remaining points in its arm; no partial L3 scoring occurs.
                    continue
                if (
                    len(matches) != 1
                    or not matches[0]["official_scored"]
                    or not resolved(matches[0])
                    or matches[0]
                    .get("scoring_recovery", {})
                    .get("previous_failure_kind")
                    != "scoring_error"
                ):
                    raise ValueError(
                        "the batch's original scoring failures have not all been recovered"
                    )
        result = summarize(
            rows,
            state["plan"],
            state["config"]["comparisons"],
            benchmark=state["config"]["benchmark"]["name"],
            formal=state["config"].get("protocol", {}).get("mode") == "formal",
        )
        terminal_coverage = (
            result["execution_coverage_complete"]
            if state["config"]["benchmark"]["name"]
            in ("apibank", "toolsandbox", "stabletoolbench")
            else result["coverage_complete"]
        )
        if summary_recovery and not terminal_coverage:
            raise ValueError("summary-only recovery requires complete scored coverage")
        # Fatal episodes still block recovery, even if some other scores were repaired.
        if args.rescore:
            pending_plan(state["plan"], rows)
        write_json(run_dir / "metrics.json", result)
        if original_error is not None and (args.rescore or summary_recovery):
            state.setdefault("scoring_recovery", {}).setdefault(
                "previous_errors", []
            ).append(original_error)
        if result["coverage_complete"]:
            state.update(
                status="completed",
                phase="finished",
                ended_at=now(),
                active=[],
                attempts=result["attempts"],
            )
            state.pop("error", None)
            write_json(run_dir / "metadata/run_status.json", state)
        elif args.rescore or summary_recovery:
            # A repaired early episode leaves the remaining plan runnable. Other
            # fatal failures still block resume and must not be cleared here.
            state.update(
                status="partial",
                phase="finished" if terminal_coverage else "awaiting_resume",
                ended_at=now(),
                active=[],
                attempts=result["attempts"],
            )
            state.pop("error", None)
            write_json(run_dir / "metadata/run_status.json", state)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("coverage_complete", "planned", "attempts", "resolved")
            }
        )
    )


if __name__ == "__main__":
    main()
