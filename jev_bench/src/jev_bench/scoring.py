"""Explicit recovery of finished inference, never retrying a model request."""

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from .benchmarks import adapter
from .cost import costs
from .io import now, read_jsonl, write_json
from .runner import directory_for, key, resolved, validate_attempts


def _index_score(rows, row, index, plan):
    identity = (key(row), row["attempt"])
    existing = [(key(r), r["attempt"]) for r in rows]
    if identity in existing:
        position = existing.index(identity)
        if rows[position] == row:
            return
        if resolved(rows[position]):
            raise ValueError("a counted outcome cannot be changed by score recovery")
        rows[position] = row
    else:
        rows.append(row)
    validate_attempts(rows, plan)
    staging = index.with_suffix(".jsonl.writing")
    staging.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )
    staging.replace(index)


def recover_scores(config, plan, tasks, answers, run_dir, *, code):
    """Caller holds run_lock. Recover only scoring-phase attempts in place.

    Keep the same attempt and cost, and leave its original failure diagnostics.
    Completed official outcomes and inference failures are never rescored.
    """
    if config["benchmark"]["name"] == "apibank":
        from .benchmarks.apibank_group_scoring import score_groups

        return score_groups(config, plan, tasks, run_dir, code=code, recovery=True)
    run_dir = Path(run_dir)
    index = run_dir / "episodes.jsonl"
    rows = read_jsonl(index)
    latest = validate_attempts(rows, plan)
    repaired = 0
    for planned in plan:
        previous = latest.get(key(planned))
        attempts = (
            [1] if previous is None else [previous["attempt"], previous["attempt"] + 1]
        )
        for attempt in attempts:
            item = {**planned, "attempt": attempt}
            directory = directory_for(run_dir, item)
            if not directory.exists():
                continue
            marker = directory / "diagnostics/attempt.json"
            if not marker.exists():
                raise ValueError(
                    f"incomplete attempt lacks ownership evidence; diagnosis required: {directory}"
                )
            with marker.open("r+") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError(
                        f"attempt worker is still active: {directory}"
                    ) from exc
                state = json.load(stream)
                if state["item"] != item:
                    raise ValueError(
                        "saved attempt identity differs from the frozen plan"
                    )
                row_path = directory / "episode.json"
                row = json.loads(row_path.read_text()) if row_path.exists() else None
                if row is not None and any(
                    row[field] != value for field, value in item.items()
                ):
                    raise ValueError("saved episode differs from the frozen plan")
                if state["phase"] != "scoring":
                    continue
                if row is not None and resolved(row):
                    if row.get("scoring_recovery") is not None:
                        # Recovery may itself have stopped between episode/index writes.
                        _index_score(rows, row, index, plan)
                    continue
                if row is not None and row["failure_kind"] not in (
                    "scoring_error",
                    "interrupted",
                ):
                    raise ValueError(
                        "non-scoring failure cannot be recovered by scoring"
                    )
                trajectory = directory / "trajectory.json"
                if (
                    not trajectory.exists()
                    or json.loads(trajectory.read_text()).get("partial") is True
                ):
                    raise ValueError(
                        f"scoring recovery requires a complete saved trajectory: {directory}"
                    )
                if row is None:
                    diagnostics = directory / "diagnostics"
                    cost = costs(
                        SimpleNamespace(
                            records=read_jsonl(diagnostics / "executor.jsonl")
                        ),
                        SimpleNamespace(
                            advisor_records=read_jsonl(diagnostics / "advisor.jsonl")
                        ),
                        SimpleNamespace(
                            records=read_jsonl(diagnostics / "user_simulator.jsonl")
                        ),
                        advisor_type=state["advisor_type"],
                        advisor_requests=read_jsonl(
                            diagnostics / "advisor_requests.jsonl"
                        ),
                    )
                    cost["accounting_complete"] = False
                    row = {
                        **item,
                        "success": None,
                        "official_scored": False,
                        "failure_kind": "scoring_error",
                        "cost": cost,
                        "decision_steps": None,
                        "tool_call_count": None,
                        "duration_seconds": None,
                        "response_anomalies": [],
                    }
                try:
                    outcome = adapter(config).score_saved(
                        config,
                        item,
                        tasks[item["category"], item["task_id"]],
                        answers[item["category"], item["task_id"]],
                        directory,
                    )
                except BaseException:
                    if config["benchmark"]["name"] == "stabletoolbench":
                        # A judge retry can incur cost and still fail. Its adapter
                        # persists the complete ledger-derived cost before raising.
                        failed = json.loads(row_path.read_text())
                        _index_score(rows, failed, index, plan)
                    raise
                row.update(
                    **outcome,
                    official_scored=True,
                    failure_kind="model_task_failure"
                    if outcome["success"] is False
                    else None,
                    scoring_recovery={
                        "at": now(),
                        "code": code,
                        "previous_failure_kind": row["failure_kind"],
                    },
                )
                write_json(row_path, row)
                repaired += 1
            # Atomic index replacement keeps this inference attempt counted once.
            _index_score(rows, row, index, plan)
    return repaired
