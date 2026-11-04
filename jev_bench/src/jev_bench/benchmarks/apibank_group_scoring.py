"""API-Bank ordered group scoring and recovery.

The caller holds run_lock and must not dispatch more generation concurrently.
Each worker ownership marker is checked before reading and again before writing.
No model client is imported or called by this module.
"""

import fcntl
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ..config import PROJECT_ROOT, asset_path, fields, number
from ..cost import costs
from ..io import append_json, now, read_jsonl, write_json
from ..runner import directory_for, key, resolved, validate_attempts

SCORING_WORKER = PROJECT_ROOT / "scripts/apibank_score_worker.py"


def scoring_python():
    value = os.environ.get("JEV_APIBANK_SCORING_PYTHON")
    if value is None or not value.strip():
        raise ValueError(
            "JEV_APIBANK_SCORING_PYTHON must specify the scoring interpreter"
        )
    asset_path(value)
    path = Path(value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(
            f"API-Bank scoring interpreter is missing or not executable: {path}"
        )
    return path


class GroupScoringError(RuntimeError):
    def __init__(self, message, scoring_attempts, scoring_groups):
        super().__init__(message)
        self.scoring_attempts = scoring_attempts
        self.scoring_groups = scoring_groups


def _worker_score(config, predictions, *, terminal_failures=()):
    payload = {
        "config": config,
        "predictions": [
            {"category": category, "task_id": task_id, "prediction": prediction}
            for (category, task_id), prediction in predictions.items()
        ],
        "terminal_failures": [
            {
                "category": category,
                "task_id": task_id,
                "failure_kind": "executor_response_error",
            }
            for category, task_id in terminal_failures
        ],
    }
    result = subprocess.run(
        [str(scoring_python()), "-B", str(SCORING_WORKER)],
        input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    return json.loads(result.stdout)


def _read_member(run_dir, planned, previous, task):
    """Discover the latest saved attempt, including a row lost before indexing."""
    attempt = 1 if previous is None else previous["attempt"]
    next_item = {**planned, "attempt": attempt + 1}
    if previous is not None and directory_for(run_dir, next_item).exists():
        if resolved(previous) or previous["failure_kind"] == "pending_scoring":
            raise ValueError("completed inference has an unauthorized later attempt")
        attempt += 1
    item = {**planned, "attempt": attempt}
    directory = directory_for(run_dir, item)
    if not directory.exists():
        return None
    marker = directory / "diagnostics/attempt.json"
    if not marker.exists():
        raise ValueError(f"API-Bank attempt lacks ownership evidence: {directory}")
    with marker.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.load(stream)
        if state["item"] != item:
            raise ValueError("API-Bank saved attempt differs from the frozen plan")
        row_path = directory / "episode.json"
        if row_path.exists():
            saved_row = json.loads(row_path.read_text())
            if any(saved_row[field] != value for field, value in item.items()):
                raise ValueError("API-Bank saved episode differs from its attempt")
            if saved_row["failure_kind"] == "executor_response_error":
                if (
                    saved_row["source_id"] != task["source_id"]
                    or saved_row["success"] is not False
                    or saved_row["official_scored"] is not False
                ):
                    raise ValueError("invalid terminal API-Bank interface failure")
                return {
                    "item": item,
                    "directory": directory,
                    "row": saved_row,
                    "trajectory": None,
                }
        if state["phase"] != "scoring":
            return None
        trajectory = json.loads((directory / "trajectory.json").read_text())
        if (
            trajectory["category"] != item["category"]
            or trajectory["task_id"] != item["task_id"]
            or trajectory["source_id"] != task["source_id"]
            or trajectory["level"] != task["level"]
            or trajectory["kind"] != task["kind"]
            or trajectory["partial"] is not False
            or not isinstance(trajectory["prediction"], str)
        ):
            raise ValueError(
                "API-Bank scoring requires matching complete original text"
            )
        row_path = directory / "episode.json"
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if any(row[field] != value for field, value in item.items()):
                raise ValueError("API-Bank saved episode differs from its attempt")
            if not resolved(row) and row["failure_kind"] not in (
                "pending_scoring",
                "scoring_error",
                "interrupted",
            ):
                raise ValueError(
                    "non-scoring API-Bank failure cannot be score-recovered"
                )
        else:
            diagnostics = directory / "diagnostics"
            cost = costs(
                SimpleNamespace(records=read_jsonl(diagnostics / "executor.jsonl")),
                SimpleNamespace(
                    advisor_records=read_jsonl(diagnostics / "advisor.jsonl")
                ),
                advisor_type=state["advisor_type"],
                advisor_requests=read_jsonl(diagnostics / "advisor_requests.jsonl"),
            )
            cost["accounting_complete"] = False
            row = {
                **item,
                "source_id": task["source_id"],
                "success": None,
                "official_scored": False,
                "failure_kind": "pending_scoring",
                "decision_steps": trajectory["decision_steps"],
                "tool_call_count": None,
                "duration_seconds": None,
                "response_anomalies": trajectory["response_anomalies"],
                "cost": cost,
            }
            write_json(row_path, row)
    return {"item": item, "directory": directory, "row": row, "trajectory": trajectory}


def _persist_member(member, row):
    marker = member["directory"] / "diagnostics/attempt.json"
    with marker.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.load(stream)
        if state["item"] != member["item"] or state["phase"] != "scoring":
            raise ValueError("API-Bank attempt ownership changed during scoring")
        write_json(member["directory"] / "episode.json", row)


def _point_scores(result, tasks, *, terminal_failures=()):
    fields(
        result, ("points", "categories", "protocol_events", "level_3_sample_accuracy")
    )
    if not isinstance(result["points"], list):
        raise TypeError("API-Bank batch points must be a list")
    points = {}
    for point in result["points"]:
        identity = point["category"], point["task_id"]
        if identity in points or identity not in tasks:
            raise ValueError("API-Bank batch score has duplicate or unknown point IDs")
        task = tasks[identity]
        if point["source_id"] != task["source_id"]:
            raise ValueError("API-Bank batch score source identity differs")
        if identity in terminal_failures:
            if (
                point["failure_kind"] != "executor_response_error"
                or point["official_scored"] is not False
                or "correct" in point
                or "rouge_l_f" in point
            ):
                raise ValueError(
                    "terminal API-Bank failure must not have an invented score"
                )
        elif task["kind"] == "api":
            if type(point["correct"]) is not bool:
                raise TypeError("API-Bank API correctness must be boolean")
        else:
            number(point["rouge_l_f"], maximum=1)
        points[identity] = point
    if points.keys() != tasks.keys():
        raise ValueError("API-Bank batch score must cover the selected points exactly")
    return points


def score_groups(config, plan, tasks, run_dir, *, code, recovery):
    """Score every fully generated condition/trial; return repaired point count.

    Incomplete groups retain completed predictions as pending_scoring. A marker
    that already says scoring but lacks its full trajectory fails explicitly.
    """
    from ..scoring import _index_score

    run_dir = Path(run_dir)
    index = run_dir / "episodes.jsonl"
    rows = read_jsonl(index)
    latest = validate_attempts(rows, plan)
    groups = {}
    for planned in plan:
        group = groups.setdefault((planned["condition"], planned["trial"]), {})
        group[planned["category"], planned["task_id"]] = planned
    repaired = 0
    for (condition, trial), planned_group in groups.items():
        if planned_group.keys() != tasks.keys():
            raise ValueError(
                "API-Bank arm/trial plan differs from the selected task set"
            )
        group_path = run_dir / "scores" / "apibank" / condition / f"trial_{trial}.json"
        members = []
        for identity, task in tasks.items():
            planned = planned_group[identity]
            member = _read_member(run_dir, planned, latest.get(key(planned)), task)
            if member is None:
                continue
            if member["row"]["official_scored"] and not group_path.exists():
                raise ValueError(
                    "completed API-Bank point lacks its canonical group score"
                )
            # This also protects a completed generation whose episode/index write
            # was interrupted, even when the rest of its group is unfinished.
            _index_score(rows, member["row"], index, plan)
            latest[key(planned)] = member["row"]
            members.append(member)
        if len(members) != len(tasks):
            if group_path.exists():
                raise ValueError(
                    "canonical API-Bank group score has missing inference members"
                )
            # Finished predictions may have been interrupted at the scoring
            # boundary before the rest of this arm was generated. Restore only
            # their pending state; never score a partial L3 group or regenerate.
            for member in members:
                row = member["row"]
                if row["failure_kind"] in ("scoring_error", "interrupted"):
                    row = {
                        **row,
                        "failure_kind": "pending_scoring",
                        "scoring_recovery": {
                            "at": now(),
                            "code": code,
                            "previous_failure_kind": row["failure_kind"],
                        },
                    }
                    _persist_member(member, row)
                    _index_score(rows, row, index, plan)
                    repaired += 1
            continue
        manifest = [member["item"] for member in members]
        terminal_failures = {
            (member["item"]["category"], member["item"]["task_id"])
            for member in members
            if member["trajectory"] is None
        }
        try:
            if group_path.exists():
                saved = json.loads(group_path.read_text())
                if (
                    saved["condition"] != condition
                    or saved["trial"] != trial
                    or saved["members"] != manifest
                ):
                    raise ValueError(
                        "saved API-Bank group score belongs to different attempts"
                    )
                result = saved["score"]
                points = _point_scores(
                    result, tasks, terminal_failures=terminal_failures
                )
            else:
                predictions = {
                    (member["item"]["category"], member["item"]["task_id"]): member[
                        "trajectory"
                    ]["prediction"]
                    for member in members
                    if member["trajectory"] is not None
                }
                result = (
                    _worker_score(
                        config, predictions, terminal_failures=terminal_failures
                    )
                    if terminal_failures
                    else _worker_score(config, predictions)
                )
                points = _point_scores(
                    result, tasks, terminal_failures=terminal_failures
                )
                write_json(
                    group_path,
                    {
                        "condition": condition,
                        "trial": trial,
                        "members": manifest,
                        "scored_at": now(),
                        "code": code,
                        "score": result,
                    },
                )
            for member in members:
                row = member["row"]
                if resolved(row):
                    continue
                identity = row["category"], row["task_id"]
                point = points[identity]
                previous_failure = row["failure_kind"]
                row = {
                    **row,
                    "official_scored": True,
                    "official_score_source": str(group_path.relative_to(run_dir)),
                }
                if tasks[identity]["kind"] == "api":
                    row.update(
                        success=point["correct"],
                        failure_kind=None if point["correct"] else "model_task_failure",
                    )
                else:
                    row.update(
                        success=None,
                        failure_kind=None,
                        rouge_l_f=point["rouge_l_f"],
                        official_score={
                            "metric": "rouge_l_f",
                            "value": point["rouge_l_f"],
                        },
                    )
                if recovery:
                    row["scoring_recovery"] = {
                        "at": now(),
                        "code": code,
                        "previous_failure_kind": previous_failure,
                    }
                _persist_member(member, row)
                member["row"] = row
                _index_score(rows, row, index, plan)
                latest[key(row)] = row
                repaired += 1
        except BaseException as exc:
            failure = {
                "at": now(),
                "condition": condition,
                "trial": trial,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, subprocess.CalledProcessError):
                failure.update(
                    returncode=exc.returncode, stdout=exc.stdout, stderr=exc.stderr
                )
            failed = []
            persistence_errors = []
            try:
                append_json(
                    run_dir
                    / "diagnostics"
                    / "apibank_scoring"
                    / condition
                    / f"trial_{trial}.jsonl",
                    failure,
                )
            except OSError as persistence_error:
                persistence_errors.append(str(persistence_error))
            for member in members:
                if resolved(member["row"]):
                    continue
                row = {**member["row"], "failure_kind": "scoring_error"}
                failed.append(member["item"])
                try:
                    _persist_member(member, row)
                    _index_score(rows, row, index, plan)
                except OSError as persistence_error:
                    persistence_errors.append(str(persistence_error))
            if not isinstance(exc, Exception):
                raise
            raise GroupScoringError(
                str(exc)
                + (
                    f"; failure persistence errors: {persistence_errors}"
                    if persistence_errors
                    else ""
                ),
                failed,
                [{"condition": condition, "trial": trial, "members": manifest}],
            ) from exc
    return repaired
