"""Attempt ownership and explicit recovery of interrupted, unscored inference."""

import fcntl
import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

from .config import advisor_type
from .cost import completion_cost, costs
from .io import now, read_jsonl, write_json


def record_attempt(function):
    @wraps(function)
    def run(config, item, task, ground_truth, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / "diagnostics/attempt.json"
        marker.parent.mkdir()
        # Keep this inode for the lifetime of the worker. Recovery must acquire it,
        # including after a coordinator dies while its worker is still running.
        with marker.open("x+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            json.dump(
                {
                    "item": item,
                    "benchmark": config["benchmark"]["name"],
                    "advisor_type": advisor_type(config, item["condition"]),
                    "started_at": now(),
                    "phase": "inference",
                },
                stream,
            )
            stream.flush()
            return function(config, item, task, ground_truth, directory, **kwargs)

    return run


def mark_scoring(directory):
    """Once inference finished, an interruption must not resample the model."""
    with (Path(directory) / "diagnostics/attempt.json").open("r+") as stream:
        state = json.load(stream)
        state["phase"] = "scoring"
        stream.seek(0)
        json.dump(state, stream)
        stream.truncate()


def recover_interrupted(directory, item):
    directory = Path(directory)
    marker = directory / "diagnostics/attempt.json"
    if not marker.exists():
        raise ValueError(
            f"incomplete attempt lacks ownership evidence; diagnosis required: {directory}"
        )
    with marker.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"attempt worker is still active: {directory}") from exc
        state = json.load(stream)
        if state["item"] != item:
            raise ValueError(
                "interrupted attempt identity differs from the frozen plan"
            )
        result = directory / "episode.json"
        if result.exists():
            row = json.loads(result.read_text())
            if row["failure_kind"] != "interrupted":
                return row
        else:
            row = None
        trajectory = directory / "trajectory.json"
        finished_trajectory = (
            trajectory.exists()
            and json.loads(trajectory.read_text()).get("partial") is not True
        )
        if (
            state["phase"] != "inference"
            or (directory / "score.json").exists()
            or finished_trajectory
        ):
            raise ValueError(
                f"inference already finished; diagnose scoring without resampling: {directory}"
            )
        if row is not None:
            return row
        diagnostics = directory / "diagnostics"
        cost = costs(
            SimpleNamespace(records=read_jsonl(diagnostics / "executor.jsonl")),
            SimpleNamespace(advisor_records=read_jsonl(diagnostics / "advisor.jsonl")),
            SimpleNamespace(records=read_jsonl(diagnostics / "user_simulator.jsonl")),
            advisor_type=state["advisor_type"],
            advisor_requests=read_jsonl(diagnostics / "advisor_requests.jsonl"),
        )
        # A killed request may never have reached the append-only diagnostics.
        # Preserve observed costs, but do not claim complete request accounting.
        cost["accounting_complete"] = False
        if state["benchmark"] == "stabletoolbench":
            for participant in ("tool_simulator", "judge"):
                cost[participant] = completion_cost(
                    SimpleNamespace(
                        records=read_jsonl(diagnostics / f"{participant}.jsonl")
                    )
                )
        row = {
            **item,
            "success": None,
            "official_scored": False,
            "failure_kind": "interrupted",
            "decision_steps": None,
            "tool_call_count": None,
            "duration_seconds": None,
            "response_anomalies": [],
            "cost": cost,
        }
        write_json(result, row)
        return row
