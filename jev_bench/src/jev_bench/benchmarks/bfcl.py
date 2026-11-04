"""Official BFCL loading, execution and scoring; no custom task-success proxy."""

import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from uuid import uuid4

from openai import InternalServerError

from ..advice import AdvisorServiceError, make_advisor
from ..attempt import mark_scoring, record_attempt
from ..client import (
    CompletionResponseError,
    ContextLimitError,
    QwenClient,
    is_service_error,
)
from ..config import CATEGORIES, WORKSPACE_ROOT, advisor_type
from ..cost import costs
from ..io import append_json, write_json
from ..services import bypass_local_proxy


def configure(config, runtime):
    # BFCL creates output/lock directories at import time. Each process gets its own root.
    os.environ["BFCL_PROJECT_ROOT"] = str(runtime)
    sys.path.insert(0, str(WORKSPACE_ROOT / config["benchmark"]["path"]))
    bypass_local_proxy()


def load_tasks(config):
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    tasks, answers = {}, {}
    for category in CATEGORIES:
        loaded = load_dataset_entry(category, include_language_specific_hint=True)
        references = load_ground_truth_entry(category)
        task_ids = [row["id"] for row in loaded]
        reference_ids = [row["id"] for row in references]
        if (
            len(task_ids) != 200
            or len(set(task_ids)) != 200
            or len(reference_ids) != 200
            or set(task_ids) != set(reference_ids)
        ):
            raise ValueError(
                f"{category}: expected 200 unique tasks and exactly matching references"
            )
        selected = config["task_ids"][category]
        if selected == "all":
            selected = task_ids
        if not set(selected) <= set(task_ids):
            raise ValueError(f"{category}: unknown configured task ID")
        tasks.update(
            ((category, row["id"]), row) for row in loaded if row["id"] in selected
        )
        answers.update(
            ((category, row["id"]), row["ground_truth"])
            for row in references
            if row["id"] in selected
        )
    return tasks, answers


def cleanup_backends(namespace, task):
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

    for suffix in ("", "_eval", "_ground_truth_eval"):
        for class_name in task["involved_classes"]:
            key = re.sub(
                r"[-./:]",
                "_",
                f"{namespace}{suffix}_{task['id']}_{class_name}_instance",
            )
            vars(multi_turn_utils).pop(key, None)


def score_saved(config, item, task, ground_truth, directory):
    """Score complete saved inference with only the official decoder and checker."""
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.utils import make_json_serializable

    directory = Path(directory)
    trajectory = json.loads((directory / "trajectory.json").read_text())
    if trajectory["id"] != task["id"] or trajectory.get("partial") is True:
        raise ValueError("saved BFCL trajectory is partial or belongs to another task")
    tool_calls = sum(
        message["role"] == "tool"
        for turn in trajectory["inference_log"]
        for name, messages in turn.items()
        if name.startswith("step_")
        for message in messages
    )
    score_path = directory / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
    else:
        handler = SimpleNamespace(is_fc_model=True)
        handler.decode_execute = MethodType(
            OpenAICompletionsHandler.decode_execute, handler
        )
        namespace = "jevbench_rescore_" + uuid4().hex
        try:
            score = _evaluate_single_multi_turn_entry(
                handler,
                task["id"],
                trajectory["result"],
                ground_truth,
                deepcopy(task),
                namespace,
                item["category"],
            )
        finally:
            cleanup_backends(namespace, task)
        score = make_json_serializable(score)
        if type(score["valid"]) is not bool:
            raise ValueError("official score.valid must be a boolean")
        write_json(score_path, score, allow_nan=True)
    if type(score["valid"]) is not bool:
        raise ValueError("official score.valid must be a boolean")
    return {"success": score["valid"], "tool_call_count": tool_calls}


@record_attempt
def run_episode(
    config,
    item,
    task,
    ground_truth,
    directory,
    *,
    pa_tool_identity=None,
    client_factory=QwenClient,
    advisor_factory=make_advisor,
):
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
    from bfcl_eval.utils import make_json_serializable

    from .handler import GuidanceHandler

    directory = Path(directory)
    policy = config["conditions"][item["condition"]]
    settings = config["executors"][policy["executor"]]
    namespace = "jevbench_" + uuid4().hex
    started = time.perf_counter()
    row = {
        **item,
        "success": None,
        "official_scored": False,
        "failure_kind": None,
        "decision_steps": 0,
        "tool_call_count": None,
        "duration_seconds": None,
        "response_anomalies": [],
    }
    client = handler = advisor = None
    phase = "inference"
    interrupted = False
    sink = lambda kind, value: append_json(
        directory / "diagnostics" / f"{kind}.jsonl", value
    )
    try:
        client = client_factory(settings, sink)
        advisor = (
            advisor_factory(
                config["advisors"][policy["advisor"]], sink, advisors=config["advisors"]
            )
            if "advisor" in policy
            else None
        )
        from ..pa_tool import load_projection

        handler = GuidanceHandler(
            condition=item["condition"],
            namespace=namespace,
            client=client,
            settings=settings,
            policy=policy,
            advisor_client=advisor,
            event_sink=sink,
            projection=load_projection(
                config, item, expected_identity=pa_tool_identity
            ),
        )
        result, metadata = handler.inference(
            deepcopy(task), include_input_log=False, exclude_state_log=True
        )
        phase = "scoring"
        mark_scoring(directory)
        write_json(
            directory / "trajectory.json",
            {"id": task["id"], "result": result, **metadata},
        )
        row["tool_call_count"] = sum(
            message["role"] == "tool"
            for turn in metadata["inference_log"]
            for key, messages in turn.items()
            if key.startswith("step_")
            for message in messages
        )
        score = _evaluate_single_multi_turn_entry(
            handler,
            task["id"],
            result,
            ground_truth,
            deepcopy(task),
            namespace,
            item["category"],
        )
        if type(score["valid"]) is not bool:
            raise ValueError("official score.valid must be a boolean")
        # Official state diagnostics can contain NaN/Infinity produced by a tool.
        # Keep that evidence; all other artifacts retain strict JSON validation.
        write_json(
            directory / "score.json", make_json_serializable(score), allow_nan=True
        )
        row.update(
            success=score["valid"],
            official_scored=True,
            failure_kind=None if score["valid"] else "model_task_failure",
        )
    except BaseException as exc:
        failure = {"phase": phase, "error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, (CompletionResponseError, ContextLimitError)):
            failure["provider_diagnostics"] = exc.diagnostics
        write_json(directory / "diagnostics/failure.json", failure)
        if isinstance(exc, KeyboardInterrupt):
            interrupted = True
            row["failure_kind"] = (
                "interrupted" if phase == "inference" else "scoring_error"
            )
            raise
        if phase == "inference" and isinstance(
            exc, (CompletionResponseError, ContextLimitError)
        ):
            row.update(
                success=False,
                failure_kind=(
                    "executor_context_limit"
                    if isinstance(exc, ContextLimitError)
                    else "executor_response_error"
                ),
            )
        elif phase == "inference" and isinstance(exc, AdvisorServiceError):
            row["failure_kind"] = exc.record["status"]
            if row["failure_kind"] == "advisor_error":
                raise
        elif (
            phase == "inference"
            and is_service_error(exc)
            and not isinstance(exc, InternalServerError)
        ):
            row["failure_kind"] = "service_error"
        else:
            row["failure_kind"] = (
                "scoring_error" if phase == "scoring" else "unexpected_error"
            )
            raise
    finally:
        if handler is not None:
            row.update(
                decision_steps=handler.decision_steps,
                response_anomalies=handler.anomalies,
            )
        row["cost"] = costs(
            client,
            handler,
            advisor_type=advisor_type(config, item["condition"]),
            advisor_requests=advisor.records if advisor else (),
        )
        row["cost"]["accounting_complete"] = not interrupted
        row["duration_seconds"] = time.perf_counter() - started
        write_json(directory / "episode.json", row)
        try:
            if advisor is not None:
                advisor.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                cleanup_backends(namespace, task)
    return row
