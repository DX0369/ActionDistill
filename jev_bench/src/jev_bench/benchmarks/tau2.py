"""τ² base domains using the official orchestrator, environments and evaluator."""

import json
import math
import os
import sys
import time
from pathlib import Path

from openai import InternalServerError

from ..advice import AdvisorServiceError, make_advisor
from ..attempt import mark_scoring, record_attempt
from ..client import (
    CompletionResponseError,
    ContextLimitError,
    QwenClient,
    SimulatorClient,
    is_service_error,
)
from ..config import TAU2_DOMAINS, WORKSPACE_ROOT, advisor_type
from ..cost import costs
from ..io import append_json, write_json
from ..services import bypass_local_proxy

BASE_COUNTS = {"retail": 114, "airline": 50, "telecom": 114}


def configure(config, runtime):
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    expected_data = (source / "data").resolve(strict=True)
    if (
        "TAU2_DATA_DIR" in os.environ
        and Path(os.environ["TAU2_DATA_DIR"]).resolve() != expected_data
    ):
        raise ValueError("TAU2_DATA_DIR must point to the pinned checkout data")
    os.environ["TAU2_DATA_DIR"] = str(expected_data)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    sys.path.insert(0, str(source / "src"))
    import tau2
    from tau2.utils import DATA_DIR

    if Path(tau2.__file__).resolve() != (source / "src/tau2/__init__.py").resolve():
        raise ValueError("tau2 import must use the pinned checkout")
    if DATA_DIR.resolve() != expected_data:
        raise ValueError("imported tau2 uses a different data directory")
    bypass_local_proxy()


def load_tasks(config):
    from tau2.registry import registry

    tasks = {}
    for domain in TAU2_DOMAINS:
        loaded = registry.get_tasks_loader(domain)(
            task_split_name=config["protocol"]["split"]
        )
        ids = [task.id for task in loaded]
        if len(ids) != BASE_COUNTS[domain] or len(set(ids)) != len(ids):
            raise ValueError(f"{domain}: base task count or uniqueness differs")
        if any(not isinstance(task_id, str) or not task_id for task_id in ids):
            raise ValueError("official task IDs must be nonempty strings")
        selected = config["task_ids"][domain]
        if selected == "all":
            selected = ids
        if not set(selected) <= set(ids):
            raise ValueError(f"{domain}: unknown base task ID")
        tasks.update(
            ((domain, task.id), task) for task in loaded if task.id in selected
        )
    return tasks, dict.fromkeys(tasks)


def score_saved(config, item, task, ground_truth, directory):
    """Recover ALL scoring from a complete SimulationRun without model clients."""
    from tau2.data_model.message import ToolMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.metrics.agent_metrics import is_successful

    directory = Path(directory)
    simulation = SimulationRun.model_validate_json(
        (directory / "trajectory.json").read_text()
    )
    if simulation.task_id != task.id or simulation.trial != item["trial"]:
        raise ValueError("saved tau2 trajectory belongs to another task/trial")
    if simulation.termination_reason == TerminationReason.USER_ERROR:
        raise ValueError("a simulator failure cannot be recovered by scoring")
    protocol = config["protocol"]
    if protocol["evaluation_type"] != "ALL":
        raise ValueError("only the fixed non-LLM ALL evaluator may be recovered")
    score_path = directory / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text())
    else:
        score = evaluate_simulation(
            simulation=simulation,
            task=task,
            domain=item["category"],
            evaluation_type=EvaluationType.ALL,
            solo_mode=protocol["solo_mode"],
        ).model_dump(mode="json")
        reward = score["reward"]
        if (
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError("official reward must be finite and in [0,1]")
        write_json(score_path, score)
    reward = score["reward"]
    if (
        type(reward) not in (int, float)
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError("official reward must be finite and in [0,1]")
    return {
        "success": bool(is_successful(reward)),
        "reward": reward,
        "termination_reason": simulation.termination_reason.value,
        "tool_call_count": sum(
            isinstance(m, ToolMessage) and m.requestor == "assistant"
            for m in simulation.messages
        ),
        "user_tool_call_count": sum(
            isinstance(m, ToolMessage) and m.requestor == "user"
            for m in simulation.messages
        ),
    }


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
    user_client_factory=SimulatorClient,
    advisor_factory=make_advisor,
):
    from tau2.data_model.message import ToolMessage
    from tau2.data_model.simulation import TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.metrics.agent_metrics import is_successful
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry

    from .tau2_participants import DeepSeekUser, GuidedAgent, UserSimulatorError

    directory = Path(directory)
    policy = config["conditions"][item["condition"]]
    settings = config["executors"][policy["executor"]]
    protocol = config["protocol"]
    started = time.perf_counter()
    row = {
        **item,
        "success": None,
        "official_scored": False,
        "failure_kind": None,
        "decision_steps": 0,
        "tool_call_count": None,
        "duration_seconds": None,
        "user_tool_call_count": None,
        "response_anomalies": [],
        "reward": None,
        "termination_reason": None,
    }
    client = user_client = agent = orchestrator = simulation = advisor = None
    phase = "inference"
    interrupted = False
    sink = lambda kind, value: append_json(
        directory / "diagnostics" / f"{kind}.jsonl", value
    )
    try:
        environment = registry.get_env_constructor(item["category"])()
        client = client_factory(settings, sink)
        user_client = user_client_factory(config["user_simulator"], sink)
        advisor = (
            advisor_factory(
                config["advisors"][policy["advisor"]], sink, advisors=config["advisors"]
            )
            if "advisor" in policy
            else None
        )
        from ..pa_tool import load_projection

        agent = GuidedAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            client=client,
            settings=settings,
            policy=policy,
            advisor_client=advisor,
            event_sink=sink,
            projection=load_projection(
                config, item, expected_identity=pa_tool_identity
            ),
        )
        user = DeepSeekUser(
            client=user_client,
            instructions=str(task.user_scenario),
            tools=environment.get_user_tools()
            if environment.user_tools is not None
            else None,
        )
        orchestrator = Orchestrator(
            domain=item["category"],
            task=task,
            agent=agent,
            user=user,
            environment=environment,
            max_steps=protocol["max_steps"],
            max_errors=protocol["max_errors"],
            seed=item["seed"],
            solo_mode=protocol["solo_mode"],
            validate_communication=protocol["validate_communication"],
        )
        simulation = orchestrator.run()
        simulation.trial = item["trial"]
        row["termination_reason"] = simulation.termination_reason.value
        if simulation.termination_reason == TerminationReason.USER_ERROR:
            write_json(
                directory / "trajectory.json", simulation.model_dump(mode="json")
            )
            raise UserSimulatorError("official orchestrator terminated with user_error")
        phase = "scoring"
        mark_scoring(directory)
        write_json(directory / "trajectory.json", simulation.model_dump(mode="json"))
        simulation.reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            domain=item["category"],
            evaluation_type=EvaluationType[protocol["evaluation_type"]],
            solo_mode=protocol["solo_mode"],
        )
        reward = simulation.reward_info.reward
        if (
            type(reward) not in (int, float)
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError("official reward must be finite and in [0,1]")
        success = bool(is_successful(reward))
        write_json(
            directory / "score.json", simulation.reward_info.model_dump(mode="json")
        )
        row.update(
            reward=reward,
            success=success,
            official_scored=True,
            failure_kind=None if success else "model_task_failure",
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
        if phase == "inference" and isinstance(exc, UserSimulatorError):
            row["failure_kind"] = "user_simulator_error"
        elif phase == "inference" and isinstance(
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
        if agent is not None:
            row.update(
                decision_steps=agent.decision_steps, response_anomalies=agent.anomalies
            )
        if orchestrator is not None:
            history = (
                simulation.messages
                if simulation is not None
                else orchestrator.trajectory
            )
            if simulation is None:
                write_json(
                    directory / "trajectory.json",
                    {
                        "partial": True,
                        "messages": [
                            message.model_dump(mode="json") for message in history
                        ],
                    },
                )
            row["tool_call_count"] = sum(
                isinstance(m, ToolMessage) and m.requestor == "assistant"
                for m in history
            )
            row["user_tool_call_count"] = sum(
                isinstance(m, ToolMessage) and m.requestor == "user" for m in history
            )
        row["cost"] = costs(
            client,
            agent,
            user_client,
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
                if user_client is not None:
                    user_client.close()
    return row
