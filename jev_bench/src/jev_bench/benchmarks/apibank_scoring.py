"""Published-JSON API-Bank scoring, preserving each level's official error rules."""

import importlib
import os
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from ..config import WORKSPACE_ROOT, number

RETRIEVER_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"


@contextmanager
def official_runtime(config, level):
    """Bind the upstream retriever to the existing local model, on CPU only.

    Upstream tools use relative source/database paths and module-local model
    constructors. Scope both changes to this context; do not edit the checkout.
    This fixture runtime is process-global and must not be used from threads.
    """
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("unsupported API-Bank level")
    source = (WORKSPACE_ROOT / config["benchmark"]["path"]).resolve(strict=True)
    model_path = Path(config["protocol"]["retriever_model_path"]).resolve(strict=True)
    required = (
        "config.json",
        "modules.json",
        "tokenizer_config.json",
        "1_Pooling/config.json",
    )
    if not model_path.is_dir() or any(
        not (model_path / name).is_file() for name in required
    ):
        raise ValueError("the local MiniLM model directory is incomplete")
    if not any(
        (model_path / name).is_file()
        for name in ("pytorch_model.bin", "model.safetensors")
    ):
        raise ValueError("the local MiniLM weights are missing")

    previous_directory = Path.cwd()
    updates = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    # Upstream Translate restores this variable by assignment, which rejects None.
    # Do not override a configured proxy or leave a change in the caller's process.
    if "ALL_PROXY" not in os.environ:
        updates["ALL_PROXY"] = ""
    previous_environment = {name: os.environ.get(name) for name in updates}
    bound = []
    try:
        os.environ.update(updates)
        os.chdir(source)
        from sentence_transformers import SentenceTransformer

        def local_model(identifier):
            if identifier != RETRIEVER_ID:
                raise ValueError("official ToolSearcher requested an unexpected model")
            return SentenceTransformer(str(model_path), device="cpu")

        for name in ("apis.tool_search", "lv3_apis.tool_search"):
            module = importlib.import_module(name)
            bound.append((module, module.SentenceTransformer))
            module.SentenceTransformer = local_model
        yield
    finally:
        for module, constructor in bound:
            module.SentenceTransformer = constructor
        os.chdir(previous_directory)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def make_manager(level):
    """Construct one official manager; score_batch owns its per-level lifecycle."""
    from tool_manager import ToolManager

    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("unsupported API-Bank level")
    return ToolManager("./lv3_apis" if level == 3 else "./apis")


def score_batch(config, predictions, *, terminal_failures=()):
    """Score one arm/trial in published order, without generating predictions.

    predictions maps canonical (category, task_id) to the original output string.
    Each invocation isolates all managers from other arms/trials. Partial input
    selections expose per-point scores but never a full-corpus Sample Accuracy.
    """
    from .apibank import load_tasks

    tasks, answers = load_tasks(config)
    terminal_failures = set(terminal_failures)
    if (
        predictions.keys() & terminal_failures
        or predictions.keys() | terminal_failures != tasks.keys()
    ):
        raise ValueError(
            "predictions and explicit failures must cover every selected point exactly once"
        )
    if any(not isinstance(prediction, str) for prediction in predictions.values()):
        raise TypeError("predictions must be original output strings")
    rows, categories, events = [], {}, Counter()
    for level in (1, 2, 3):
        selected = [
            (key, task) for key, task in tasks.items() if task["level"] == level
        ]
        if not selected:
            continue
        with official_runtime(config, level):
            manager = make_manager(level) if level == 3 else None
            for key, task in selected:
                gold = answers[key]["ground_truth"]
                if key in terminal_failures:
                    score = {
                        "failure_kind": "executor_response_error",
                        "official_scored": False,
                    }
                elif task["kind"] == "api":
                    score = score_api(
                        manager if level == 3 else make_manager(level),
                        level,
                        gold,
                        predictions[key],
                    )
                else:
                    score = score_response(level, gold["text"], predictions[key])
                row = {
                    "category": key[0],
                    "task_id": key[1],
                    "source_id": task["source_id"],
                    **score,
                }
                rows.append(row)
                categories.setdefault(key[0], []).append(row)
                events.update(score.get("protocol_events", {}))
    metrics = {}
    for category, points in categories.items():
        field = "correct" if category.endswith("_api") else "rouge_l_f"
        metrics[category] = {
            "points": len(points),
            "mean": sum(row[field] for row in points) / len(points)
            if all(field in row for row in points)
            else None,
        }
        if any(field not in row for row in points):
            metrics[category]["missing_scores"] = sum(
                field not in row for row in points
            )
    return {
        "points": rows,
        "categories": metrics,
        "protocol_events": dict(events),
        "level_3_sample_accuracy": sample_accuracy(categories.get("level_3_api", [])),
    }


def sample_accuracy(level3):
    """Official L3 conversation accuracy, defined only on all 245 scored API points."""
    samples = {row["source_id"]["sample_id"] for row in level3}
    complete_l3 = (
        len(level3) == 245
        and samples == set(range(50))
        and all("correct" in row for row in level3)
    )
    failed_samples = sorted(
        {row["source_id"]["sample_id"] for row in level3 if row.get("correct") is False}
    )
    return {
        "complete": complete_l3,
        "denominator": 50 if complete_l3 else None,
        "accuracy": (50 - len(failed_samples)) / 50 if complete_l3 else None,
        "failed_sample_ids": failed_samples,
    }


def score_api(manager, level, ground_truth, prediction):
    """Use evaluator_by_json (L1/2) and lv3_evaluator (L3), not evaluator.py.

    Known model errors are incorrect. Corrupt targets, infrastructure errors and
    exceptions not mapped by that official entrypoint still raise.
    """
    from api_call_extraction import get_api_call, parse_api_call

    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("unsupported API-Bank level")
    if not isinstance(prediction, str):
        raise TypeError("prediction must be the original output string")
    if ground_truth["role"] != "API":
        raise ValueError("API scoring requires an API source target")
    name, gold = ground_truth["api_name"], ground_truth["result"]
    if not isinstance(name, str) or not name or not isinstance(gold, dict):
        raise TypeError("API target requires its canonical name and result object")
    # Validate the score target before a name mismatch can short-circuit. Output
    # may legitimately be null and CheckToken has no result-level api_name.
    for field in ("input", "output", "exception"):
        if field not in gold:
            raise KeyError(field)
    if not (
        isinstance(gold["input"], dict)
        or (name == "GetToday" and gold["input"] is None)
    ) or (gold["exception"] is not None and not isinstance(gold["exception"], str)):
        raise TypeError(
            "canonical result requires an input object (GetToday permits null) and string/null exception"
        )
    api_text = get_api_call(prediction)
    if api_text is None:
        return {"correct": False, "reason": "NO_API_CALL", "result": None}
    try:
        predicted_name, parameters = parse_api_call(api_text)
    except Exception as exc:  # noqa: BLE001 -- the official parser maps all parse exceptions
        return {
            "correct": False,
            "reason": "FAILED_PARSE_API_CALL",
            "result": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    if level < 3 and predicted_name != name:
        return {"correct": False, "reason": "API_NAME_MISMATCH", "result": None}
    result = None
    if level < 3:
        phase = "call"
        try:
            result = manager.api_call(predicted_name, **parameters)
            phase = "checker"
            checker = manager.init_tool(name)
            try:
                correct = checker.check_api_call_correctness(
                    deepcopy(result), deepcopy(gold)
                )
            except KeyError:
                return {"correct": False, "reason": "KEY_ERROR", "result": result}
        except Exception as exc:
            if (
                phase == "call"
                and isinstance(exc, AssertionError)
                and (
                    str(exc) == "invalid parameter type."
                    or str(exc).startswith("invalid parameter type. parameter: ")
                )
            ):
                return {
                    "correct": False,
                    "reason": "INVALID_PARAMETER_TYPE",
                    "result": None,
                    "protocol_events": {"apibank_parameter_type_error": 1},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            if isinstance(
                exc, AssertionError
            ) and "The API name is not correct." in str(exc):
                reason = "API_NAME_MISMATCH"
            elif isinstance(exc, AssertionError) and "invalid parameter name" in str(
                exc
            ):
                reason = "INVALID_INPUT_PARAMETER"
            elif "missing" in str(exc) and "required positional argument" in str(exc):
                reason = "MISS_INPUT_ARGUMENT"
            else:
                raise
            return {
                "correct": False,
                "reason": reason,
                "result": result,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    else:
        try:
            if predicted_name == "ToolSearcher":
                from lv3_evaluator import split_by_uppercase

                parameters["keywords"] = split_by_uppercase(parameters["keywords"])
            result = manager.api_call(predicted_name, **parameters)
        except Exception as exc:
            if (
                not isinstance(exc, (TypeError, AssertionError))
                and str(exc) != "invalid tool name."
            ):
                raise
            return {
                "correct": False,
                "reason": "official_call_error",
                "result": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        # In the official L3 driver init_tool is outside the checker exception map.
        checker = manager.init_tool(name)
        try:
            correct = checker.check_api_call_correctness(
                deepcopy(result), deepcopy(gold)
            )
        except (KeyError, AssertionError) as exc:
            return {
                "correct": False,
                "reason": "official_checker_error",
                "result": result,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    if type(correct) is not bool:
        raise TypeError("official API checker must return a boolean")
    reason = "official_checker"
    if level < 3 and not correct:
        if result["exception"]:
            reason = "HAS_EXCEPTION"
        elif result["output"] != gold["output"]:
            reason = "OUTPUT_MISMATCH"
        elif result["input"] != gold["input"]:
            reason = "INPUT_MISMATCH"
    return {"correct": correct, "reason": reason, "result": result}


def score_response(level, reference, prediction):
    """Apply the corresponding official driver's text cleanup and per-point score."""
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("unsupported API-Bank level")
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise TypeError("response reference and prediction must be strings")
    if level < 3:
        from evaluator_by_json import calculate_rouge_l_score

        value = (
            0
            if prediction == ""
            else round(calculate_rouge_l_score(reference, prediction), 4)
        )
    else:
        from lv3_evaluator import calculate_rouge_l_score

        prediction = prediction.replace("User:", "").replace("AI:", "").strip()
        value = calculate_rouge_l_score(reference, prediction)
    number(value, maximum=1)
    return {"rouge_l_f": value}
