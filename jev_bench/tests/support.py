"""CPU fixtures; all generated files live under this project's tmp directory."""

import atexit
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parents[1]))
(PROJECT / "tmp").mkdir(exist_ok=True)
RUNTIME = tempfile.TemporaryDirectory(prefix="cpu_tests_", dir=PROJECT / "tmp")
atexit.register(RUNTIME.cleanup)
os.environ["BFCL_PROJECT_ROOT"] = RUNTIME.name
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from jev_bench.benchmarks.bfcl import configure


def config():
    result = json.loads((PROJECT / "configs/bfcl_mt_native_smoke.json").read_text())
    configure(result, RUNTIME.name)
    return result


def advisor_settings():
    root = Path(RUNTIME.name)
    (root / "model").mkdir(exist_ok=True)
    (root / "temperature.json").write_text('{"temperature": 1.3}')
    return {
        "type": "jev_local",
        "base_url": "http://127.0.0.1:18770",
        "timeout_seconds": 120,
        "model": str(root / "model"),
        "temperature_file": str(root / "temperature.json"),
    }


def prediction(names=("echo", "RESPOND"), values=(0.8, 0.2)):
    from jev_bench.advice import OPTION_ORDER

    settings = advisor_settings()
    return {
        "model": settings["model"],
        "temperature_file": settings["temperature_file"],
        "temperature": 1.3,
        "prompt_budget": 32704,
        "option_order": OPTION_ORDER,
        "duplicate_candidates": "positional",
        "n_prompt_tokens": 30,
        "messages_dropped": 0,
        "truncation_counts": {"images_replaced": 0, "skipped_framework_greeting": 0},
        "truncated": False,
        "latency_seconds": 0.01,
        "probabilities": [
            {"name": name, "probability": value} for name, value in zip(names, values)
        ],
        "top1": names[max(range(len(names)), key=values.__getitem__)],
    }


def completion(content="Done", calls=None, reasoning="I considered the next step."):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "cpu_fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-4b",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning": reasoning,
                        "tool_calls": [] if calls is None else calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def sdk(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
