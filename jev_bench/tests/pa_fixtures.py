"""Explicit interface fixtures shared by the independent CPU adapter suites."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from jev_bench.client import CompletionResult
from jev_bench.io import write_json
from jev_bench.pa_tool import Projection, asset_identity, task_key
from jev_bench.pa_tool_prepare import PROTOCOL


def tool(name="lookup"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up a record, preserving the description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Record key"},
                    "payload": {
                        "type": "object",
                        "description": "Unchanged payload",
                        "properties": {"key": {"type": "string"}},
                    },
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }


def rows_for(tools):
    return [
        {
            "name": f"pa_tool_{index}"
            if t["function"]["name"] != "Finish"
            else "Finish",
            "parameters": {
                name: f"pa_arg_{position}"
                if t["function"]["name"] != "Finish"
                else name
                for position, name in enumerate(
                    t["function"]["parameters"]["properties"]
                )
            },
        }
        for index, t in enumerate(tools)
    ]


def freeze(config, item, tools, directory):
    directory = Path(directory)
    rows = rows_for(tools)
    policy = config["conditions"][item["condition"]]
    catalog = {
        "version": 1,
        "benchmark": config["benchmark"],
        "executor": config["executors"][policy["executor"]],
        "control_names": ["Finish"]
        if config["benchmark"]["name"] == "stabletoolbench"
        else [],
        "scopes": {"fixture": tools},
        "bindings": {task_key(item): "fixture"},
    }
    if config["benchmark"]["name"] == "toolsandbox":
        catalog["description_augmentations"] = {"fixture": []}
    write_json(directory / "catalog.json", catalog)
    write_json(
        directory / "mapping.json",
        {"version": 1, "protocol": PROTOCOL, "scopes": {"fixture": rows}, "description_preservations": {}},
    )
    write_json(directory / "metadata/run_status.json", {"status": "completed"})
    policy["pa_tool_mapping"] = str(directory / "mapping.json")
    projection = Projection(tools, rows)
    projection.asset_identity = asset_identity(directory / "mapping.json")
    return projection


def result(calls=(), content=None):
    raw_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
        for index, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=[
            SimpleNamespace(**{**call, "function": SimpleNamespace(**call["function"])})
            for call in raw_calls
        ],
    )
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": deepcopy(raw_calls),
                }
            }
        ]
    }
    return CompletionResult(
        message,
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        0,
        "stop",
        raw,
    )


def alias_message(message, tools, rows):
    """The test oracle emits declared aliases; it is not the runtime mapper."""
    value = deepcopy(message)
    by_name = {tool["function"]["name"]: row for tool, row in zip(tools, rows)}
    for call in value.get("tool_calls") or []:
        function = call["function"]
        row = by_name[function["name"]]
        function["name"] = row["name"]
        arguments = json.loads(function["arguments"])
        function["arguments"] = json.dumps(
            {row["parameters"][key]: item for key, item in arguments.items()}
        )
    return value
