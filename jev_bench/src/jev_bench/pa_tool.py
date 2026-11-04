"""Frozen PA-Tool interfaces; execution and saved trajectories use official names.

Only the request copy is projected. Model interface violations are participant
errors; missing/ambiguous assets are configuration errors, never native fallbacks.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .client import ParticipantResponseError

VERSION = 1
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class InterfaceViolation(ParticipantResponseError):
    """An executor used an interface that the PA-Tool condition did not expose."""


def legal_name(value):
    return (
        isinstance(value, str)
        and len(value) <= 64
        and NAME.fullmatch(value) is not None
    )


def task_key(item):
    return json.dumps([item["category"], item["task_id"]], ensure_ascii=False)


def asset_identity(path):
    path = Path(path)
    return {
        "mapping_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(
            (path.parent / "catalog.json").read_bytes()
        ).hexdigest(),
    }


def load_projection(config, item, *, expected_identity=None):
    policy = config["conditions"][item["condition"]]
    if "pa_tool_mapping" not in policy:
        return None
    if expected_identity is None:
        raise ValueError("PA-Tool loading requires the run's frozen asset identity")
    path = Path(policy["pa_tool_mapping"])
    mapping_bytes = path.read_bytes()
    catalog_bytes = (path.parent / "catalog.json").read_bytes()
    actual_identity = {
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
    }
    if actual_identity != expected_identity:
        raise ValueError("PA-Tool assets differ from the run's frozen identity")
    mapping = json.loads(mapping_bytes)
    catalog = json.loads(catalog_bytes)
    status = json.loads((path.parent / "metadata/run_status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("PA-Tool preparation must be completed")
    if mapping["version"] != VERSION or catalog["version"] != VERSION:
        raise ValueError("unsupported PA-Tool asset version")
    from .pa_tool_prepare import (
        check_protocol,
        description_preservation,
        description_preservations,
    )

    check_protocol(mapping["protocol"])
    if catalog["benchmark"] != config["benchmark"]:
        raise ValueError("PA-Tool benchmark source differs")
    expected_controls = (
        ["Finish"] if config["benchmark"]["name"] == "stabletoolbench" else []
    )
    if catalog["control_names"] != expected_controls:
        raise ValueError("PA-Tool control-name exclusions differ")
    if set(mapping["scopes"]) != set(catalog["scopes"]):
        raise ValueError("PA-Tool mapping scopes differ from catalog")
    settings = config["executors"][policy["executor"]]
    for key in ("model", "model_path"):
        if catalog["executor"][key] != settings[key]:
            raise ValueError(f"PA-Tool executor {key} differs")
    scope = catalog["bindings"][task_key(item)]
    if mapping["description_preservations"] != description_preservations(catalog):
        raise ValueError("PA-Tool description preservation records differ from catalog")
    for tool, row in zip(catalog["scopes"][scope], mapping["scopes"][scope]):
        function = tool["function"]
        if (
            description_preservation(catalog, scope, function) is not None
            and row["name"] != function["name"]
        ) or any(
            description_preservation(catalog, scope, function, parameter) is not None
            and row["parameters"][parameter] != parameter
            for parameter in function["parameters"]["properties"]
        ):
            raise ValueError(
                "PA-Tool hidden-description components must retain visible names"
            )
        if function["name"] in expected_controls and row != {
            "name": function["name"],
            "parameters": {key: key for key in function["parameters"]["properties"]},
        }:
            raise ValueError("PA-Tool protocol controls must remain unchanged")
    return Projection(catalog["scopes"][scope], mapping["scopes"][scope])


def remap_schema(function, row):
    """Rename declared top-level inputs and their required references only."""
    result = deepcopy(function)
    result["name"] = row["name"]
    schema = result["parameters"]
    names = row["parameters"]
    if set(names) != set(schema["properties"]):
        raise ValueError("PA-Tool mapping does not cover the input properties")
    # These keywords may refer to property names. Supporting them requires an
    # explicit schema projection, not treating every nested string as a name.
    unsupported = set(schema) & {
        "$ref",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
        "propertyNames",
        "patternProperties",
        "unevaluatedProperties",
        "enum",
        "const",
    }
    if unsupported:
        raise ValueError(
            f"PA-Tool top-level schema references need support: {sorted(unsupported)}"
        )

    def check_references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef") and (
                    not isinstance(item, str) or item.startswith("#/properties/")
                ):
                    raise ValueError(
                        "PA-Tool schema reference to renamed root properties needs support"
                    )
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)

    check_references(schema)
    schema["properties"] = {
        names[key]: value for key, value in schema["properties"].items()
    }
    if "required" in schema:
        # Preserve upstream undeclared entries (Stable includes an empty-name
        # sentinel). They are not declared input parameters to be renamed.
        schema["required"] = [names.get(key, key) for key in schema["required"]]
    return result


class Projection:
    def __init__(self, tools, rows):
        if len(tools) != len(rows):
            raise ValueError("PA-Tool schema/mapping lengths differ")
        self.tools, self.rows = deepcopy(tools), deepcopy(rows)
        aliases = [row["name"] for row in rows]
        if len(set(aliases)) != len(aliases) or not all(map(legal_name, aliases)):
            raise ValueError("PA-Tool tool aliases must be unique identifiers")
        self.aliases = dict(zip(aliases, range(len(rows))))
        self.originals = {}
        for index, (tool, row) in enumerate(zip(tools, rows)):
            function = tool["function"]
            self.originals.setdefault(function["name"], []).append(index)
            names = list(row["parameters"].values())
            if len(set(names)) != len(names) or not all(map(legal_name, names)):
                raise ValueError("PA-Tool parameter aliases must be unique identifiers")
            remap_schema(function, row)
        self.visible = []
        # Calls retain their occurrence even when canonical tool names repeat.
        self.calls = {}
        self.next_assistant = 0

    def _indices(self, tools):
        available = list(range(len(self.tools)))
        result = []
        for tool in tools:
            matches = [i for i in available if self.tools[i] == tool]
            if not matches:
                raise ValueError("visible tool has no exact PA-Tool schema mapping")
            index = matches[0]
            available.remove(index)
            result.append(index)
        return result

    def project_arguments(self, arguments, index):
        names = self.rows[index]["parameters"]
        result = {names.get(key, key): value for key, value in arguments.items()}
        if len(result) != len(arguments):
            raise ValueError("PA-Tool historical arguments collide after projection")
        return result

    def restore_arguments(self, arguments, index):
        names = self.rows[index]["parameters"]
        reverse = {alias: original for original, alias in names.items()}
        # Unknown arguments still belong to the official tool validator unless
        # they would bypass the declared interface or overwrite another key.
        if any(key in names and key not in reverse for key in arguments):
            raise InterfaceViolation(
                "PA-Tool call used an unexposed parameter name", role="agent"
            )
        result = {reverse.get(key, key): value for key, value in arguments.items()}
        if len(result) != len(arguments):
            raise InterfaceViolation(
                "PA-Tool arguments collide after translation", role="agent"
            )
        return result

    def _history_index(self, call, position):
        if position in self.calls:
            index = self.calls[position]
            if call["function"]["name"] != self.tools[index]["function"]["name"]:
                raise ValueError("PA-Tool canonical history changed tool occurrence")
            return index
        indices = self.originals.get(call["function"]["name"], [])
        if len(indices) != 1:
            raise ValueError("PA-Tool history needs an unambiguous tool occurrence")
        return indices[0]

    def request(self, messages, tools):
        self.visible = self._indices(tools)
        projected = []
        for tool, index in zip(tools, self.visible):
            row = deepcopy(tool)
            row["function"] = remap_schema(tool["function"], self.rows[index])
            projected.append(row)
        history = deepcopy(messages)
        assistant_index = 0
        call_names = {}
        for message in history:
            if message["role"] == "tool" and "name" in message:
                index = call_names.get(message["tool_call_id"])
                if index is None:
                    raise ValueError("PA-Tool tool reply lacks a corresponding call")
                if message["name"] != self.tools[index]["function"]["name"]:
                    raise ValueError("PA-Tool tool reply changed canonical name")
                message["name"] = self.rows[index]["name"]
            if message["role"] != "assistant":
                continue
            for call_index, call in enumerate(message.get("tool_calls") or []):
                index = self._history_index(call, (assistant_index, call_index))
                call_names[call["id"]] = index
                function = call["function"]
                function["name"] = self.rows[index]["name"]
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, TypeError):
                    continue  # Preserve malformed historical arguments verbatim.
                if isinstance(arguments, dict):
                    projected_arguments = self.project_arguments(arguments, index)
                    if projected_arguments != arguments:
                        function["arguments"] = json.dumps(
                            projected_arguments, ensure_ascii=False
                        )
            assistant_index += 1
        self.next_assistant = assistant_index
        return history, projected

    def restore_call(self, name, arguments, *, position=None, suffix=False):
        if not isinstance(name, str) or not name:
            raise InterfaceViolation(
                "PA-Tool call requires a nonempty tool name", role="agent"
            )
        visible_aliases = {self.rows[i]["name"] for i in self.visible}
        if name in self.originals and name not in visible_aliases:
            raise InterfaceViolation(
                "PA-Tool call used an unexposed original tool name", role="agent"
            )
        # Stable dispatches exact Finish before its ordered business suffix search.
        indices = [
            i
            for i in self.visible
            if (
                self.rows[i]["name"].endswith(name)
                if suffix and name != "Finish"
                else self.rows[i]["name"] == name
            )
        ]
        if not indices:
            raise InterfaceViolation(
                f"PA-Tool call used an unexposed tool name: {name!r}", role="agent"
            )
        index = indices[0]  # StableToolBench's ordered suffix resolver is preserved.
        if position is not None:
            self.calls[position] = index
        if isinstance(arguments, dict):
            arguments = self.restore_arguments(arguments, index)
        return self.tools[index]["function"]["name"], arguments

    def response(self, result, *, suffix=False):
        translated = deepcopy(result)
        for call_index, call in enumerate(translated.message.tool_calls or []):
            try:
                arguments = json.loads(call.function.arguments)
            except (ValueError, TypeError):
                arguments = None
            name, restored = self.restore_call(
                call.function.name,
                arguments,
                position=(self.next_assistant, call_index),
                suffix=suffix,
            )
            call.function.name = name
            if isinstance(restored, dict) and restored != arguments:
                call.function.arguments = json.dumps(restored, ensure_ascii=False)
        # ToolSandbox consumes this route copy; the client's original ledger
        # remains byte-for-byte unchanged.
        raw_message = translated.raw_response["choices"][0]["message"]
        for raw_call, call in zip(
            raw_message.get("tool_calls") or [], translated.message.tool_calls or []
        ):
            raw_call["function"]["name"] = call.function.name
            raw_call["function"]["arguments"] = call.function.arguments
        return translated
