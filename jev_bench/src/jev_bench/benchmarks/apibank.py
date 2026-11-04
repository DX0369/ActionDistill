"""Published API-Bank decision points, with visible input separated from score targets."""

import ast
import importlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from ..config import APIBANK_CATEGORIES, WORKSPACE_ROOT, fields, integer

FILES = {
    "level_1_api": "level-1-api.json",
    "level_1_response": "level-1-response.json",
    "level_2_api": "level-2-api.json",
    "level_2_response": "level-2-response.json",
    "level_3_api": "level-3-batch-inf.json",
    "level_3_response": "level-3-batch-inf-response.json",
}
DESCRIPTION_MARKER = "\nAPI descriptions:\n"
SEARCH_RESULT = re.compile(r"^API-Request: \[ToolSearcher\(.*\)\]->(.*)$")


def configure(config, runtime):
    source = (WORKSPACE_ROOT / config["benchmark"]["path"]).resolve(strict=True)
    sys.path.insert(0, str(source))
    # These unqualified names belong to the upstream checkout, not another adapter.
    for name in ("apis", "tool_manager", "evaluator_by_json", "lv3_evaluator"):
        if name in sys.modules and not Path(
            sys.modules[name].__file__
        ).resolve().is_relative_to(source):
            raise ValueError(f"API-Bank module {name} was imported from another source")
    parser = importlib.import_module("api_call_extraction")
    if Path(parser.__file__).resolve() != source / "api_call_extraction.py":
        raise ValueError("API-Bank parser must use the pinned checkout")
    Path(config["protocol"]["data_root"]).resolve(strict=True)
    Path(config["protocol"]["retriever_model_path"]).resolve(strict=True)


def _text(value):
    if not isinstance(value, str):
        raise TypeError("published text fields must be strings")


def _json_list(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array: {path}")
    return value


def point_id(*parts):
    """Lossless composite ID; retain the original fields alongside it."""
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def visible_tools(level, instruction, history):
    """Keep every visible definition and its source, without choosing among conflicts.

    This is candidate evidence, not an OpenAI schema conversion. In particular,
    optional parameters and conflicting descriptions are not silently rewritten.
    Neither target text nor future source history is accepted by this function.
    """
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("unsupported API-Bank level")
    _text(instruction)
    _text(history)
    occurrences = []

    def add(schema, format_name, field, line):
        if format_name == "level3_initial":
            fields(schema, ("apiCode", "description", "parameters", "response"))
            name = schema["apiCode"]
            inputs, outputs = schema["parameters"], schema["response"]
        else:
            fields(
                schema,
                ("name", "description", "input_parameters", "output_parameters"),
                ("desc_for_search",),
            )
            name = schema["name"]
            inputs, outputs = schema["input_parameters"], schema["output_parameters"]
        if not isinstance(name, str) or not name:
            raise ValueError("visible tools require nonempty canonical names")
        _text(schema["description"])
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise TypeError("visible parameter definitions must be objects")
        for definitions in (inputs, outputs):
            for parameter, definition in definitions.items():
                if not isinstance(parameter, str) or not parameter:
                    raise ValueError("parameter names must be nonempty strings")
                fields(definition, ("type", "description"))
                _text(definition["type"])
                _text(definition["description"])
        occurrences.append(
            {
                "name": name,
                "format": format_name,
                "source": {"field": field, "line": line},
                "schema": deepcopy(schema),
            }
        )

    if level in (1, 2):
        if instruction.count(DESCRIPTION_MARKER) != 1:
            raise ValueError("expected exactly one official API descriptions section")
        prefix, descriptions = instruction.split(DESCRIPTION_MARKER)
        offset = prefix.count("\n") + 3
        for line, text in enumerate(descriptions.splitlines(), offset):
            if text.strip():
                add(json.loads(text), "api_definition", "instruction", line)
    elif level == 3:
        if not history.splitlines():
            raise ValueError("level 3 input must start with a tool definition")
        add(json.loads(history.splitlines()[0]), "level3_initial", "input", 1)
    if not occurrences:
        raise ValueError("published input has no initial visible tools")

    for line, text in enumerate(history.splitlines(), 1):
        if not text.startswith("API-Request: [ToolSearcher("):
            continue
        match = SEARCH_RESULT.fullmatch(text)
        if match is None:
            raise ValueError("malformed visible ToolSearcher history")
        value = ast.literal_eval(match.group(1))
        if level == 3:
            fields(value, ("api_name", "input", "output", "exception"))
            fields(value["input"], ("keywords",))
            _text(value["input"]["keywords"])
            if value["api_name"] != "ToolSearcher":
                raise ValueError(
                    "ToolSearcher history wrapper has a different API name"
                )
            if value["exception"] is not None:
                _text(value["exception"])
                if value["output"] is not None:
                    raise ValueError("failed ToolSearcher output must be null")
                continue
            value = value["output"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not value:
            raise ValueError("ToolSearcher output must contain visible API definitions")
        for schema in value:
            add(schema, "api_definition", "input", line)

    names = list(dict.fromkeys(row["name"] for row in occurrences))
    variants = {}
    for name in names:
        indices = [i for i, row in enumerate(occurrences) if row["name"] == name]
        definitions = {
            json.dumps(occurrences[i]["schema"], sort_keys=True, ensure_ascii=False)
            for i in indices
        }
        if len(definitions) > 1:
            variants[name] = indices
    return {
        "names": names,
        "occurrences": occurrences,
        "multiple_definitions": variants,
    }


def load_tasks(config):
    # Reuse the official sample enumeration. Sample.apis scans the *full* source
    # conversation and is intentionally never exposed as input or candidates.
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    # Sample is pure data enumeration; importing its module also imports the old
    # ToolManager/torch stack, which belongs only in the scoring worker.
    filename = source / "evaluator_by_json.py"
    tree = ast.parse(filename.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Sample"
    )
    namespace = {}
    exec(  # noqa: S102 -- pinned pure Sample class without importing old Torch
        compile(ast.Module(body=[node], type_ignores=[]), str(filename), "exec"),
        namespace,
    )
    Sample = namespace["Sample"]
    data = Path(config["protocol"]["data_root"])
    originals = {}
    for level, folder in ((1, "level-1-given-desc"), (2, "level-2-toolsearcher")):
        paths = sorted((source / "lv1-lv2-samples" / folder).glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no canonical source conversations: {folder}")
        for path in paths:
            history = [json.loads(line) for line in path.read_text().splitlines()]
            for index, sample in enumerate(Sample.from_chat_history(history)):
                originals[level, point_id(path.name, index)] = sample.ground_truth
    level3 = _json_list(data / "level-3.json")
    tasks, answers, published_api3 = {}, {}, {}
    for category in APIBANK_CATEGORIES:
        _, level_text, kind = category.split("_")
        level = int(level_text)
        rows = _json_list(data / FILES[category])
        seen = set()
        expected = (
            {
                identity
                for (source_level, identity), gold in originals.items()
                if source_level == level
                and gold["role"] == ("API" if kind == "api" else "AI")
            }
            if level < 3
            else {
                point_id(sample_id, api_id)
                for sample_id, sample in enumerate(level3)
                for api_id in (
                    range(len(sample["apis"]))
                    if kind == "api"
                    else [len(sample["apis"]) - 1]
                )
            }
        )
        for row in rows:
            fields(
                row,
                ("file", "id", "instruction", "input", "expected_output")
                if level < 3
                else ("sample_id", "api_id", "instruction", "input", "output"),
            )
            for field in (
                "instruction",
                "input",
                "expected_output" if level < 3 else "output",
            ):
                _text(row[field])
            if level < 3:
                integer(row["id"])
                if (
                    not isinstance(row["file"], str)
                    or Path(row["file"]).name != row["file"]
                ):
                    raise ValueError(
                        "source filename must be a plain canonical filename"
                    )
                identity = point_id(row["file"], row["id"])
                source_id = {"file": row["file"], "id": row["id"]}
                gold = originals[level, identity]
                target = row["expected_output"]
            else:
                integer(row["sample_id"])
                integer(row["api_id"])
                identity = point_id(row["sample_id"], row["api_id"])
                source_id = {"sample_id": row["sample_id"], "api_id": row["api_id"]}
                sample = level3[row["sample_id"]]
                api = sample["apis"][row["api_id"]]
                gold = (
                    {
                        "role": "API",
                        "api_name": api["api_name"],
                        "result": api["output"],
                    }
                    if kind == "api"
                    else {"role": "AI", "text": sample["response"]}
                )
                target = row["output"]
                if kind == "api":
                    published_api3[identity] = row
            if identity in seen or identity not in expected:
                raise ValueError(
                    f"{category}: duplicate ID or source role/index mismatch: {identity}"
                )
            seen.add(identity)
            suffix = (
                "Generate API Request:" if kind == "api" else "Generate AI Response:"
            ) + (" \n" if level == 3 else "\n")
            if not row["input"].endswith(suffix):
                raise ValueError(f"{category}: official decision marker is missing")
            visible = visible_tools(level, row["instruction"], row["input"])
            task_key = category, identity
            tasks[task_key] = {
                "source_id": source_id,
                "level": level,
                "kind": kind,
                "instruction": row["instruction"],
                "input": row["input"],
                "visible_tools": visible,
            }
            answers[task_key] = {
                "published_target": target,
                "ground_truth": deepcopy(gold),
            }
        if seen != expected:
            raise ValueError(
                f"{category}: published records do not cover official source IDs"
            )

    # ICL is an alternate instruction for the same L3 API points, not extra tasks.
    icl_seen = set()
    for row in _json_list(data / "level-3-batch-inf-icl.json"):
        fields(row, ("sample_id", "api_id", "instruction", "input", "output"))
        integer(row["sample_id"])
        integer(row["api_id"])
        _text(row["instruction"])
        identity = point_id(row["sample_id"], row["api_id"])
        if identity in icl_seen or identity not in published_api3:
            raise ValueError("ICL has duplicate or unknown source IDs")
        base = published_api3[identity]
        if row["input"] != base["input"] or row["output"] != base["output"]:
            raise ValueError("ICL and base have different input/target mappings")
        icl_seen.add(identity)
    if icl_seen != published_api3.keys():
        raise ValueError("ICL and base source coverage differs")

    selected = set()
    for category, ids in config["task_ids"].items():
        available = {
            identity for task_category, identity in tasks if task_category == category
        }
        if ids != "all" and not set(ids) <= available:
            raise ValueError(f"{category}: unknown published decision-point ID")
        selected.update(
            (category, identity) for identity in (available if ids == "all" else ids)
        )
    return (
        {key: value for key, value in tasks.items() if key in selected},
        {key: value for key, value in answers.items() if key in selected},
    )


def run_episode(config, *args, **kwargs):
    if config["protocol"]["mode"] == "cpu_only":
        raise ValueError("API-Bank CPU preview cannot generate an episode")
    from .apibank_runtime import run_episode as generate

    return generate(config, *args, **kwargs)
