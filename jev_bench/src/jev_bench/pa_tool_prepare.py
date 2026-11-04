"""PA-Tool Appendix L / Algorithm 1 and the explicitly approved supplements.

Preparation runs in llm-efficiency. Runtime projection never imports RapidFuzz.
"""

import hashlib
import json
from collections import Counter
from fractions import Fraction

from .pa_tool import VERSION, legal_name, remap_schema

PROTOCOL = {
    "version": VERSION,
    "candidates": 16,
    "minimum_valid_candidates": 12,
    "temperature": 0.4,
    "reference_temperature": 0,
    "alpha": 0.2,
    "top_p": 1,
    "top_k": -1,
    "min_p": 0,
    "max_tokens": 8192,
    "seed": 20261015,
    "enable_thinking": True,
    "parameter_context": "selected_tool_alias",
    "unresolved": "stop",
    "candidate_failure": "exclude_length_or_invalid_final_name",
    "complete_all_slots": True,
    "density": "neighbors_over_valid_candidates_minus_one",
    "generation_failure": "original_name_on_invalid_reference_or_insufficient_candidates",
    "tie_break": {
        "source": "user_approved_project_supplement_not_paper",
        "candidate_order": [
            "density_desc",
            "reference_distance_asc",
            "first_sample_slot_asc",
        ],
        "equal_density_tool_collision": "original_tool_order",
        "equal_density_parameter_collision": "original_property_order",
    },
    "toolhop_array_description": "items.description_when_top_level_missing",
    "toolhop_descendant_description": "properties_items_depth_first_original_order_newline",
    "apibank_duplicate_description": "shared_alias_from_instruction",
    "stable_empty_parameter_description": "empty_text_with_tool_context",
    "toolsandbox_hidden_description": "preserve_current_visible_name",
}


class NameGenerationError(ValueError):
    """Only these verified model output failures permit excluding a naming slot."""

    def __init__(self, reason, *, failures=()):
        if reason not in {"length", "invalid_final_name"}:
            raise ValueError("unsupported PA-Tool naming failure reason")
        self.reason = reason
        self.failures = list(failures)
        super().__init__(f"PA-Tool naming output failed: {reason}")


def generated_name(finish_reason, content, tool_calls):
    if content is not None and not isinstance(content, str):
        raise ValueError("PA-Tool naming content must be a string or null")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise ValueError("PA-Tool naming tool_calls must be a list or null")
    if finish_reason == "length":
        raise NameGenerationError("length")
    if finish_reason != "stop":
        raise ValueError(f"unexpected PA-Tool naming finish reason: {finish_reason!r}")
    if tool_calls or content is None or not legal_name(content.strip()):
        raise NameGenerationError("invalid_final_name")
    return content.strip()


def check_name_failure(record, model):
    """A fallback label is valid only with the corresponding provider evidence."""
    provider = record["provider"]
    if provider["error"] is not None:
        raise ValueError("PA-Tool service failure cannot become a naming fallback")
    response = provider["response"]
    choices = response["choices"]
    if response["model"] != model or not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("PA-Tool naming response model or choices differ")
    choice = choices[0]
    message = choice["message"]
    if message["role"] != "assistant":
        raise ValueError("PA-Tool naming response role differs")
    try:
        generated_name(
            choice["finish_reason"], message["content"], message["tool_calls"]
        )
    except NameGenerationError as exc:
        expected = {
            "type": type(exc).__name__,
            "message": str(exc),
            "reason": exc.reason,
        }
        if record["name"] is not None or record["error"] != expected:
            raise ValueError(
                "PA-Tool naming failure differs from provider response"
            ) from exc
    else:
        raise ValueError("PA-Tool fallback requires a recorded naming output failure")


def identity(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def check_protocol(value):
    if json.dumps(value, sort_keys=True) != json.dumps(PROTOCOL, sort_keys=True):
        raise ValueError("use the approved PA-Tool generation protocol")


def prompt(
    component, description, *, tool_name=None, tool_description=None, allow_empty=False
):
    if not isinstance(description, str) or (
        not description.strip()
        and not (allow_empty and component == "parameter" and description == "")
    ):
        raise ValueError("PA-Tool components require an explicit description")
    result = (
        f"Generate a {component} name from the description below.\n"
        f"The {component} will be used in a tool agent scenario.\n"
        f"Description:\n{description}\n"
    )
    if component == "tool":
        result += (
            "Example:\nDescription: A tool that manages files and directories on the system.\n"
            "Output: file_manager\n"
        )
    elif component == "parameter":
        if not legal_name(tool_name) or not isinstance(tool_description, str):
            raise ValueError(
                "parameter prompt requires the selected tool alias and description"
            )
        result += (
            "Example:\nContext:\n"
            "Tool: file_manager - A tool for managing files and directories\n"
            "Output: file_path\nContext:\n"
            f"Tool: {tool_name} - {tool_description}\n"
        )
    else:
        raise ValueError("unknown PA-Tool component")
    return result + "Generate only the name without additional explanation."


def parameter_description(schema, *, allow_toolhop=False, allow_empty=False):
    if "description" in schema:
        value, source = schema["description"], "description"
    elif (
        allow_toolhop
        and schema.get("type") == "array"
        and isinstance(schema.get("items"), dict)
        and "description" in schema["items"]
        and isinstance(schema["items"]["description"], str)
        and schema["items"]["description"].strip()
    ):
        value, source = schema["items"]["description"], "items.description"
    elif allow_toolhop and schema.get("type") in ("object", "array"):
        descriptions, paths = [], []

        def visit(node, path):
            if "description" in node:
                description = node["description"]
                if not isinstance(description, str):
                    raise ValueError("PA-Tool descendant description must be a string")
                if description.strip():
                    descriptions.append(description)
                    paths.append([*path, "description"])
            for key, child in node.items():
                if key == "properties" and node.get("type") == "object":
                    for name, definition in child.items():
                        visit(definition, [*path, key, name])
                elif (
                    key == "items"
                    and node.get("type") == "array"
                    and isinstance(child, dict)
                ):
                    visit(child, [*path, key])

        visit(schema, [])
        if not descriptions:
            raise ValueError("PA-Tool parameter has no approved description source")
        value = "\n".join(descriptions)
        source = {"rule": "descendant_descriptions", "paths": paths}
    else:
        raise ValueError("PA-Tool parameter has no approved description source")
    if not isinstance(value, str) or (
        not value.strip()
        and not (allow_empty and source == "description" and value == "")
    ):
        raise ValueError(f"PA-Tool parameter {source} is not a nonempty string")
    return value, source


def rank_names(samples, reference):
    from rapidfuzz.distance import Levenshtein

    if (
        len(samples) != PROTOCOL["candidates"]
        or any(name is not None and not legal_name(name) for name in samples)
        or sum(name is not None for name in samples)
        < PROTOCOL["minimum_valid_candidates"]
        or not legal_name(reference)
    ):
        raise ValueError(
            "PA-Tool requires 16 slots, at least 12 legal samples and a legal reference"
        )
    valid = [name for name in samples if name is not None]
    counts = Counter(valid)
    threshold = PROTOCOL["alpha"] * max(map(len, valid))
    result = []
    for name, count in counts.items():
        score = (
            sum(
                n
                for other, n in counts.items()
                if Levenshtein.distance(name, other) <= threshold
            )
            - 1
        )
        result.append(
            {
                "name": name,
                "count": count,
                "score": Fraction(score, len(valid) - 1),
                "reference_distance": Levenshtein.distance(name, reference),
                "first_sample_slot": samples.index(name),
            }
        )
    # The final key is the approved project supplement, not a paper rule.
    return sorted(
        result,
        key=lambda row: (
            -row["score"],
            row["reference_distance"],
            row["first_sample_slot"],
        ),
    )


def resolve_names(rankings, *, reserved=(), original_names=None):
    """Lock by density, then original tool/property order; never rescore losers."""
    originals = (
        {}
        if original_names is None
        else {
            key: original_names[key] for key, rows in rankings.items() if rows is None
        }
    )
    if (
        not all(map(legal_name, originals.values()))
        or len(set(originals.values())) != len(originals)
        or set(originals.values()) & set(reserved)
    ):
        raise ValueError("PA-Tool fallback original names must be legal and unique")
    reserved = (*reserved, *originals.values())
    rankings = {key: rows for key, rows in rankings.items() if key not in originals}
    remaining = {key: list(rows) for key, rows in rankings.items()}
    locked, used = {}, set(reserved)
    while remaining:
        proposals = {}
        for key, rows in remaining.items():
            options = [row for row in rows if row["name"] not in used]
            if not options:
                raise ValueError(f"PA-Tool candidates exhausted for {key}")
            first = options[0]
            proposals.setdefault(first["name"], []).append((key, first))
        winners = []
        for name, contenders in proposals.items():
            # Dict insertion order is the original tool/property order supplied
            # by build_mapping. Equal density must not introduce a new sort.
            key, _ = max(contenders, key=lambda contender: contender[1]["score"])
            winners.append((key, name))
        for key, name in winners:
            locked[key] = name
            used.add(name)
            del remaining[key]
    return {**originals, **locked}


def description_preservation(catalog, scope, function, parameter=None):
    """Only the two official description-hiding augmentations permit this."""
    if catalog["benchmark"]["name"] != "toolsandbox":
        return None
    augmentations = catalog["description_augmentations"][scope]
    if parameter is None:
        if (
            "TOOL_DESCRIPTION_SCRAMBLED" in augmentations
            and function["description"] == ""
        ):
            return "tool_description_scrambled"
    elif (
        "ARG_DESCRIPTION_SCRAMBLED" in augmentations
        and "description" not in function["parameters"]["properties"][parameter]
    ):
        return "arg_description_scrambled"
    return None


def description_preservations(catalog):
    """Unique retained components, with no invented generation slots or usage."""
    if catalog["benchmark"]["name"] != "toolsandbox":
        return {}
    augmentations = catalog["description_augmentations"]
    if set(augmentations) != set(catalog["scopes"]) or any(
        not isinstance(values, list)
        or any(
            value not in {"TOOL_DESCRIPTION_SCRAMBLED", "ARG_DESCRIPTION_SCRAMBLED"}
            for value in values
        )
        or len(values) != len(set(values))
        for values in augmentations.values()
    ):
        raise ValueError("invalid ToolSandbox description augmentation metadata")
    result = {}
    for scope, tools in catalog["scopes"].items():
        for index, tool in enumerate(tools):
            function = tool["function"]
            spec = {
                "function": function,
                "occurrence": sum(other == tool for other in tools[:index]),
            }
            for parameter in (None, *function["parameters"]["properties"]):
                reason = description_preservation(catalog, scope, function, parameter)
                if reason is None:
                    continue
                component = "tool" if parameter is None else "parameter"
                key = identity({**spec, "component": component, "parameter": parameter})
                result[key] = {
                    "component": component,
                    "visible_name": function["name"]
                    if parameter is None
                    else parameter,
                    "reason": reason,
                }
    return result


def validate_catalog(catalog):
    if catalog["version"] != VERSION:
        raise ValueError("unsupported PA-Tool catalog version")
    if set(catalog["bindings"].values()) - set(catalog["scopes"]):
        raise ValueError("PA-Tool bindings reference missing scopes")
    controls = catalog["control_names"]
    if controls != (
        ["Finish"] if catalog["benchmark"]["name"] == "stabletoolbench" else []
    ):
        raise ValueError("PA-Tool control name exclusions differ")
    description_preservations(catalog)
    for scope, tools in catalog["scopes"].items():
        for tool in tools:
            if tool["type"] != "function":
                raise ValueError("PA-Tool catalog requires function tools")
            function = tool["function"]
            properties = function["parameters"]["properties"]
            if function["name"] not in controls:
                if description_preservation(catalog, scope, function) is None:
                    prompt("tool", function["description"])
                for parameter, schema in properties.items():
                    if (
                        description_preservation(catalog, scope, function, parameter)
                        is not None
                    ):
                        continue
                    try:
                        description, _ = parameter_description(
                            schema,
                            allow_toolhop=catalog["benchmark"]["name"] == "toolhop",
                            allow_empty=catalog["benchmark"]["name"]
                            == "stabletoolbench",
                        )
                        prompt(
                            "parameter",
                            description,
                            tool_name="checked_tool",
                            tool_description=function["description"],
                            allow_empty=catalog["benchmark"]["name"]
                            == "stabletoolbench",
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"scope={scope}, tool={function['name']}, parameter={parameter}: {exc}"
                        ) from exc
            remap_schema(
                function,
                {
                    "name": function["name"],
                    "parameters": {key: key for key in properties},
                },
            )


def estimate(catalog):
    """Exact component count before context-dependent cross-scope reuse."""
    validate_catalog(catalog)
    count = sum(
        1 + len(tool["function"]["parameters"]["properties"])
        for tools in catalog["scopes"].values()
        for tool in tools
        if tool["function"]["name"] not in catalog["control_names"]
    )
    preserved = sum(
        description_preservation(catalog, scope, tool["function"], parameter)
        is not None
        for scope, tools in catalog["scopes"].items()
        for tool in tools
        for parameter in (None, *tool["function"]["parameters"]["properties"])
    )
    preservations = description_preservations(catalog)
    return {
        "scopes": len(catalog["scopes"]),
        "bindings": len(catalog["bindings"]),
        "components_upper_bound": count,
        "requests_upper_bound": (PROTOCOL["candidates"] + 1) * (count - preserved),
        "description_preservations": {
            "unique_components": len(preservations),
            "by_component": dict(
                Counter(row["component"] for row in preservations.values())
            ),
            "by_reason": dict(Counter(row["reason"] for row in preservations.values())),
        },
        "empty_parameter_descriptions": sum(
            schema.get("description") == ""
            for tools in catalog["scopes"].values()
            for tool in tools
            if tool["function"]["name"] not in catalog["control_names"]
            for schema in tool["function"]["parameters"]["properties"].values()
        ),
    }


def build_mapping(catalog, generate):
    validate_catalog(catalog)
    scopes, cached, sources = {}, {}, {}
    fallbacks = {}
    statistics = {}

    def candidates(spec, text):
        key = identity(spec)
        if key not in cached:
            samples, reference = generate(key, text)
            if (
                len(samples) != PROTOCOL["candidates"]
                or any(name is not None and not legal_name(name) for name in samples)
                or reference is not None
                and not legal_name(reference)
            ):
                raise ValueError("PA-Tool generator returned an invalid candidate set")
            excluded = [slot for slot, name in enumerate(samples) if name is None]
            valid_count = len(samples) - len(excluded)
            statistics[key] = {
                "component": spec["component"],
                "valid_candidates": valid_count,
                "discarded_candidate_slots": excluded,
                "reference_valid": reference is not None,
            }
            reason = (
                "reference_failed"
                if reference is None
                else "insufficient_valid_candidates"
                if valid_count < PROTOCOL["minimum_valid_candidates"]
                else None
            )
            if reason is not None:
                cached[key] = None
                fallbacks[key] = {
                    "component": spec["component"],
                    "original_name": spec["function"]["name"]
                    if spec["component"] == "tool"
                    else spec["parameter"],
                    "reason": reason,
                    "failed_slots": excluded
                    + ([PROTOCOL["candidates"]] if reference is None else []),
                }
            else:
                cached[key] = rank_names(samples, reference)
        return cached[key]

    for scope, tools in sorted(catalog["scopes"].items()):
        controls, rankings, specs = {}, {}, {}
        for index, tool in enumerate(tools):
            function = tool["function"]
            # Occurrence preserves distinct candidates with identical schemas.
            occurrence = sum(other == tool for other in tools[:index])
            specs[index] = {"function": function, "occurrence": occurrence}
            if function["name"] in catalog["control_names"]:
                controls[index] = function["name"]
            elif description_preservation(catalog, scope, function) is not None:
                rankings[index] = None
            else:
                rankings[index] = candidates(
                    {**specs[index], "component": "tool"},
                    prompt("tool", function["description"]),
                )
        names = {
            **controls,
            **resolve_names(
                rankings,
                reserved=controls.values(),
                original_names={
                    i: spec["function"]["name"] for i, spec in specs.items()
                },
            ),
        }
        rows, scope_sources = [], []
        for index, tool in enumerate(tools):
            function = tool["function"]
            properties = function["parameters"]["properties"]
            if index in controls:
                parameters = {key: key for key in properties}
                descriptions = {}
            else:
                descriptions = {
                    key: parameter_description(
                        schema,
                        allow_toolhop=catalog["benchmark"]["name"] == "toolhop",
                        allow_empty=catalog["benchmark"]["name"] == "stabletoolbench",
                    )
                    for key, schema in properties.items()
                    if description_preservation(catalog, scope, function, key) is None
                }
                parameters = resolve_names(
                    {
                        key: None
                        if key not in descriptions
                        else candidates(
                            {
                                **specs[index],
                                "component": "parameter",
                                "parameter": key,
                                "tool_alias": names[index],
                                "description_source": descriptions[key][1],
                            },
                            prompt(
                                "parameter",
                                descriptions[key][0],
                                tool_name=names[index],
                                tool_description=function["description"],
                                allow_empty=catalog["benchmark"]["name"]
                                == "stabletoolbench",
                            ),
                        )
                        for key, schema in properties.items()
                    },
                    original_names={key: key for key in properties},
                )
            rows.append({"name": names[index], "parameters": parameters})
            scope_sources.append({key: value[1] for key, value in descriptions.items()})
        scopes[scope] = rows
        sources[scope] = scope_sources
    return {
        "version": VERSION,
        "protocol": PROTOCOL,
        "scopes": scopes,
        "parameter_description_sources": sources,
        "generation_fallbacks": fallbacks,
        "candidate_statistics": statistics,
        "description_preservations": description_preservations(catalog),
    }
