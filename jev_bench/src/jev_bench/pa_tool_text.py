"""API-Bank's textual interface: edit declared spans, never business strings."""

import ast
import io
import json
import re
import tokenize
from copy import deepcopy

from .pa_tool import InterfaceViolation

# Exactly the pinned api_call_extraction.get_api_call framing.
CALL = re.compile(r"\[(\w+)\((.*)\)\]")


def interface_tools(task):
    """Approved API-Bank exception: one interface for description-only variants."""
    from .benchmarks.apibank_runtime import advisor_tools

    occurrences = task["visible_tools"]["occurrences"]
    grouped = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence["name"], []).append(occurrence)
    chosen, sources = [], []
    for name, group in grouped.items():
        first = group[0]
        structural = {
            key: value
            for key, value in first["schema"].items()
            if key not in ("description", "desc_for_search")
        }
        if any(
            row["format"] != first["format"]
            or {
                key: value
                for key, value in row["schema"].items()
                if key not in ("description", "desc_for_search")
            }
            != structural
            for row in group
        ):
            raise ValueError(f"PA-Tool conflicting API-Bank interface: {name}")
        if any(row["schema"] != first["schema"] for row in group):
            initial = [row for row in group if row["source"]["field"] == "instruction"]
            if len(initial) != 1:
                raise ValueError(
                    f"PA-Tool description variants need one instruction definition: {name}"
                )
            first = initial[0]
        chosen.append(first)
        sources.append(
            {"name": name, "source": first["source"], "occurrences": len(group)}
        )
    tools = advisor_tools({"visible_tools": {"occurrences": chosen}})
    positions = {row["name"]: index for index, row in enumerate(chosen)}
    return tools, [positions[row["name"]] for row in occurrences], sources


def replace_spans(text, edits):
    end = len(text)
    for start, stop, value in sorted(edits, reverse=True):
        if stop > end:
            raise ValueError("overlapping PA-Tool text edits")
        text = text[:start] + value + text[stop:]
        end = start
    return text


def node_span(text, node):
    """AST columns are UTF-8 byte offsets, not character offsets."""
    lines = text.splitlines(keepends=True)
    start = sum(map(len, lines[: node.lineno - 1]))
    stop = sum(map(len, lines[: node.end_lineno - 1]))
    return (
        start + len(lines[node.lineno - 1].encode()[: node.col_offset].decode()),
        stop + len(lines[node.end_lineno - 1].encode()[: node.end_col_offset].decode()),
    )


def dict_fields(node):
    if not isinstance(node, ast.Dict):
        raise TypeError("PA-Tool expects a literal API definition")
    return {
        ast.literal_eval(key): (key, value)
        for key, value in zip(node.keys, node.values)
    }


def definition_edits(text, node, row, format_name):
    fields = dict_fields(node)
    name_key, parameter_key = (
        ("apiCode", "parameters")
        if format_name == "level3_initial"
        else ("name", "input_parameters")
    )
    edits = [(*node_span(text, fields[name_key][1]), json.dumps(row["name"]))]
    for key, (key_node, _) in dict_fields(fields[parameter_key][1]).items():
        edits.append((*node_span(text, key_node), json.dumps(row["parameters"][key])))
    return edits


def call_text(text, projection, *, restore):
    match = CALL.search(text)
    if match is None:
        return text
    name = match.group(1)
    if restore:
        canonical, _ = projection.restore_call(name, None)
        index = projection.aliases[name]
        names = {
            alias: original
            for original, alias in projection.rows[index]["parameters"].items()
        }
    else:
        indices = projection.originals.get(name, [])
        if len(indices) != 1:
            raise ValueError(
                "PA-Tool textual history needs an unambiguous tool definition"
            )
        index = indices[0]
        canonical = projection.rows[index]["name"]
        names = projection.rows[index]["parameters"]
    edits = [(match.start(1), match.end(1), canonical)]
    arguments = match.group(2)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(arguments).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        error = "PA-Tool cannot safely translate malformed textual arguments"
        if restore:
            raise InterfaceViolation(error, role="agent") from exc
        raise ValueError(error) from exc
    depth = 0
    keys = []
    for position, token in enumerate(tokens):
        if (
            token.type == tokenize.NAME
            and depth == 0
            and tokens[position + 1].string == "="
        ):
            keys.append(token.string)
            # CALL cannot span newlines; tokenize columns are character offsets.
            edits.append(
                (
                    match.start(2) + token.start[1],
                    match.start(2) + token.end[1],
                    names.get(token.string, token.string),
                )
            )
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
    from api_call_extraction import parse_api_call

    _, extracted = parse_api_call(match.group(0))
    if restore:
        # The upstream regex can extract keyword-like text inside nested
        # expressions. Never let such text bypass the exposed input interface.
        expected = projection.restore_arguments(extracted, index)
        if set(extracted) - set(keys):
            raise InterfaceViolation(
                "PA-Tool official parser extracted a non-top-level argument",
                role="agent",
            )
        projection.restore_arguments(dict.fromkeys(keys), index)
    else:
        expected = projection.project_arguments(extracted, index)
        projection.project_arguments(dict.fromkeys(keys), index)
    translated = replace_spans(text, edits)
    if parse_api_call(translated) != (canonical, expected):
        message = "PA-Tool text translation changes the official parsed call"
        if restore:
            raise InterfaceViolation(message, role="agent")
        raise ValueError(message)
    return translated


def project_task(task, projection):
    """Tool definitions and API-Request calls only; policies/examples stay exact."""
    tools, occurrence_positions, _ = interface_tools(task)
    indices = projection._indices(tools)
    projection.visible = indices
    result = deepcopy(task)
    lines = {
        field: task[field].splitlines(keepends=True)
        for field in ("instruction", "input")
    }
    grouped = {}
    for occurrence, position in zip(
        task["visible_tools"]["occurrences"], occurrence_positions
    ):
        index = indices[position]
        source = occurrence["source"]
        grouped.setdefault((source["field"], source["line"] - 1), []).append(
            (occurrence, index)
        )
    for (field, line_index), occurrences in grouped.items():
        text = lines[field][line_index]
        offset = text.index("->") + 2 if text.startswith("API-Request: ") else 0
        literal = text[offset:].strip()
        offset += len(text[offset:]) - len(text[offset:].lstrip())
        node = ast.parse(literal, mode="eval").body
        if text.startswith("API-Request: ") and task["level"] == 3:
            node = dict_fields(node)["output"][1]
        nodes = node.elts if isinstance(node, ast.List) else [node]
        if len(nodes) != len(occurrences):
            raise ValueError("PA-Tool discovery definition count changed")
        edits = []
        for definition, (occurrence, index) in zip(nodes, occurrences):
            edits.extend(
                (start + offset, stop + offset, value)
                for start, stop, value in definition_edits(
                    literal, definition, projection.rows[index], occurrence["format"]
                )
            )
        lines[field][line_index] = replace_spans(text, edits)
    for index, line in enumerate(lines["input"]):
        if line.startswith("API-Request: "):
            call, separator, observation = line.partition("->")
            lines["input"][index] = (
                call_text(call, projection, restore=False) + separator + observation
            )
    for field, content in lines.items():
        result[field] = "".join(content)
    return result
