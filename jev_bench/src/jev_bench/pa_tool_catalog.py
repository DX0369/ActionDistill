"""Extract interface schemas only, using each pinned benchmark's conversion."""

from contextlib import ExitStack
from copy import deepcopy

from .benchmarks import load_tasks
from .pa_tool import VERSION, task_key
from .pa_tool_prepare import identity, validate_catalog


def build_catalog(config, executor):
    tasks, _ = load_tasks(config)
    benchmark = config["benchmark"]["name"]
    scopes, bindings, domains, definition_sources = {}, {}, {}, {}
    description_augmentations = {}
    stack = ExitStack()
    if benchmark == "stabletoolbench":
        from .benchmarks.stabletoolbench import Archive
        from .benchmarks.stabletoolbench_runtime import catalog_order

        archive = stack.enter_context(Archive(config))
        order = catalog_order(archive)
    try:
        for (category, task_id), task in tasks.items():
            if benchmark == "bfcl":
                from bfcl_eval.constants.enums import ModelStyle
                from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
                from bfcl_eval.model_handler.utils import convert_to_tool

                functions = deepcopy(task["function"])
                for turn, extra in sorted(
                    task.get("missed_function", {}).items(), key=lambda row: int(row[0])
                ):
                    functions.extend(deepcopy(extra))
                tools = convert_to_tool(
                    functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
                )
            elif benchmark == "tau2":
                from tau2.registry import registry

                if category not in domains:
                    env = registry.get_env_constructor(category)()
                    domains[category] = [tool.openai_schema for tool in env.get_tools()]
                tools = domains[category]
            elif benchmark == "toolhop":
                tools = [
                    {"type": "function", "function": tool} for tool in task["tools"]
                ]
            elif benchmark == "apibank":
                from .pa_tool_text import interface_tools

                tools, _, sources = interface_tools(task)
                definition_sources[
                    task_key({"category": category, "task_id": task_id})
                ] = sources
            elif benchmark == "toolsandbox":
                from tool_sandbox.common.execution_context import new_context
                from tool_sandbox.common.tool_conversion import convert_to_openai_tools
                from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent

                with new_context(task["scenario"].starting_context):
                    tools = convert_to_openai_tools(
                        OpenAIAPIAgent.get_available_tools()
                    )
                augmentations = sorted(
                    str(value)
                    for value in task[
                        "scenario"
                    ].starting_context.tool_augmentation_list
                    if str(value)
                    in {"TOOL_DESCRIPTION_SCRAMBLED", "ARG_DESCRIPTION_SCRAMBLED"}
                )
            elif benchmark == "stabletoolbench":

                def forbidden_transport(*args, **kwargs):
                    raise RuntimeError("PA-Tool catalog must not execute tools")

                env = archive.environment(
                    task,
                    catalog_order=order,
                    transport=forbidden_transport,
                    max_observation_length=config["protocol"]["max_observation_length"],
                    observ_compress_method="truncate",
                )
                tools = env.functions
            else:
                raise ValueError("unsupported PA-Tool benchmark")
            scope = (
                identity([tools, augmentations])
                if benchmark == "toolsandbox"
                else identity(tools)
            )
            if benchmark == "toolsandbox":
                description_augmentations[scope] = augmentations
            scopes[scope] = deepcopy(tools)
            bindings[task_key({"category": category, "task_id": task_id})] = scope
    finally:
        stack.close()
    catalog = {
        "version": VERSION,
        "benchmark": deepcopy(config["benchmark"]),
        "executor": deepcopy(config["executors"][executor]),
        "control_names": ["Finish"] if benchmark == "stabletoolbench" else [],
        "scopes": scopes,
        "bindings": bindings,
        "definition_sources": definition_sources,
    }
    if benchmark == "toolsandbox":
        catalog["description_augmentations"] = description_augmentations
    validate_catalog(catalog)
    return catalog
