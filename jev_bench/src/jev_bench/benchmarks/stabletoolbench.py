"""StableToolBench source inputs, read-only ZIP access and official wrapper."""

import ast
import importlib.util
import io
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from ..config import STABLETOOLBENCH_CATEGORIES, WORKSPACE_ROOT, fields, integer


def configure(config, runtime):
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    (source / "solvable_queries/test_instruction").resolve(strict=True)
    Path(config["protocol"]["cache_archive"]).resolve(strict=True)


def official_functions(config):
    """Load only pure naming functions, avoiding server initialization and I/O."""
    source = WORKSPACE_ROOT / config["benchmark"]["path"]
    spec = importlib.util.spec_from_file_location(
        "stabletoolbench_names", source / "server/utils.py"
    )
    names = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(names)
    tree = ast.parse((source / "server/main.py").read_text())
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_tool_name_and_url"
    ]
    namespace = {"standardize": names.standardize, "change_name": names.change_name}
    exec(  # noqa: S102 -- fixed upstream function, no server initialization
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(source / "server/main.py"),
            "exec",
        ),
        namespace,
    )
    return names, namespace["prepare_tool_name_and_url"]


def _text(value):
    if not isinstance(value, str):
        raise TypeError("published text must be a string")


def _parameters(value):
    if not isinstance(value, list):
        raise TypeError("parameter definitions must be lists")
    for parameter in value:
        fields(parameter, ("name", "type", "description", "default"))
        for name in ("name", "type", "description"):
            _text(parameter[name])
        # False, zero, empty strings and lists are real published defaults.


def validate_api(api):
    fields(
        api,
        (
            "category_name",
            "tool_name",
            "api_name",
            "api_description",
            "required_parameters",
            "optional_parameters",
            "method",
        ),
        ("template_response",),
    )
    for field in (
        "category_name",
        "tool_name",
        "api_name",
        "api_description",
        "method",
    ):
        _text(api[field])
    for field in ("category_name", "tool_name", "api_name"):
        if not api[field]:
            raise ValueError("candidate identity must not be empty")
    for field in ("required_parameters", "optional_parameters"):
        _parameters(api[field])


def load_tasks(config):
    source = WORKSPACE_ROOT / config["benchmark"]["path"] / "solvable_queries"
    tasks, answers = {}, {}
    for category in STABLETOOLBENCH_CATEGORIES:
        rows = json.loads(
            (source / "test_instruction" / (category + ".json")).read_text()
        )
        published_ids = json.loads(
            (source / "test_query_ids" / (category + ".json")).read_text()
        )
        if not isinstance(rows, list) or not isinstance(published_ids, dict):
            raise TypeError("invalid published task/ID collection")
        seen = set()
        selected = config["task_ids"][category]
        for row in rows:
            fields(row, ("query_id", "query", "api_list", "relevant APIs"))
            integer(row["query_id"])
            identity = str(row["query_id"])
            if identity in seen:
                raise ValueError("duplicate published query ID")
            seen.add(identity)
            _text(row["query"])
            if not isinstance(row["api_list"], list) or not row["api_list"]:
                raise ValueError("published api_list must be nonempty")
            for api in row["api_list"]:
                validate_api(api)
            relevant = row["relevant APIs"]
            if not isinstance(relevant, list) or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(value, str) for value in pair)
                for pair in relevant
            ):
                raise ValueError("invalid relevant APIs target")
            if selected == "all" or identity in selected:
                key = (category, identity)
                tasks[key] = {
                    name: deepcopy(row[name])
                    for name in ("query_id", "query", "api_list")
                }
                answers[key] = {"relevant APIs": deepcopy(relevant)}
        if seen != published_ids.keys():
            raise ValueError(f"{category}: task and published ID coverage differ")
        if selected != "all" and not set(selected) <= seen:
            raise ValueError(f"{category}: unknown published ID")
    return tasks, answers


class CacheMiss(KeyError):
    """No exact stored response; never invoke the online fallback in CPU work."""


class Archive:
    def __init__(self, config):
        self.names, self.prepare = official_functions(config)
        self.source = WORKSPACE_ROOT / config["benchmark"]["path"]
        self.zip = ZipFile(config["protocol"]["cache_archive"], "r")
        source = (
            WORKSPACE_ROOT
            / config["benchmark"]["path"]
            / "toolbench/inference/Downstream_tasks/rapidapi.py"
        )
        tree = ast.parse(source.read_text())
        (wrapper,) = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "rapidapi_wrapper"
        ]
        methods = [
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"fetch_api_json", "api_json_to_openai_json"}
        ]
        namespace = {
            "json": json,
            "os": os,
            "standardize": self.names.standardize,
            "change_name": self.names.change_name,
            # Only catalog reads reach this local namespace, never the filesystem/server.
            "open": lambda path, mode: io.BytesIO(self.zip.read(path)),
        }
        exec(  # noqa: S102 -- fixed pure methods, no pipeline/server initialization
            compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self._fetch = namespace["fetch_api_json"]
        self._convert = namespace["api_json_to_openai_json"]
        self.tool_root_dir = "tools"
        self.cache_events = []
        self._whitelists = {}

    def environment(
        self,
        task,
        *,
        catalog_order,
        transport,
        max_observation_length,
        observ_compress_method,
        sleep=None,
    ):
        """Official wrapper with explicit catalog order and transport.

        Categories must be contiguous, reflecting two-level directory traversal.
        No filesystem ordering, live endpoint, retrieval or Finish mapping is
        selected here. The transport receives the official virtual-server payload.
        Official request/backoff waits remain active unless a test supplies sleep.
        """
        integer(max_observation_length, 1)
        _text(observ_compress_method)
        catalog_order = tuple(catalog_order)
        directories = {"tools": []}
        for member in catalog_order:
            parts = member.split("/")
            if len(parts) != 3 or parts[0] != "tools" or not member.endswith(".json"):
                raise ValueError(
                    "catalog_order requires explicit tools/category/file.json members"
                )
            self.zip.getinfo(member)
            category, filename = parts[1:]
            path = f"tools/{category}"
            if path not in directories:
                directories["tools"].append(category)
                directories[path] = []
            elif directories["tools"][-1] != category:
                raise ValueError("catalog_order must keep each category contiguous")
            if filename in directories[path]:
                raise ValueError("duplicate member in catalog_order")
            directories[path].append(filename)
        source = self.source / "toolbench/inference/Downstream_tasks/rapidapi.py"
        tree = ast.parse(source.read_text())
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "get_white_list",
                "contain",
            }:
                selected.append(node)
            elif isinstance(node, ast.ClassDef) and node.name == "rapidapi_wrapper":
                node.body = [
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef)
                    and method.name
                    in {
                        "__init__",
                        "build_tool_description",
                        "fetch_api_json",
                        "api_json_to_openai_json",
                        "check_success",
                        "to_json",
                        "restart",
                        "get_score",
                        "step",
                        "_step",
                    }
                ]
                selected.append(node)
        import requests

        def post(url, *, json, headers, timeout):
            response = transport(deepcopy(json))
            return SimpleNamespace(status_code=200, json=lambda: response)

        namespace = {
            "json": json,
            "base_env": object,
            "os": SimpleNamespace(
                path=SimpleNamespace(
                    join=os.path.join, isdir=lambda path: path in directories
                ),
                listdir=lambda path: directories[path],
                getenv=lambda name, default: "http://cpu.invalid/virtual",
            ),
            "open": lambda path, mode="r": io.BytesIO(self.zip.read(path)),
            "standardize": self.names.standardize,
            "change_name": self.names.change_name,
            "tqdm": lambda values: values,
            "colored": lambda text, **kwargs: text,
            "time": time if sleep is None else SimpleNamespace(sleep=sleep),
            "requests": SimpleNamespace(post=post, exceptions=requests.exceptions),
        }
        exec(  # noqa: S102 -- fixed official wrapper; transport and catalog I/O are scripted
            compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        # Catalog order determines the official last-wins behavior for duplicate names.
        if catalog_order not in self._whitelists:
            self._whitelists[catalog_order] = namespace["get_white_list"]("tools")
        whitelist = self._whitelists[catalog_order]
        # The official constructor may look up descriptions again for omitted APIs.
        namespace["get_white_list"] = lambda tool_root_dir: whitelist
        descriptions = namespace["contain"](
            [self.names.standardize(api["tool_name"]) for api in task["api_list"]],
            whitelist,
        )
        if descriptions is False:
            raise ValueError(
                "candidate tool is absent from the explicitly ordered whitelist"
            )
        wrapper = namespace["rapidapi_wrapper"](
            deepcopy(task),
            [[row["standard_tool_name"], row["description"]] for row in descriptions],
            None,
            SimpleNamespace(
                tool_root_dir="tools",
                toolbench_key="cpu",
                rapidapi_key="",
                use_rapidapi_key=False,
                api_customization=False,
                max_observation_length=max_observation_length,
                observ_compress_method=observ_compress_method,
            ),
            process_id=1,
        )
        wrapper.candidate_diagnostics = [
            {
                "index": index,
                "category_name": api["category_name"],
                "tool_name": api["tool_name"],
                "api_name": api["api_name"],
                "reason": "catalog_api_not_found",
            }
            for index, api in enumerate(task["api_list"])
            if not self.api_definitions(api)
        ]
        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()

    def tool_document(self, category_name, tool_name):
        category = self.names.standardize_category(category_name)
        tool = self.names.standardize(tool_name)
        path = f"tools/{category}/{tool}.json"
        document = json.loads(self.zip.read(path))
        for field in ("tool_name", "tool_description", "api_list"):
            if field not in document:
                raise KeyError(field)
        _text(document["tool_name"])
        if document["tool_description"] is not None:
            _text(document["tool_description"])
        if not isinstance(document["api_list"], list):
            raise TypeError("tool catalog api_list must be a list")
        return document

    def api_definitions(self, api):
        """Official first catalog match; an unmatched candidate is omitted upstream."""
        return self.catalog([api])["api_list"]

    def catalog(self, api_list):
        """Reuse official matching/order and raw category paths, without extra coercion."""
        return self._fetch(self, {"api_list": api_list})

    def openai_function(self, api_json, standard_tool_name):
        """Pinned official conversion, including alias collisions and schema quirks.

        The caller supplies the official whitelist's standard_tool_name. No
        alias deduplication, Finish/RESPOND mapping or live dispatch is chosen here.
        """
        return self._convert(self, api_json, standard_tool_name)

    def lookup(self, *, category, tool_name, api_name, tool_input):
        """Official virtual-server parse/cache branch, with cache misses blocked."""
        for value in (category, tool_name, api_name):
            _text(value)
        if not isinstance(tool_input, (str, dict)):
            raise TypeError("tool_input must be string or object")
        source = self.source / "server/main.py"
        tree = ast.parse(source.read_text())
        (function,) = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_virtual_response"
        ]
        function.decorator_list = []

        def blocked_post(*args, **kwargs):
            raise CacheMiss((category, tool_name, api_name, tool_input))

        def exists(path):
            return path in self.zip.NameToInfo or any(
                name.startswith(path + "/") for name in self.zip.NameToInfo
            )

        namespace = {
            "Request": object,
            "Info": object,
            "json": json,
            "prepare_tool_name_and_url": self.prepare,
            "CACHE_FOLDER": "tool_response_cache",
            "CONFIG": {"toolbench_url": "cpu-cache-miss"},
            "os": SimpleNamespace(
                path=SimpleNamespace(exists=exists, join=os.path.join),
                environ={},
                # CPU cache lookup never creates the official cache directory.
                mkdir=lambda path: None,
            ),
            "open": lambda path, mode="r": io.BytesIO(self.zip.read(path)),
            "requests": SimpleNamespace(post=blocked_post),
            "write_log": lambda **event: self.cache_events.append(
                {
                    "type": event["type"],
                    "request": deepcopy(vars(event["request"])),
                    "response": deepcopy(event["response"]),
                }
            ),
        }
        exec(  # noqa: S102 -- whole official function; no server initialization/network
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        return namespace["get_virtual_response"](
            None,
            SimpleNamespace(
                category=category,
                tool_name=tool_name,
                api_name=api_name,
                tool_input=tool_input,
                toolbench_key="cpu",
            ),
        )


def run_episode(config, *args, **kwargs):
    if config["protocol"]["mode"] == "cpu_only":
        raise ValueError(
            "StableToolBench CPU preview cannot execute an episode"
        )
    from .stabletoolbench_runtime import run_episode as execute

    return execute(config, *args, **kwargs)


def score_saved(*args, **kwargs):
    from .stabletoolbench_runtime import score_saved as score

    return score(*args, **kwargs)
