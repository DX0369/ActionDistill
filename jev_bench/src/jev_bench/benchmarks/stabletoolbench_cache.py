"""official StableToolBench resolver with a shared writable overlay.

The published ZIP is read-only.
"""

import ast
import fcntl
import io
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import requests

from ..advice import redact_secret
from ..client import CompletionClient
from ..io import write_json


class ToolbenchTransportError(RuntimeError):
    """A transport failure is not an official simulator-fallback trigger."""


def require_key(name):
    value = os.environ[name]
    if not value.strip():
        raise ValueError(f"API key environment variable is empty: {name}")
    if any(not 33 <= ord(character) <= 126 for character in value):
        raise ValueError(f"API key contains invalid HTTP credential characters: {name}")
    return value


def _response(value):
    if not isinstance(value, dict) or not isinstance(value["error"], str):
        raise TypeError("tool response requires a string error and a response field")
    value["response"]
    return value


class SharedCache:
    def __init__(
        self,
        archive,
        directory,
        settings,
        sink,
        *,
        client_factory=CompletionClient,
        post=None,
    ):
        self.archive = archive
        self.directory = Path(directory)
        self.settings = settings
        self.sink = sink
        self.key = require_key(settings["toolbench_key_env"])
        simulator_settings = settings["simulator"]
        self.simulator_key = require_key(simulator_settings["api_key_env"])
        self.simulator = client_factory(
            simulator_settings,
            self._safe_sink,
            api_key=self.simulator_key,
            participant="tool_simulator",
        )
        # Proxy changes stay local to this session. No process/global env mutation.
        self.session = None
        if post is None:
            self.session = requests.Session()
            self.session.trust_env = False
            self.post = self.session.post
        else:
            self.post = post
        self.events = []
        self._merged = {}
        self._published = {}
        self._overlay = {}
        self._cache_member = None
        self._overlay_file = None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.directory / "cache.lock"
        self._load_resolver()

    def _safe(self, value):
        value = redact_secret(value, self.key)
        return redact_secret(value, self.simulator_key)

    def _safe_sink(self, kind, event):
        cleaned = self._safe(event)
        event.clear()
        event.update(cleaned)
        self.sink(kind, event)

    def close(self):
        self.simulator.close()
        if self.session is not None:
            self.session.close()

    def _load_resolver(self):
        source = self.archive.source / "server/main.py"
        tree = ast.parse(source.read_text())
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "get_virtual_response",
                "check_result",
                "is_valid_json",
                "fake_response_function_chat",
            }:
                node.decorator_list = []
                selected.append(node)
        simulator_settings = self.settings["simulator"]
        namespace = {
            "Request": object,
            "Info": object,
            "json": json,
            "time": time,
            "print": lambda *args, **kwargs: None,
            "standardize": self.archive.names.standardize,
            "prepare_tool_name_and_url": self.archive.prepare,
            "CACHE_FOLDER": "tool_response_cache",
            "CONFIG": {
                "toolbench_url": self.settings["toolbench_url"],
                "tools_folder": "tools",
                "is_save": True,
                "model": simulator_settings["model"],
                "temperature": simulator_settings["temperature"],
            },
            "os": SimpleNamespace(
                path=SimpleNamespace(exists=self._exists, join=os.path.join),
                mkdir=lambda path: None,
                environ={},
            ),
            "open": self._open,
            "requests": SimpleNamespace(post=self._post),
            "write_log": self._log,
            "save_cache": self._save,
            # The real credential belongs only to CompletionClient, not this
            # upstream namespace, its Info object, or its request artifacts.
            "OPENAI_API_KEY": "environment",
            "OPENAI_API_BASE": "environment",
            "OpenAI": lambda **kwargs: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=self._simulate))
            ),
        }
        exec(  # noqa: S102 -- fixed official functions with explicit I/O bindings
            compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self._resolve = namespace["get_virtual_response"]

    def _exists(self, path):
        if path == self._cache_member or path == "tool_response_cache":
            return True
        if self._cache_member is not None and self._cache_member.startswith(path + "/"):
            return True
        return path in self.archive.zip.NameToInfo or any(
            name.startswith(path + "/") for name in self.archive.zip.NameToInfo
        )

    def _open(self, path, mode="r"):
        if mode != "r":
            raise ValueError("official resolver may only read through this binding")
        if path == self._cache_member:
            return io.StringIO(json.dumps(self._merged, ensure_ascii=False))
        return io.BytesIO(self.archive.zip.read(path))

    def _post(self, url, *, headers, data):
        payload = json.loads(data)
        payload["toolbench_key"] = self.key
        headers = {**headers, "toolbench_key": self.key}
        started = time.perf_counter()
        event = {"source": "real_api", "status_code": None, "error_type": None}
        try:
            response = self.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.settings["timeout_seconds"],
            )
            event["status_code"] = response.status_code
            # Preserve upstream: non-200 goes to simulation; a broken 200 body
            # is a failure, not another reason to fabricate a tool observation.
            body = self._safe(response.json()) if response.status_code == 200 else None
            return SimpleNamespace(status_code=response.status_code, json=lambda: body)
        except requests.RequestException as exc:
            event["error_type"] = type(exc).__name__
            raise ToolbenchTransportError(
                f"ToolBench request failed: {type(exc).__name__}"
            ) from None
        finally:
            event["latency_seconds"] = time.perf_counter() - started
            self._safe_sink("tool_backend_requests", event)

    def _simulate(self, **request):
        settings = self.settings["simulator"]
        request.update(
            model=settings["model"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            extra_body={"thinking": {"type": settings["thinking"]}},
        )
        result = self.simulator.request(request)
        if result.finish_reason != "stop" or not isinstance(
            result.message.content, str
        ):
            raise ValueError("tool simulator requires a complete text response")
        content = self._safe(result.message.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _log(self, *, request, response, type):
        event = {
            "type": type,
            "category": request.category,
            "tool_name": request.tool_name,
            "api_name": request.api_name,
            "tool_input": deepcopy(request.tool_input),
        }
        self.events.append(self._safe(event))
        self._safe_sink("tool_backend", event)

    def _save(self, cache, tool_input, result, category, tool, api):
        value = json.loads(result) if isinstance(result, str) else result
        value = _response(self._safe(value))
        key = str(tool_input)
        if key in self._published or key in self._overlay:
            raise ValueError("resolver attempted to overwrite an existing cache key")
        self._overlay[key] = value
        write_json(self._overlay_file, self._overlay)
        self._merged[key] = value

    def __call__(self, payload):
        info = SimpleNamespace(
            category=payload["category"],
            tool_name=payload["tool_name"],
            api_name=payload["api_name"],
            tool_input=payload["tool_input"],
            strip=payload["strip"],
            toolbench_key="environment",
        )
        tool, category, api, _ = self.archive.prepare(info)
        relative = Path(category) / tool / (api + ".json")
        self._cache_member = "tool_response_cache/" + relative.as_posix()
        self._overlay_file = self.directory / relative
        if not self._overlay_file.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("normalized cache path escapes its overlay")
        # Stable lock inode: write_json atomically replaces a separate data file.
        # This serializes first resolution across executor worker processes.
        with self._lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._published = (
                    json.loads(self.archive.zip.read(self._cache_member))
                    if self._cache_member in self.archive.zip.NameToInfo
                    else {}
                )
                self._overlay = (
                    json.loads(self._overlay_file.read_text())
                    if self._overlay_file.exists()
                    else {}
                )
                if not isinstance(self._published, dict) or not isinstance(
                    self._overlay, dict
                ):
                    raise TypeError("API caches must be JSON objects")
                if self._published.keys() & self._overlay.keys():
                    raise ValueError("overlay shadows a published cache key")
                # Python insertion order is part of the official few-shot prompt.
                self._merged = {**self._published, **self._overlay}
                return _response(self._resolve(None, info))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
