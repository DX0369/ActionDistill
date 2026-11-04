"""Explicit, read-only import of matching PA naming requests into a new run."""

import json
from copy import deepcopy
from pathlib import Path

from .pa_tool_generation import slots
from .pa_tool_prepare import (
    PROTOCOL,
    NameGenerationError,
    generated_name,
    identity,
    validate_catalog,
)


class ReuseSource:
    def __init__(self, directory, catalog):
        self.directory = Path(directory).resolve(strict=True)
        previous = json.loads((self.directory / "catalog.json").read_text())
        state = json.loads((self.directory / "metadata/run_status.json").read_text())
        validate_catalog(previous)
        validate_catalog(catalog)
        if identity(state["protocol"]) != identity(PROTOCOL):
            raise ValueError("PA-Tool reuse source generation protocol differs")
        if state["status"] not in ("completed", "failed", "interrupted"):
            raise ValueError("PA-Tool reuse requires a stopped source run")
        self.source_status = state["status"]
        for field in ("version", "benchmark", "control_names"):
            if identity(previous[field]) != identity(catalog[field]):
                raise ValueError(f"PA-Tool reuse catalog {field} differs")
        # Different domain scopes/bindings are allowed: the requested component
        # identity already includes its exact function and occurrence, and for
        # parameters also the final tool alias and description source.
        expected = dict(previous["executor"])
        expected["base_url"] = catalog["executor"]["base_url"]
        if identity(expected) != identity(catalog["executor"]):
            raise ValueError("PA-Tool reuse executor settings differ")
        self.settings = catalog["executor"]
        service = {
            "id": self.settings["model"],
            "root": str(Path(self.settings["model_path"]).resolve()),
            "max_model_len": self.settings["max_model_len"],
        }
        if identity(state["service"]) != identity(service):
            raise ValueError("PA-Tool reuse source model identity differs")
        ledger = self.directory / "diagnostics/generation.jsonl"
        if not ledger.is_file():
            raise ValueError("PA-Tool reuse source generation ledger is missing")
        self.saved = slots(ledger)

    def __call__(self, key, text):
        selected = {}
        for (component, slot), saved in self.saved.items():
            if component != key:
                continue
            if type(slot) is not int or not 0 <= slot <= PROTOCOL["candidates"]:
                raise ValueError("PA-Tool reuse source slot is invalid")
            if saved["phase"] not in ("finished", "reused") or (
                saved["error"] is not None
                and saved["error"]["type"] != "NameGenerationError"
            ):
                raise ValueError(
                    "PA-Tool reuse has a failed or uncertain request; do not resample"
                )
            seed = int(identity([PROTOCOL["seed"], key, slot])[:15], 16)
            if saved["text"] != text or saved["seed"] != seed:
                raise ValueError("PA-Tool reuse prompt or seed differs")
            request = {
                "model": self.settings["model"],
                "messages": [{"role": "user", "content": text}],
                "temperature": PROTOCOL["reference_temperature"]
                if slot == PROTOCOL["candidates"]
                else PROTOCOL["temperature"],
                "max_tokens": PROTOCOL["max_tokens"],
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": PROTOCOL["enable_thinking"]
                    },
                    "top_k": PROTOCOL["top_k"],
                    "min_p": PROTOCOL["min_p"],
                    "seed": seed,
                },
                "top_p": PROTOCOL["top_p"],
            }
            provider = saved["provider"]
            if provider["error"] is not None or identity(
                provider["request"]
            ) != identity(request):
                raise ValueError("PA-Tool reuse actual request differs or failed")
            response = provider["response"]
            if (
                response["model"] != self.settings["model"]
                or len(response["choices"]) != 1
            ):
                raise ValueError("PA-Tool reuse response model or choices differ")
            choice = response["choices"][0]
            message = choice["message"]
            if message["role"] != "assistant":
                raise ValueError("PA-Tool reuse response role differs")
            try:
                name = generated_name(
                    choice["finish_reason"], message["content"], message["tool_calls"]
                )
            except NameGenerationError as exc:
                expected_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "reason": exc.reason,
                }
                if saved["name"] is not None or saved["error"] != expected_error:
                    raise ValueError(
                        "PA-Tool reuse naming failure differs from response"
                    ) from exc
            else:
                if saved["name"] != name or saved["error"] is not None:
                    raise ValueError(
                        "PA-Tool reuse response is not a successful naming result"
                    )
            usage = provider["usage"]
            response_usage = response["usage"]
            if usage is None:
                if response_usage is not None:
                    raise ValueError("PA-Tool reuse usage differs from response")
            elif any(
                type(usage[field]) is not int
                or usage[field] < 0
                or response_usage[field] != usage[field]
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise ValueError("PA-Tool reuse usage differs from response")
            latency = provider["latency_seconds"]
            if latency is not None and (
                type(latency) not in (int, float) or latency < 0
            ):
                raise ValueError("PA-Tool reuse latency is invalid")
            source = {
                "directory": str(self.directory),
                "component": component,
                "slot": slot,
            }
            if saved["phase"] == "reused":
                origin = saved["source"]
                if (
                    not isinstance(origin["directory"], str)
                    or origin["component"] != component
                    or origin["slot"] != slot
                ):
                    raise ValueError("PA-Tool reused source origin differs")
                source["origin"] = deepcopy(origin)
            selected[slot] = {**deepcopy(saved), "source": source}
        return selected
