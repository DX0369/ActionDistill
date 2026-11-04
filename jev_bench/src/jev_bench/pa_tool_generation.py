"""Audited, bounded PA naming slots; no retries or silent resampling."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from .client import QwenClient
from .cost import completion_cost
from .io import append_json, read_jsonl
from .pa_tool import legal_name
from .pa_tool_prepare import (
    PROTOCOL,
    NameGenerationError,
    check_name_failure,
    generated_name,
    identity,
)


def slots(path):
    result = {}
    for event in read_jsonl(path):
        key = (event["component"], event["slot"])
        if event["phase"] in {"started", "reused"}:
            if key in result:
                raise ValueError("PA-Tool generation slot was sampled twice")
            if event["phase"] == "reused":
                if (
                    event["error"] is not None
                    and event["error"]["type"] != "NameGenerationError"
                ):
                    raise ValueError("PA-Tool cannot reuse a failed request")
                if not isinstance(event["source"], dict) or not event["source"]:
                    raise ValueError("PA-Tool reused request requires its source")
        elif (
            event["phase"] != "finished"
            or key not in result
            or result[key]["phase"] != "started"
        ):
            raise ValueError("PA-Tool generation ledger has an invalid transition")
        result[key] = event
    return result


def generation_cost(path, *, origin=None):
    """Separate inherited cost from new calls, including uncertain started slots."""
    if origin not in {None, "new", "reused"}:
        raise ValueError("PA-Tool cost origin must be new, reused, or None")
    records = [
        event["provider"]
        for event in slots(path).values()
        if origin is None or (event["phase"] == "reused") == (origin == "reused")
    ]
    return completion_cost(SimpleNamespace(records=records))


class CandidateGenerator:
    def __init__(
        self,
        settings,
        path,
        *,
        concurrency=1,
        client_factory=QwenClient,
        reuse=None,
        additional_settings=(),
    ):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("PA-Tool concurrency must be a positive integer")
        self.path = Path(path)
        self.settings = settings
        self.saved = slots(path)
        self._check_saved()
        self.request_settings = (settings, *additional_settings)
        self.concurrency = concurrency
        self.client_factory = client_factory
        self.reuse = reuse

    def _check_saved(self):
        for event in self.saved.values():
            error = event["error"] if event["phase"] in {"finished", "reused"} else None
            if event["phase"] not in {"finished", "reused"} or (
                error is not None and error["type"] != "NameGenerationError"
            ):
                raise ValueError(
                    "PA-Tool has a failed or uncertain request; stop for diagnosis, do not resample"
                )
            if error is not None:
                check_name_failure(event, self.settings["model"])
            elif not legal_name(event["name"]):
                raise ValueError("PA-Tool successful saved slot requires a legal name")

    def close(self):
        # Each request owns and closes its client before the component returns.
        pass

    def __call__(self, key, text):
        self._check_saved()
        names = [None] * (PROTOCOL["candidates"] + 1)
        pending = []
        for slot in range(PROTOCOL["candidates"] + 1):
            seed = int(identity([PROTOCOL["seed"], key, slot])[:15], 16)
            saved = self.saved.get((key, slot))
            if saved is not None:
                if saved["text"] != text or saved["seed"] != seed:
                    raise ValueError("PA-Tool resumed naming request differs")
                names[slot] = saved["name"]
                continue
            pending.append(
                {
                    "component": key,
                    "slot": slot,
                    "seed": seed,
                    "text": text,
                    "phase": "started",
                    "name": None,
                    "error": None,
                    "provider": {"usage": None, "latency_seconds": None},
                }
            )
        if pending and self.reuse is not None:
            reused = self.reuse(key, text)
            for event in pending[:]:
                slot = event["slot"]
                if slot not in reused:
                    continue
                source = reused[slot]
                if (
                    source["phase"] not in {"finished", "reused"}
                    or (
                        source["error"] is not None
                        and source["error"]["type"] != "NameGenerationError"
                    )
                    or any(
                        source[field] != event[field]
                        for field in ("component", "slot", "seed", "text")
                    )
                ):
                    raise ValueError("PA-Tool reused naming request differs")
                if not isinstance(source["source"], dict) or not source["source"]:
                    raise ValueError("PA-Tool reused request requires its source")
                inherited = {**source, "phase": "reused", "source": source["source"]}
                append_json(self.path, inherited)
                self.saved[key, slot] = inherited
                names[slot] = inherited["name"]
                pending.remove(event)
        if not pending:
            return names[:-1], names[-1]

        stopped = Event()
        running = set()
        failure = None

        def finish(future):
            nonlocal failure
            event, error = future.result()
            append_json(self.path, event)
            self.saved[key, event["slot"]] = event
            names[event["slot"]] = event["name"]
            if (
                error is not None
                and not isinstance(error, NameGenerationError)
                and failure is None
            ):
                failure = error

        with ThreadPoolExecutor(
            max_workers=min(self.concurrency, len(pending))
        ) as pool:
            try:
                while pending or running:
                    while (
                        pending
                        and len(running) < self.concurrency
                        and not stopped.is_set()
                    ):
                        event = pending.pop(0)
                        settings = self.request_settings[
                            event["slot"] % len(self.request_settings)
                        ]
                        event["provider_base_url"] = settings["base_url"]
                        append_json(self.path, event)
                        self.saved[key, event["slot"]] = event
                        running.add(pool.submit(self._request, event, stopped))
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        running.remove(future)
                        finish(future)
                    if stopped.is_set():
                        break
            except BaseException as exc:  # noqa: BLE001 - re-raised after draining.
                stopped.set()
                if failure is None:
                    failure = exc
            finally:
                # A failed or interrupted component still accounts for every
                # submitted request. No cancellation, retry, or replacement slot.
                for future in running:
                    try:
                        finish(future)
                    except BaseException as exc:  # noqa: BLE001 - drain other futures.
                        if failure is None:
                            failure = exc
        if failure is not None:
            raise failure
        return names[:-1], names[-1]

    def _request(self, started, stopped):
        event = dict(started)
        provider = None
        client = None
        failure = None

        def record(kind, value):
            nonlocal provider
            provider = value

        try:
            settings = self.request_settings[event["slot"] % len(self.request_settings)]
            client = self.client_factory(settings, record)
            result = client.complete(
                messages=[{"role": "user", "content": event["text"]}],
                tools=None,
                tool_choice=None,
                max_tokens=PROTOCOL["max_tokens"],
                temperature=PROTOCOL["reference_temperature"]
                if event["slot"] == PROTOCOL["candidates"]
                else PROTOCOL["temperature"],
                top_p=PROTOCOL["top_p"],
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": PROTOCOL["enable_thinking"]
                    },
                    "top_k": PROTOCOL["top_k"],
                    "min_p": PROTOCOL["min_p"],
                    "seed": event["seed"],
                },
            )
            event["name"] = generated_name(
                result.finish_reason, result.message.content, result.message.tool_calls
            )
        except BaseException as exc:  # noqa: BLE001 - caller saves then re-raises.
            if not isinstance(exc, NameGenerationError):
                stopped.set()
            failure = exc
        finally:
            if client is not None:
                try:
                    client.close()
                except BaseException as exc:  # noqa: BLE001 - retain request failure.
                    stopped.set()
                    if failure is None or isinstance(failure, NameGenerationError):
                        failure = exc
            if failure is not None:
                event["error"] = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
                if isinstance(failure, NameGenerationError):
                    event["error"]["reason"] = failure.reason
            event["phase"] = "finished"
            if provider is not None:
                event["provider"] = provider
            if isinstance(failure, NameGenerationError):
                try:
                    check_name_failure(event, settings["model"])
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    stopped.set()
                    failure = exc
                    event["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return event, failure
