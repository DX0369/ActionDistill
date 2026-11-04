"""One-step advice; texts and gate decisions match the historical runs."""

import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from openai import APIConnectionError, APIStatusError

from .client import AssistantChoiceError, CompletionClient, ContextLimitError
from .config import GATES, PROJECT_ROOT, fields, integer, number

DATASET_INVENTORY_SRC = str(PROJECT_ROOT.parent / "dataset_tool_inventory/src")

NEUTRAL_MESSAGE = (
    "Continue with the user's request and decide the next action yourself."
)
PREFER_TOOL = (
    "Recommended next action: call the tool `{name}` first. Construct its arguments yourself "
    "from the available information. You may depart from this recommendation if it conflicts "
    "with the current request or cannot be executed with valid arguments."
)
PREFER_RESPOND = (
    "Recommended next action: reply to the user without calling a tool. You may depart from "
    "this recommendation if the current request still requires a tool call."
)
RANKED_HINT = (
    'Likely next actions, ranked by estimated probability:\n{items}\n'
    'These are alternatives for your next action, not a sequence to execute. '
    'Choose the one that best fits the current request and construct any arguments yourself. '
    'You may take a different action if none of them fits.'
)
OPTION_ORDER = "request tool order, then RESPOND; no shuffle"
POSITIONAL_TOOL_NAMES_MESSAGE = (
    "Some tools share original names. For this request, each tool has a unique "
    "candidate_XXXX name in the tools list. If you call a tool, use that exact "
    "candidate name in tool_calls[].function.name. Original tool names in "
    "descriptions and earlier messages are historical labels, not valid names "
    "for your next tool call. Choose the candidate using its description and "
    "parameters. If no tool is needed, reply normally without a tool call."
)


class AdvisorFormatError(ValueError):
    """A received advisor response violates its action-selection contract."""


@contextmanager
def response_format():
    """Action parsing only; exclude inputs, identity, metadata and transport."""
    try:
        yield
    except (KeyError, TypeError, ValueError) as exc:
        raise AdvisorFormatError(str(exc)) from exc


def candidate_ids(names):
    """Preserve existing unique-name requests; duplicates get positional IDs."""
    if "RESPOND" in names:
        raise ValueError("RESPOND is reserved for the reply option")
    if len(names) == len(set(names)):
        return names
    return [f"candidate_{index:04d}" for index in range(len(names))]


def aggregate_candidates(names, values):
    grouped = {}
    for name, value in zip(names, values, strict=True):
        grouped.setdefault(name, []).append(value)
    return [
        {"name": name, "probability": math.fsum(items)}
        for name, items in grouped.items()
    ]


class AdvisorServiceError(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record

    def __reduce__(self):
        # Spawn workers must restore both arguments without breaking their pool.
        return type(self), (str(self), self.record)


class HTTPResponseError(ValueError):
    def __init__(self, status_code, body):
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class ProviderHTTPError(HTTPResponseError):
    """A remote provider error cannot trigger the local-renderer fallback."""


def api_key(settings):
    value = os.environ.get(settings["api_key_env"])
    if value is None or not value.strip():
        raise ValueError("advisor API key environment variable is missing or empty")
    if not re.fullmatch(r"[!-~]+", value):
        raise ValueError("advisor API key contains invalid HTTP credential characters")
    return value


def redact_secret(value, secret):
    """Never persist an exact credential echoed by a provider or transport error."""
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, dict):
        return {
            redact_secret(k, secret): redact_secret(v, secret) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secret(v, secret) for v in value]
    return value


def render_overlength(exc):
    if not isinstance(exc, HTTPResponseError) or isinstance(exc, ProviderHTTPError):
        return False
    detail = exc.body.get("error") if isinstance(exc.body, dict) else None
    return (
        exc.status_code == 422
        and isinstance(detail, dict)
        and detail.get("type") == "ValueError"
        and detail.get("message") == "over_length_after_drop"
    )


def json_request(url, timeout, payload=None, *, headers=None):
    body = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    )
    request = Request(
        url, data=body, headers={"Content-Type": "application/json", **(headers or {})}
    )
    local = urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")
    opener = build_opener(ProxyHandler({})) if local else build_opener()
    authorization = (headers or {}).get("Authorization", "")
    secret = authorization[7:] if authorization.startswith("Bearer ") else None
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                error = AdvisorFormatError("advisor returned a non-JSON response")
                text = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else raw
                )
                error.raw_response = redact_secret(text, secret) if secret else text
                raise error from exc
            return redact_secret(result, secret) if secret else result
    except HTTPError as exc:
        with exc:
            detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail)
        except ValueError:
            pass  # Preserve a non-JSON error body verbatim for diagnosis.
        if secret:
            detail = redact_secret(detail, secret)
        error = HTTPResponseError if local else ProviderHTTPError
        raise error(exc.code, detail) from exc


class JevClient:
    def __init__(self, settings):
        self.settings = settings
        self.last_response = None
        self.last_usage = None
        self.last_metadata = None
        self.records = []  # Only remote clients persist separate request records.
        self.model = str(Path(settings["model"]).resolve(strict=True))
        self.temperature_file = str(
            Path(settings["temperature_file"]).resolve(strict=True)
        )
        self.temperature = json.loads(Path(self.temperature_file).read_text())[
            "temperature"
        ]
        number(self.temperature)
        if self.temperature == 0:
            raise ValueError("Jev temperature must be positive")

    def identity(self, response):
        if (
            response["model"] != self.model
            or response["temperature_file"] != self.temperature_file
            or type(response["temperature"]) not in (int, float)
            or response["temperature"] != self.temperature
        ):
            raise ValueError(
                "Jev model/temperature identity differs from the configuration"
            )
        if (
            type(response["prompt_budget"]) is not int
            or response["prompt_budget"] != 32704
        ):
            raise ValueError("unexpected Jev prompt budget")
        if response["option_order"] != OPTION_ORDER:
            raise ValueError("unexpected Jev option ordering")
        if response["duplicate_candidates"] != "positional":
            raise ValueError("Jev service must support positional duplicate candidates")

    def health(self):
        response = json_request(
            self.settings["base_url"] + "/health", self.settings["timeout_seconds"]
        )
        if response["status"] != "ready":
            raise ValueError("Jev is not ready")
        self.identity(response)
        return response

    def close(self):
        pass  # urllib responses are closed per request.

    def render(self, messages, tools):
        response = json_request(
            self.settings["base_url"] + "/render",
            self.settings["timeout_seconds"],
            {
                "messages": messages,
                "tools": [tool["function"] for tool in tools],
                "respond": True,
            },
        )
        self.identity(response)
        self.render_metadata(response)
        return response

    def predict(self, messages, tools):
        self.last_response = None
        self.last_usage = None
        self.last_metadata = None
        definitions = [
            {
                key: tool["function"][key]
                for key in ("name", "description", "parameters")
            }
            for tool in tools
        ]
        names = [tool["name"] for tool in definitions] + ["RESPOND"]
        if "RESPOND" in names[:-1]:
            raise ValueError("RESPOND is reserved and cannot be an input tool")
        try:
            response = json_request(
                self.settings["base_url"] + "/advise",
                self.settings["timeout_seconds"],
                {"messages": messages, "tools": definitions, "respond": True},
            )
        except AdvisorFormatError as exc:
            self.last_response = exc.raw_response
            raise
        self.last_response = response
        self.identity(response)
        self.render_metadata(response)
        number(response["latency_seconds"])
        # Metadata failures are fatal; valid costs survive malformed actions.
        self.last_metadata = {
            key: response[key]
            for key in (
                "n_prompt_tokens",
                "truncated",
                "messages_dropped",
                "latency_seconds",
            )
        }
        self.last_usage = {
            "prompt_tokens": response["n_prompt_tokens"],
            "completion_tokens": 0,
            "total_tokens": response["n_prompt_tokens"],
        }
        with response_format():
            if [x["name"] for x in response["probabilities"]] != names:
                raise ValueError(
                    "Jev probabilities must match the exact candidate order; RESPOND is reserved"
                )
            values = [x["probability"] for x in response["probabilities"]]
            for value in values:
                number(value, maximum=1)
            if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
                raise ValueError("Jev probabilities do not sum to one")
            if (
                response["top1"]
                != names[max(range(len(names)), key=values.__getitem__)]
            ):
                raise ValueError("Jev top1 must match the stable argmax")
        # Sum positional probabilities, not logits. First occurrence fixes ties.
        grouped = {}
        for name, probability in zip(names, values, strict=True):
            grouped.setdefault(name, []).append(probability)
        probabilities = [
            {"name": name, "probability": math.fsum(items)}
            for name, items in grouped.items()
        ]
        top1 = max(probabilities, key=lambda item: item["probability"])["name"]
        return {
            **response,
            "raw_probabilities": response["probabilities"],
            "raw_top1": response["top1"],
            "probabilities": probabilities,
            "top1": top1,
            "aggregation": "sum_by_tool_name",
            "remote": None,
        }

    @staticmethod
    def render_metadata(response):
        integer(response["n_prompt_tokens"], 1)
        if (
            response["n_prompt_tokens"] > 32704
            or type(response["truncated"]) is not bool
        ):
            raise ValueError("invalid Jev token count or truncation flag")
        integer(response["messages_dropped"])
        fields(
            response["truncation_counts"],
            ("images_replaced", "skipped_framework_greeting"),
            ("tool_outputs_truncated", "tool_call_arguments_truncated"),
        )
        for value in response["truncation_counts"].values():
            integer(value)
        counts = response["truncation_counts"]
        # The service serializes these two Counter keys sparsely: absence means zero.
        truncated = bool(
            response["messages_dropped"]
            or counts["images_replaced"]
            or counts.get("tool_outputs_truncated", 0)
            or counts.get("tool_call_arguments_truncated", 0)
        )
        if response["truncated"] != truncated:
            raise ValueError("truncated contradicts the canonical truncation counts")


def remote_record(kind):
    """One logical advice request, including render-only failures before an API call."""
    return {
        "advisor_type": kind,
        "provider_called": False,
        "request": None,
        "response": None,
        "usage": None,
        "local_prompt_tokens": None,
        "remote_input_tokens": None,
        "remote_output_tokens": None,
        "reasoning_tokens": None,
        "render_latency_seconds": None,
        "model_latency_seconds": None,
        "truncated": None,
        "messages_dropped": None,
        "latency_seconds": None,
        "error": None,
        "fallback": None,
    }


def typesafe_usage(response):
    usage = response["usage"]
    integer(usage["input_tokens"])
    integer(usage["output_tokens"])
    return {
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
    }


class RenderClient(JevClient):
    """Explicit tokenizer-only renderer; never substitutes for a Jev predictor."""
    def __init__(self, settings):
        self.settings = settings
        self.last_response = self.last_usage = self.last_metadata = None
        self.records = []
        source = str(PROJECT_ROOT.parent/'jev_train/src')
        if source not in sys.path:
            sys.path.insert(0, source)
        from jev_train.tokenizer_identity import tokenizer_identity
        self.tokenizer_identity = tokenizer_identity(settings['tokenizer'])

    def identity(self, response):
        if response['service_mode'] != 'render_only' or any(
                response[k] != v for k, v in self.tokenizer_identity.items()):
            raise ValueError('renderer tokenizer identity differs from the configuration')
        if (type(response['prompt_budget']) is not int or response['prompt_budget'] != 32704
                or response['option_order'] != OPTION_ORDER or response['duplicate_candidates'] != 'positional'):
            raise ValueError('renderer budget/order/candidate capability differs')

    def predict(self, messages, tools):
        raise ValueError('a tokenizer renderer cannot predict actions')

    def render(self, messages, tools):
        self.last_response = None
        try:
            response = json_request(self.settings['base_url']+'/render', self.settings['timeout_seconds'],
                                    {'messages': messages, 'tools': [t['function'] for t in tools], 'respond': True})
        except AdvisorFormatError as exc:
            self.last_response = exc.raw_response
            raise ValueError('renderer returned invalid response metadata') from exc
        self.last_response = response
        self.identity(response)
        self.render_metadata(response)
        number(response['latency_seconds'])
        return response


class TypeSafeClient:
    def __init__(self, settings, sink, render_settings):
        self.settings, self.sink = settings, sink
        self.key = api_key(settings)
        self.renderer = (RenderClient(render_settings) if render_settings['type'] == 'jev_render'
                         else JevClient(render_settings))
        self.last_response = self.last_usage = None
        self.records = []

    def health(self):
        return {
            "model": self.settings["model"],
            "remote_checked": False,
            "renderer": self.renderer.health(),
        }

    def close(self):
        self.renderer.close()

    def predict(self, messages, tools):
        self.last_response = self.last_usage = None
        record = remote_record("typesafe")
        started = time.perf_counter()
        try:
            original_names = [tool["function"]["name"] for tool in tools]
            names = candidate_ids(original_names) + ["RESPOND"]
            original_names += ["RESPOND"]
            record["candidate_mapping"] = dict(zip(names, original_names, strict=True))
            rendered = self.renderer.render(messages, tools)
            self.last_response = rendered
            record.update(
                local_prompt_tokens=rendered["n_prompt_tokens"],
                truncated=rendered["truncated"],
                messages_dropped=rendered["messages_dropped"],
                render_latency_seconds=time.perf_counter() - started,
            )
            row = rendered["rendered"]
            if (
                not isinstance(row["options"], list)
                or len(row["options"]) != len(names)
                or any(not isinstance(x, str) or not x for x in row["options"])
                or not isinstance(row["context"], str)
                or not isinstance(row["question"], str)
                or not row["question"]
            ):
                raise ValueError("invalid rendered context/question/options")
            record["request"] = {
                "model": self.settings["model"],
                "state": row["context"],
                "questions": {
                    "next_action": {
                        "type": "choice",
                        "instructions": row["question"],
                        "criteria": dict(zip(names, row["options"], strict=True)),
                    }
                },
            }
            provider_started = time.perf_counter()
            try:
                record["provider_called"] = True
                response = json_request(
                    self.settings["base_url"] + "/systemone",
                    self.settings["timeout_seconds"],
                    record["request"],
                    headers={"Authorization": "Bearer " + self.key},
                )
            except HTTPResponseError as exc:
                exc.body = redact_secret(exc.body, self.key)
                exc.args = (f"HTTP {exc.status_code}: {exc.body}",)
                record["response"] = self.last_response = exc.body
                if isinstance(exc.body, dict) and "usage" in exc.body:
                    try:
                        self.last_usage = typesafe_usage(exc.body)
                    except (KeyError, ValueError, TypeError) as usage_error:
                        record["usage_error"] = redact_secret(
                            str(usage_error), self.key
                        )
                # Provider errors must never masquerade as the approved renderer fallback.
                raise ProviderHTTPError(exc.status_code, exc.body) from exc
            finally:
                record["model_latency_seconds"] = time.perf_counter() - provider_started
            record["response"] = self.last_response = response
            self.last_usage = typesafe_usage(response)
            if response["model"] != self.settings["model"]:
                raise ValueError("TypeSafe returned a different model version")
            with response_format():
                if set(response["answers"]) != {"next_action"}:
                    raise ValueError("TypeSafe answer IDs differ from the request")
                answer = response["answers"]["next_action"]
                if answer["type"] != "choice" or answer["choice"] not in names:
                    raise ValueError("invalid TypeSafe choice")
                number(answer["confidence"], maximum=1)
                probabilities = answer["probabilities"]
                if not isinstance(probabilities, dict) or set(probabilities) != set(
                    names
                ):
                    raise ValueError(
                        "TypeSafe probabilities differ from the candidates"
                    )
                for value in probabilities.values():
                    number(value, maximum=1)
                # Same local validation tolerance as Jev; never renormalize API output.
                if not math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-5):
                    raise ValueError("TypeSafe probabilities do not sum to one")
                if probabilities[answer["choice"]] != max(probabilities.values()):
                    raise ValueError(
                        "TypeSafe choice is not a maximum-probability candidate"
                    )
            raw_probabilities = [
                {"name": name, "probability": probabilities[identity]}
                for name, identity in zip(original_names, names, strict=True)
            ]
            grouped = aggregate_candidates(
                original_names, [probabilities[identity] for identity in names]
            )
            # Unique-name ties preserve the provider's selected maximum.
            top1 = (
                answer["choice"]
                if names == original_names
                else max(grouped, key=lambda item: item["probability"])["name"]
            )
            return {
                "top1": top1,
                "probabilities": grouped,
                "raw_probabilities": raw_probabilities,
                "raw_top1": record["candidate_mapping"][answer["choice"]],
                "candidate_mapping": record["candidate_mapping"],
                "aggregation": "sum_by_tool_name",
                "n_prompt_tokens": record["local_prompt_tokens"],
                "truncated": record["truncated"],
                "messages_dropped": record["messages_dropped"],
                "truncation_counts": rendered["truncation_counts"],
                "latency_seconds": record["model_latency_seconds"],
                "remote": response,
            }
        except (OSError, HTTPException) as exc:
            message = redact_secret(str(exc), self.key)
            record["error"] = {"type": type(exc).__name__, "message": message}
            # URLError/OSError format .reason/.strerror, not just .args.
            raise OSError(message) from None
        except Exception as exc:
            if not record['provider_called'] and isinstance(self.renderer, RenderClient):
                record['response'] = self.last_response = self.renderer.last_response
            if isinstance(exc, AdvisorFormatError) and hasattr(exc, "raw_response"):
                record["response"] = self.last_response = exc.raw_response
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            if isinstance(exc, HTTPResponseError):
                exc.body = redact_secret(exc.body, self.key)
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if render_overlength(exc):
                record["fallback"] = "over_length_after_drop_neutral"
            elif isinstance(exc, AdvisorFormatError):
                record["fallback"] = "advisor_format_error_neutral"
            raise
        finally:
            record["usage"] = self.last_usage
            if self.last_usage is not None:
                record["remote_input_tokens"] = self.last_usage["prompt_tokens"]
                record["remote_output_tokens"] = self.last_usage["completion_tokens"]
            record["latency_seconds"] = time.perf_counter() - started
            record = redact_secret(record, self.key)
            self.last_response = redact_secret(self.last_response, self.key)
            self.records.append(record)
            self.sink("advisor_requests", record)


class DeepSeekClient:
    def __init__(self, settings, sink):
        self.settings, self.sink = settings, sink
        self.key = api_key(settings)
        # Reuse the exact visible-history boundary used by the Jev renderer.
        if DATASET_INVENTORY_SRC not in sys.path:
            sys.path.insert(0, DATASET_INVENTORY_SRC)
        from dataset_tool_inventory.adapters.review_common import visible

        self.visible = visible
        self.completion = CompletionClient(
            settings, lambda *_: None, api_key=self.key, participant="advisor"
        )
        self.last_response = self.last_usage = None
        self.records = []

    def health(self):
        response = json_request(
            self.settings["base_url"] + "/models",
            self.settings["timeout_seconds"],
            headers={"Authorization": "Bearer " + api_key(self.settings)},
        )
        if self.settings["model"] not in [item["id"] for item in response["data"]]:
            raise ValueError("DeepSeek model is absent from /models")
        return {
            "model": self.settings["model"],
            "models_checked": True,
            "inference_checked": False,
        }

    def close(self):
        self.completion.close()

    def predict(self, messages, tools, *, output_protocol="tool_calls"):
        self.last_response = self.last_usage = None
        record = remote_record("deepseek")
        record["output_protocol"] = output_protocol
        started = time.perf_counter()
        before = len(self.completion.records)
        try:
            if output_protocol not in {"tool_calls", "stabletoolbench", "apibank_text"}:
                raise ValueError("unsupported DeepSeek output protocol")
            names = [tool["function"]["name"] for tool in tools]
            if "RESPOND" in names or any(
                not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", x)
                for x in names
            ):
                raise ValueError("invalid or duplicate DeepSeek tool name")
            identities = (
                names if output_protocol == "apibank_text" else candidate_ids(names)
            )
            # Copy occurrences independently even when the caller reused one dict.
            wire_tools = [deepcopy(tool) for tool in tools]
            record["candidate_mapping"] = dict(zip(identities, names, strict=True))
            if identities != names:
                for tool, identity, name in zip(
                    wire_tools, identities, names, strict=True
                ):
                    function = tool["function"]
                    function["name"] = identity
                    function["description"] = (
                        f"Original tool name: {name}.\n" + function["description"]
                    )
            visible_messages = [
                self.visible(message, strip_think=True) for message in messages
            ]
            assistant_count = 0
            for message in visible_messages:
                if message["role"] == "assistant":
                    # External executor history has no DeepSeek reasoning to
                    # return. The approved wire format marks it explicitly empty.
                    message["reasoning_content"] = ""
                    assistant_count += 1
            if identities != names:
                # History may contain ambiguous original names. Preserve it and
                # explain the request-local names instead of guessing a mapping.
                visible_messages.append(
                    {"role": "system", "content": POSITIONAL_TOOL_NAMES_MESSAGE}
                )
            record["history_compatibility"] = {
                "rule": "external_assistant_empty_reasoning",
                "assistant_messages": assistant_count,
            }
            request = {
                "model": self.settings["model"],
                "messages": visible_messages,
                "max_tokens": self.settings["max_tokens"],
                "stream": False,
                "extra_body": {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": self.settings["reasoning_effort"],
                },
            }
            if tools and output_protocol != "apibank_text":
                request.update(tools=wire_tools, tool_choice="auto")
            record["request"] = request
            record["provider_called"] = True
            result = self.completion.request(request)
            self.last_response = result.raw_response
            self.last_usage = result.usage
            model = result.raw_response["model"]
            if not isinstance(model, str) or not model:
                raise ValueError("DeepSeek response must identify its actual model")
            raw_usage = result.raw_response.get("usage")
            if raw_usage is not None:
                details = raw_usage.get("completion_tokens_details")
                if details is not None and not isinstance(details, dict):
                    raise ValueError(
                        "completion_tokens_details must be an object or null"
                    )
                if details is not None and details.get("reasoning_tokens") is not None:
                    integer(details["reasoning_tokens"])
                    record["reasoning_tokens"] = details["reasoning_tokens"]
            with response_format():
                calls = result.raw_response["choices"][0]["message"].get("tool_calls")
                if calls is not None and not isinstance(calls, list):
                    raise ValueError("DeepSeek tool_calls must be a list or null")
                if (
                    output_protocol != "apibank_text"
                    and result.finish_reason == "tool_calls"
                    and calls
                ):
                    if calls[0]["type"] != "function":
                        raise ValueError("DeepSeek advisor requires a function call")
                    top1 = calls[0]["function"]["name"]
                    if top1 not in identities:
                        raise ValueError(
                            "DeepSeek advised a tool outside the candidates"
                        )
                elif (
                    result.finish_reason == "stop"
                    and not calls
                    and isinstance(result.message.content, str)
                    and result.message.content.strip()
                ):
                    if output_protocol == "apibank_text":
                        # configure() pins this official module to API-Bank.
                        # Extract only the action name; generated arguments are
                        # never executed or repaired by the advisor.
                        from api_call_extraction import get_api_call, parse_api_call

                        call = get_api_call(result.message.content)
                        record["text_call_extracted"] = call is not None
                        if call is None:
                            top1 = "RESPOND"
                        else:
                            top1 = parse_api_call(call)[0]
                            if top1 not in identities:
                                raise ValueError(
                                    "DeepSeek advised a tool outside the candidates"
                                )
                    elif output_protocol == "tool_calls" and re.search(
                        r"\[\s*[A-Za-z_]\w*\s*\(.*?\)\s*\]",
                        result.message.content,
                        re.DOTALL,
                    ):
                        raise ValueError(
                            "textual API call must use structured tool_calls"
                        )
                    else:
                        top1 = "RESPOND"
                else:
                    raise ValueError(
                        f"invalid advisor completion: finish_reason={result.finish_reason!r}"
                    )
            raw_top1 = top1
            if top1 != "RESPOND":
                top1 = record["candidate_mapping"][top1]
            if output_protocol == "stabletoolbench" and top1 == "RESPOND":
                # Plain text is an official Thought, not a Finish action.
                # Record before the provider ledger is persisted for recovery.
                record["fallback"] = "stable_thought_neutral"
                return {"top1": None, "raw_action": "Thought"}
            return {
                "top1": top1,
                "raw_top1": raw_top1,
                "candidate_mapping": record["candidate_mapping"],
                "probabilities": None,
                "n_prompt_tokens": None,
                "truncated": None,
                "messages_dropped": None,
                "latency_seconds": result.latency_seconds,
                "remote": {
                    "model": model,
                    "usage": raw_usage,
                    "system_fingerprint": result.raw_response.get("system_fingerprint"),
                },
            }
        except APIStatusError as exc:
            # SDK .body may discard the envelope; retain the actual response body.
            try:
                self.last_response = exc.response.json()
            except ValueError:
                self.last_response = exc.response.text
            self.last_response = redact_secret(self.last_response, self.key)
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            exc.message = redact_secret(exc.message, self.key)
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise ProviderHTTPError(exc.status_code, self.last_response) from exc
        except APIConnectionError as exc:
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            exc.message = redact_secret(exc.message, self.key)
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise OSError(str(exc)) from exc
        except AssistantChoiceError as exc:
            message = redact_secret(str(exc), self.key)
            record["error"] = {"type": type(exc).__name__, "message": message}
            raw = exc.diagnostics["response"]
            if not isinstance(raw["model"], str) or not raw["model"]:
                raise ValueError(
                    "DeepSeek response must identify its actual model"
                ) from exc
            record["fallback"] = "advisor_format_error_neutral"
            raise AdvisorFormatError(message) from exc
        except ContextLimitError as exc:
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise ValueError(str(exc)) from exc
        except Exception as exc:
            exc.args = tuple(redact_secret(list(exc.args), self.key))
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, AdvisorFormatError):
                record["fallback"] = "advisor_format_error_neutral"
            raise
        finally:
            if len(self.completion.records) > before:
                raw = self.completion.records[-1]
                self.last_usage = raw["usage"]
                if raw["response"] is not None:
                    self.last_response = raw["response"]
                record["model_latency_seconds"] = raw["latency_seconds"]
            record["response"] = self.last_response
            record["usage"] = self.last_usage
            if self.last_usage is not None:
                record["remote_input_tokens"] = self.last_usage["prompt_tokens"]
                record["remote_output_tokens"] = self.last_usage["completion_tokens"]
            record["latency_seconds"] = time.perf_counter() - started
            record = redact_secret(record, self.key)
            self.last_response = redact_secret(self.last_response, self.key)
            self.records.append(record)
            self.sink("advisor_requests", record)


def make_advisor(settings, sink, *, advisors):
    if settings["type"] == "jev_local":
        return JevClient(settings)
    if settings["type"] == "typesafe":
        return TypeSafeClient(settings, sink, advisors[settings["render_advisor"]])
    if settings["type"] == "deepseek":
        return DeepSeekClient(settings, sink)
    raise ValueError("unsupported advisor type")


def shown_actions(policy, prediction, gate):
    if gate is not None and gate['triggered'] != 'none':
        return []
    if 'hint_display' in policy:
        probabilities = prediction['probabilities']
        if max(p['probability'] for p in probabilities) < policy['hint_display']['threshold']:
            return [p['name'] for p in sorted(probabilities, key=lambda p: -p['probability'])[:3]]
    return [prediction['top1']]


def choose_hint(policy, prediction, recent_calls, *, action_names=None):
    """No request or history mutation; a single configured gate is sufficient."""
    top1 = prediction["top1"]
    reason = "none"
    details = None
    if policy["advice"] == "gate":
        probabilities = {
            item["name"]: item["probability"] for item in prediction["probabilities"]
        }
        gate = policy["gate"]
        last_two = recent_calls[-2:]
        normalize = lambda value: json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            # BFCL's official decoder accepts NaN/Infinity inside argument strings.
            # This is only an in-memory equality key, never a JSON wire payload.
            allow_nan=True,
        )
        loop = (
            top1 != "RESPOND"
            and len(last_two) == 2
            and last_two[0]["name"] == last_two[1]["name"] == top1
            and normalize(last_two[0]["arguments"])
            == normalize(last_two[1]["arguments"])
        )
        decisions = {
            "respond_prob_ge": top1 != "RESPOND"
            and probabilities["RESPOND"] >= GATES["respond_prob_ge"],
            "top1_prob_lt": top1 != "RESPOND"
            and max(probabilities.values()) < GATES["top1_prob_lt"],
        }
        reason = "loop" if loop else "confidence" if decisions[gate["rule"]] else "none"
        details = {
            "rule": gate["rule"],
            "threshold": gate["threshold"],
            "triggered": reason,
            "loop": loop,
            "confidence_by_rule": decisions,
        }
    shown = shown_actions(policy, prediction, details)
    action_names = {} if action_names is None else action_names
    if not shown:
        hint = NEUTRAL_MESSAGE
    elif len(shown) > 1:
        probabilities = {p['name']: p['probability'] for p in prediction['probabilities']}
        items = []
        for position, name in enumerate(shown, 1):
            if name == 'RESPOND' and name not in action_names:
                action = 'reply to the user without calling a tool'
            else:
                action = f'call the tool `{action_names.get(name, name)}`'
                if name == 'RESPOND':
                    action += ' (RESPOND; the same terminating action as Finish, a separate logical candidate)'
            items.append(f'{position}. {action} (p={probabilities[name]:.2f})')
        hint = RANKED_HINT.format(items='\n'.join(items))
    else:
        hint = (PREFER_RESPOND if top1 == 'RESPOND' and top1 not in action_names
                else PREFER_TOOL.format(name=action_names.get(top1, top1)))
    return hint, details


def advise(policy, client, messages, tools, recent_calls, *, output_protocol=None, action_names=None):
    started = time.perf_counter()
    record = {
        "status": "requesting",
        "prediction": None,
        "latency_seconds": None,
        "hint": None,
        "gate": None,
        "first_action_matches": None,
        "fallback": None,
        'hint_kind': None,
        'shown': [],
    }
    try:
        prediction = (
            client.predict(messages, tools)
            if output_protocol is None
            else client.predict(messages, tools, output_protocol=output_protocol)
        )
        if output_protocol == "stabletoolbench" and prediction["top1"] is None:
            record.update(
                status="neutral_fallback",
                fallback="stable_thought_neutral",
                raw_action=prediction["raw_action"],
                hint=NEUTRAL_MESSAGE,
                hint_kind='neutral',
            )
            return record
        record["prediction"] = prediction
        hint, gate = choose_hint(policy, prediction, recent_calls, action_names=action_names)
        shown = shown_actions(policy, prediction, gate)
        record.update(status="recommended", hint=hint, gate=gate, shown=shown,
                      hint_kind='neutral' if not shown else 'ranked' if len(shown) > 1 else 'top1')
        if action_names is not None and len(shown) == 1 and shown[0] in action_names:
            record['hint_action'] = action_names[shown[0]]
        return record
    except (OSError, HTTPException, ValueError, KeyError, TypeError) as exc:
        transport = isinstance(exc, (OSError, HTTPException)) or (
            isinstance(exc, HTTPResponseError)
            and (exc.status_code == 429 or exc.status_code >= 500)
        )
        record.update(
            status="service_error" if transport else "advisor_error",
            error_type=type(exc).__name__,
            error=str(exc),
            raw_response=client.last_response,
        )
        if isinstance(exc, AdvisorFormatError):
            record.update(
                status="neutral_fallback",
                fallback="advisor_format_error_neutral",
                prediction=None,
                gate=None,
                hint=NEUTRAL_MESSAGE,
                hint_kind='neutral',
            )
            return record
        if isinstance(exc, HTTPResponseError):
            record["http_error"] = {"status_code": exc.status_code, "body": exc.body}
            if render_overlength(exc):
                record.update(
                    status="neutral_fallback",
                    fallback="over_length_after_drop_neutral",
                    hint=NEUTRAL_MESSAGE,
                    hint_kind='neutral',
                )
                return record
        raise AdvisorServiceError(str(exc), record) from exc
    finally:
        record["latency_seconds"] = time.perf_counter() - started
        record["usage"] = client.last_usage
        if isinstance(client, JevClient):
            record["validated_metadata"] = client.last_metadata
