"""Qwen non-streaming requests, preserving the historical request semantics."""

import os
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI


@dataclass
class CompletionResult:
    message: object
    usage: dict | None
    latency_seconds: float
    finish_reason: str | None
    raw_response: dict


class CompletionResponseError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class ParticipantResponseError(CompletionResponseError):
    def __init__(self, message, *, role):
        super().__init__(message, {"role": role})


class AssistantChoiceError(CompletionResponseError):
    """A response with usable accounting but no valid assistant choice."""


class ContextLimitError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def is_context_limit_error(exc):
    if (
        not isinstance(exc, APIStatusError)
        or exc.status_code != 400
        or not isinstance(exc.body, dict)
    ):
        return False
    body = exc.body
    message = body.get("message")
    return (
        body.get("type") == "BadRequestError"
        and body.get("param") in ("input_tokens", "input_text")
        and isinstance(message, str)
        and message.startswith("This model's maximum context length is ")
        and "Please reduce the length of the input prompt or the number of requested output tokens."
        in message
    )


def is_service_error(exc):
    return isinstance(exc, APIConnectionError) or (
        isinstance(exc, APIStatusError) and exc.status_code in {429, 500, 502, 503, 504}
    )


class CompletionClient:
    def __init__(self, settings, event_sink, *, api_key, participant):
        self.model = settings["model"]
        self.participant = participant
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=api_key,
            max_retries=0,
            timeout=settings["timeout_seconds"],
        )
        self.event_sink = event_sink
        self.records = []

    def close(self):
        self.client.close()

    def request(self, request):
        started = time.perf_counter()
        record = {
            "request": request,
            "usage": None,
            "response": None,
            "error": None,
            "latency_seconds": None,
        }
        try:
            response = self.client.chat.completions.create(**request)
            raw = response.model_dump(mode="json")
            record["response"] = raw
            # Usage can be absent; absence is measured, never replaced by zero.
            usage = raw.get("usage")
            if usage is not None:
                try:
                    usage = {
                        key: usage[key]
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    }
                    if any(type(x) is not int or x < 0 for x in usage.values()):
                        raise ValueError("invalid canonical completion usage")
                except (KeyError, TypeError, ValueError) as exc:
                    raise CompletionResponseError(
                        str(exc), {"role": self.participant, **record}
                    ) from exc
            record["usage"] = usage
            choices = getattr(response, "choices", None)
            message = (
                getattr(choices[0], "message", None)
                if isinstance(choices, list) and choices
                else None
            )
            if message is None or getattr(message, "role", None) != "assistant":
                raise AssistantChoiceError(
                    "completion requires an assistant choice",
                    {"role": self.participant, **record},
                )
            choice = choices[0]
            return CompletionResult(
                choice.message,
                usage,
                time.perf_counter() - started,
                choice.finish_reason,
                raw,
            )
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if is_context_limit_error(exc):
                raise ContextLimitError(
                    str(exc),
                    {
                        "role": self.participant,
                        "error_body": exc.body,
                        "request": request,
                    },
                ) from exc
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            self.records.append(record)
            self.event_sink(self.participant, record)


class QwenClient(CompletionClient):
    def __init__(self, settings, event_sink):
        super().__init__(settings, event_sink, api_key="EMPTY", participant="executor")

    def complete(
        self,
        *,
        messages,
        tools,
        tool_choice,
        max_tokens,
        temperature,
        top_p,
        extra_body,
    ):
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
            "top_p": top_p,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return self.request(request)


class SimulatorClient(CompletionClient):
    def __init__(self, settings, event_sink):
        api_key = os.environ[settings["api_key_env"]]
        if not api_key.strip():
            raise ValueError("user simulator API key environment variable is empty")
        super().__init__(
            settings, event_sink, api_key=api_key, participant="user_simulator"
        )
        self.settings = settings

    def complete(self, *, messages, tools):
        request = {"model": self.model, "messages": messages}
        request.update({k: self.settings[k] for k in ("temperature", "max_tokens")})
        request["extra_body"] = {"thinking": {"type": self.settings["thinking"]}}
        if tools is not None:
            request.update(tools=tools, tool_choice="auto")
        return self.request(request)
