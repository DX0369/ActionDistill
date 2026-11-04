import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from jev_bench.client import CompletionClient, QwenClient
from openai.types.chat import ChatCompletion


def call(name, arguments, call_id="cpu_call"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def response(*, content=None, calls=None, reasoning=None):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": calls,
        "reasoning": reasoning,
    }


class CpuUserClient(CompletionClient):
    """Scripted simulator with measured requests; no remote model is selected."""

    def __init__(self, settings, sink):
        super().__init__(settings, sink, api_key="cpu", participant="user_simulator")

    def complete(self, *, messages, tools):
        request = {"model": self.model, "messages": messages}
        if tools is not None:
            request.update(tools=tools, tool_choice="auto")
        return self.request(request)


class Endpoint:
    def __init__(self, steps):
        self.steps = list(steps)
        self.requests = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        if not self.steps:
            raise RuntimeError("scripted endpoint exhausted")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        message = step(request) if callable(step) else step
        return ChatCompletion.model_validate(
            {
                "id": "cpu",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    def client(self, settings, sink, *, user=False):
        sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)),
            close=lambda: None,
        )
        with patch("jev_bench.client.OpenAI", return_value=sdk):
            return (CpuUserClient if user else QwenClient)(settings, sink)


class Advisor:
    def __init__(self, *, top1="RESPOND", probabilities=None):
        self.top1, self.probabilities = top1, probabilities
        self.requests = []
        self.last_response = None
        self.last_usage = None
        self.records = []

    def predict(self, messages, tools):
        self.requests.append(deepcopy({"messages": messages, "tools": tools}))
        names = [tool["function"]["name"] for tool in tools] + ["RESPOND"]
        probabilities = self.probabilities or {
            name: int(name == self.top1) for name in names
        }
        return {
            "top1": self.top1,
            "probabilities": [
                {"name": name, "probability": probabilities[name]} for name in names
            ],
            "n_prompt_tokens": 10,
            "truncated": False,
            "messages_dropped": 0,
            "latency_seconds": 0,
            "model": "cpu-advisor",
            "temperature": 1,
        }
