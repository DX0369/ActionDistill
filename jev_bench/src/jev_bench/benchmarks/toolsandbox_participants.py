"""Keep official roles/routing; replace only their inference hooks for CPU fixtures."""

import json
from collections import Counter
from copy import deepcopy

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

from ..advice import NEUTRAL_MESSAGE, AdvisorServiceError, advise
from ..client import ParticipantResponseError


def completion(result, tools, *, role, normalizations):
    """Reject malformed responses before official Python conversion and routing."""
    try:
        choices = result.raw_response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("one completion choice is required")
        raw = choices[0]["message"]
        if raw["role"] != "assistant":
            raise ValueError("completion role must be assistant")
        # OpenAI content/tool_calls are optional, but both absent is invalid here.
        content, calls = raw.get("content"), raw.get("tool_calls")
        if content is not None and not isinstance(content, str):
            raise ValueError("completion content must be a string or null")
        empty_calls = isinstance(calls, list) and not calls and isinstance(content, str)
        if calls is None or empty_calls:
            if not isinstance(content, str):
                raise ValueError("completion contains neither text nor tools")
        else:
            if not isinstance(calls, list) or not calls:
                raise ValueError("tool_calls must be null or a nonempty list")
            available = {tool["function"]["name"] for tool in tools}
            ids = set()
            for call in calls:
                if call["type"] != "function":
                    raise ValueError("only function calls are supported")
                call_id = call["id"]
                if not isinstance(call_id, str) or not call_id or call_id in ids:
                    raise ValueError("tool calls require distinct nonempty IDs")
                ids.add(call_id)
                function = call["function"]
                if function["name"] not in available:
                    raise ValueError("tool call is not among the visible tools")
                if not isinstance(function["arguments"], str):
                    raise TypeError("tool arguments must be a JSON string")
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must decode to an object")
                json.dumps(arguments, allow_nan=False)
        reasoning = raw.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning must be a string or null when present")
        # Official roles interpolate IDs into Python variable names and recover
        # them from that code when rebuilding history. Rename only the routing
        # copy; UTF-8 hex is injective, and escaping our prefix prevents clashes
        # with an already-valid ID. Raw provider evidence stays unchanged.
        renamed_ids = {
            call["id"]: "ts_" + call["id"].encode("utf-8").hex()
            for call in calls or []
            if not (call["id"].isascii() and call["id"].isidentifier())
            or call["id"].startswith("ts_")
        }
        payload = result.raw_response
        if empty_calls or renamed_ids:
            payload = deepcopy(payload)
        if empty_calls:
            # vLLM represents a text-only answer with []; the official roles use
            # `is None`. Normalize only the routing copy, preserving raw evidence.
            payload["choices"][0]["message"]["tool_calls"] = None
        if renamed_ids:
            for call in payload["choices"][0]["message"]["tool_calls"]:
                call["id"] = renamed_ids.get(call["id"], call["id"])
        response = ChatCompletion.model_validate(payload)
        if empty_calls:
            normalizations["empty_tool_calls_to_none"] += 1
        if renamed_ids:
            normalizations["tool_call_id_to_python_identifier"] += len(renamed_ids)
        return response
    except (KeyError, TypeError, ValueError) as exc:
        raise ParticipantResponseError(str(exc), role=role) from exc


def assistant_signature(message):
    """Official reconstruction changes argument whitespace and discards tool-call text."""
    if message.get("tool_calls"):
        return [
            (
                call["id"],
                call["function"]["name"],
                json.loads(call["function"]["arguments"]),
            )
            for call in message["tool_calls"]
        ]
    return message["content"]


class GuidedAgent(OpenAIAPIAgent):
    def __init__(
        self, *, client, settings, policy, advisor_client, event_sink, projection=None
    ):
        # BaseRole has no constructor state. The official subclass constructor
        # would create a hardcoded external OpenAI client; use the injected client.
        self.client, self.settings, self.policy = client, settings, policy
        self.advisor_client, self.event_sink = advisor_client, event_sink
        self.advisor_records = []
        self.decision_steps = 0
        self.reasoning_history = []
        self.normalizations = Counter()
        self.projection = projection

    def model_inference(self, openai_messages, openai_tools):
        messages = deepcopy(openai_messages)
        tools = [] if openai_tools is NOT_GIVEN else deepcopy(list(openai_tools))
        # Official Message has no reasoning field. Restore reasoning only to the
        # request copy, tied to the exact assistant event, never to the user DB.
        for index, signature, reasoning in self.reasoning_history:
            if (
                index >= len(messages)
                or messages[index]["role"] != "assistant"
                or assistant_signature(messages[index]) != signature
            ):
                raise ValueError(
                    "official assistant history differs from reasoning sidecar"
                )
            messages[index]["reasoning"] = reasoning

        recent_calls = []
        turn, step = -1, 0
        for message in messages:
            if message["role"] == "user":
                turn += 1
                step = 0
                recent_calls = []
            elif message["role"] == "assistant":
                step += 1
                recent_calls.extend(
                    {
                        "name": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"]),
                    }
                    for call in message.get("tool_calls", [])
                )

        record = None
        mode = self.policy["advice"]
        if mode == "neutral":
            messages.append({"role": "system", "content": NEUTRAL_MESSAGE})
        elif mode in ("prefer", "gate"):
            try:
                record = advise(
                    self.policy, self.advisor_client, messages, tools, recent_calls
                )
            except AdvisorServiceError as exc:
                exc.record.update(turn=turn, step=step)
                self.advisor_records.append(exc.record)
                self.event_sink("advisor", exc.record)
                raise
            record.update(turn=turn, step=step)
            self.advisor_records.append(record)
            messages.append({"role": "system", "content": record["hint"]})

        sampling = self.settings["thinking"]
        self.decision_steps += 1
        try:
            request_tools = tools
            if self.projection is not None:
                messages, request_tools = self.projection.request(messages, tools)
            result = self.client.complete(
                messages=messages,
                tools=request_tools or None,
                tool_choice="auto" if tools else None,
                max_tokens=self.settings["max_tokens"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": sampling["top_k"],
                    "min_p": sampling["min_p"],
                },
            )
            if self.projection is not None:
                result = self.projection.response(result)
            response = completion(
                result, tools, role="executor", normalizations=self.normalizations
            )
            raw = response.choices[0].message.model_dump()
            reasoning = raw.get("reasoning")
            if reasoning:
                self.reasoning_history.append(
                    (len(openai_messages), assistant_signature(raw), reasoning)
                )
            if record is not None and record["prediction"] is not None:
                actual = (
                    raw["tool_calls"][0]["function"]["name"]
                    if raw["tool_calls"]
                    else "RESPOND"
                )
                record["first_action_matches"] = actual == record["prediction"]["top1"]
            return response
        finally:
            if record is not None:
                self.event_sink("advisor", record)


class ConfiguredUser(OpenAIAPIUser):
    def __init__(self, *, client):
        self.client = client
        self.normalizations = Counter()

    def model_inference(self, openai_messages, openai_tools):
        tools = [] if openai_tools is NOT_GIVEN else deepcopy(list(openai_tools))
        result = self.client.complete(
            messages=deepcopy(openai_messages), tools=tools or None
        )
        return completion(
            result, tools, role="user_simulator", normalizations=self.normalizations
        )
