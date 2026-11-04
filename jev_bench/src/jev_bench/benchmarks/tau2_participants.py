"""Official participant states with request-local Jev hints and explicit clients."""

import json
from copy import deepcopy

from openai import APIError
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.user.user_simulator import UserSimulator

from ..advice import NEUTRAL_MESSAGE, AdvisorServiceError, advise
from ..client import (
    CompletionResponseError,
    ContextLimitError,
    ParticipantResponseError,
)


class UserSimulatorError(RuntimeError):
    """The simulated user failed; the executor task outcome remains unresolved."""


def to_openai(message):
    if isinstance(message, ToolMessage):
        if not isinstance(message.content, str):
            raise TypeError("tool output requires its canonical string content")
        return {"role": "tool", "tool_call_id": message.id, "content": message.content}
    if not isinstance(message, (SystemMessage, AssistantMessage, UserMessage)):
        raise TypeError(f"unsupported tau2 message: {type(message).__name__}")
    result = {"role": message.role, "content": message.content}
    if (
        isinstance(message, (AssistantMessage, UserMessage))
        and message.tool_calls is not None
    ):
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        call.arguments, ensure_ascii=False, allow_nan=False
                    ),
                },
            }
            for call in message.tool_calls
        ]
    if isinstance(message, AssistantMessage) and message.raw_data is not None:
        for field in ("reasoning", "reasoning_content"):
            if field in message.raw_data:
                value = message.raw_data[field]
                if value is not None and not isinstance(value, str):
                    raise TypeError(f"{field} must be a string or null")
                result[field] = value
    return result


def parse_message(result, *, user=False):
    message = result.message
    role = "user_simulator" if user else "agent"
    content, calls = message.content, message.tool_calls
    if content is not None and not isinstance(content, str):
        raise ParticipantResponseError("content must be a string or null", role=role)
    if not calls and (content is None or not content.strip()):
        raise ParticipantResponseError(
            "completion contains neither text nor tools", role=role
        )
    parsed = []
    for call in calls or []:
        if (
            getattr(call, "type", None) != "function"
            or getattr(call, "function", None) is None
        ):
            raise ParticipantResponseError("tool call requires a function", role=role)
        if not isinstance(getattr(call, "id", None), str) or not call.id:
            raise ParticipantResponseError(
                "tool call requires its canonical id", role=role
            )
        name, raw = (
            getattr(call.function, "name", None),
            getattr(call.function, "arguments", None),
        )
        if not isinstance(name, str) or not name or not isinstance(raw, str):
            raise ParticipantResponseError(
                "tool name and JSON arguments are required", role=role
            )
        try:
            arguments = json.loads(raw)
            json.dumps(arguments, allow_nan=False)
        except ValueError:
            arguments = None
        if not isinstance(arguments, dict):
            raise ParticipantResponseError(
                "tool arguments must decode to a JSON object", role=role
            )
        parsed.append(
            ToolCall(
                id=call.id,
                name=name,
                arguments=arguments,
                requestor="user" if user else "assistant",
            )
        )
    raw_data = {}
    for field in ("reasoning", "reasoning_content"):
        if hasattr(message, field):
            value = getattr(message, field)
            if value is not None and not isinstance(value, str):
                raise ParticipantResponseError(
                    f"{field} must be a string or null", role=role
                )
            raw_data[field] = value
    cls = UserMessage if user else AssistantMessage
    return cls(
        role="user" if user else "assistant",
        content=content,
        tool_calls=parsed or None,
        usage=result.usage,
        raw_data=raw_data or None,
    )


class GuidedAgent(LLMAgent):
    def __init__(
        self,
        *,
        tools,
        domain_policy,
        client,
        settings,
        policy,
        advisor_client,
        event_sink,
        projection=None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=client.model)
        self.client, self.settings, self.policy = client, settings, policy
        self.advisor_client, self.event_sink = advisor_client, event_sink
        self.turn_tool_calls = []
        self.advisor_records, self.anomalies = [], []
        self.decision_steps = self.turn_index = self.step_index = 0
        self.projection = projection

    def set_seed(self, seed):
        # Match BFCL: the recorded episode seed is not an executor sampling seed.
        pass

    def _observe(self, message):
        if isinstance(message, UserMessage):
            self.turn_tool_calls = []
        elif isinstance(message, AssistantMessage) and message.tool_calls:
            self.turn_tool_calls = (
                self.turn_tool_calls
                + [
                    {"name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ]
            )[-2:]

    def get_init_state(self, message_history=None):
        state = super().get_init_state(deepcopy(message_history))
        self.turn_tool_calls = []
        for message in state.messages:
            self._observe(message)
        return state

    def generate_next_message(self, message, state):
        from tau2.agent.base import is_valid_agent_history_message

        incoming = (
            message.tool_messages
            if isinstance(message, MultiToolMessage)
            else [message]
        )
        if not all(is_valid_agent_history_message(item) for item in incoming):
            raise ValueError("agent input includes private user tool activity")
        for item in incoming:
            self._observe(item)
            if isinstance(item, UserMessage):
                self.turn_index += 1
                self.step_index = 0
        state.messages.extend(incoming)
        messages = [to_openai(item) for item in state.system_messages + state.messages]
        tools = [tool.openai_schema for tool in self.tools]
        record = None
        mode = self.policy["advice"]
        if mode == "neutral":
            messages.append({"role": "system", "content": NEUTRAL_MESSAGE})
        elif mode in ("prefer", "gate"):
            try:
                record = advise(
                    self.policy,
                    self.advisor_client,
                    messages,
                    tools,
                    self.turn_tool_calls,
                )
            except AdvisorServiceError as exc:
                exc.record.update(turn=self.turn_index, step=self.step_index)
                self.advisor_records.append(exc.record)
                self.event_sink("advisor", exc.record)
                raise
            record.update(turn=self.turn_index, step=self.step_index)
            self.advisor_records.append(record)
            messages.append({"role": "system", "content": record["hint"]})
        sampling = self.settings["thinking"]
        if self.projection is not None:
            messages, tools = self.projection.request(messages, tools)
        self.decision_steps += 1
        try:
            result = self.client.complete(
                messages=messages,
                tools=tools if tools else None,
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
            flags = []
            if result.finish_reason == "length":
                flags.append("output_limit")
            if not result.message.tool_calls and not result.message.content:
                flags.append("empty_response")
            if flags:
                self.anomalies.append(
                    {"turn": self.turn_index, "step": self.step_index, "flags": flags}
                )
            response = parse_message(result)
            self._observe(response)
            if record is not None and record["prediction"] is not None:
                actual = (
                    response.tool_calls[0].name if response.tool_calls else "RESPOND"
                )
                record["first_action_matches"] = actual == record["prediction"]["top1"]
            state.messages.append(response)
            return response, state
        finally:
            if record is not None:
                self.event_sink("advisor", record)
            self.step_index += 1


class DeepSeekUser(UserSimulator):
    def __init__(self, *, client, tools, instructions):
        super().__init__(tools=tools, instructions=instructions, llm=client.model)
        self.client = client

    def set_seed(self, seed):
        # No provider seed contract: never claim identical user branches across conditions.
        pass

    def _generate_next_message(self, message, state):
        state.messages.extend(
            message.tool_messages
            if isinstance(message, MultiToolMessage)
            else [message]
        )
        flipped = state.flip_roles()
        # Upstream flips roles one-to-one but omits raw_data. Only the simulator's
        # own past outputs become assistant messages in its private request.
        for original, converted in zip(state.messages, flipped, strict=True):
            if isinstance(original, UserMessage):
                converted.raw_data = deepcopy(original.raw_data)
        messages = [to_openai(item) for item in state.system_messages + flipped]
        tools = (
            [tool.openai_schema for tool in self.tools]
            if self.tools is not None
            else None
        )
        try:
            result = self.client.complete(messages=messages, tools=tools)
            response = parse_message(result, user=True)
            allowed = (
                {tool.name for tool in self.tools} if self.tools is not None else set()
            )
            if any(call.name not in allowed for call in response.tool_calls or []):
                raise ParticipantResponseError(
                    "user called a tool outside its visible tools",
                    role="user_simulator",
                )
        except Exception as exc:
            if isinstance(exc, (CompletionResponseError, ContextLimitError, APIError)):
                raise UserSimulatorError(str(exc)) from exc
            raise
        state.messages.append(response)
        return response, state
