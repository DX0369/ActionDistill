"""Inject one request-local hint into BFCL's unmodified official loop."""

import json
from copy import deepcopy

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.base_handler import BaseHandler

from ..advice import NEUTRAL_MESSAGE, AdvisorServiceError, advise
from ..client import ParticipantResponseError


class GuidanceHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        *,
        condition,
        namespace,
        client,
        settings,
        policy,
        advisor_client,
        event_sink,
        projection=None,
    ):
        BaseHandler.__init__(
            self,
            model_name=client.model,
            temperature=0,
            registry_name=condition,
            is_fc_model=True,
        )
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.model_name_underline_replaced = namespace
        self.request_client = client
        self.settings = settings
        self.policy = policy
        self.advisor_client = advisor_client
        self.event_sink = event_sink
        self.turn_index = self.step_index = 0
        self.turn_tool_calls = []
        self.advisor_records = []
        self.anomalies = []
        self.decision_steps = 0
        self.projection = projection

    def _add_next_turn_user_message_FC(self, inference_data, user_message):
        self.turn_index += 1
        self.step_index = 0
        self.turn_tool_calls = []
        return super()._add_next_turn_user_message_FC(inference_data, user_message)

    def _query_FC(self, inference_data):
        messages = deepcopy(inference_data["message"])
        tools = inference_data["tools"]
        mode = self.policy["advice"]
        record = None
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
        inference_data["inference_input_log"] = {
            "message": repr(messages),
            "tools": tools,
        }
        try:
            self.decision_steps += 1
            result = self.request_client.complete(
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
            if record is not None and record["prediction"] is not None:
                calls = result.message.tool_calls
                actual = (
                    getattr(getattr(calls[0], "function", None), "name", None)
                    if calls
                    else (
                        "RESPOND"
                        if isinstance(result.message.content, str)
                        and result.message.content.strip()
                        else None
                    )
                )
                record["first_action_matches"] = actual == record["prediction"]["top1"]
            return result, result.latency_seconds
        finally:
            if record is not None:
                self.event_sink("advisor", record)
            self.step_index += 1

    def _parse_query_response_FC(self, result):
        message = result.message
        calls = message.tool_calls
        history_message = {"role": "assistant", "content": message.content}
        flags = []
        if result.finish_reason == "length":
            flags.append("output_limit")
        if not calls and (
            not isinstance(message.content, str) or not message.content.strip()
        ):
            flags.append("empty_response")
        if flags:
            self.anomalies.append(
                {"turn": self.turn_index, "step": self.step_index - 1, "flags": flags}
            )
        if calls:
            parsed_calls = []
            recent_calls = []
            for call in calls:
                if (
                    getattr(call, "type", None) != "function"
                    or getattr(call, "function", None) is None
                ):
                    raise ParticipantResponseError(
                        "tool call must contain a function", role="agent"
                    )
                if not isinstance(getattr(call, "id", None), str) or not call.id:
                    raise ParticipantResponseError(
                        "tool call lacks its canonical id", role="agent"
                    )
                if (
                    not isinstance(getattr(call.function, "name", None), str)
                    or not call.function.name
                ):
                    raise ParticipantResponseError(
                        "tool call lacks its canonical function name", role="agent"
                    )
                if not isinstance(getattr(call.function, "arguments", None), str):
                    raise ParticipantResponseError(
                        "tool arguments must be a JSON string", role="agent"
                    )
                try:
                    arguments = json.loads(call.function.arguments)
                except ValueError:
                    arguments = None
                if isinstance(arguments, dict):
                    recent_calls.append(
                        {"name": call.function.name, "arguments": arguments}
                    )
                # Pass the original arguments to the official decoder. It owns
                # malformed JSON/non-object handling and turn advancement.
                parsed_calls.append(
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
            history_message["tool_calls"] = parsed_calls
            if self.policy["advice"] == "gate":
                self.turn_tool_calls = (self.turn_tool_calls + recent_calls)[-2:]
        responses = (
            [{call.function.name: call.function.arguments} for call in calls]
            if calls is not None
            else message.content
        )
        usage = result.usage
        parsed = {
            "model_responses": responses,
            "model_responses_message_for_chat_history": history_message,
            "tool_call_ids": [call.id for call in calls] if calls else [],
            "input_token": usage["prompt_tokens"] if usage is not None else None,
            "output_token": usage["completion_tokens"] if usage is not None else None,
        }
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            parsed["reasoning_content"] = reasoning
            history_message["reasoning"] = reasoning
        return parsed

    def _add_reasoning_content_if_available_FC(self, response, parsed):
        pass  # The parser above preserves the same reasoning in diagnostics and history.
