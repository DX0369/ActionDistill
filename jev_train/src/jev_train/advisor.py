"""Local advice through the same visible-history renderer as sample construction."""

import json
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

from dataset_tool_inventory.adapters.review_common import visible

from .render import prompt_item, render_event
from .tokenizer_identity import tokenizer_identity


def render_request(request, tokenizer):
    if not isinstance(request, dict) or set(request) != {"messages", "tools", "respond"}:
        raise ValueError("request requires exactly messages, tools, respond")
    if request["respond"] is not True:
        raise ValueError("respond must be true: RESPOND is always an option")
    if not isinstance(request["messages"], list) or not request["messages"]:
        raise ValueError("messages must be a nonempty list")
    tools = request["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a nonempty list")
    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            raise ValueError("each tool requires its canonical name")
        # Source declarations may carry extra schema metadata; option() defines
        # exactly which canonical fields become model input.
    history = [visible(message, strip_think=True) for message in request["messages"]]
    for message in history:
        for call in message.get("tool_calls", []):
            if message["role"] != "assistant":
                raise ValueError("only assistant messages may contain tool_calls")
            if "arguments" not in call["function"]:
                raise ValueError("historical tool call lacks arguments")
            if not isinstance(call["function"]["arguments"], (str, dict)):
                raise ValueError("historical arguments must be a string or object")
    # render_event requires a label, but prompt_item never inserts it in the
    # prompt. No option subsampling is allowed here; this placeholder is not a
    # prediction, an input feature, or a training target.
    event = SimpleNamespace(
        metadata={"_history": history, "_candidate_definitions": tools},
        candidate_tools=tuple(tool["name"] for tool in tools),
        action_kind="RESPOND", action_names=(),
    )
    row, reason = render_event(event, tokenizer, max_len=32768, length_margin=64,
                               description_chars=300, head=2048, tail=2048,
                               allow_duplicate_candidates=True)
    if reason is not None:
        raise ValueError(reason)
    item = prompt_item(row["context"], row["question"], row["options"], row["gold"], tokenizer)
    return row, item, [tool["name"] for tool in tools] + ["RESPOND"]


def load_temperature(path):
    value = json.loads(Path(path).read_text())["temperature"]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be positive and finite")
    return float(value)


def response_metadata(description, row):
    counts = row['truncation_counts']
    return {**description, 'n_prompt_tokens': row['n_prompt_tokens'],
            'messages_dropped': row['messages_dropped'], 'truncation_counts': counts,
            'truncated': bool(row['messages_dropped'] or any(counts.get(k, 0) for k in
                ('tool_outputs_truncated', 'tool_call_arguments_truncated', 'images_replaced')))}


class JevRenderer:
    """Tokenizer-only service; it has no prediction endpoint or model weights."""
    def __init__(self, tokenizer_path):
        from transformers import AutoTokenizer
        self.identity = tokenizer_identity(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.identity['tokenizer'], local_files_only=True)

    def describe(self):
        return {**self.identity, 'service_mode': 'render_only', 'prompt_budget': 32704,
                'option_order': 'request tool order, then RESPOND; no shuffle',
                'duplicate_candidates': 'positional'}

    def handle(self, request, *, render_only=False):
        if not render_only:
            raise ValueError('render-only service cannot predict actions')
        started = time.perf_counter()
        row, item, _ = render_request(request, self.tokenizer)
        result = response_metadata(self.describe(), row)
        result.update(rendered=row, prompt_token_ids=item['ids'],
                      prompt=self.tokenizer.decode(item['ids'], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False),
                      latency_seconds=time.perf_counter()-started)
        return result


class JevAdvisor:
    def __init__(self, model_path, temperature_path, device="cuda"):
        from decider.model import DecisionModel
        self.model_path = str(Path(model_path).resolve(strict=True))
        self.temperature_path = str(Path(temperature_path).resolve(strict=True))
        self.temperature = load_temperature(self.temperature_path)
        self.model = DecisionModel(self.model_path, grad_ckpt=False).to(device).eval()
        self.model.lm.config.use_cache = False

    def describe(self):
        return {"model": self.model_path, "temperature_file": self.temperature_path,
                "temperature": self.temperature, "prompt_budget": 32704,
                "option_order": "request tool order, then RESPOND; no shuffle",
                "duplicate_candidates": "positional"}

    def handle(self, request, *, render_only=False):
        started = time.perf_counter()
        row, item, names = render_request(request, self.model.tok)
        result = response_metadata(self.describe(), row)
        if render_only:
            # Diagnostic endpoint uses the very same renderer as /advise.
            result.update(rendered=row, prompt_token_ids=item["ids"],
                          prompt=self.model.tok.decode(item["ids"], skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False))
        else:
            import torch
            from decider.model import collate
            batch = collate([item], self.model.tok.pad_token_id)
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.inference_mode():
                logits = self.model(batch)[0, :len(names)].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite candidate logits")
                probabilities = torch.softmax(logits / self.temperature, dim=0).cpu().tolist()
            result.update(
                probabilities=[{"name": n, "probability": p} for n, p in zip(names, probabilities)],
                top1=names[max(range(len(names)), key=probabilities.__getitem__)],
            )
        result["latency_seconds"] = time.perf_counter() - started
        return result


def make_server(advisor, port):
    """Single-request GPU execution; fixed loopback address, never 0.0.0.0."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in 0..65535")

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ready", **advisor.describe()})
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path not in ("/advise", "/render"):
                self.send_json(404, {"error": "unknown endpoint"})
                return
            if self.path == '/advise' and isinstance(advisor, JevRenderer):
                self.send_json(405, {'error': {'type': 'MethodNotAllowed', 'message': 'render-only service'}})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError("request body must be 1..67108864 bytes")
                def invalid_constant(value):
                    raise ValueError(f"invalid JSON constant: {value}")
                request = json.loads(self.rfile.read(size), parse_constant=invalid_constant)
                result = advisor.handle(request, render_only=self.path == "/render")
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(422, {"error": {"type": type(exc).__name__, "message": str(exc)}})
            except Exception as exc:
                self.send_json(500, {"error": {"type": type(exc).__name__, "message": str(exc)}})
            else:
                self.send_json(200, result)

    return HTTPServer(("127.0.0.1", port), Handler)
