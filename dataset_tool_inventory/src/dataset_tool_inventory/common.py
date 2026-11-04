"""Streaming and defensive helpers used by dataset adapters."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Iterate a top-level JSON array without loading the complete file."""

    decoder = json.JSONDecoder()
    with path.open("rt", encoding="utf-8") as handle:
        buffer = ""
        started = False
        done = False
        expect_value = True
        while not done:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            end_of_file = not chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    buffer = buffer[1:]
                    continue
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if expect_value:
                    if buffer[0] == "]":
                        done = True
                        buffer = buffer[1:]
                        break
                    try:
                        item, end = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    yield item
                    buffer = buffer[end:]
                    expect_value = False
                    continue
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    expect_value = True
                    continue
                if buffer[0] == "]":
                    done = True
                    buffer = buffer[1:]
                    break
                raise ValueError(f"invalid JSON array separator in {path}")
            if done:
                if buffer.strip():
                    raise ValueError(f"trailing data after JSON array in {path}")
                break
            if end_of_file:
                raise ValueError(f"truncated JSON array in {path}")


def string_name(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def tool_schema(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        return value
    if isinstance(value.get("name"), str):
        return value
    return None

