"""Small, JSON-serializable records shared by all adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


def schema_hash(schema: Any) -> str | None:
    if schema is None:
        return None
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolEvent:
    dataset: str
    trajectory_id: str
    event: str  # declared, called, returned
    raw_name: str
    domain: str | None = None
    model: str | None = None
    harness: str | None = None
    server: str | None = None
    turn_index: int | None = None
    schema: Any = None
    call_id: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_hash"] = schema_hash(self.schema)
        return value


@dataclass(frozen=True)
class DecisionEvent:
    dataset: str
    trajectory_id: str
    turn_index: int
    candidate_tools: tuple[str, ...]
    action_kind: str  # TOOL, TOOL_SET, RESPOND, UNKNOWN
    action_names: tuple[str, ...]
    candidate_set_complete: bool
    domain: str | None = None
    model: str | None = None
    harness: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IssueEvent:
    dataset: str
    issue: str
    message: str
    trajectory_id: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryEvent:
    dataset: str
    trajectory_id: str
    domain: str | None = None
    model: str | None = None
    harness: str | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

