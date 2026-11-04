"""Streaming aggregation and output writing."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

from .schema import DecisionEvent, IssueEvent, ToolEvent, TrajectoryEvent, schema_hash


class InventoryWriter:
    def __init__(self, output: Path, dataset_names: Iterable[str]):
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        output.mkdir(parents=True)
        self.output = output
        self.tools_path = output / "tools.jsonl"
        self.states_path = output / "states.jsonl"
        self.issues_path = output / "issues.jsonl"
        self._tools: dict[tuple, dict] = {}
        self._summary: dict[str, dict] = {name: self._blank(name) for name in dataset_names}
        self._trajectory_ids: dict[str, set[str]] = {name: set() for name in dataset_names}
        self._declared_by_trajectory: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._called_by_trajectory: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._states = 0
        self._issues = 0

    @staticmethod
    def _blank(name: str) -> dict:
        return {
            "dataset": name,
            "trajectory_count": 0,
            "event_count": 0,
            "declared_event_count": 0,
            "called_event_count": 0,
            "returned_event_count": 0,
            "declared_tool_union": 0,
            "called_tool_union": 0,
            "returned_tool_union": 0,
            "decision_state_count": 0,
            "complete_candidate_state_count": 0,
            "incomplete_candidate_state_count": 0,
            "action_kind_counts": Counter(),
            "domains": Counter(),
            "models": Counter(),
            "harnesses": Counter(),
        }

    def _context(self, event: ToolEvent) -> tuple:
        # Model and harness are provenance, not tool identity. A schema change
        # or server change must still create a separate exact identity.
        return (event.dataset, event.domain, event.server, event.raw_name, schema_hash(event.schema))

    def consume(self, event: ToolEvent | DecisionEvent | IssueEvent | TrajectoryEvent) -> None:
        if isinstance(event, TrajectoryEvent):
            self._trajectory_ids[event.dataset].add(event.trajectory_id)
            return
        if isinstance(event, ToolEvent):
            summary = self._summary[event.dataset]
            self._trajectory_ids[event.dataset].add(event.trajectory_id)
            if event.event == "declared":
                self._declared_by_trajectory[(event.dataset, event.trajectory_id)].add(event.raw_name)
            elif event.event == "called":
                self._called_by_trajectory[(event.dataset, event.trajectory_id)].add(event.raw_name)
            summary["event_count"] += 1
            summary[f"{event.event}_event_count"] += 1
            if event.domain:
                summary["domains"][event.domain] += 1
            if event.model:
                summary["models"][event.model] += 1
            if event.harness:
                summary["harnesses"][event.harness] += 1
            key = self._context(event)
            item = self._tools.setdefault(key, {
                "dataset": event.dataset,
                "domain": event.domain,
                "server": event.server,
                "raw_name": event.raw_name,
                "schema": event.schema,
                "schema_hash": schema_hash(event.schema),
                "declared_count": 0,
                "called_count": 0,
                "returned_count": 0,
                "trajectory_ids": set(),
                "models": set(),
                "harnesses": set(),
                "first_source": event.source,
            })
            item[f"{event.event}_count"] += 1
            item["trajectory_ids"].add(event.trajectory_id)
            if event.model:
                item["models"].add(event.model)
            if event.harness:
                item["harnesses"].add(event.harness)
            return
        if isinstance(event, DecisionEvent):
            summary = self._summary[event.dataset]
            self._trajectory_ids[event.dataset].add(event.trajectory_id)
            summary["decision_state_count"] += 1
            if event.candidate_set_complete:
                summary["complete_candidate_state_count"] += 1
            else:
                summary["incomplete_candidate_state_count"] += 1
            summary["action_kind_counts"][event.action_kind] += 1
            with self.states_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            self._states += 1
            return
        summary = self._summary[event.dataset]
        summary.setdefault("issue_counts", Counter())[event.issue] += 1
        with self.issues_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._issues += 1

    def finalize(self) -> dict:
        # Only report this mismatch when a trajectory actually exposes a
        # declaration set. Some formats intentionally hide schemas or use
        # dynamic MCP servers, for which absence is not evidence of an error.
        with self.issues_path.open("a", encoding="utf-8") as handle:
            for key, declared in self._declared_by_trajectory.items():
                unexpected = sorted(self._called_by_trajectory.get(key, set()) - declared)
                if not unexpected:
                    continue
                dataset, trajectory_id = key
                issue = {
                    "dataset": dataset,
                    "issue": "called_not_declared",
                    "message": "called tool names were not present in the trajectory declaration",
                    "trajectory_id": trajectory_id,
                    "source": "aggregate",
                    "metadata": {"tool_names": unexpected},
                }
                handle.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")
                self._summary[dataset].setdefault("issue_counts", Counter())["called_not_declared"] += 1
                self._issues += 1
        with self.tools_path.open("w", encoding="utf-8") as handle:
            for item in sorted(self._tools.values(), key=lambda x: (x["dataset"], x["raw_name"], x["schema_hash"] or "")):
                item = dict(item)
                item["trajectory_ids"] = sorted(item["trajectory_ids"])
                item["models"] = sorted(item["models"])
                item["harnesses"] = sorted(item["harnesses"])
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        for name, summary in self._summary.items():
            summary["trajectory_count"] = len(self._trajectory_ids[name])
            summary["declared_tool_union"] = len({k for k, v in self._tools.items() if k[0] == name and v["declared_count"]})
            summary["called_tool_union"] = len({k for k, v in self._tools.items() if k[0] == name and v["called_count"]})
            summary["returned_tool_union"] = len({k for k, v in self._tools.items() if k[0] == name and v["returned_count"]})
            for key in ("action_kind_counts", "domains", "models", "harnesses", "issue_counts"):
                if isinstance(summary.get(key), Counter):
                    summary[key] = dict(summary[key])
        result = {
            "datasets": self._summary,
            "global": {
                "dataset_count": len(self._summary),
                "exact_tool_identity_count": len(self._tools),
                "decision_state_count": self._states,
                "issue_count": self._issues,
            },
        }
        (self.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

