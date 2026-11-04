"""Review statistics and bounded samples on top of the existing InventoryWriter."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import heapq
import json
from pathlib import Path
import shutil

from .adapters.review_common import fingerprint
from .aggregate import InventoryWriter
from .schema import DecisionEvent, IssueEvent, TrajectoryEvent
from .review_policy import eligibility

# Review sheets show the tail of the history; full inputs stay out of the sample file.
REVIEW_HISTORY_MESSAGES = 8
REVIEW_MESSAGE_CHARS = 4000
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def percentile(counter, fraction):
    total = sum(counter.values())
    if not total: return None
    threshold = max(1, int(total * fraction + 0.999999))
    seen = 0
    for value, count in sorted(counter.items()):
        seen += count
        if seen >= threshold: return value


def review_view(history, tools):
    shown = []
    for message in history[-REVIEW_HISTORY_MESSAGES:]:
        text = json.dumps(message, ensure_ascii=False)
        shown.append(message if len(text) <= REVIEW_MESSAGE_CHARS else
                     {"truncated_json_prefix": text[:REVIEW_MESSAGE_CHARS], "original_chars": len(text)})
    return {"omitted_earlier_messages": max(0, len(history) - REVIEW_HISTORY_MESSAGES),
            "visible_history_tail": shown, "candidate_tools": [t["name"] for t in tools],
            "additional_candidate": "RESPOND"}


IMAGE_BLOCK_TYPES = {"image", "input_image", "image_url"}
IMAGE_PLACEHOLDER = "[image omitted]"


def head_tail(text, tokenizer, head, tail):
    """Keep the first `head` and last `tail` tokens; mark the omitted middle."""
    # A token covers at least one character, so short texts need no encoding.
    if len(text) <= head + tail:
        return text, False
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= head + tail:
        return text, False
    return (tokenizer.decode(ids[:head]) + f"\n[... {len(ids) - head - tail} tokens omitted ...]\n"
            + tokenizer.decode(ids[-tail:])), True


def truncate_history(history, tokenizer, head, tail):
    """Jev input view: images become a placeholder (the advisor is text-only);
    each tool output and each historical tool-call argument keeps head/tail tokens."""
    result = []; counts = Counter()
    for message in history:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") in IMAGE_BLOCK_TYPES for b in content):
            content = [{"type": "text", "text": IMAGE_PLACEHOLDER}
                       if isinstance(b, dict) and b.get("type") in IMAGE_BLOCK_TYPES else b for b in content]
            counts["images_replaced"] += sum(b == {"type": "text", "text": IMAGE_PLACEHOLDER} for b in content)
            message = dict(message, content=content)
        if message["role"] == "tool" and content is not None:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            text, cut = head_tail(text, tokenizer, head, tail)
            if cut: message = dict(message, content=text); counts["tool_outputs_truncated"] += 1
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                arguments = call["function"].get("arguments")
                if arguments is not None:
                    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                    text, cut = head_tail(text, tokenizer, head, tail)
                    if cut:
                        call = dict(call, function=dict(call["function"], arguments=text))
                        counts["tool_call_arguments_truncated"] += 1
                calls.append(call)
            message = dict(message, tool_calls=calls)
        result.append(message)
    return result, counts


def render_input(history, tools):
    return json.dumps({"visible_history": history, "candidate_tools": tools,
                       "additional_candidate": "RESPOND"}, ensure_ascii=False, sort_keys=True)


class ReviewWriter(InventoryWriter):
    def __init__(self, output, names, policy, entries, tokenizer_path, seed=20261014):
        super().__init__(output, names)
        from transformers import AutoTokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.truncation = policy["tool_output_truncation"]
        self.policy = policy
        self.entries = {e["name"]: e for e in entries}
        self.seed = seed
        self.groups = {}
        self.trajectories = {}
        self.review_samples = defaultdict(dict)
        self.length_samples = []
        self.blocked = defaultdict(Counter)
        self.contamination = []
        self.test_scenarios = self.load_test_tasks(policy)
        self.contaminated_trajectories = set()
        self.harness_tool_sets = {}

    @staticmethod
    def load_test_tasks(policy):
        spec = policy["tau2_retail_test"]
        split = json.loads((WORKSPACE_ROOT / spec["split"]).read_text())
        tasks = json.loads((WORKSPACE_ROOT / spec["tasks"]).read_text())
        ids = {str(v) for v in split["test"]}
        scenarios = {str(t["id"]): fingerprint(t["user_scenario"]) for t in tasks if str(t["id"]) in ids}
        if len(scenarios) != len(ids): raise ValueError("retail test split/task coverage mismatch")
        return scenarios

    def consume(self, event):
        if isinstance(event, IssueEvent):
            if event.issue in {"record_parse_error", "unresolved_dataset", "missing_input", "input_file_parse_error"}:
                self.blocked[event.dataset][event.issue] += 1
            super().consume(event)
            return
        if isinstance(event, TrajectoryEvent):
            key = (event.dataset, event.trajectory_id)
            if "trajectory_hash" in event.metadata:
                previous = self.trajectories.get(key, {})
                item = {**event.to_dict(), "decision_count": previous.get("decision_count", 0),
                        "selection_counts": previous.get("selection_counts", Counter())}
                self.trajectories[key] = item
            super().consume(event)
            return
        if not isinstance(event, DecisionEvent):
            super().consume(event)
            return
        metadata = dict(event.metadata)
        history = metadata.pop("_history")
        tools = metadata.pop("_candidate_definitions")
        entry = self.entries[event.dataset]
        if entry.get("check_tau2_retail"):
            # Only tau2-native records carry the embedded task scenario; other
            # formats have no verifiable field, so their status stays unknown.
            if entry["review_adapter"] != "tau2":
                status = "unknown"
            elif event.domain is None or metadata["task_id"] is None:
                status = "unknown"
            elif event.domain != "retail" or str(metadata["task_id"]) not in self.test_scenarios:
                status = "clear"
            elif metadata["task_scenario_hash"] == self.test_scenarios[str(metadata["task_id"])]:
                status = "confirmed"
            else:
                status = "task_id_only"
            metadata["eval_contamination"] = status
            if status in {"confirmed", "task_id_only"} and (event.dataset, event.trajectory_id) not in self.contaminated_trajectories:
                self.contaminated_trajectories.add((event.dataset, event.trajectory_id))
                self.contamination.append({"dataset": event.dataset, "trajectory_id": event.trajectory_id,
                                           "task_id": metadata["task_id"], "status": status, "source": event.source})
        event = replace(event, metadata=metadata)
        metadata.update(eligibility(event, self.policy))
        event = replace(event, metadata=metadata)
        key = (event.dataset, event.model, event.harness)
        group = self.groups.setdefault(key, {"dataset": event.dataset, "model": event.model,
            "harness": event.harness, "trajectories": set(), "states": 0,
            "actions": Counter(), "candidate_levels": Counter(), "candidate_counts": Counter(),
            "history_lengths": Counter(), "outcomes": Counter(), "eligibility": Counter(),
            "exclusion_reasons": Counter(), "tools": Counter(), "approved_teacher_states": 0,
            "outcome_trajectories": defaultdict(set), "declared_names": set(), "called_names": set()})
        group["trajectories"].add(event.trajectory_id)
        group["states"] += 1
        group["actions"][event.action_kind] += 1
        group["candidate_levels"][metadata["candidate_level"]] += 1
        group["candidate_counts"][len(event.candidate_tools)] += 1
        group["history_lengths"][metadata["history_chars"]] += 1
        group["outcomes"][metadata["outcome"]] += 1
        group["outcome_trajectories"][metadata["outcome"]].add(event.trajectory_id)
        if metadata["candidate_level"] != "I": group["declared_names"].update(event.candidate_tools)
        group["called_names"].update(event.action_names)
        group["eligibility"][metadata["eligibility"]] += 1
        group["exclusion_reasons"].update(metadata["exclusion_reasons"])
        if metadata["model_mapping"]["status"] == "approved": group["approved_teacher_states"] += 1
        if event.action_kind == "TOOL": group["tools"].update(event.action_names)
        tkey = (event.dataset, event.trajectory_id)
        trajectory = self.trajectories.setdefault(tkey, {"decision_count": 0})
        trajectory["decision_count"] += 1
        counts = trajectory.setdefault("selection_counts", Counter())
        counts["all_states"] += 1
        if metadata["model_mapping"]["status"] == "approved":
            counts["after_whitelist"] += 1
            if len(event.candidate_tools) >= 2: counts["after_multi_tool"] += 1
        if metadata["eligibility"] in {"confirmed", "conditional"}:
            counts["eligible_before_dedup"] += 1
            counts[metadata["eligibility"]] += 1
            if metadata.get("eval_contamination") in {"confirmed", "task_id_only"}:
                counts["eligible_contamination_" + metadata["eval_contamination"]] += 1
        input_text = None
        score = int(hashlib.sha256(f"{self.seed}:{event.dataset}:{event.trajectory_id}:{event.turn_index}:{event.source}".encode()).hexdigest(), 16)
        # Length estimate uses 1000 deterministic states, without writing every
        # cumulative history to states.jsonl.
        if metadata["eligibility"] in {"confirmed", "conditional"} and (len(self.length_samples) < self.policy["length_sample_size"] or score < -self.length_samples[0][0]):
            shown, truncated = truncate_history(history, self.tokenizer, self.truncation["head_tokens"],
                                                self.truncation["tail_tokens"])
            sample = {"dataset": event.dataset, "trajectory_id": event.trajectory_id,
                      "turn_index": event.turn_index, "truncation_counts": dict(truncated),
                      "input": render_input(shown, tools)}
            value = (-score, f"{event.dataset}:{event.trajectory_id}:{event.turn_index}", sample)
            if len(self.length_samples) < self.policy["length_sample_size"]: heapq.heappush(self.length_samples, value)
            else: heapq.heapreplace(self.length_samples, value)
        if metadata["eligibility"] in {"confirmed", "conditional"}:
            slots = self.review_samples[(event.dataset, event.model)]
            tscore = int(hashlib.sha256(f"{self.seed}:{event.dataset}:{event.trajectory_id}".encode()).hexdigest(), 16)
            if event.trajectory_id not in slots and (len(slots) < 10 or tscore < max(v["score"] for v in slots.values())):
                if len(slots) >= 10: del slots[max(slots, key=lambda k: slots[k]["score"])]
                slots[event.trajectory_id] = {"score": tscore, "states": []}
            if event.trajectory_id in slots:
                states = slots[event.trajectory_id]["states"]
                if len(states) < 3 or score < max(s["score"] for s in states):
                    if len(states) >= 3: states.remove(max(states, key=lambda s: s["score"]))
                    states.append({"score": score, "event": event.to_dict(),
                                   "input": review_view(history, tools),
                                   "checks": {"action_matches_source": "pending_manual_review",
                                              "candidate_visibility": "pending_manual_review",
                                              "no_current_or_future_leakage": "pending_manual_review",
                                              "reasoning_removed": "pending_manual_review"}})
        super().consume(event)

    # Per-dataset worker state. Everything except the dedup index and the
    # length sample is keyed by dataset, so merging partial writers in config
    # order reproduces a single-process run.
    STATE_FIELDS = ("_tools", "_declared_by_trajectory", "_called_by_trajectory", "_states", "_issues",
                    "groups", "trajectories", "review_samples", "length_samples", "blocked",
                    "contamination", "contaminated_trajectories", "harness_tool_sets")

    def export_state(self):
        return {"summary": self._summary, "trajectory_ids": self._trajectory_ids,
                **{name: getattr(self, name) for name in self.STATE_FIELDS}}

    def absorb(self, state, partial_dir):
        self._summary.update(state["summary"])
        self._trajectory_ids.update(state["trajectory_ids"])
        for name in ("_tools", "_declared_by_trajectory", "_called_by_trajectory", "groups", "trajectories",
                     "review_samples", "blocked", "harness_tool_sets"):
            getattr(self, name).update(state[name])
        self._states += state["_states"]; self._issues += state["_issues"]
        self.contamination.extend(state["contamination"])
        self.contaminated_trajectories.update(state["contaminated_trajectories"])
        # Each worker keeps its own smallest-score states; the global smallest are among them.
        merged = self.length_samples + state["length_samples"]
        self.length_samples = heapq.nlargest(self.policy["length_sample_size"], merged)
        heapq.heapify(self.length_samples)
        for name in ("states.jsonl", "issues.jsonl"):
            source = Path(partial_dir) / name
            if source.exists():
                with source.open("rb") as src, (self.output / name).open("ab") as dst:
                    shutil.copyfileobj(src, dst)

    def finalize(self):
        result = super().finalize()
        duplicates = []
        duplicate_counts = Counter()
        index = defaultdict(list)
        for key, value in self.trajectories.items():
            meta = value.get("metadata", {})
            if "trajectory_hash" in meta and meta["trace_scope"] == "published_record":
                # Model labels and call IDs differ between republished copies of one trajectory.
                index[(meta["first_user_hash"], meta["trajectory_hash"])].append(key)
        removed = defaultdict(Counter)
        for keys in index.values():
            keys = sorted(set(keys))
            if len(keys) < 2: continue
            duplicates.append({"keep": keys[0], "duplicates": keys[1:],
                               "basis": "exact_normalized_messages_without_call_ids"})
            for key in keys[1:]:
                duplicate_counts[key[0]] += self.trajectories[key]["decision_count"]
                removed[key[0]].update(self.trajectories[key]["selection_counts"])
        groups = []
        for group in self.groups.values():
            g = {k: dict(v) if isinstance(v, Counter) else v for k, v in group.items()
                 if k not in {"trajectories", "candidate_counts", "history_lengths", "outcome_trajectories", "declared_names", "called_names"}}
            g["outcome_state_counts"] = g.pop("outcomes")
            g["outcome_trajectory_counts"] = {k: len(v) for k, v in group["outcome_trajectories"].items()}
            g["declared_not_called_count"] = len(group["declared_names"] - group["called_names"])
            g["called_not_declared_count"] = len(group["called_names"] - group["declared_names"])
            g["trajectory_count"] = len(group["trajectories"])
            g["candidate_count_p50"] = percentile(group["candidate_counts"], .5)
            g["candidate_count_p90"] = percentile(group["candidate_counts"], .9)
            g["single_tool_fraction"] = group["candidate_counts"][1] / group["states"]
            g["history_chars_p50"] = percentile(group["history_lengths"], .5)
            g["history_chars_p90"] = percentile(group["history_lengths"], .9)
            top = group["tools"].most_common(1)
            g["most_common_tool"] = top[0][0] if top else None
            g["most_common_tool_fraction"] = top[0][1] / sum(group["tools"].values()) if top else None
            groups.append(g)
        unknown_models = sorted({e[1] for e, g in self.groups.items()
                                 if e[1] is not None and e[1] not in self.policy["models"]})
        totals = defaultdict(Counter)
        for (dataset, _), value in self.trajectories.items(): totals[dataset].update(value["selection_counts"])
        pipeline = []
        for name in self.entries:
            counts = totals[name]
            adapter = self.entries[name]["review_adapter"]
            issues = self.blocked[name]
            coverage = ("excluded_by_scope" if adapter == "excluded" else "unresolved" if adapter == "unresolved"
                        else "failed" if issues["missing_input"] or issues["input_file_parse_error"]
                        else "partial" if issues["record_parse_error"] else "scanned")
            item = {"dataset": name, "coverage": coverage, "record_parse_errors": issues["record_parse_error"]}
            if coverage in {"scanned", "partial"}:
                after = lambda k: counts[k] - removed[name][k]
                item.update({k: counts[k] for k in ("all_states", "after_whitelist", "after_multi_tool",
                                                    "eligible_before_dedup", "confirmed", "conditional")},
                            eligible_after_dedup=after("eligible_before_dedup"),
                            confirmed_after_dedup=after("confirmed"), conditional_after_dedup=after("conditional"),
                            contamination_confirmed_after_dedup=after("eligible_contamination_confirmed"),
                            contamination_task_id_only_after_dedup=after("eligible_contamination_task_id_only"),
                            manual_quality_gate="pending")
            else:
                # Not scanned: no count is reported, so absence is never read as zero.
                item.update(eligible_after_dedup=None, confirmed_after_dedup=None, conditional_after_dedup=None)
            pipeline.append(item)
        report = {"groups": groups, "pipeline": pipeline, "unknown_models": unknown_models,
                  "harness_tool_sets": self.harness_tool_sets,
                  "blocked_datasets": {k: dict(v) for k, v in self.blocked.items() if v},
                  "duplicate_state_counts": dict(duplicate_counts),
                  "manual_review_status": "pending", "length_rendering": "JSON visible_history (images replaced; tool outputs and call arguments truncated head/tail) + candidate schemas + RESPOND; not a finalized training prompt"}
        for name, summary in result["datasets"].items():
            summary["review_status"] = "blocked_or_partial" if sum(self.blocked[name].values()) else "scanned"
            if self.entries[name]["review_adapter"] == "unresolved": summary["decision_state_count"] = None
            if self.entries[name]["review_adapter"] == "excluded": summary["review_status"] = "excluded_by_scope"
            entry = self.entries[name]
            if entry["review_adapter"] in {"otel", "mcp"} or "target_field" in entry:
                summary["returned_event_count"] = None
                summary["return_count_note"] = "request prefixes contain historical results; no repeated historical return events counted"
        self.write_json("summary.json", result)
        self.write_json("review_summary.json", report)
        same_task = defaultdict(set)
        for key, value in self.trajectories.items():
            meta = value.get("metadata", {})
            if "first_user_hash" in meta: same_task[meta["first_user_hash"]].add(key)
        # Same first request across datasets: candidates for manual review, not duplicates.
        cross = [sorted(v) for v in same_task.values() if len({k[0] for k in v}) > 1]
        self.write_json("duplicates.json", {"exact": duplicates,
            "cross_dataset_same_first_request": cross,
            "cross_dataset_pair_counts": dict(Counter(" | ".join(sorted({k[0] for k in v})) for v in cross))})
        self.write_json("contamination.json", self.contamination)
        with (self.output / "trajectories.jsonl").open("w") as handle:
            for value in self.trajectories.values(): handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        with (self.output / "review_samples.jsonl").open("w") as handle:
            for slots in self.review_samples.values():
                for slot in slots.values():
                    for sample in slot["states"]:
                        sample = {k: v for k, v in sample.items() if k != "score"}
                        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        lengths = []
        for _, _, sample in sorted(self.length_samples):
            lengths.append({k: v for k, v in sample.items() if k != "input"} |
                           {"tokens": len(self.tokenizer.encode(sample["input"], add_special_tokens=False))})
        counts = Counter(v["tokens"] for v in lengths)
        self.write_json("length_stats.json", {"tokenizer": str(self.tokenizer_path), "sample_count": len(lengths),
            "tool_output_truncation": self.truncation,
            "p50": percentile(counts, .5), "p90": percentile(counts, .9),
            # Measured after image/tool-output/argument truncation, before dropping earliest middle turns.
            "over_32768_fraction": sum(v["tokens"] > 32768 for v in lengths) / len(lengths) if lengths else None,
            "states_with_any_truncation_fraction":
                sum(bool(v["truncation_counts"]) for v in lengths) / len(lengths) if lengths else None,
            "states": lengths, "rendering": report["length_rendering"],
            "population": "eligible before dedup; confirmed and conditional; manual quality gate pending"})
        return result

    def write_json(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
