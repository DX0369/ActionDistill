"""Exact model mapping and explicit review eligibility (not training export)."""
from __future__ import annotations


def model_status(raw_model, policy):
    if raw_model is None:
        return {"status": "missing", "canonical": None}
    if not isinstance(raw_model, str) or not raw_model:
        return {"status": "invalid", "canonical": None}
    item = policy["models"].get(raw_model)
    if item is None:
        return {"status": "unmapped", "canonical": None}
    if not item["approved"]:
        return {"status": "excluded", "canonical": item["canonical"], "reason": item["reason"]}
    return {"status": "approved", "canonical": item["canonical"],
            "release_basis": item["release_basis"]}


def eligibility(event, policy):
    mapping = model_status(event.model, policy)
    meta = event.metadata
    reasons = []
    if mapping["status"] != "approved":
        reasons.append("teacher_" + mapping["status"])
    if len(event.candidate_tools) < 2:
        reasons.append("single_tool" if len(event.candidate_tools) == 1 else "empty_candidates")
    if event.action_kind == "UNKNOWN":
        reasons.append("unknown_action")
    if set(event.action_names) - set(event.candidate_tools):
        reasons.append("action_not_in_candidates")
    if meta.get("after_context_compaction"):
        # History before the boundary is the uncompacted text, not what the model saw.
        reasons.append("after_context_compaction")
    # Contamination is only marked (see ReviewWriter), never an exclusion.
    status = "excluded" if reasons else "confirmed" if event.candidate_set_complete else "conditional"
    return {"model_mapping": mapping, "eligibility": status, "exclusion_reasons": reasons}
