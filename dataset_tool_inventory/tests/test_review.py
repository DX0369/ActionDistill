import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from dataset_tool_inventory.adapters.review_common import (
    emit_trajectory,
    fingerprint,
    outcome,
)
from dataset_tool_inventory.adapters.review_formats import (
    native_claude,
    native_codex,
    scan,
    tau2_messages,
)
from dataset_tool_inventory.review_aggregate import ReviewWriter
from dataset_tool_inventory.review_policy import eligibility, model_status
from dataset_tool_inventory.schema import DecisionEvent, IssueEvent

TMP = PROJECT / "tmp"
DECIDER = Path("/data/wangsq/model/Mapika-decider-0.8b")


def decisions(messages, tools=None, **kw):
    ctx = {"source": "fixture", "model": "gpt-5.5", "harness": "fixture", **kw}
    return [e for e in emit_trajectory("fixture", "t", messages, tools, ctx) if isinstance(e, DecisionEvent)]


class ReviewBoundaryTests(unittest.TestCase):
    def test_tau2_test_paths_resolve_from_workspace_and_require_both_fields(self):
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory)
            (root / "split.json").write_text(json.dumps({"test": ["1"]}))
            (root / "tasks.json").write_text(json.dumps([
                {"id": "1", "user_scenario": "test scenario"},
            ]))
            for absolute in (False, True):
                spec = {key: str(root / name) if absolute else name
                        for key, name in (("split", "split.json"), ("tasks", "tasks.json"))}
                with self.subTest(absolute=absolute), patch(
                    "dataset_tool_inventory.review_aggregate.WORKSPACE_ROOT", root
                ):
                    result = ReviewWriter.load_test_tasks({"tau2_retail_test": spec})
                    self.assertEqual(result, {"1": fingerprint("test scenario")})
                    for missing in ("split", "tasks"):
                        incomplete = {key: value for key, value in spec.items() if key != missing}
                        with self.assertRaises(KeyError):
                            ReviewWriter.load_test_tasks({"tau2_retail_test": incomplete})

    def test_historical_candidates_exclude_current_and_future_calls(self):
        call = lambda n: {"role": "assistant", "tool_calls": [{"function": {"name": n}}]}
        ds = decisions([call("read"), call("bash"), call("read")])
        self.assertEqual([d.candidate_tools for d in ds], [(), ("read",), ("read", "bash")])
        self.assertEqual(ds[-1].metadata["candidate_level"], "I")
        self.assertFalse(ds[-1].candidate_set_complete)

    def test_distinct_tools_and_repeated_calls_have_different_labels(self):
        tools = [{"name": "read", "parameters": {}}, {"name": "bash", "parameters": {}}]
        ds = decisions([{"role": "assistant", "tool_calls": [{"function": {"name": n}} for n in names]}
                        for names in [["read"], ["read", "read"], ["read", "bash"]]], tools)
        self.assertEqual([d.action_kind for d in ds], ["TOOL", "TOOL", "TOOL_SET"])
        self.assertEqual(ds[1].metadata["call_counts"], {"read": 2})

    def test_visible_history_strips_reasoning_and_preserves_think_tool(self):
        ds = decisions([{"role": "assistant", "content": "<think>SECRET</think>visible",
                         "reasoning_content": "HIDDEN", "tool_calls": [{"function": {
                             "name": "think", "arguments": {"thought": "user-approved tool input"}}}]},
                        {"role": "assistant", "content": "done"}], strip_think=True)
        history = ds[1].metadata["_history"]
        self.assertNotIn("SECRET", json.dumps(history))
        self.assertNotIn("HIDDEN", json.dumps(history))
        self.assertEqual(history[0]["tool_calls"][0]["function"]["arguments"]["thought"], "user-approved tool input")
        self.assertEqual(ds[1].action_kind, "RESPOND")
        self.assertEqual(ds[1].metadata["history_chars"], len(json.dumps(history, ensure_ascii=False)))

    def test_claude_fragments_freeze_history_before_interleaved_result(self):
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p = Path(d) / "s.jsonl"
            rows = [{"type": "user", "sessionId": "s", "message": {"role": "user", "content": [{"type": "text", "text": "question"}]}},
                    {"type": "assistant", "sessionId": "s", "message": {"id": "m", "role": "assistant", "model": "claude-fable-5", "content": [{"type": "thinking", "thinking": "secret"}, {"type": "tool_use", "id": "a", "name": "Read", "input": {}}]}},
                    {"type": "user", "sessionId": "s", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "FUTURE_RESULT"}]}},
                    {"type": "assistant", "sessionId": "s", "message": {"id": "m", "role": "assistant", "model": "claude-fable-5", "content": [{"type": "tool_use", "id": "b", "name": "Bash", "input": {}}]}}]
            p.write_text("\n".join(json.dumps(r) for r in rows))
            _, messages, _ = native_claude(p)
            ds = decisions(messages, [{"name": "Read"}, {"name": "Bash"}])
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds[0].action_kind, "TOOL_SET")
            self.assertNotIn("FUTURE_RESULT", json.dumps(ds[0].metadata["_history"]))

    def test_gpt_prefix_target_excludes_old_action_and_requires_target(self):
        entry = {"review_adapter": "openai", "storage": "jsonl", "glob": "*.jsonl", "harness": "codex",
                 "trajectory_field": "source_trajectory_id", "model_field": "teacher_model",
                 "tools_field": "tools", "messages_field": "messages", "target_field": "target_message_index"}
        item = {"source_trajectory_id": "t", "assistant_step": 0, "teacher_model": "gpt-5.6-sol", "tools": [], "target_message_index": 2,
                "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "tool_calls": [{"function": {"name": "OLD"}}]},
                             {"role": "assistant", "tool_calls": [{"function": {"name": "CURRENT"}}]}]}
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p = Path(d) / "x.jsonl"; p.write_text(json.dumps(item))
            ds = [e for e in scan("test", Path(d), entry) if isinstance(e, DecisionEvent)]
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds[0].action_names, ("CURRENT",))
            del item["target_message_index"]; p.write_text(json.dumps(item))
            events = list(scan("test", Path(d), entry))
            self.assertTrue(any(isinstance(e, IssueEvent) and "target_message_index" in e.message for e in events))
            item["target_message_index"] = 2; item["source_trajectory_id"] = None
            p.write_text(json.dumps(item))
            self.assertTrue(any(isinstance(e, IssueEvent) and "trajectory identifier" in e.message
                                for e in scan("test", Path(d), entry)))

    def test_outcome_is_explicit_and_false_is_not_missing(self):
        spec = {"field": "resolved", "values": [{"value": False, "outcome": "failure"}, {"value": True, "outcome": "success"}]}
        self.assertEqual(outcome({"resolved": False}, spec), "failure")
        with self.assertRaises(ValueError): outcome({}, spec)
        with self.assertRaises(ValueError): outcome({"resolved": 0}, spec)

    def test_readme_snapshot_preserves_parameters_and_recovered_level(self):
        tools = [{"name": "read", "description": "Read a file", "parameters": {
            "type": "object", "properties": {
                "path": {"type": "string"},
                "options": {"type": "object", "properties": {
                    "recursive": {"type": "boolean", "default": False}}}},
            "required": ["path"], "additionalProperties": False}},
            {"name": "ping", "parameters": {"type": "object", "properties": {}, "required": []}}]
        messages = [{"role": "user", "content": "Read the file"},
                    {"role": "assistant", "tool_calls": [{"id": "c", "function": {
                        "name": "read", "arguments": {"path": "a.txt"}}}]}]
        rows = [{"type": "session_meta", "payload": {"id": "s"}},
                {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
                {"type": "response_item", "payload": {"type": "message", **messages[0]}},
                {"type": "response_item", "payload": {"type": "function_call", "call_id": "c",
                    "name": "read", "arguments": '{"path":"a.txt"}'}}]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            (root / "README.md").write_text("```json\n" + json.dumps([
                {"type": "function", "function": tools[0]}, tools[1]]) + "\n```\n")
            (root / "record.jsonl").write_text(json.dumps({"messages": messages}) + "\n")
            (root / "native.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            base = {"harness": "codex", "model_constant": "gpt-5.5", "candidate_complete": False,
                    "candidate_source": "README snapshot",
                    "snapshot_json_block": {"file": "README.md", "index": 0}}
            for family, name in (("openai", "record.jsonl"), ("native_codex", "native.jsonl")):
                with self.subTest(family=family):
                    entry = dict(base, review_adapter=family, glob=name,
                                 storage="jsonl", messages_field="messages")
                    events = list(scan("fixture", root, entry))
                    self.assertFalse(any(isinstance(e, IssueEvent) for e in events))
                    ds = [e for e in events if isinstance(e, DecisionEvent)]
                    self.assertEqual(len(ds), 1)
                    self.assertEqual(ds[0].metadata["_candidate_definitions"], tools)
                    self.assertEqual(ds[0].metadata["candidate_level"], "N")
                    self.assertEqual(ds[0].metadata["candidate_source"], "README snapshot")
                    self.assertFalse(ds[0].candidate_set_complete)
                    self.assertEqual(ds[0].candidate_tools, ("read", "ping"))
                    self.assertEqual(ds[0].action_names, ("read",))
                    self.assertEqual(ds[0].metadata["_history"], messages[:1])

    def test_readme_snapshot_invalid_declarations_are_reported(self):
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            root = Path(d)
            (root / "record.jsonl").write_text(json.dumps({"messages": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "done"}]}))
            entry = {"review_adapter": "openai", "storage": "jsonl", "glob": "record.jsonl",
                     "harness": "fixture", "messages_field": "messages",
                     "snapshot_json_block": {"file": "README.md", "index": 0}}
            for tool, error in (({"description": "No name"}, "missing name"),
                                ({"name": "read", "parameters": None}, "parameters must be an object"),
                                ({"name": "read", "parameters": []}, "parameters must be an object")):
                with self.subTest(tool=tool):
                    (root / "README.md").write_text("```json\n" + json.dumps([tool]) + "\n```\n")
                    events = list(scan("fixture", root, entry))
                    self.assertEqual(len(events), 1)
                    self.assertIsInstance(events[0], IssueEvent)
                    self.assertEqual(events[0].issue, "record_parse_error")
                    self.assertIn(error, events[0].message)

    def test_codex_consecutive_calls_share_one_pre_action_history(self):
        rows = [{"type": "session_meta", "payload": {"id": "session"}},
                {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}},
                {"type": "event_msg", "payload": {"type": "user_message", "message": "MIRROR"}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "analysis", "content": [{"type": "output_text", "text": "PRIVATE"}]}},
                {"type": "response_item", "payload": {"type": "function_call", "name": "read", "call_id": "read", "arguments": "{}"}},
                {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "bash", "call_id": "bash", "input": "pwd"}},
                *[{"type": "response_item", "payload": {"type": "function_call_output", "call_id": n, "output": n+" result"}} for n in ["read", "bash"]],
                {"type": "response_item", "payload": {"type": "message", "role": "assistant", "channel": "final", "content": [{"type": "output_text", "text": "done"}]}}]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p = Path(d) / "s.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows))
            _, messages, _, session_tools = native_codex(p)
            ds = decisions(messages, [{"name": "read"}, {"name": "bash"}])
        self.assertIsNone(session_tools)
        self.assertEqual([e.action_kind for e in ds], ["TOOL_SET", "RESPOND"])
        self.assertEqual(ds[0].action_names, ("read", "bash"))
        self.assertEqual(ds[0].metadata["call_count"], 2)
        self.assertEqual(ds[0].metadata["_history"], [{"role":"user","content":[{"type":"input_text","text":"q"}]}])
        self.assertEqual([m['tool_call_id'] for m in ds[-1].metadata['_history'] if m['role']=='tool'], ['read','bash'])
        self.assertNotIn("PRIVATE", json.dumps(ds[-1].metadata["_history"]))
        self.assertNotIn("MIRROR", json.dumps(ds[-1].metadata["_history"]))

    def test_codex_calls_do_not_merge_across_visible_boundaries(self):
        header=[{"type":"session_meta","payload":{"id":"s"}}, {"type":"turn_context","payload":{"model":"gpt-5.5"}}]
        call=lambda cid:{"type":"response_item","payload":{"type":"function_call","name":"read","call_id":cid,"arguments":"{}"}}
        boundaries=[None, {"type":"response_item","payload":{"type":"function_call_output","call_id":"a","output":"result"}},
                    {"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"explain"}]}},
                    {"type":"turn_context","payload":{"model":"gpt-5.6-sol"}}]
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            for boundary in boundaries:
                with self.subTest(boundary=boundary):
                    p=Path(d)/'s.jsonl';p.write_text('\n'.join(json.dumps(r) for r in header+[call('a')]+([] if boundary is None else [boundary])+[call('b')]))
                    _,messages,_,_=native_codex(p)
                    ds=decisions(messages,[{'name':'read'},{'name':'bash'}])
                    calls=[e for e in ds if e.action_kind=='TOOL']
                    self.assertEqual(len(calls),1 if boundary is None else 2)
                    self.assertEqual(sum(e.metadata['call_count'] for e in calls),2)

    def test_exact_models_official_fp8_and_small_qwen(self):
        policy = json.loads((PROJECT / "configs/review_policy.json").read_text())
        self.assertEqual(model_status("zai-org/GLM-5.1-FP8", policy)["status"], "approved")
        self.assertEqual(model_status("Qwen/Qwen3.8-27B", policy)["status"], "excluded")
        self.assertEqual(model_status("glm-5.1-FP8-unofficial", policy)["status"], "unmapped")

    def test_otel_literal_attribute_keys_and_separate_system_parts(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": "claude-opus-4-5",
                 "gen_ai.input.messages": [{"role": "user", "parts": [{"type": "text", "content": "q"}]}],
                 "gen_ai.output.messages": [{"role": "assistant", "parts": [{"type": "tool_call", "id": "c", "name": "read", "arguments": {}}]}],
                 "gen_ai.tool.definitions": [{"name": "read", "parameters": {}}, {"name": "bash", "parameters": {}}],
                 "gen_ai.system_instructions": [{"type": "text", "content": "SYSTEM_POLICY"}]}
        entry = {"review_adapter": "otel", "storage": "jsonl", "glob": "*.jsonl", "source_record_id_field": "session_id", "harness": "OTel", "candidate_complete": True}
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            row = json.dumps({"session_id": "s", "spans": [{"attributes": attrs}]})
            (Path(d) / "r.jsonl").write_text(row + "\n" + row)
            ds = [e for e in scan("test", Path(d), entry) if isinstance(e, DecisionEvent)]
        self.assertEqual(ds[0].action_names, ("read",))
        self.assertEqual(ds[0].metadata["_history"][0], {"role": "system", "content": "SYSTEM_POLICY"})
        self.assertEqual(len({e.trajectory_id for e in ds}), 2)
        self.assertEqual([e.metadata["source_record_id"] for e in ds], ["s", "s"])
        self.assertTrue(all(e.metadata["candidate_level"] == "S" and e.candidate_set_complete for e in ds))

    def test_codex_builtin_items_and_recovered_session_tool_schema(self):
        rows = [{"type": "session_meta", "payload": {"id": "s"}},
                {"type": "turn_context", "payload": {"model": "gpt-5.5"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}},
                {"type": "response_item", "payload": {"type": "web_search_call", "status": "completed", "action": {"query": "x"}}},
                {"type": "response_item", "payload": {"type": "tool_search_call", "call_id": "c", "arguments": {"query": "y"}}},
                {"type": "response_item", "payload": {"type": "tool_search_output", "call_id": "c", "tools": [{"name": "found"}]}},
                {"type": "compacted", "payload": {}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}},
                {"type": "response_item", "payload": {"type": "tool_schema", "name": "web_search", "schema": {"description": "d", "parameters": {}}}},
                {"type": "response_item", "payload": {"type": "tool_schema", "name": "tool_search", "schema": {"description": "d", "parameters": {}}}}]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p = Path(d) / "s.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows))
            (Path(d) / "README.md").write_text('```json\n[{"name":"snapshot_only","parameters":{}}]\n```\n')
            entry = {"review_adapter": "native_codex", "glob": "s.jsonl", "harness": "codex",
                     "candidate_source": "README snapshot",
                     "snapshot_json_block": {"file": "README.md", "index": 0}}
            ds = [e for e in scan("fixture", Path(d), entry) if isinstance(e, DecisionEvent)]
            tools = ds[0].metadata["_candidate_definitions"]
            del entry["snapshot_json_block"]
            without_snapshot = [e for e in scan("fixture", Path(d), entry) if isinstance(e, DecisionEvent)]
        self.assertEqual(ds, without_snapshot)
        self.assertEqual(tools, [{"name": name, "description": "d", "parameters": {}}
                                 for name in ("web_search", "tool_search")])
        self.assertTrue(all(d.metadata["candidate_level"] == "N" and not d.candidate_set_complete for d in ds))
        self.assertEqual(ds[0].metadata["candidate_source"], "recovered session tool_schema items (trajectory-level)")
        policy = {"models": {"gpt-5.5": {"approved": True, "canonical": "x", "release_basis": "x"}}}
        self.assertEqual(eligibility(ds[0], policy)["eligibility"], "conditional")
        self.assertEqual([d.action_names for d in ds], [("web_search",), ("tool_search",), ()])
        self.assertEqual([d.metadata["after_context_compaction"] for d in ds], [False, False, True])
        self.assertIn("after_context_compaction", eligibility(ds[2], {"models": {"gpt-5.5": {"approved": True, "canonical": "x", "release_basis": "x"}}})["exclusion_reasons"])

    def test_tau2_drops_user_device_tools_and_orchestrator_greeting(self):
        record = {"_info": {"environment_info": {"policy": "P"}}, "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": None, "tool_calls": [{"id": "u", "name": "check_status_bar", "arguments": {}}]},
            {"role": "tool", "id": "u", "requestor": "user", "content": "DEVICE_ONLY"},
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "name": "get_user", "arguments": {}}]},
            {"role": "tool", "id": "a", "requestor": "assistant", "content": "ok"},
            {"role": "assistant", "content": "done"}]}
        events = list(emit_trajectory("t", "t", tau2_messages(record), [{"name": "get_user"}, {"name": "x"}],
                                      {"source": "f", "model": "m", "harness": "h"}))
        ds = [e for e in events if isinstance(e, DecisionEvent)]
        self.assertEqual([d.action_kind for d in ds], ["TOOL", "RESPOND"])
        self.assertNotIn("DEVICE_ONLY", json.dumps(ds[-1].metadata["_history"]))
        self.assertEqual(events[-1].metadata["non_teacher_steps_skipped"], 1)

    def test_hermes_tool_call_quoted_in_reasoning_is_not_a_call(self):
        from dataset_tool_inventory.adapters.review_formats import hermes_messages
        text = '<think>maybe <tool_call>{"name": "bad", "arguments": {}}</tool_call></think>answer'
        m = hermes_messages([{"from": "gpt", "value": text}])[0]
        self.assertNotIn("tool_calls", m)
        self.assertEqual(m["content"], "answer")

    def test_toolace_names_with_parentheses_and_custom_format_is_not_blocking(self):
        tools = [{"name": "User Feed (Video Posts) V2", "parameters": {}}, {"name": "Other", "parameters": {}}]
        records = [{"system": "Here is a list of functions in JSON format that you can invoke:\n" + json.dumps(tools),
                    "conversations": [{"from": "user", "value": "q"},
                                      {"from": "assistant", "value": '[User Feed (Video Posts) V2(id="1"), Other(x=1)]'}]},
                   {"system": "Here are the tools you can use:\n<tool_name>T</tool_name>", "conversations": []}]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            p = Path(d) / "data.json"; p.write_text(json.dumps(records))
            events = list(scan("ToolACE", p, {"review_adapter": "toolace", "storage": "json_array", "glob": "*", "harness": "toolace", "candidate_complete": True}))
        ds = [e for e in events if isinstance(e, DecisionEvent)]
        self.assertEqual(ds[0].action_names, ("User Feed (Video Posts) V2", "Other"))
        self.assertIn("custom_format_tool_declaration", [e.issue for e in events if isinstance(e, IssueEvent)])
        self.assertNotIn("record_parse_error", [e.issue for e in events if isinstance(e, IssueEvent)])

    def test_late_parse_error_emits_no_partial_trajectory(self):
        item = {"tools": [], "messages": [{"role": "assistant", "content": "first"}, {"role": "bogus"}]}
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            (Path(d) / "x.jsonl").write_text(json.dumps(item))
            events = list(scan("t", Path(d), {"review_adapter": "openai", "storage": "jsonl", "glob": "*.jsonl", "harness": "h",
                                              "model_constant": "gpt-5.5", "messages_field": "messages", "tools_field": "tools"}))
        self.assertEqual([type(e).__name__ for e in events], ["IssueEvent"])
        self.assertEqual(events[0].issue, "record_parse_error")

    def test_writer_marks_contamination_dedupes_copies_and_reports_coverage(self):
        policy = json.loads((PROJECT / "configs/review_policy.json").read_text())
        policy["length_sample_size"] = 5
        canonical = json.loads((PROJECT.parents[1] / policy["tau2_retail_test"]["tasks"]).read_text())
        scenario = {str(t["id"]): t["user_scenario"] for t in canonical}
        tools = [{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]
        def trajectory(dataset, task_hash, task="5", call_id="x", user="."):
            msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": None,
                     "tool_calls": [{"id": call_id, "function": {"name": "a"}}]}]
            ctx = {"source": "f", "model": "gpt-5.5", "harness": "h", "domain": "retail", "task_id": task,
                   "task_scenario_hash": task_hash}
            return list(emit_trajectory(dataset, "t", msgs, tools, ctx))
        entries = [{"name": "tau", "review_adapter": "tau2", "check_tau2_retail": True},
                   {"name": "copy_a", "review_adapter": "openai"},
                   {"name": "copy_b", "review_adapter": "openai"},
                   {"name": "other", "review_adapter": "openai", "check_tau2_retail": True},
                   {"name": "iso", "review_adapter": "unresolved"}]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            writer = ReviewWriter(Path(d) / "out", [e["name"] for e in entries], policy, entries, DECIDER)
            for e in (trajectory("tau", fingerprint(scenario["5"])) + trajectory("copy_a", None, user="q", call_id="y")
                      + trajectory("copy_b", None, user="q", call_id="z") + trajectory("other", None, user="r")):
                writer.consume(e)
            writer.consume(IssueEvent("iso", "unresolved_dataset", "r"))
            writer.finalize()
            report = json.loads((Path(d) / "out/review_summary.json").read_text())
            contamination = json.loads((Path(d) / "out/contamination.json").read_text())
        pipeline = {p["dataset"]: p for p in report["pipeline"]}
        # Contamination is a mark, never an exclusion; "." in history does not match anything.
        self.assertEqual(pipeline["tau"]["eligible_before_dedup"], 1)
        self.assertEqual(pipeline["tau"]["contamination_confirmed_after_dedup"], 1)
        self.assertEqual([(c["dataset"], c["status"]) for c in contamination], [("tau", "confirmed")])
        # Copies differing only in call ID are deduplicated; other formats stay "unknown", not contaminated.
        self.assertEqual(pipeline["copy_a"]["eligible_after_dedup"] + pipeline["copy_b"]["eligible_after_dedup"], 1)
        self.assertEqual(pipeline["other"]["eligible_after_dedup"], 1)
        self.assertEqual(pipeline["iso"]["coverage"], "unresolved")
        self.assertIsNone(pipeline["iso"]["eligible_after_dedup"])

    def test_history_truncation_images_tool_outputs_and_arguments(self):
        from dataset_tool_inventory.review_aggregate import truncate_history
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(DECIDER, local_files_only=True)
        long = " ".join(f"w{i}" for i in range(5000))
        history = [{"role": "user", "content": [{"type": "input_text", "text": long},
                                                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]},
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"function": {"name": "apply_patch", "arguments": "BEGIN " + long + " END"}}]},
                   {"role": "tool", "content": "HEAD " + long + " TAIL"},
                   {"role": "tool", "content": "short"}]
        shown, counts = truncate_history(history, tok, 20, 20)
        self.assertEqual(dict(counts), {"images_replaced": 1, "tool_outputs_truncated": 1,
                                        "tool_call_arguments_truncated": 1})
        self.assertEqual(shown[0]["content"], [{"type": "input_text", "text": long},
                                               {"type": "text", "text": "[image omitted]"}])  # user text untouched
        args = shown[1]["tool_calls"][0]["function"]["arguments"]
        self.assertTrue(args.startswith("BEGIN") and args.endswith("END") and "tokens omitted" in args)
        self.assertTrue(shown[2]["content"].startswith("HEAD") and shown[2]["content"].endswith("TAIL"))
        self.assertLessEqual(len(tok.encode(shown[2]["content"], add_special_tokens=False)), 60)
        self.assertEqual(shown[3]["content"], "short")
        self.assertEqual(history[1]["tool_calls"][0]["function"]["arguments"], "BEGIN " + long + " END")  # input not mutated

    def test_contamination_missing_identity_is_unknown_but_non_matches_clear(self):
        from dataclasses import replace
        policy=json.loads((PROJECT/'configs/review_policy.json').read_text());policy['length_sample_size']=1
        cases=[(None,'5','unknown'),('retail',None,'unknown'),('airline','5','clear'),('retail','not-a-test-task','clear')]
        entries=[{'name':str(i),'review_adapter':'tau2','check_tau2_retail':True} for i in range(len(cases))]
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            writer=ReviewWriter(Path(d)/'out',[e['name'] for e in entries],policy,entries,DECIDER)
            for i,(domain,task,_) in enumerate(cases):
                event=decisions([{'role':'user','content':'request'},{'role':'assistant','content':'reply'}],
                                [{'name':'a'},{'name':'b'}],domain=domain,task_id=task)[0]
                writer.consume(replace(event,dataset=str(i)))
            with (Path(d)/'out/states.jsonl').open() as handle:
                actual=[json.loads(line)['metadata']['eval_contamination'] for line in handle]
            self.assertEqual(actual,[c[2] for c in cases])
    def test_parallel_workers_reproduce_single_process_outputs(self):
        import filecmp
        import subprocess
        tools = [{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]
        def record(user, call_id):
            return {"tools": tools, "messages": [{"role": "user", "content": user},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "function": {"name": "a"}}]},
                    {"role": "tool", "tool_call_id": call_id, "content": "x"}, {"role": "assistant", "content": "done"}]}
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            d = Path(d); data = d / "data"
            entries = []
            # ds2 repeats a ds1 trajectory with another call ID: a cross-dataset duplicate.
            for name, rows in {"ds1": [record("q1", "c1"), record("q2", "c2")],
                               "ds2": [record("q1", "z9"), record("q3", "c3")],
                               "ds3": [record(f"q{i}", f"c{i}") for i in range(4, 9)]}.items():
                (data / name).mkdir(parents=True)
                (data / name / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                entries.append({"name": name, "path": name, "review_adapter": "openai", "storage": "jsonl",
                                "glob": "*.jsonl", "harness": "h", "model_constant": "gpt-5.5",
                                "messages_field": "messages", "tools_field": "tools", "candidate_complete": True})
            (d / "config.json").write_text(json.dumps({"datasets": entries}))
            policy = json.loads((PROJECT / "configs/review_policy.json").read_text()); policy["length_sample_size"] = 3
            (d / "policy.json").write_text(json.dumps(policy))
            outs = []
            for workers in (1, 2):
                out = d / f"out{workers}"; outs.append(out)
                subprocess.run([sys.executable, str(PROJECT / "scripts/review_datasets.py"), "--data-root", str(data),
                                "--config", str(d / "config.json"), "--policy", str(d / "policy.json"),
                                "--tokenizer", str(DECIDER), "--output", str(out), "--workers", str(workers)],
                               check=True, capture_output=True)
            files = sorted(p.name for p in outs[0].iterdir() if p.is_file())
            match, mismatch, errors = filecmp.cmpfiles(outs[0], outs[1], files, shallow=False)
            self.assertEqual((mismatch, errors), ([], []))
            self.assertFalse((outs[1] / "partials").exists())
            duplicates = json.loads((outs[1] / "duplicates.json").read_text())["exact"]
            self.assertEqual(len(duplicates), 1)

    def test_anthropic_null_content_is_an_unknown_step_not_a_record_failure(self):
        from dataset_tool_inventory.adapters.review_formats import anthropic_messages
        messages = anthropic_messages([{"role": "user", "content": "q"}, {"role": "assistant", "content": None},
                                       {"role": "assistant", "content": [{"type": "text", "text": "done"}]}])
        self.assertEqual([d.action_kind for d in decisions(messages, [{"name": "a"}, {"name": "b"}])], ["UNKNOWN", "RESPOND"])

    def test_harness_candidates_use_dataset_union_and_only_prior_session_tools(self):
        call = lambda n: {"role": "assistant", "tool_calls": [{"function": {"name": n}}]}
        sets = {("cc", None): ["Bash", "Read"]}
        ds = decisions([call("Bash"), call("mcp_x"), call("Read"), call("mcp_y")], harness="cc", harness_tool_sets=sets)
        self.assertEqual([d.candidate_tools for d in ds],
                         [("Bash", "Read"), ("Bash", "Read"), ("Bash", "Read", "mcp_x"), ("Bash", "Read", "mcp_x")])
        self.assertEqual({d.metadata["candidate_level"] for d in ds}, {"H"})
        self.assertFalse(any(d.candidate_set_complete for d in ds))
        self.assertNotIn("mcp_y", ds[-1].candidate_tools)  # the current call never enters its own candidates

    def test_harness_union_requires_min_trajectories(self):
        from dataset_tool_inventory.adapters.review_formats import harness_tool_sets
        call = lambda n: {"role": "assistant", "content": None, "tool_calls": [{"id": n, "function": {"name": n}}]}
        rows = [{"messages": [{"role": "user", "content": "q"}, call("Bash")] + ([call("rare")] if i == 0 else [])} for i in range(3)]
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            (Path(d) / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            entry = {"review_adapter": "openai", "storage": "jsonl", "glob": "*.jsonl", "harness": "h",
                     "model_constant": "gpt-5.5", "messages_field": "messages"}
            sets = harness_tool_sets("t", Path(d), entry, None, 3)
            ds = [e for e in scan("t", Path(d), dict(entry, harness_tool_sets=sets)) if isinstance(e, DecisionEvent)]
        self.assertEqual(sets, {("h", None): ["Bash"]})
        self.assertEqual(ds[0].candidate_tools, ("Bash",))
        self.assertEqual(ds[1].candidate_tools, ("Bash",))  # "rare" (1 trajectory) is not offered before its use

    def test_tau2_results_without_tool_defs_use_official_domain_tools(self):
        from dataset_tool_inventory.adapters.review_formats import record_events
        official = PROJECT / "configs/tau2_agent_tools.json"
        record = {"_info": {"environment_info": {"policy": "P", "tool_defs": None, "domain_name": "retail"}},
                  "_task_scenario_hash": "h", "task_id": "1", "messages": [
                      {"role": "user", "content": "q"},
                      {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "name": "get_user_details", "arguments": {}}]}]}
        entry = {"review_adapter": "tau2", "harness": "h", "model_constant": "m", "task_field": "task_id",
                 "domain_field": "_info.environment_info.domain_name", "tau2_official_tools": str(official)}
        ds = [e for e in record_events("t", PROJECT, entry, "tau2", PROJECT / "x.json", 1, record) if isinstance(e, DecisionEvent)]
        self.assertEqual(len(ds[0].candidate_tools), 16)
        self.assertIn("get_user_details", ds[0].candidate_tools)
        self.assertEqual(ds[0].metadata["candidate_level"], "S")

    def _claude_session(self, rows):
        TMP.mkdir(exist_ok=True)
        d = tempfile.mkdtemp(dir=TMP); path = Path(d) / "s.jsonl"
        path.write_text("\n".join(json.dumps({"sessionId": "s", **r}) for r in rows))
        try:
            _, messages, _ = native_claude(path)
        finally:
            import shutil
            shutil.rmtree(d)
        return decisions(messages, [{"name": "Bash"}, {"name": "Read"}])

    def test_claude_edited_resend_hides_abandoned_user_turn(self):
        user = lambda u, p, text: {"type": "user", "uuid": u, "parentUuid": p, "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        asst = lambda u, p, mid, block: {"type": "assistant", "uuid": u, "parentUuid": p, "message": {"id": mid, "role": "assistant", "model": "claude-fable-5", "content": [block]}}
        ds = self._claude_session([
            user("u1", None, "task"),
            asst("u2", "u1", "a", {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}),
            {"type": "user", "uuid": "u3", "parentUuid": "u2", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "R1"}]}},
            user("u4", "u3", "ABANDONED"),
            user("u5", "u3", "resent instruction"),
            asst("u6", "u5", "b", {"type": "tool_use", "id": "t2", "name": "Read", "input": {}})])
        history = json.dumps(ds[-1].metadata["_history"])
        self.assertNotIn("ABANDONED", history)
        self.assertIn("resent instruction", history)
        self.assertIn("R1", history)
        self.assertEqual(ds[-1].metadata["abandoned_branch_messages_removed"], 1)

    def test_claude_parallel_tool_results_stay_visible(self):
        frag = lambda u, p, t: {"type": "assistant", "uuid": u, "parentUuid": p, "message": {"id": "a", "role": "assistant", "model": "claude-fable-5", "content": [{"type": "tool_use", "id": t, "name": "Bash", "input": {}}]}}
        result = lambda u, p, t, text: {"type": "user", "uuid": u, "parentUuid": p, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": t, "content": text}]}}
        ds = self._claude_session([
            {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}},
            frag("u2", "u1", "t1"), frag("u3", "u2", "t2"),
            result("u4", "u2", "t1", "FIRST"), result("u5", "u3", "t2", "SECOND"),
            {"type": "assistant", "uuid": "u6", "parentUuid": "u5", "message": {"id": "b", "role": "assistant", "model": "claude-fable-5", "content": [{"type": "text", "text": "done"}]}}])
        history = json.dumps(ds[-1].metadata["_history"])
        self.assertIn("FIRST", history); self.assertIn("SECOND", history)
        self.assertEqual(ds[-1].metadata["abandoned_branch_messages_removed"], 0)


if __name__ == "__main__": unittest.main()
