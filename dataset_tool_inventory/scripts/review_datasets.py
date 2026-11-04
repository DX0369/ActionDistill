#!/usr/bin/env python3
"""Local-only review of teacher provenance, candidates and action boundaries."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_tool_inventory.adapters.review_formats import harness_tool_sets, scan
from dataset_tool_inventory.review_aggregate import ReviewWriter
from dataset_tool_inventory.schema import DecisionEvent, IssueEvent


def profile(entry, data_root, events):
    root = data_root / entry["path"]
    lines = [f"# {entry['name']} 格式档案", "", "仅记录本轮实际探查证据；不等同于人工质量门槛通过。", "",
             f"- 格式族：`{entry['review_adapter']}`。", f"- 来源：`{entry['path']}`。",
             f"- harness：`{entry['harness']}`。", f"- 记录单位/读取方式：`{entry.get('storage', 'native session file')}`。",
             f"- 配置字段：`{json.dumps({k:v for k,v in entry.items() if k not in {'name','path','notes','harness_tool_sets'}}, ensure_ascii=False)}`。",
             f"- 已知限制：{entry.get('notes', entry.get('reason', '待抽样核对'))}", "", "## 有界探查证据", ""]
    if root.exists():
        suffixes = {}; sample_paths = []
        for path in root.rglob("*") if root.is_dir() else [root]:
            if not path.is_file() or ".cache" in path.parts or ".git" in path.parts: continue
            suffix = "".join(path.suffixes) or "(no extension)"
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
            if len(sample_paths) < 3: sample_paths.append({"path": str(path), "bytes": path.stat().st_size})
        lines += [f"- 文件类型计数：`{json.dumps(suffixes)}`。", f"- 文件大小示例：`{json.dumps(sample_paths, ensure_ascii=False)}`。"]
    for event in events:
        if isinstance(event, DecisionEvent):
            lines.append(f"- `{event.source}`：model=`{event.model}`，action=`{event.action_kind}`，候选={len(event.candidate_tools)}，等级={event.metadata['candidate_level']}，完整={event.candidate_set_complete}。")
        elif isinstance(event, IssueEvent): lines.append(f"- `{event.source}`：issue=`{event.issue}`，{event.message}。")
    return "\n".join(lines) + "\n"


def review_dataset(entry, data_root, policy, tokenizer, output, limit):
    """Scan one dataset into its own partial writer; returns the mergeable state."""
    partial = output / "partials" / entry["name"]
    writer = ReviewWriter(partial, [entry["name"]], policy, [entry], tokenizer, policy["seed"])
    evidence = []
    root = data_root / entry["path"]
    if not root.exists():
        event = IssueEvent(entry["name"], "missing_input", str(root), source=str(root))
        writer.consume(event); evidence.append(event)
    else:
        try:
            if entry.get("candidate_reconstruction") == "harness_union":
                sets = harness_tool_sets(entry["name"], root, entry, limit, policy["harness_tool_min_trajectories"])
                entry = dict(entry, harness_tool_sets=sets)
                writer.harness_tool_sets[entry["name"]] = [
                    {"harness": h, "domain": d, "tools": tools} for (h, d), tools in sets.items()]
            for event in scan(entry["name"], root, entry, limit):
                if len(evidence) < 3 and isinstance(event, (DecisionEvent, IssueEvent)): evidence.append(event)
                writer.consume(event)
        except (OSError, ValueError) as exc:
            event = IssueEvent(entry["name"], "input_file_parse_error", str(exc), source=str(root))
            writer.consume(event); evidence.append(event)
    (output / "profiles" / f"{entry['name']}.md").write_text(profile(entry, data_root, evidence))
    return entry["name"], str(partial), writer.export_state()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=WORKSPACE_ROOT / "data")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/review_datasets.json")
    p.add_argument("--policy", type=Path, default=PROJECT_ROOT / "configs/review_policy.json")
    p.add_argument("--tokenizer", type=Path, default=Path("/data/wangsq/model/Mapika-decider-0.8b"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", action="append")
    p.add_argument("--limit-trajectories", type=int, help="Bounded smoke only; results are partial")
    p.add_argument("--workers", type=int, required=True, help="Datasets scanned in parallel processes")
    args = p.parse_args()
    if args.limit_trajectories is not None and args.limit_trajectories <= 0: p.error("limit must be positive")
    if args.workers <= 0: p.error("workers must be positive")
    entries = json.loads(args.config.read_text())["datasets"]
    policy = json.loads(args.policy.read_text())
    known = {e["name"] for e in entries}
    if args.dataset:
        if set(args.dataset) - known: p.error("unknown dataset selection")
        entries = [e for e in entries if e["name"] in args.dataset]
    writer = ReviewWriter(args.output.resolve(), [e["name"] for e in entries], policy, entries, args.tokenizer, policy["seed"])
    output = writer.output
    (output / "metadata").mkdir()
    (output / "profiles").mkdir()
    (output / "metadata/command.txt").write_text(f"cd {shlex.quote(os.getcwd())}\n{shlex.join([sys.executable, *sys.argv])}\n")
    status_path = output / "metadata/run_status.json"
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "bounded": args.limit_trajectories is not None, "workers": args.workers,
              "input": str(args.data_root.resolve()),
              "config": str(args.config.resolve()), "policy": str(args.policy.resolve())}
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    try:
        jobs = [(entry, args.data_root, policy, args.tokenizer, output, args.limit_trajectories) for entry in entries]
        # spawn: the tokenizer must not be shared across fork; one dataset per
        # child releases its memory. imap keeps config order for the merge.
        with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for name, partial, state in pool.imap(review_job, jobs):
                writer.absorb(state, partial)
                print(f"reviewed {name}", flush=True)
        shutil.rmtree(output / "partials")
        summary = writer.finalize()
        blocked = {name: dict(issues) for name, issues in writer.blocked.items() if issues}
        isolated = sorted(name for name, issues in blocked.items() if set(issues) == {"unresolved_dataset"})
        status.update(status="partial" if args.limit_trajectories is not None or blocked else "scan_completed_manual_review_pending",
                      ended_at=datetime.now().astimezone().isoformat(), summary=summary["global"],
                      isolated_by_config=isolated,
                      failed_or_partial_datasets={k: v for k, v in blocked.items() if k not in isolated},
                      manual_review="pending")
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        status.update(status="failed", ended_at=datetime.now().astimezone().isoformat(), error_type=type(exc).__name__)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        raise


def review_job(job):
    return review_dataset(*job)


if __name__ == "__main__":
    raise SystemExit(main())
