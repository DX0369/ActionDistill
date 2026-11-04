#!/usr/bin/env python
"""Read saved CPU logits and stream samples to analyze RESPOND; no model inference."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
for path in (PROJECT/"src", WORKSPACE/"third_party/decider", PROJECT.parent/"dataset_tool_inventory/src"):
    sys.path.insert(0, str(path))
import torch
from jev_train.data import SampleIndex, load_selected, tool_name, validate
from jev_train.evaluation import completed_parts
from jev_train.loss import check_eval_logits


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def rate(a, b):
    return a/b if b else None


def quantiles(values):
    values = sorted(values)
    result = {"count": len(values)}
    for p in (.1, .25, .5, .75, .9):
        h = (len(values)-1)*p
        lo, hi = math.floor(h), math.ceil(h)
        result[f"p{int(p*100)}"] = ((values[lo]*(hi-h)+values[hi]*(h-lo)) if lo != hi else values[lo]) if values else None
    return result


def gold_respond(row):
    names = [tool_name(row["options"][i]) for i in row["gold"]]
    require(names and ("RESPOND" not in names or names == ["RESPOND"]), f"invalid RESPOND gold: {row['id']}")
    return names == ["RESPOND"]


def previous_kind(names):
    require(isinstance(names, list) and all(isinstance(x, str) and x for x in names), "invalid prev_action_names")
    require(len(names) == len(set(names)), "duplicated previous action names")
    if not names:
        return "NONE"
    if "RESPOND" in names:
        require(names == ["RESPOND"], "mixed previous RESPOND/tool")
        return "RESPOND"
    return "TOOL" if len(names) == 1 else "TOOL_SET"


def composition(rows):
    groups = defaultdict(Counter)
    outcomes = defaultdict(Counter)
    for row in rows:
        r = gold_respond(row)
        groups[row["dataset"]].update(count=1, respond=int(r))
        outcomes[row["outcome"]].update(count=1, respond=int(r))
    def finish(value):
        return {"count": value["count"], "respond": value["respond"], "respond_share": rate(value["respond"], value["count"])}
    return {"overall": finish(sum(groups.values(), Counter())),
            "by_dataset": {d: finish(v) for d, v in sorted(groups.items())},
            "by_trajectory_outcome": {d: finish(v) for d, v in sorted(outcomes.items())}}


def slim(row):
    return {k: row[k] for k in ("id", "dataset", "outcome", "options", "gold")}


def read_population(path, wanted, smoke_limit=None, features=False):
    """Stream large JSONL, retain only aggregates and specifically requested rows."""
    seen = set()
    selected = {}
    groups, outcomes = defaultdict(Counter), defaultdict(Counter)
    all_keys = None
    with path.open("rb") as stream:
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            line_number += 1
            row = validate(json.loads(line))
            require(row["id"] not in seen, f"duplicate id: {path}:{line_number}")
            seen.add(row["id"])
            all_keys = set(row) if all_keys is None else all_keys.intersection(row)
            r = gold_respond(row)
            groups[row["dataset"]].update(count=1, respond=int(r))
            outcomes[row["outcome"]].update(count=1, respond=int(r))
            if row["id"] in wanted:
                retained = slim(row)
                if features:
                    retained.update({k: row[k] for k in ("prev_action_names", "transition", "loop_state", "trajectory_position",
                                                          "trajectory_decisions", "harness", "action_kind")})
                    retained["context_tail"] = row["context"][-1000:]
                retained["source"] = f"{path}:{line_number} (byte_offset={offset})"
                selected[row["id"]] = retained
            if smoke_limit is not None and len(seen) >= smoke_limit and wanted <= selected.keys():
                break
    def finish(value):
        return {"count": value["count"], "respond": value["respond"], "respond_share": rate(value["respond"], value["count"])}
    return {"overall": finish(sum(groups.values(), Counter())),
            "by_dataset": {d: finish(v) for d, v in sorted(groups.items())},
            "by_trajectory_outcome": {d: finish(v) for d, v in sorted(outcomes.items())}}, selected, sorted(all_keys or [])


def validate_part(data):
    n = len(data["id"])
    require(n > 0 and len(set(data["id"])) == n, "empty/duplicate evaluation ids")
    require(data["logits"].shape == data["targets"].shape == (n,255), "invalid tensor shape")
    check_eval_logits(data["logits"])
    require(all(len(data[k]) == n for k in ("dataset", "outcome", "candidate_level", "nopts", "tool_names", "transition", "copy_prev_hit")), "part metadata lengths differ")
    positions = []
    for i, names in enumerate(data["tool_names"]):
        count = data["nopts"][i]
        require(len(names) == count and names.count("RESPOND") == 1, "invalid option list / RESPOND count")
        require(torch.isfinite(data["logits"][i,:count]).all().item(), "nonfinite valid logits")
        require(torch.isneginf(data["logits"][i,count:]).all().item() and (data["targets"][i,count:] == 0).all().item(), "invalid padding")
        require((data["targets"][i] >= 0).all().item() and torch.isclose(data["targets"][i].sum(), torch.tensor(1.)).item(), "invalid targets")
        pos = names.index("RESPOND")
        require(not data["targets"][i,pos] > 0 or int((data["targets"][i] > 0).sum()) == 1, "RESPOND mixed with tool gold")
        positions.append(pos)
    return torch.tensor(positions, dtype=torch.long)


def adjusted_metrics(data, temperature, bias, respond_positions):
    logits = data["logits"].float()/temperature
    logits = logits.clone()
    logits[torch.arange(len(logits)), respond_positions] += bias
    check_eval_logits(logits)
    targets = data["targets"]
    predictions = logits.argmax(1)
    gold = targets[torch.arange(len(logits)),respond_positions] > 0
    predicted = predictions == respond_positions
    tp = int((gold & predicted).sum())
    logp = torch.log_softmax(logits, 1).masked_fill(~torch.isfinite(logits),0)
    return {"count": len(logits), "acc": float((targets[torch.arange(len(logits)),predictions] > 0).float().mean()),
            "nll": float(-(targets*logp).sum(1).mean()), "gold_respond": int(gold.sum()), "predicted_respond": int(predicted.sum()),
            "respond_tp": tp, "respond_precision": rate(tp,int(predicted.sum())), "respond_recall": rate(tp,int(gold.sum()))}


def miss_analysis(data, rows, temperature, respond_positions):
    probs = torch.softmax(data["logits"].float()/temperature, 1)
    pred = probs.argmax(1)
    states = []
    for i, value in enumerate(data["id"]):
        row = rows[value]
        require(row["dataset"] == data["dataset"][i] and row["outcome"] == data["outcome"][i] and row["transition"] == data["transition"][i], "part/sample metadata mismatch")
        position = int(respond_positions[i])
        gold = bool(data["targets"][i,position] > 0)
        require(gold == gold_respond(row), "part/sample RESPOND label differs")
        if not gold:
            continue
        names = data["tool_names"][i]
        prediction = names[int(pred[i])]
        rank = sorted(range(data["nopts"][i]), key=lambda j: (-float(probs[i,j]),j)).index(position)+1
        total, place = row["trajectory_decisions"], row["trajectory_position"]
        require(type(total) is int and total > 0 and type(place) is int and 0 <= place < total, "invalid trajectory position")
        previous = row["prev_action_names"]
        kind = previous_kind(previous)
        states.append({"id": value, "dataset": row["dataset"], "miss": prediction != "RESPOND", "prediction": prediction,
                       "p_respond": float(probs[i,position]), "respond_rank": rank,
                       "respond_rank_tied": int((probs[i,:data["nopts"][i]] == probs[i,position]).sum()) > 1,
                       "previous_action_names": previous, "previous_action_type_derived": kind,
                       "prediction_in_previous_tool_names": prediction != "RESPOND" and prediction in previous,
                       "transition": row["transition"], "loop_state": row["loop_state"],
                       "trajectory_outcome": row["outcome"], "trajectory_position": place, "trajectory_decisions": total,
                       "relative_position": place/total, "position_decile": int(10*place/total),
                       "harness": row["harness"], "source": row["source"]})
    def summarize(group):
        misses = [r for r in group if r["miss"]]
        return {"gold_respond": len(group), "misses": len(misses), "recall": rate(len(group)-len(misses),len(group)),
                "miss_predictions": dict(Counter(r["prediction"] for r in misses)),
                "miss_copy_previous_tool": sum(r["prediction_in_previous_tool_names"] for r in misses),
                "miss_copy_previous_tool_rate": rate(sum(r["prediction_in_previous_tool_names"] for r in misses),len(misses)),
                "miss_copy_single_previous_tool": sum(r["prediction_in_previous_tool_names"] and r["previous_action_type_derived"] == "TOOL" for r in misses),
                "miss_copy_member_of_previous_tool_set": sum(r["prediction_in_previous_tool_names"] and r["previous_action_type_derived"] == "TOOL_SET" for r in misses),
                "gold_p_respond": quantiles([r["p_respond"] for r in group]), "miss_p_respond": quantiles([r["p_respond"] for r in misses]),
                "gold_rank_counts": dict(sorted(Counter(r["respond_rank"] for r in group).items())),
                "miss_rank_counts": dict(sorted(Counter(r["respond_rank"] for r in misses).items())),
                "respond_rank_ties": sum(r["respond_rank_tied"] for r in group)}
    result = {"overall": summarize(states), "slices": {}, "false_negative_states": [r for r in states if r["miss"]]}
    for key in ("dataset", "harness", "transition", "previous_action_type_derived", "loop_state", "trajectory_outcome", "position_decile"):
        groups = defaultdict(list)
        if key == "dataset":
            for dataset in data["dataset"]:
                groups[dataset] = []
        for row in states:
            groups[str(row[key])].append(row)
        result["slices"][key] = {k: summarize(v) for k,v in sorted(groups.items())}
    first = [i for i,x in enumerate(data["transition"]) if x == "first"]
    result["all_first_transition"] = {"count": len(first), "correct": sum(bool(data["targets"][i,int(pred[i])] > 0) for i in first)}
    result["all_first_transition"]["acc"] = rate(result["all_first_transition"]["correct"], len(first))
    result["examples"] = [{**r, "visible_context_tail": rows[r["id"]]["context_tail"]} for r in result["false_negative_states"][:8]]
    return result


def training_selection(run, smoke_limit):
    lines = (run/"metadata/command.txt").read_text().splitlines()
    require(len(lines) == 2 and lines[0].startswith("cd "), "unexpected training command format")
    tokens = shlex.split(lines[1])
    def argument(name):
        require(tokens.count(name) == 1, f"missing/duplicate {name} in training command")
        return tokens[tokens.index(name)+1]
    require("--train_ids" not in tokens, "this A analysis expects load_selected, not explicit train_ids")
    samples, limit, seed = Path(argument("--samples")), int(argument("--train_limit")), int(argument("--seed"))
    require(samples.is_absolute(), "training samples must be an absolute canonical path")
    print(f"index training source {samples/'train.jsonl'}", flush=True)
    index = SampleIndex(samples/"train.jsonl")
    entries = load_selected(index,limit,seed)
    require(len(entries) == limit, "training limit not fully met")
    full_count = len(entries)
    if smoke_limit:
        entries = entries[:smoke_limit]
    rows = [slim(index.read(e["offset"])) for e in entries]
    return rows, {"command": str(run/"metadata/command.txt"), "samples": str(samples), "seed": seed, "train_limit": limit,
                  "source_count": len(index.entries), "reconstructed_count": full_count, "analyzed_count": len(rows),
                  "selection": "existing SampleIndex + load_selected; no train_ids argument"}


def analyze(evaluation, training_run, smoke_limit):
    torch.set_num_threads(4)
    state = read_json(evaluation/"metadata/run_status.json")
    require(state["status"] == "completed", "evaluation not complete")
    config = state["configuration"]
    require(Path(config["model"]).resolve() == (training_run/"model").resolve(), "eval model/training run differ")
    pool = Path(config["samples"])
    temperature = read_json(evaluation/"temperature.json")["temperature"]
    require(type(temperature) in (float,int) and temperature > 0 and math.isfinite(temperature), "invalid temperature")
    previous_report = read_json(evaluation/"eval.json")
    parts, positions, feature_rows, population_stats, schema = {}, {}, {}, {}, {}
    for split in ("val", "test_id", "test_ood"):
        data, seed = completed_parts(evaluation,split)
        require(seed == config["seed"], "evaluation seed differs")
        all_positions = validate_part(data)
        baseline = adjusted_metrics(data,temperature,0,all_positions)
        for metric in ("acc", "nll"):
            require(abs(baseline[metric]-previous_report[split]["overall"][metric]) < 1e-6, f"saved {split} {metric} not reproduced")
        if smoke_limit:
            indices = torch.linspace(0,len(data["id"])-1,min(smoke_limit,len(data["id"]))).round().long().tolist()
            data = {k:v[indices] if torch.is_tensor(v) else [v[i] for i in indices] for k,v in data.items()}
            all_positions = all_positions[indices]
        parts[split] = data
        positions[split] = all_positions[:len(data["id"])]
        wanted = set(data["id"])
        print(f"stream {split} population / join {len(wanted)} evaluated states", flush=True)
        stats, rows, keys = read_population(pool/f"{split}.jsonl", wanted, smoke_limit, features=True)
        require(wanted == rows.keys(), f"{split} feature join incomplete")
        feature_rows[split], population_stats[split], schema[split] = rows, stats, keys
    grid = [{"bias": round(i/10,1), **adjusted_metrics(parts["val"],temperature,i/10,positions["val"])} for i in range(-10,31)]
    best = min(grid, key=lambda r:r["nll"])
    chosen = best["bias"]
    bias_results = {}
    for split in parts:
        baseline = adjusted_metrics(parts[split],temperature,0,positions[split])
        fitted = adjusted_metrics(parts[split],temperature,chosen,positions[split])
        bias_results[split] = {"baseline": baseline, "fitted": fitted,
                              "delta": {k:fitted[k]-baseline[k] if fitted[k] is not None and baseline[k] is not None else None
                                        for k in ("acc","nll","respond_precision","respond_recall")}}
    train, selection = training_selection(training_run,smoke_limit)
    print("stream pool training population", flush=True)
    pool_stats, joined, schema["pool_train"] = read_population(pool/"train.jsonl", {r["id"] for r in train}, smoke_limit)
    for row in train:
        if row["id"] in joined:
            other = joined[row["id"]]
            require(all(row[k] == other[k] for k in ("dataset","outcome","options","gold")), "old/pool common sample differs")
    populations = {"selected_A_training": composition(train), "pool_train": pool_stats,
                   **{f"full_{k}": v for k,v in population_stats.items()},
                   **{f"evaluated_{k}": composition(rows.values()) for k,rows in feature_rows.items()}}
    datasets = sorted({d for pop in populations.values() for d in pop["by_dataset"]})
    comparison = [{"dataset": d, "populations": {k: (v["by_dataset"][d] if d in v["by_dataset"] else None)
                                                 for k,v in populations.items()}} for d in datasets]
    summary = {"evaluation": str(evaluation), "training_run": str(training_run), "pool": str(pool), "smoke_limit": smoke_limit,
               "temperature": temperature, "training_selection": selection, "sample_schema": schema,
               "definitions": {
                   "nll": "Existing soft-target CE: -sum(target*log_softmax(logits/T)); padding is masked. Bias is added to RESPOND logit AFTER division by fixed saved T. Choose minimum val NLL on [-1,3] in steps of 0.1; tied grid minima choose smaller b.",
                   "metrics": "Accuracy is top1 in positive-target set. RESPOND gold is singleton; precision TP/predicted_RESPOND, recall TP/gold_RESPOND; zero denominators null.",
                   "rank": "Descending calibrated probability, tied by stored shuffled-option index; quantiles linear interpolation. Bias cannot change ordering among tool options.",
                   "previous": "Previous type is explicitly derived from canonical prev_action_names: [] NONE, [RESPOND] RESPOND, one distinct tool TOOL, multiple TOOL_SET. Copy means predicted name belongs to previous tool names, with single/multiple separated; it is not copy_prev_acc's set-exact baseline.",
                   "position": "canonical zero-based trajectory_position / trajectory_decisions, decile=floor(10*ratio), strictly0<=position<decisions",
                   "outcome": "trajectory/task outcome only; NOT previous-action success/failure",
                   "composition": "Training8k reconstructed from actual command using existing selector. Full pool train/full holdouts streamed separately from evaluated1000 subsets. Smoke limits evaluated/selected-training states; index/join scans can still cover full files, and actual population counts are explicitly reported.",
                   "missing_dataset": "null in cross-population table means absent dataset, never zero RESPOND share; designated OOD datasets must stay held out"},
               "unavailable": {"previous_action_success_failure": "No canonical field in old training, pool or saved parts. Not inferred from trajectory outcome or rendered text.",
                               "full_training_history_features": {"selected": len(train), "joinable_pool_ids": len(joined), "missing_pool_ids": len(train)-len(joined),
                                                                  "decision": "Do not replace actual training set with join intersection; complete8k comparison uses base fields only."}},
               "respond_misses": {s:miss_analysis(parts[s],feature_rows[s],temperature,positions[s]) for s in parts},
               "bias": {"selected_on": "val NLL only", "grid": grid, "selected_bias": chosen, "at_grid_boundary": chosen in (-1,3),
                        "splits": bias_results, "ood_sensitivity": [{"bias": b, **adjusted_metrics(parts["test_ood"],temperature,b,positions["test_ood"])} for b in (.5,1.,1.5)]},
               "composition": populations, "dataset_comparison": comparison}
    return summary


def render_report(summary):
    lines = ["# 离线 RESPOND 漏判与训练构成", "", f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z %z')}。CPU读取既有logits，无模型推理。", "",
             "| split | gold RESPOND | 漏判 | 召回率 | 漏判中重复前工具 | 漏判P(RESPOND)中位数 |", "|---|---:|---:|---:|---:|---:|"]
    for split, value in summary["respond_misses"].items():
        v = value["overall"]
        lines.append(f"| {split} | {v['gold_respond']} | {v['misses']} | {v['recall']} | {v['miss_copy_previous_tool']} | {v['miss_p_respond']['p50']} |")
    lines += ["", "按数据集、transition、上一步名称派生类型、位置、loop和轨迹outcome切分，以及全部漏判ID/原始定位见 summary.respond_misses。",
              f"OOD first：{summary['respond_misses']['test_ood']['all_first_transition']}。", "",
              f"固定原温度 T={summary['temperature']}，仅由val NLL选出的全局偏置 b={summary['bias']['selected_bias']}；是在除温度后添加。",
              "", "| split | acc 原→偏置 | RESPOND precision 原→偏置 | recall 原→偏置 | NLL 原→偏置 |", "|---|---|---|---|---|"]
    for split,r in summary["bias"]["splits"].items():
        pair = lambda k: f"{r['baseline'][k]} → {r['fitted'][k]}"
        lines.append(f"| {split} | {pair('acc')} | {pair('respond_precision')} | {pair('respond_recall')} | {pair('nll')} |")
    lines += ["", "OOD固定b=0.5/1/1.5只作敏感性检查，见summary.bias.ood_sensitivity；不能据此选择测试集最优参数。", "", "## 构成", "",
              "| 人群 | 状态数 | RESPOND数 | 比例 |", "|---|---:|---:|---:|"]
    for name,p in summary["composition"].items():
        v = p["overall"]
        lines.append(f"| {name} | {v['count']} | {v['respond']} | {v['respond_share']} |")
    lines += ["", "| 数据集 | A训练总数 | A训练RESPOND | 比例 | pool比例 | full val比例 |", "|---|---:|---:|---:|---:|---:|"]
    for d in summary["dataset_comparison"]:
        p = d["populations"]
        if p["selected_A_training"] is None:
            continue
        t = p["selected_A_training"]
        ratio = lambda k: p[k]["respond_share"] if p[k] is not None else None
        lines.append(f"| {d['dataset']} | {t['count']} | {t['respond']} | {t['respond_share']} | {ratio('pool_train')} | {ratio('full_val')} |")
    lines += ["", "上一步动作成败没有canonical字段，因此无法统计；轨迹outcome单列，不替代它。",
              f"旧训练与pool交集审计：{summary['unavailable']['full_training_history_features']}。",
              "组成表里的null表示该数据集不在该人群；OOD来源不应被拉回训练。所有新数字与旧正式eval分开保存，未改写旧产物。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    output = (args.output or PROJECT/"outputs"/("analysis_respond_"+datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))).resolve()
    require(output.is_relative_to(PROJECT/"tmp") or output.is_relative_to(PROJECT/"outputs"), "output outside project")
    require(not output.is_relative_to(args.evaluation.resolve()) and not output.is_relative_to(args.training_run.resolve()), "cannot write inside source")
    require(args.smoke_limit is None or args.smoke_limit > 0 and output.is_relative_to(PROJECT/"tmp"), "smoke requires positive limit and tmp output")
    output.mkdir(parents=True, exist_ok=False)
    (output/"metadata").mkdir()
    (output/"metadata/command.txt").write_text(f"cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n")
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "evaluation": str(args.evaluation.resolve()),
             "training_run": str(args.training_run.resolve()), "smoke_limit": args.smoke_limit}
    write_json(output/"metadata/run_status.json",state)
    try:
        summary = analyze(args.evaluation.resolve(),args.training_run.resolve(),args.smoke_limit)
        write_json(output/"summary.json", summary)
        (output/"report.md").write_text(render_report(summary))
        state.update(status="completed")
        print(json.dumps({"output": str(output), "bias": summary["bias"]["selected_bias"], **state}),flush=True)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state["ended_at"] = datetime.now().astimezone().isoformat()
        write_json(output/"metadata/run_status.json",state)


if __name__ == "__main__":
    main()
