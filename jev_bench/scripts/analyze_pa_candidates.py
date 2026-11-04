#!/usr/bin/env python
"""CPU-only, fixed-reference PA32 -> PA16 replay of five approved ledgers.

This reads frozen code and naming evidence; it never generates a production
mapping or calls a model. Changed parameter contexts remain explicitly unknown.
"""

import argparse
import json
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from jev_bench import pa_tool_prepare as current
from jev_bench.cost import completion_cost
from jev_bench.io import now, write_json
from jev_bench.pa_tool_generation import slots

OLD_TREE = "133e81ddac4d2129e92507ba28dba88c0a9c44e6"
INPUTS = (
    "tmp/protocol_smoke_20261103_000630/pa_apibank/preparation",
    "tmp/protocol_smoke_20261103_000630/pa_toolhop/preparation",
    "tmp/protocol_smoke_20261103_000630/pa_wifi/preparation",
    "tmp/pa_call_smoke_20261103_112823/pa_apibank/preparation",
    "tmp/pa_call_smoke_20261103_112823/pa_toolhop/preparation",
)
SEED = 20261015
SUBSETS = 1000


def frozen_prepare():
    source = subprocess.check_output(
        ["git", "show", f"{OLD_TREE}:jev_bench/pa_tool_prepare.py"],
        cwd=PROJECT.parent,
        text=True,
    )
    module = ModuleType("pa32_analysis_only")
    module.__package__ = "jev_bench"
    exec(  # noqa: S102 - fixed, committed project source is the historical oracle.
        compile(source, f"git:{OLD_TREE}:jev_bench/pa_tool_prepare.py", "exec"),
        module.__dict__,
    )
    return module


def specs_for(catalog, mapping):
    """Recover exact identities/prompts at the recorded, resolved tool aliases."""
    specs = {}
    for scope, tools in sorted(catalog["scopes"].items()):
        for index, tool in enumerate(tools):
            function = tool["function"]
            if function["name"] in catalog["control_names"]:
                continue
            base = {"function": function, "occurrence": tools[:index].count(tool)}
            if current.description_preservation(catalog, scope, function) is None:
                spec = {**base, "component": "tool"}
                specs[current.identity(spec)] = {
                    "spec": spec,
                    "original_name": function["name"],
                    "text": current.prompt("tool", function["description"]),
                }
            alias = mapping["scopes"][scope][index]["name"]
            for parameter, schema in function["parameters"]["properties"].items():
                if (
                    current.description_preservation(
                        catalog, scope, function, parameter
                    )
                    is not None
                ):
                    continue
                spec, text = parameter_input(catalog, base, parameter, schema, alias)
                specs[current.identity(spec)] = {
                    "spec": spec,
                    "original_name": parameter,
                    "text": text,
                }
    return specs


def parameter_input(catalog, base, parameter, schema, alias):
    benchmark = catalog["benchmark"]["name"]
    description, source = current.parameter_description(
        schema,
        allow_toolhop=benchmark == "toolhop",
        allow_empty=benchmark == "stabletoolbench",
    )
    return (
        {
            **base,
            "component": "parameter",
            "parameter": parameter,
            "tool_alias": alias,
            "description_source": source,
        },
        current.prompt(
            "parameter",
            description,
            tool_name=alias,
            tool_description=base["function"]["description"],
            allow_empty=benchmark == "stabletoolbench",
        ),
    )


def validate_event(event, key, text, slot, settings, protocol, old):
    seed = int(old.identity([protocol["seed"], key, slot])[:15], 16)
    if (
        event["phase"] != "finished"
        or event["component"] != key
        or type(event["slot"]) is not int
        or event["slot"] != slot
        or event["seed"] != seed
        or event["text"] != text
    ):
        raise ValueError(
            "Replay requires complete original slots with exact prompts/seeds"
        )
    expected = {
        "model": settings["model"],
        "messages": [{"role": "user", "content": text}],
        "temperature": protocol["reference_temperature"]
        if slot == 32
        else protocol["temperature"],
        "max_tokens": protocol["max_tokens"],
        "top_p": protocol["top_p"],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": protocol["enable_thinking"]},
            "top_k": protocol["top_k"],
            "min_p": protocol["min_p"],
            "seed": seed,
        },
    }
    provider = event["provider"]
    if provider["error"] is not None or old.identity(
        provider["request"]
    ) != old.identity(expected):
        raise ValueError("Recorded provider request differs or failed")
    response = provider["response"]
    if response["model"] != settings["model"] or len(response["choices"]) != 1:
        raise ValueError("Recorded response identity/choices differ")
    choice = response["choices"][0]
    message = choice["message"]
    if message["role"] != "assistant":
        raise ValueError("Recorded response role differs")
    try:
        name = old.generated_name(
            choice["finish_reason"], message["content"], message["tool_calls"]
        )
    except old.NameGenerationError:
        old.check_name_failure(event, settings["model"])
    else:
        if event["name"] != name or event["error"] is not None:
            raise ValueError("Recorded name differs from successful response")
    usage = provider["usage"]
    if usage is None:
        if response["usage"] is not None:
            raise ValueError("Recorded usage differs")
    elif any(
        type(usage[f]) is not int or usage[f] < 0 or usage[f] != response["usage"][f]
        for f in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise ValueError("Recorded usage differs or is invalid")
    latency = provider["latency_seconds"]
    if latency is not None and (
        type(latency) not in (int, float) or not np.isfinite(latency) or latency < 0
    ):
        raise ValueError("Recorded latency is invalid")


def read_baseline(directory, old):
    catalog = json.loads((directory / "catalog.json").read_text())
    mapping = json.loads((directory / "mapping.json").read_text())
    state = json.loads((directory / "metadata/run_status.json").read_text())
    if (
        state["status"] != "completed"
        or state["code"]["trees"]["jev_bench/src"] != OLD_TREE
        or state["code"]["dirty_paths"]
        or state["protocol"] != old.PROTOCOL
    ):
        raise ValueError("Input is not the approved completed PA32 source")
    saved = slots(directory / "diagnostics/generation.jsonl")
    components = specs_for(catalog, mapping)
    expected = {(key, slot) for key in components for slot in range(33)}
    if set(saved) != expected:
        raise ValueError("Old component/slot coverage differs; stop before PA16 replay")
    consumed = set()

    def generate(key, text):
        if text != components[key]["text"]:
            raise ValueError("Reconstructed old prompt differs")
        consumed.add(key)
        events = [saved[key, slot] for slot in range(33)]
        for slot, event in enumerate(events):
            validate_event(
                event, key, text, slot, catalog["executor"], old.PROTOCOL, old
            )
        failures = [
            {"slot": e["slot"], "reason": e["error"]["reason"]}
            for e in events
            if e["error"] is not None
        ]
        components[key]["events"] = events
        components[key]["old_ranking"] = (
            None
            if failures
            else old.rank_names([e["name"] for e in events[:32]], events[32]["name"])
        )
        if failures:
            raise old.NameGenerationError(failures[0]["reason"], failures=failures)
        return [e["name"] for e in events[:32]], events[32]["name"]

    reconstructed = old.build_mapping(catalog, generate)
    if reconstructed != mapping or consumed != set(components):
        raise ValueError(
            "Old32 mapping field-for-field replay differs; PA16 analysis stopped"
        )
    return {
        "catalog": catalog,
        "mapping": mapping,
        "components": components,
        "source": str(directory.relative_to(PROJECT)),
        "baseline_exact": True,
        "old_cost": cost([e for c in components.values() for e in c["events"]]),
    }


def cost(events):
    return completion_cost(SimpleNamespace(records=[e["provider"] for e in events]))


def selection(component, subset):
    events = component["events"]
    samples = [events[slot]["name"] for slot in subset]
    reference = events[32]["name"]  # Always the old reference, never old random slot16.
    valid = sum(name is not None for name in samples)
    reason = (
        "reference_failed"
        if reference is None
        else "insufficient_valid_candidates"
        if valid < 12
        else None
    )
    ranking = None if reason else current.rank_names(samples, reference)
    old = component["old_ranking"]
    old_local = component["original_name"] if old is None else old[0]["name"]
    local = component["original_name"] if ranking is None else ranking[0]["name"]
    return {
        "ranking": ranking,
        "local": local,
        "old_local": old_local,
        "valid": valid,
        "fallback": reason,
        "selected_slots": list(subset),
        "reference_slot": 32,
    }


def resolve(rankings, originals, reserved=()):
    try:
        return current.resolve_names(
            rankings, reserved=reserved, original_names=originals
        ), None
    except ValueError as exc:
        if not str(exc).startswith("PA-Tool candidates exhausted"):
            raise
        return None, str(exc)


def replay_scopes(data, selected):
    catalog, old_mapping, components = (
        data["catalog"],
        data["mapping"],
        data["components"],
    )
    rows, scopes, reached, unknown = [], [], set(), set()
    for scope, tools in sorted(catalog["scopes"].items()):
        rankings, controls, bases = {}, {}, {}
        for index, tool in enumerate(tools):
            function = tool["function"]
            bases[index] = {
                "function": function,
                "occurrence": tools[:index].count(tool),
            }
            key = current.identity({**bases[index], "component": "tool"})
            if function["name"] in catalog["control_names"]:
                controls[index] = function["name"]
            elif current.description_preservation(catalog, scope, function) is not None:
                rankings[index] = None
            else:
                rankings[index] = selected[key]["ranking"]
                reached.add(key)
        aliases, error = resolve(
            rankings,
            {i: t["function"]["name"] for i, t in enumerate(tools)},
            controls.values(),
        )
        if error:
            scopes.append(
                {
                    "scope": scope,
                    "status": "candidate_exhaustion",
                    "tool_mapping_same": False,
                    "complete_mapping_same": False,
                    "error": error,
                    "parameter_errors": [],
                }
            )
            for index, tool in enumerate(tools):
                rows.append(
                    {
                        "scope": scope,
                        "tool": index,
                        "component": "tool",
                        "status": "blocked_by_candidate_exhaustion",
                    }
                )
                rows.extend(
                    {
                        "scope": scope,
                        "tool": index,
                        "component": "parameter",
                        "parameter": p,
                        "status": "blocked_by_candidate_exhaustion",
                    }
                    for p in tool["function"]["parameters"]["properties"]
                )
            continue
        aliases.update(controls)
        scope_rows = []
        parameter_errors = []
        for index, tool in enumerate(tools):
            function, alias = tool["function"], aliases[index]
            old_row = old_mapping["scopes"][scope][index]
            tool_key = current.identity({**bases[index], "component": "tool"})
            tool_choice = (
                selected[tool_key]["local"]
                if tool_key in selected
                else function["name"]
            )
            scope_rows.append(
                {
                    "scope": scope,
                    "tool": index,
                    "component": "tool",
                    "status": "known",
                    "old": old_row["name"],
                    "selected": alias,
                    "collision_changed": alias != tool_choice,
                }
            )
            param_ranks, missing, fixed, param_keys = {}, {}, {}, {}
            for parameter, schema in function["parameters"]["properties"].items():
                if (
                    index in controls
                    or current.description_preservation(
                        catalog, scope, function, parameter
                    )
                    is not None
                ):
                    fixed[parameter] = parameter
                    continue
                spec, text = parameter_input(
                    catalog, bases[index], parameter, schema, alias
                )
                key = current.identity(spec)
                param_keys[parameter] = key
                if key not in components:
                    missing[parameter] = "unknown_context_changed"
                    unknown.add(key)
                else:
                    if components[key]["text"] != text:
                        raise ValueError(
                            "Matching parameter identity has a different prompt"
                        )
                    param_ranks[parameter] = selected[key]["ranking"]
                    reached.add(key)
            resolved, param_error = (
                (None, None)
                if missing
                else resolve(param_ranks, {p: p for p in param_ranks}, fixed.values())
            )
            if param_error:
                parameter_errors.append({"tool": index, "error": param_error})
            for parameter in function["parameters"]["properties"]:
                row = {
                    "scope": scope,
                    "tool": index,
                    "component": "parameter",
                    "parameter": parameter,
                    "old": old_row["parameters"][parameter],
                }
                if parameter in fixed:
                    row.update(
                        status="known", selected=parameter, collision_changed=False
                    )
                elif missing:
                    row["status"] = missing.get(
                        parameter, "unknown_collision_dependency"
                    )
                elif param_error:
                    row.update(
                        status="blocked_by_candidate_exhaustion", error=param_error
                    )
                else:
                    name = resolved[parameter]
                    row.update(
                        status="known",
                        selected=name,
                        collision_changed=name
                        != selected[param_keys[parameter]]["local"],
                    )
                scope_rows.append(row)
            # All generated parameters in this namespace are unknown if even
            # one ranking is missing: collision competition cannot be replayed.
        known = all(row["status"] == "known" for row in scope_rows)
        scopes.append(
            {
                "scope": scope,
                "status": "candidate_exhaustion"
                if parameter_errors
                else "complete"
                if known
                else "unknown_context",
                "tool_mapping_same": all(
                    r["old"] == r["selected"]
                    for r in scope_rows
                    if r["component"] == "tool"
                ),
                "complete_mapping_same": False
                if parameter_errors
                else all(r["old"] == r["selected"] for r in scope_rows)
                if known
                else None,
                "error": None,
                "parameter_errors": parameter_errors,
            }
        )
        rows.extend(scope_rows)
    return rows, scopes, reached, unknown


def metrics(data, selected, rows, scopes, reached, unknown):
    result = {}
    for kind in ("tool", "parameter"):
        choices = [
            selected[k]
            for k, c in data["components"].items()
            if c["spec"]["component"] == kind
        ]
        actual = [r for r in rows if r["component"] == kind]
        known = [r for r in actual if r["status"] == "known"]
        result[kind] = {
            "unique_recorded_components": len(choices),
            "local_same": sum(r["local"] == r["old_local"] for r in choices),
            "original_name_fallbacks": sum(r["fallback"] is not None for r in choices),
            "excluded_random_slots": sum(16 - r["valid"] for r in choices),
            "scope_components": len(actual),
            "known": len(known),
            "same_after_collision": sum(r["old"] == r["selected"] for r in known),
            "collision_reselected": sum(r["collision_changed"] for r in known),
            "unknown": sum(r["status"].startswith("unknown_") for r in actual),
            "components_blocked_by_candidate_exhaustion": sum(
                r["status"] == "blocked_by_candidate_exhaustion" for r in actual
            ),
        }
    result["scopes"] = {
        "count": len(scopes),
        "tool_mapping_same": sum(r["tool_mapping_same"] for r in scopes),
        "complete_mapping_known": sum(r["status"] == "complete" for r in scopes),
        "complete_mapping_same": sum(
            r["complete_mapping_same"] is True for r in scopes
        ),
        "unknown_context": sum(r["status"] == "unknown_context" for r in scopes),
        "candidate_exhaustion": sum(
            r["status"] == "candidate_exhaustion" for r in scopes
        ),
        "candidate_exhaustion_namespaces": sum(
            (r["error"] is not None) + len(r["parameter_errors"]) for r in scopes
        ),
    }
    all_events = [
        data["components"][key]["events"][slot]
        for key, value in selected.items()
        for slot in [*value["selected_slots"], 32]
    ]
    reachable_events = [
        data["components"][key]["events"][slot]
        for key in sorted(reached)
        for slot in [*selected[key]["selected_slots"], 32]
    ]
    result["fixed_old_context_cost"] = cost(all_events)
    result["reachable_recorded_cost"] = cost(reachable_events)
    result["unknown_new_parameter_components"] = len(unknown)
    return result


def numeric_leaves(value, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from numeric_leaves(item, name)
        else:
            yield name, item


def distributions(values):
    return {
        key: {
            "mean": float(np.mean(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.5)),
            "p95": float(np.quantile(v, 0.95)),
            "min": min(v),
            "max": max(v),
        }
        for key, v in values.items()
    }


def experiment(data, rng):
    components = data["components"]
    selected = {k: selection(c, range(16)) for k, c in components.items()}
    rows, scopes, reached, unknown = replay_scopes(data, selected)
    prefix = metrics(data, selected, rows, scopes, reached, unknown)
    prefix_details = {
        "components": {
            k: {
                **{f: v for f, v in selected[k].items() if f != "ranking"},
                "component": c["spec"]["component"],
                "original_name": c["original_name"],
            }
            for k, c in components.items()
        },
        "rows": rows,
        "scopes": scopes,
    }
    trials, frequencies = {}, {k: Counter() for k in components}
    for _ in range(SUBSETS):
        selected = {
            k: selection(components[k], sorted(rng.sample(range(32), 16)))
            for k in sorted(components)
        }
        for key, value in selected.items():
            frequencies[key][value["local"]] += 1
        rows, scopes, reached, unknown = replay_scopes(data, selected)
        for key, value in numeric_leaves(
            metrics(data, selected, rows, scopes, reached, unknown)
        ):
            trials.setdefault(key, []).append(value)
    return {
        "source": data["source"],
        "baseline_exact": True,
        "old_cost": data["old_cost"],
        "prefix": prefix,
        "prefix_details": prefix_details,
        "random_subsets": distributions(trials),
        "local_selection_frequencies": {
            k: dict(v.most_common()) for k, v in frequencies.items()
        },
    }, trials


def report(summary):
    prefix, randoms = summary["prefix_totals"], summary["random_subset_totals"]
    lines = [
        "# PA16 CPU 重放",
        "",
        f"时间：{summary['completed_at']}（CST +0800）。",
        "",
        "五份旧32＋1映射全部逐字段复现；未修改旧资产、未调用模型或生成正式映射。",
        "16随机＋1参考；至少12合法候选，失败候选排除；密度按有效候选数减一归一化。thinking及8192不变。",
        "",
        "| 同一旧提示下的组件选名 | 数量 | 前16槽与旧结果一致 | 1000子集一致数均值（p05–p95） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind, label in (("tool", "工具"), ("parameter", "参数（固定旧工具别名）")):
        n, same = (
            prefix[f"{kind}.unique_recorded_components"],
            prefix[f"{kind}.local_same"],
        )
        v = randoms[f"{kind}.local_same"]
        lines.append(
            f"| {label} | {n} | {same}/{n}（{same / n:.1%}） | {v['mean']:.2f}（{v['p05']:g}–{v['p95']:g}） |"
        )
    lines += [
        "",
        "| 作用域内重名处理后 | 总数 | 可判 | 与旧名一致 | 上下文未知 | 因耗尽受阻的组件 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, label in (("tool", "工具"), ("parameter", "参数")):
        fields = [
            "scope_components",
            "known",
            "same_after_collision",
            "unknown",
            "components_blocked_by_candidate_exhaustion",
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(str(prefix[f"{kind}.{f}"]) for f in fields)
            + " |"
        )
    old = summary["old_totals"]
    requests = prefix["fixed_old_context_cost.requests"]
    output = prefix["fixed_old_context_cost.completion_tokens.observed_total"]
    lines += [
        "",
        f"前16槽：完整工具映射一致 {prefix['scopes.tool_mapping_same']}/{prefix['scopes.count']} 个scope；完整工具＋参数映射可判 {prefix['scopes.complete_mapping_known']} 个，其中一致 {prefix['scopes.complete_mapping_same']} 个。",
        f"固定旧提示的账本抽样成本：{old['requests']}→{requests} 请求（减少 {1 - requests / old['requests']:.1%}），输出token {old['completion_tokens']}→{output}（减少 {1 - output / old['completion_tokens']:.1%}）。含已失败请求成本。",
        f"前16槽有 {prefix['unknown_new_parameter_components']} 个新的参数上下文无记录，其真实生成成本未知；不能把上述节省当作完整新运行实测。",
        "",
        "这是小样本工程敏感性分析，不能证明benchmark分数不变或效果等价。参数的组件级一致率固定旧工具别名；工具别名改变后，缺少精确新提示的参数候选即记未知，不能拿旧参数结果代替。一个参数候选缺失会使同工具内其他生成参数的重名处理也不可判。",
        "每个试验按固定输入顺序、排序后的组件ID独立抽16槽，无放回，保留槽位顺序及重复名字，同组件跨scope使用同一抽样。1000组使用seed=20261015；分位数是子集敏感性区间，不是统计置信区间。",
        "始终使用旧slot32的参考名。新生成16＋1的参考是slot16，派生seed与旧slot32不同，因此本分析不等于新生成协议的逐token复现。",
        "同时改变候选数量、失败排除与密度协议，差异不能全归因于候选数量。服务/数据错误仍停止；本次数据不覆盖参考失败或少于12个有效候选，这些边界只由CPU合成测试验证。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "started_at": now()}
    write_json(args.output_dir / "metadata/run_status.json", state)
    (args.output_dir / "metadata/command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    try:
        old = frozen_prepare()
        data = [read_baseline(PROJECT / path, old) for path in INPUTS]
        # No PA16 result is computed until every old mapping passes exactly.
        rng, results, trial_totals, prefix_totals = (
            random.Random(SEED),
            [],
            {},
            Counter(),
        )
        for entry in data:
            result, trials = experiment(entry, rng)
            results.append(result)
            prefix_totals.update(dict(numeric_leaves(result["prefix"])))
            for key, values in trials.items():
                previous = trial_totals.setdefault(key, [0] * SUBSETS)
                trial_totals[key] = [
                    a + b for a, b in zip(previous, values, strict=True)
                ]
        summary = {
            "completed_at": now(),
            "old_source_tree": OLD_TREE,
            "new_protocol": current.PROTOCOL,
            "seed": SEED,
            "subsets": SUBSETS,
            "old_reference_slot": 32,
            "new_real_generation_reference_slot": 16,
            "all_old_mappings_exact": True,
            "old_totals": {
                "requests": sum(d["old_cost"]["requests"] for d in data),
                "completion_tokens": sum(
                    d["old_cost"]["completion_tokens"]["observed_total"] for d in data
                ),
            },
            "prefix_totals": dict(prefix_totals),
            "random_subset_totals": distributions(trial_totals),
            "inputs": results,
        }
        write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "report.md").write_text(report(summary))
        state.update(status="completed", completed_at=summary["completed_at"])
    except BaseException as exc:
        state.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        state["updated_at"] = now()
        write_json(args.output_dir / "metadata/run_status.json", state)


if __name__ == "__main__":
    main()
