#!/usr/bin/env python
"""Create a full-benchmark configuration without choosing the final Ours model."""

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jev_bench.config import EXTENDED_BENCHMARKS, load_config, validate


def set_ours_guidance(policy, *, gate=None, adaptive_top3_threshold=None):
    """Apply explicit Ours controls without changing the evaluation scope."""
    if gate is not None:
        if gate not in ('off', 'G1', 'G2'):
            raise ValueError('Ours gate must be off/G1/G2')
        if gate == 'off':
            policy['advice'] = 'prefer'
            policy.pop('gate', None)
        else:
            rule, threshold = (('respond_prob_ge', 0.1) if gate == 'G1' else ('top1_prob_lt', 0.3))
            policy.update(advice='gate', gate={'rule': rule, 'threshold': threshold, 'loop_guard': True})
    if adaptive_top3_threshold is not None:
        policy['hint_display'] = {'mode': 'adaptive_top3', 'threshold': adaptive_top3_threshold}


def prepare(
    template, conditions, *, model=None, temperature=None, gate=None, pa_mapping=None,
    adaptive_top3_threshold=None
):
    value = copy.deepcopy(template)
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("select unique conditions explicitly")
    if "pa_tool" in conditions:
        if pa_mapping is None:
            raise ValueError("PA-Tool requires an explicit frozen mapping")
        value["conditions"]["pa_tool"] = {
            "executor": "qwen3_4b",
            "advice": "none",
            "pa_tool_mapping": str(Path(pa_mapping).resolve(strict=True)),
        }
    elif pa_mapping is not None:
        raise ValueError("PA mapping supplied without a PA-Tool condition")
    if set(conditions) - set(value["conditions"]):
        raise ValueError("selected condition is absent from the template")
    value["conditions"] = {name: value["conditions"][name] for name in conditions}
    if "ours" in conditions:
        if model is None or temperature is None or gate is None:
            raise ValueError(
                "Ours requires an explicit model, temperature file and off/G1/G2"
            )
        advisor = value["advisors"][value["conditions"]["ours"]["advisor"]]
        advisor.update(
            model=str(Path(model).resolve(strict=True)),
            temperature_file=str(Path(temperature).resolve(strict=True)),
        )
        policy = value['conditions']['ours']
        policy.pop('hint_display', None)
        set_ours_guidance(policy, gate=gate, adaptive_top3_threshold=adaptive_top3_threshold)
    elif any(item is not None for item in (model, temperature, gate, adaptive_top3_threshold)):
        raise ValueError("Ours settings supplied without an Ours condition")
    executors = {policy["executor"] for policy in value["conditions"].values()}
    advisors = {
        policy["advisor"]
        for policy in value["conditions"].values()
        if "advisor" in policy
    }
    advisors.update(
        value["advisors"][name]["render_advisor"]
        for name in tuple(advisors)
        if "render_advisor" in value["advisors"][name]
    )
    value["executors"] = {
        name: settings
        for name, settings in value["executors"].items()
        if name in executors
    }
    value["advisors"] = {
        name: settings
        for name, settings in value["advisors"].items()
        if name in advisors
    }
    comparisons = (
        [["ours", name] for name in conditions if name != "ours"]
        if "ours" in conditions
        else []
    )
    value.update(output_root="outputs", trials=1, comparisons=comparisons)
    if value["benchmark"]["name"] in EXTENDED_BENCHMARKS:
        value["protocol"]["mode"] = "formal"
    value["task_ids"] = dict.fromkeys(value["task_ids"], "all")
    return validate(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--ours-model", type=Path)
    parser.add_argument("--ours-temperature", type=Path)
    parser.add_argument("--ours-gate", choices=("off", "G1", "G2"))
    parser.add_argument('--ours-adaptive-top3-threshold', type=float)
    parser.add_argument("--pa-mapping", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(
        load_config(args.template),
        args.conditions,
        model=args.ours_model,
        temperature=args.ours_temperature,
        gate=args.ours_gate,
        pa_mapping=args.pa_mapping,
        adaptive_top3_threshold=args.ours_adaptive_top3_threshold,
    )
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
