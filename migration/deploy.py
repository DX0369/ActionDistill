#!/usr/bin/env python3
"""Create explicit target-machine benchmark configs; never start a service or run."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPOSITORY/'jev_bench/src'), str(REPOSITORY/'jev_bench/scripts')]
from jev_bench.config import EXTENDED_BENCHMARKS, validate
from prepare_formal_config import prepare, set_ours_guidance

MODELS = {'qwen3-4b': ('Qwen3-4B', 'qwen'),
          'actionopd-4b': ('Qwen3-4B-ActionOPD-DeepSeek-V4-Flash', 'actionopd'),
          'star-4b': ('STAR-4b', 'star')}


def relocate(template, workspace, model_root, endpoints, *, ours_model=None, ours_temperature=None,
             pa_mapping=None):
    if not workspace.is_absolute() or not model_root.is_absolute():
        raise ValueError('workspace/model roots must be absolute')
    value = deepcopy(template)
    for settings in value['executors'].values():
        directory, role = MODELS[settings['model']]
        if not endpoints.get(role):
            raise ValueError(f'missing explicit {role} service endpoint')
        settings.update(model_path=str(model_root/directory), base_url=endpoints[role])
    used = {p['advisor'] for p in value['conditions'].values() if 'advisor' in p}
    for name in tuple(used):
        settings = value['advisors'][name]
        if settings['type'] == 'typesafe':
            if not endpoints.get('renderer'):
                raise ValueError('TypeSafe requires a standalone renderer endpoint')
            settings['render_advisor'] = 'renderer'
            value['advisors']['renderer'] = dict(type='jev_render', base_url=endpoints['renderer'],
                                                tokenizer=str(model_root/'Mapika-decider-2b'), timeout_seconds=120)
    used.update(value['advisors'][name]['render_advisor'] for name in tuple(used)
                if value['advisors'][name]['type'] == 'typesafe')
    value['advisors'] = {name: settings for name, settings in value['advisors'].items() if name in used}
    for settings in value['advisors'].values():
        if settings['type'] == 'jev_local':
            if ours_model is None or ours_temperature is None or not endpoints.get('ours'):
                raise ValueError('local advice requires explicit final weights, temperature and endpoint')
            settings.update(model=str(ours_model.resolve(strict=True)),
                            temperature_file=str(ours_temperature.resolve(strict=True)), base_url=endpoints['ours'])
    for policy in value['conditions'].values():
        if 'pa_tool_mapping' in policy:
            if pa_mapping is None:
                raise ValueError('PA requires the explicit completed target-machine mapping')
            policy['pa_tool_mapping'] = str(pa_mapping.resolve(strict=True))
    if value['benchmark']['name'] == 'apibank':
        value['protocol'].update(data_root=str(workspace/'experiments/jev_bench/data/API-Bank/test-data'),
                                 retriever_model_path=str(model_root/'paraphrase-MiniLM-L3-v2'))
    if value['benchmark']['name'] == 'stabletoolbench':
        value['protocol']['cache_archive'] = str(workspace/'experiments/jev_bench/data/StableToolBench/server_cache.zip')
    return validate(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    for role in ('qwen', 'actionopd', 'star', 'renderer', 'ours'):
        parser.add_argument('--'+role+'-url')
    parser.add_argument('--ours-model', type=Path)
    parser.add_argument('--ours-temperature', type=Path)
    parser.add_argument('--ours-gate', choices=('off', 'G1', 'G2'))
    parser.add_argument('--ours-adaptive-top3-threshold', type=float)
    parser.add_argument('--pa-mapping', type=Path)
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.template.read_text())
    if args.conditions is not None:
        if not args.conditions or len(args.conditions) != len(set(args.conditions)):
            raise ValueError('select unique conditions explicitly')
        if set(args.conditions)-value['conditions'].keys()-{'pa_tool'}:
            raise ValueError('unknown condition')
        if not args.formal and 'pa_tool' not in value['conditions'] and 'pa_tool' in args.conditions:
            raise ValueError('adding PA requires --formal and a complete explicit mapping')
        value['conditions'] = {name: policy for name, policy in value['conditions'].items() if name in args.conditions}
        value['comparisons'] = [pair for pair in value['comparisons'] if set(pair) <= value['conditions'].keys()]
    endpoints = {role: getattr(args, role+'_url') for role in ('qwen', 'actionopd', 'star', 'renderer', 'ours')}
    used_executors = {p['executor'] for p in value['conditions'].values()}
    value['executors'] = {k: v for k, v in value['executors'].items() if k in used_executors}
    # PA's original Qwen executor is required even when adding PA to a template.
    if args.formal and args.conditions and 'pa_tool' in args.conditions and 'qwen3_4b' not in value['executors']:
        value['executors']['qwen3_4b'] = json.loads(args.template.read_text())['executors']['qwen3_4b']
    if args.formal and args.conditions and 'pa_tool' in args.conditions and 'pa_tool' not in value['conditions']:
        if args.pa_mapping is None:
            raise ValueError('adding PA requires an explicit completed mapping')
        value['conditions']['pa_tool'] = dict(executor='qwen3_4b', advice='none',
                                              pa_tool_mapping=str(args.pa_mapping.resolve(strict=True)))
    value = relocate(value, args.workspace_root, args.model_root, endpoints, ours_model=args.ours_model,
                     ours_temperature=args.ours_temperature, pa_mapping=args.pa_mapping)
    if args.formal:
        value = prepare(value, list(value['conditions']), model=args.ours_model,
                        temperature=args.ours_temperature, gate=args.ours_gate, pa_mapping=args.pa_mapping,
                        adaptive_top3_threshold=args.ours_adaptive_top3_threshold)
    elif args.ours_gate is not None or args.ours_adaptive_top3_threshold is not None:
        if 'ours' not in value['conditions']:
            parser.error('gate/display changes require an Ours condition')
        if (value['output_root'] != 'tmp' or value['trials'] != 1 or value['comparisons']
                or any(ids == 'all' for ids in value['task_ids'].values())
                or (value['benchmark']['name'] in EXTENDED_BENCHMARKS and value['protocol']['mode'] != 'smoke')):
            parser.error('gate/display changes without --formal require a tmp, one-trial smoke template '
                         'with explicit task IDs and no comparisons')
        set_ours_guidance(value['conditions']['ours'], gate=args.ours_gate,
                          adaptive_top3_threshold=args.ours_adaptive_top3_threshold)
        value = validate(value)
    if not args.output.resolve().is_relative_to((args.workspace_root/'experiments').resolve()):
        parser.error('deployment config must be inside the target experiments Git repository')
    with args.output.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
