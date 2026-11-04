#!/usr/bin/env python3
"""Print one explicit local service command. No GPU selection or startup is automatic."""
import argparse
import json
import shlex
from pathlib import Path

EXECUTORS = {'qwen': ('Qwen3-4B', 'qwen3-4b'),
             'actionopd': ('Qwen3-4B-ActionOPD-DeepSeek-V4-Flash', 'actionopd-4b'),
             'star': ('STAR-4b', 'star-4b')}


def service_command(args):
    if not args.python.is_absolute() or not args.workspace_root.is_absolute() or not args.model_root.is_absolute():
        raise ValueError('Python, workspace and model roots must be absolute')
    if type(args.port) is not int or not 1 <= args.port <= 65535:
        raise ValueError('port must be in 1..65535')
    env = ['env', '-u', 'HTTP_PROXY', '-u', 'HTTPS_PROXY', '-u', 'ALL_PROXY',
           '-u', 'http_proxy', '-u', 'https_proxy', '-u', 'all_proxy',
           'NO_PROXY=localhost,127.0.0.1', 'no_proxy=localhost,127.0.0.1']
    if args.kind in EXECUTORS:
        if args.gpu is None or args.memory_utilization is None or not 0 < args.memory_utilization <= 1:
            raise ValueError('vLLM requires an explicit GPU and memory utilization in (0,1]')
        directory, alias = EXECUTORS[args.kind]
        overrides = dict(rope_parameters=dict(rope_type='yarn', factor=4,
                                              original_max_position_embeddings=32768), rope_theta=1000000)
        return [*env, 'CUDA_VISIBLE_DEVICES='+args.gpu, str(args.python), '-m',
                'vllm.entrypoints.openai.api_server', '--model', str(args.model_root/directory),
                '--served-model-name', alias, '--host', '127.0.0.1', '--port', str(args.port),
                '--dtype', 'bfloat16', '--tensor-parallel-size', '1', '--max-model-len', '131072',
                '--max-num-seqs', '1', '--gpu-memory-utilization', str(args.memory_utilization),
                '--enable-auto-tool-choice', '--tool-call-parser', 'hermes', '--reasoning-parser', 'qwen3',
                '--hf-overrides', json.dumps(overrides, separators=(',', ':'))]
    command = [str(args.python), '-B', str(args.workspace_root/'experiments/jev_train/scripts/serve_jev.py'),
               '--port', str(args.port)]
    if args.kind == 'renderer':
        if args.gpu is not None:
            raise ValueError('CPU renderer does not take a GPU')
        return [*env, 'CUDA_VISIBLE_DEVICES=', *command, '--render-only', '--tokenizer',
                str(args.model_root/'Mapika-decider-2b')]
    if args.gpu is None or args.ours_model is None or args.ours_temperature is None:
        raise ValueError('Jev requires an explicit GPU, final model and calibrated temperature')
    if not args.ours_model.is_absolute() or not args.ours_temperature.is_absolute():
        raise ValueError('Jev weights and temperature paths must be absolute')
    return [*env, 'CUDA_VISIBLE_DEVICES='+args.gpu, *command, '--model', str(args.ours_model),
            '--temperature', str(args.ours_temperature)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=(*EXECUTORS, 'jev', 'renderer'), required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu')
    parser.add_argument('--memory-utilization', type=float)
    parser.add_argument('--ours-model', type=Path)
    parser.add_argument('--ours-temperature', type=Path)
    args = parser.parse_args()
    print(shlex.join(service_command(args)))


if __name__ == '__main__':
    main()
