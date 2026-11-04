#!/usr/bin/env python3
"""Check source import locations and record a freshly installed environment."""
import argparse
import importlib
import json
import os
import platform
import sys
from importlib.metadata import distributions
from importlib.util import find_spec
from pathlib import Path

from environments import PYTHON, editable_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=tuple(PYTHON), required=True)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--write-resolved', action='store_true')
    args = parser.parse_args()
    if platform.python_version() != PYTHON[args.role]:
        raise ValueError('Python version differs from the role definition')
    observed = {}
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name, root in editable_sources(args.role, args.workspace_root):
        spec = find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root.resolve()):
            raise ValueError(f'{name}: imported source is outside the fixed checkout')
        observed[name] = spec.origin
    if args.role == 'stabletoolbench':
        for name in ('numpy', 'openai', 'httpx', 'pydantic', 'requests', 'yaml', 'tenacity'):
            observed[name] = importlib.import_module(name).__file__
    binary = []
    if args.role in ('train', 'train-runner', 'runner', 'vllm', 'apibank'):
        torch = importlib.import_module('torch')
        expected = '12.8' if args.role in ('train', 'train-runner', 'vllm') else None
        if torch.version.cuda != expected:
            raise ValueError('Torch CUDA/CPU variant differs from the role definition')
        binary.append('torch')
    if args.role in ('train', 'train-runner'):
        for name in ('causal_conv1d', 'causal_conv1d_cuda'):
            importlib.import_module(name)
            binary.append(name)
        fla = importlib.import_module('fla.ops.gated_delta_rule')
        for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
            if not callable(getattr(fla, name)):
                raise TypeError(f'fla.ops.gated_delta_rule.{name} must be callable')
        binary.append('fla.ops.gated_delta_rule')
    if args.write_resolved:
        path = Path(sys.prefix)/'resolved-packages.txt'
        path.write_text('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in distributions()))+'\n')
    print(json.dumps(dict(role=args.role, python=platform.python_version(), libc=platform.libc_ver(),
                          imports=observed, binary_imports=binary, gpu_inference_checked=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
