#!/usr/bin/env python3
"""Print role-specific rebuild commands; only --execute creates a new environment."""
import argparse
import os
import shlex
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

PYTHON = {'train': '3.11.16', 'runner': '3.11.16', 'train-runner': '3.11.16',
          'vllm': '3.11.15', 'apibank': '3.10.22', 'toolsandbox': '3.11.18',
          'stabletoolbench': '3.10.22'}
ROOT = Path(__file__).resolve().parent


def validate_causal_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('wheel must contain exactly one package metadata record')
        metadata = Parser().parsestr(archive.read(names[0]).decode())
    if (not isinstance(metadata['Name'], str) or not isinstance(metadata['Version'], str)
            or metadata['Name'].replace('_', '-').lower() != 'causal-conv1d' or metadata['Version'] != '1.7.0'):
        raise ValueError('explicit wheel must be causal-conv1d==1.7.0')


def editable_sources(role, workspace):
    sources = []
    if role in ('train', 'train-runner'):
        sources.append(('decider', workspace/'third_party/decider'))
    if role in ('runner', 'train-runner'):
        sources += [('bfcl_eval', workspace/'third_party/gorilla/berkeley-function-call-leaderboard'),
                    ('tau2', workspace/'third_party/tau2-bench-verified')]
    if role == 'toolsandbox':
        sources.append(('tool_sandbox', workspace/'third_party/ToolSandbox'))
    return sources


def installation_plan(args):
    if not args.prefix.is_absolute() or not args.conda_executable.is_absolute():
        raise ValueError('prefix and conda executable must be absolute paths')
    roles = ['train', 'runner'] if args.role == 'train-runner' else [args.role]
    if 'train' not in roles and any((args.causal_mode, args.cuda_home, args.build_dir, args.causal_wheel)):
        raise ValueError('causal build options only apply to training roles')
    python = args.prefix/'bin/python'
    pip = [str(python), '-m', 'pip']
    commands = [[str(args.conda_executable), 'create', '--yes', '--prefix', str(args.prefix),
                 '--override-channels', '--channel', 'conda-forge', 'python='+PYTHON[args.role], 'pip']]
    commands.append([*pip, 'install', 'wheel==0.45.1', 'setuptools>=77,<81'])
    if 'train' in roles or args.role == 'vllm':
        packages = ['torch==2.10.0']
        if args.role == 'vllm':
            packages += ['torchvision==0.25.0', 'torchaudio==2.10.0']
        commands.append([*pip, 'install', '--index-url', 'https://download.pytorch.org/whl/cu128', *packages])
    elif args.role in ('runner', 'apibank'):
        packages = (['torch==2.10.0'] if args.role == 'runner'
                    else ['torch==2.2.2+cpu', 'torchvision==0.17.2+cpu'])
        commands.append([*pip, 'install', '--index-url', 'https://download.pytorch.org/whl/cpu', *packages])
    if 'train' in roles:
        commands.append([*pip, 'install', 'ninja==1.13.0', 'packaging==26.3'])
        if args.causal_mode == 'source':
            if args.cuda_home is None or args.build_dir is None:
                raise ValueError('source build requires --cuda-home and --build-dir')
            if args.causal_wheel is not None:
                raise ValueError('source mode cannot also select a wheel')
            if (not args.cuda_home.is_absolute() or not args.build_dir.is_absolute()
                    or not args.build_dir.resolve().is_relative_to((args.workspace_root/'experiments/jev_train/tmp').resolve())):
                raise ValueError('CUDA path must be absolute; build directory must be inside jev_train/tmp')
            commands.append(['mkdir', '-p', str(args.build_dir/'tmp'), str(args.build_dir/'wheels')])
            commands.append(['env', 'CUDA_VISIBLE_DEVICES=', 'CUDA_HOME='+str(args.cuda_home),
                             'TMPDIR='+str(args.build_dir/'tmp'), 'TORCH_CUDA_ARCH_LIST=8.0',
                             'CAUSAL_CONV1D_FORCE_BUILD=TRUE', *pip, 'wheel', '--no-cache-dir', '--no-deps',
                             '--no-build-isolation', '--no-binary=causal-conv1d',
                             '--wheel-dir', str(args.build_dir/'wheels'), 'causal-conv1d==1.7.0'])
            commands.append([*pip, 'install', '--no-deps', '--no-index', '--find-links',
                             str(args.build_dir/'wheels'), 'causal-conv1d==1.7.0'])
        elif args.causal_mode == 'wheel':
            if args.causal_wheel is None or not args.causal_wheel.is_absolute():
                raise ValueError('wheel mode requires an explicit --causal-wheel')
            if args.cuda_home is not None or args.build_dir is not None:
                raise ValueError('wheel mode cannot also select a source build directory/CUDA SDK')
            commands.append([*pip, 'install', '--no-deps', str(args.causal_wheel)])
        else:
            raise ValueError('training requires explicit --causal-mode source or wheel')
    install = [*pip, 'install']
    for role in roles:
        install += ['-r', str(ROOT/'envs/requirements'/f'{role}.txt'),
                    '-c', str(ROOT/'envs/constraints'/f'{role}.txt')]
    sources = editable_sources(args.role, args.workspace_root)
    for _, path in sources:
        install += ['-e', str(path)]
    commands += [install, [*pip, 'check']]
    commands.append([str(python), '-B', str(ROOT/'check_env.py'), '--role', args.role,
                     '--workspace-root', str(args.workspace_root), '--write-resolved'])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=tuple(PYTHON), required=True)
    parser.add_argument('--conda-executable', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--causal-mode', choices=('source', 'wheel'))
    parser.add_argument('--cuda-home', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--causal-wheel', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.workspace_root.is_absolute():
        parser.error('--workspace-root must be absolute')
    commands = installation_plan(args)
    print('# Candidate constraints; clean installation and target GPU validation are still required.', flush=True)
    if args.execute:
        if args.prefix.exists():
            raise ValueError('environment prefix already exists; no overwrite or implicit repair')
        if not args.conda_executable.is_file() or not os.access(args.conda_executable, os.X_OK):
            raise ValueError('Conda executable is missing')
        if args.causal_mode == 'source' and (not (args.cuda_home/'bin/nvcc').is_file() or args.build_dir.exists()):
            raise ValueError('CUDA compiler missing or source build output already exists')
        if args.causal_mode == 'source':
            compiler = subprocess.check_output([str(args.cuda_home/'bin/nvcc'), '--version'], text=True)
            if 'release 12.8,' not in compiler:
                raise ValueError('source build requires the explicit CUDA 12.8 SDK matching Torch cu128')
        if args.causal_mode == 'wheel' and not args.causal_wheel.is_file():
            raise ValueError('explicit causal-conv1d wheel is missing')
        if args.causal_mode == 'wheel':
            validate_causal_wheel(args.causal_wheel)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
