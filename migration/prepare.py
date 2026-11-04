#!/usr/bin/env python3
"""Print migration commands or audit existing assets; never download or execute a plan."""
import argparse
import json
import shlex
import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def command(*args):
    return shlex.join([str(arg) for arg in args])


def validate_assets(assets):
    if assets['format'] != 'actiondistill_migration_v1':
        raise ValueError('unsupported migration asset manifest')
    for group in ('git_checkouts', 'hf_datasets', 'models'):
        if not isinstance(assets[group], list) or not assets[group]:
            raise ValueError(f'{group} must be a nonempty asset list')
        for asset in assets[group]:
            if any(not isinstance(asset[k], str) or not asset[k] for k in ('path', 'source', 'revision')):
                raise ValueError('asset path/source/revision must be nonempty strings')
            if Path(asset['path']).is_absolute() or '..' in Path(asset['path']).parts:
                raise ValueError('asset path must stay relative to its root')
            if len(asset['revision']) != 40 or any(c not in '0123456789abcdef' for c in asset['revision']):
                raise ValueError('asset revision must be an exact commit')
    return assets


def hf_command(asset, root, dataset=False):
    args = ['env', '-u', 'HTTP_PROXY', '-u', 'HTTPS_PROXY', '-u', 'ALL_PROXY',
            '-u', 'http_proxy', '-u', 'https_proxy', '-u', 'all_proxy',
            'HF_ENDPOINT=https://hf-mirror.com', 'hf', 'download', asset['source']]
    args += asset['files'] if dataset else []
    if dataset:
        args += ['--repo-type', 'dataset']
    args += ['--revision', asset['revision'], '--local-dir', str(root/asset['path'])]
    if 'exclude' in asset:
        args += ['--exclude', *asset['exclude']]
    return command(*args)


def plan(assets, workspace, models, stage):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail']
    if stage in ('sources', 'all'):
        lines.append(command('mkdir', '-p', workspace/'third_party'))
        for asset in assets['git_checkouts']:
            target = workspace/asset['path']
            lines.append(command('test', '!', '-e', target))
            lines.append(command('git', 'clone', '--filter=blob:none', '--no-checkout', asset['source'], target))
            lines.append(command('git', '-C', target, 'fetch', 'origin', asset['revision']))
            lines.append(command('git', '-C', target, 'checkout', '--detach', asset['revision']))
    if stage in ('benchmarks', 'all'):
        lines.extend(hf_command(a, workspace, dataset=True) for a in assets['hf_datasets'])
    if stage in ('models', 'all'):
        lines.extend(hf_command(a, models) for a in assets['models'])
    return lines


def check(assets, workspace):
    """Audit sources and official files without importing benchmark packages."""
    for asset in assets['git_checkouts']:
        target = workspace/asset['path']
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != asset['revision']:
            raise ValueError(f'{target}: revision differs')
        for name in asset['required_files']:
            if not (target/name).exists():
                raise ValueError(f'{target/name}: missing')
    for asset in assets['hf_datasets']:
        target = workspace/asset['path']
        for name in asset['files']:
            if not (target/name).is_file():
                raise ValueError(f'{target/name}: missing')
        for name in asset['metadata_files']:
            metadata = target/'.cache/huggingface/download'/(name+'.metadata')
            with metadata.open() as handle:
                if handle.readline().strip() != asset['revision']:
                    raise ValueError(f'{metadata}: revision differs')
        if 'size_bytes' in asset and (target/asset['files'][0]).stat().st_size != asset['size_bytes']:
            raise ValueError(f'{target}: published cache size differs')
    # Reuse the existing runtime checks, including untracked source and the
    # narrow Python-cache exception, instead of weakening them for migration.
    source = Path(__file__).resolve().parents[1]/'jev_bench/src/jev_bench/config.py'
    spec = spec_from_file_location('migration_source_checks', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKSPACE_ROOT = workspace
    for name in assets['benchmarks']:
        path, revision, _ = module.BENCHMARKS[name]
        module.check_checkout({'benchmark': dict(name=name, path=path, commit=revision)})
    if module.git_dirty(workspace/'third_party/decider', ('decider',)):
        raise ValueError('decider runtime has uncommitted source files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path(__file__).with_name('assets.json'))
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--stage', choices=('sources', 'benchmarks', 'models', 'all'), default='all')
    parser.add_argument('--check', action='store_true', help='audit existing source/data assets only')
    args = parser.parse_args()
    assets = validate_assets(json.loads(args.assets.read_text()))
    workspace = args.workspace_root.resolve()
    if args.check:
        check(assets, workspace)
        print('Source and benchmark assets match; environments/services/models are not checked.')
        return
    if args.stage in ('all', 'models') and args.model_root is None:
        parser.error('--model-root is required for model download commands')
    models = args.model_root.resolve() if args.model_root else None
    print('\n'.join(plan(assets, workspace, models, args.stage)))


if __name__ == '__main__':
    main()
