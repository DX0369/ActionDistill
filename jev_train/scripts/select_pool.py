#!/usr/bin/env python3
"""Select IDs from a frozen portable pool, without raw data, relabeling or rendering."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT.parents[1]/'third_party/decider'),
               str(PROJECT.parent/'dataset_tool_inventory/src')]
from jev_train.common import write_json
from jev_train.portable_pool import select_pool, validate_recipe, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-files', action='store_true', help='also hash all body/eval files after transfer')
    args = parser.parse_args()
    recipe = validate_recipe(json.loads(args.config.read_text()))
    manifest = json.loads((args.pool/'manifest.json').read_text())
    status = json.loads((args.pool/'metadata/run_status.json').read_text())
    if status['status'] != 'completed' or status['smoke'] != manifest['smoke']:
        parser.error('pool export must be completed with a matching smoke status')
    if manifest['smoke'] and not args.output.resolve().is_relative_to(PROJECT/'tmp'):
        parser.error('partial/smoke pool selections must remain under jev_train/tmp')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = args.output/'metadata'
    metadata.mkdir()
    (metadata/'command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable, *sys.argv])}\n')
    status = dict(status='running', started_at=datetime.now().astimezone().isoformat(), smoke=manifest['smoke'])
    write_json(metadata/'run_status.json', status)
    try:
        if args.verify_files:
            verify_files(args.pool)
        ids, summary = select_pool(args.pool, recipe)
        summary['pool'] = str(args.pool.resolve())
        summary['full_files_verified'] = args.verify_files
        write_json(args.output/'summary.json', summary)
        if not manifest['smoke'] and not summary['respond_gate_passed']:
            raise ValueError('RESPOND fraction outside [0.08,0.20]; no IDs written, no automatic adjustment')
        (args.output/'train_ids.txt').write_text('\n'.join(ids)+'\n')
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      ended_at=datetime.now().astimezone().isoformat())
        write_json(metadata/'run_status.json', status)
        raise
    status.update(status='completed', ended_at=datetime.now().astimezone().isoformat(), count=len(ids))
    write_json(metadata/'run_status.json', status)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
