#!/usr/bin/env python3
"""Prepare publication; remote writes require --execute and completed local assets."""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY = Path(__file__).resolve().parents[1]
GITHUB = 'https://github.com/DX0369/ActionDistill.git'
DATASET = 'dx0369/ActionDistill'
POOL_FILES = {'README.md', 'pool/train.jsonl', 'pool/annotations.jsonl', 'pool/selection_reference.json',
              'eval/val.jsonl', 'eval/test_id.jsonl', 'eval/test_ood.jsonl',
              'selections/task_reply_v2_38k_ids.txt', 'recipes/task_reply_v2_38k.json'}


def git(repository, *args, check=True):
    return subprocess.run(['git', '-C', str(repository), *args], check=check,
                          capture_output=True, text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})


def public_code_check(repository, source_ref):
    if git(repository, 'status', '--porcelain').stdout.strip():
        raise ValueError('commit the reviewed code before publishing')
    pattern = r'(ms-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{24,})'
    for commit in git(repository, 'rev-list', source_ref).stdout.splitlines():
        result = git(repository, 'grep', '-I', '-l', '-E', pattern, commit, check=False)
        if result.returncode not in (0, 1):
            raise ValueError('could not inspect tracked history for publication')
        if result.returncode == 0:
            # Only commit/path locators, never the matching credential value.
            raise ValueError('possible credential in reachable history: '+result.stdout.strip())


def completed_pool(pool):
    sys.path[:0] = [str(REPOSITORY/'jev_train/src'), str(REPOSITORY.parent/'third_party/decider'),
                    str(REPOSITORY/'dataset_tool_inventory/src')]
    from jev_train.portable_pool import DEDUP_SEED, FORMAT, verify_files
    status = json.loads((pool/'metadata/run_status.json').read_text())
    manifest = json.loads((pool/'manifest.json').read_text())
    if (status['status'] != 'completed' or status['smoke'] is not False or manifest['smoke'] is not False
            or manifest['format'] != FORMAT or manifest['dedup_seed'] != DEDUP_SEED
            or manifest['count'] != 150234 or manifest['construction_dirty']
            or set(manifest['files']) != POOL_FILES or manifest['validation']['all_rows_readback_equal'] is not True
            or manifest['validation']['delivered_rows_equal'] != 38000
            or manifest['validation']['golden_replays'] != [8000, 38000, 76000]):
        raise ValueError('data publication requires the completed full frozen pool and golden replays')
    verify_files(pool)
    return manifest


def private_hub(dataset):
    if version('modelscope') != '1.34.0':
        raise ValueError('this publication entry uses the pinned modelscope==1.34.0 API')
    from modelscope.hub.api import HubApi
    token = os.environ['MODELSCOPE_API_TOKEN']
    if not token.strip():
        raise ValueError('MODELSCOPE_API_TOKEN must be nonempty')
    api = HubApi()
    api.login(token)
    info = api.dataset_info(dataset)
    if type(info.visibility) is not int or info.visibility != 1:
        raise ValueError('the existing ModelScope dataset must be private; no automatic creation/change')
    return api, token


def checked_revision(api, dataset, revision):
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('the dataset operation must return an immutable 40-character revision')
    info = api.dataset_info(dataset, revision=revision)
    if type(info.visibility) is not int or info.visibility != 1 or info.sha != revision:
        raise ValueError('private visibility and the exact dataset revision could not be confirmed')
    return revision


def upload_revision(api, dataset, result):
    if isinstance(result, list):
        if not result:
            raise ValueError('dataset upload did not return any commits')
        result = result[-1]
    return checked_revision(api, dataset, result.oid)


def publish_data(pool, dataset):
    manifest = completed_pool(pool)
    api, token = private_hub(dataset)
    files = [*manifest['files'], 'manifest.json', 'metadata/command.txt', 'metadata/run_status.json']
    result = api.upload_folder(repo_id=dataset, repo_type='dataset', folder_path=str(pool), path_in_repo='',
                               allow_patterns=files, token=token,
                               commit_message='Publish verified frozen Jev selection pool')
    print('Private dataset:', dataset, 'revision:', upload_revision(api, dataset, result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('code', 'data', 'data-download', 'check-data'), required=True)
    parser.add_argument('--repository', type=Path, default=REPOSITORY)
    parser.add_argument('--github-url', default=GITHUB)
    parser.add_argument('--source-branch', help='explicit local branch to publish; required for the code stage')
    parser.add_argument('--branch', default='main')
    parser.add_argument('--dataset-id', default=DATASET)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.dataset_id):
        raise ValueError('dataset-id must be namespace/name, without credentials')
    if args.stage == 'code':
        url = urlsplit(args.github_url)
        if url.scheme != 'https' or url.hostname != 'github.com' or url.username or url.password:
            raise ValueError('use a credential-free HTTPS GitHub URL and local Git authentication')
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]*', args.branch):
            raise ValueError('invalid target branch')
        if args.source_branch is None or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]*', args.source_branch):
            parser.error('code publication requires an explicit --source-branch')
        source_ref = 'refs/heads/'+args.source_branch
        if args.execute:
            public_code_check(args.repository, source_ref)
            remote = git(args.repository, 'remote', 'get-url', 'origin', check=False)
            if remote.returncode == 0 and remote.stdout.strip() != args.github_url:
                raise ValueError('origin differs from the explicitly selected repository')
            if remote.returncode != 0:
                git(args.repository, 'remote', 'add', 'origin', args.github_url)
            result = git(args.repository, 'push', '--set-upstream', 'origin', source_ref+':refs/heads/'+args.branch)
            print(result.stdout, result.stderr, end='')
        else:
            print('# Public code publication; checked history and a clean commit are required.')
            print(shlex.join([sys.executable, '-B', str(Path(__file__).resolve()), '--stage', 'code',
                              '--repository', str(args.repository), '--github-url', args.github_url,
                              '--source-branch', args.source_branch, '--branch', args.branch, '--execute']))
    elif args.stage in ('data', 'check-data'):
        if args.pool is None:
            parser.error('--pool is required')
        if args.stage == 'check-data':
            completed_pool(args.pool)
            print('Full pool, checksums and golden replay evidence are complete; no upload performed.')
        elif args.execute:
            publish_data(args.pool, args.dataset_id)
        else:
            print('# Uses MODELSCOPE_API_TOKEN; token values never belong in command arguments.')
            print(shlex.join([sys.executable, '-B', str(Path(__file__).resolve()), '--stage', 'data',
                              '--pool', str(args.pool), '--dataset-id', args.dataset_id, '--execute']))
    else:
        if args.destination is None or args.revision is None or not re.fullmatch(r'[0-9a-f]{40}', args.revision):
            parser.error('download requires --destination and an explicit immutable --revision')
        command = ['modelscope', 'download', '--dataset', args.dataset_id, '--revision', args.revision,
                   '--local_dir', str(args.destination), '--include',
                   *sorted(POOL_FILES | {'manifest.json', 'metadata/command.txt', 'metadata/run_status.json'})]
        print(shlex.join(command))
        if args.execute:
            api, _ = private_hub(args.dataset_id)
            checked_revision(api, args.dataset_id, args.revision)
            subprocess.run(command, check=True)
            completed_pool(args.destination)
            print('Downloaded pool verified at revision:', args.revision)


if __name__ == '__main__':
    main()
