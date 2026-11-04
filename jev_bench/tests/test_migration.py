"""Migration boundaries: no installation, remote API, weights or GPU execution."""
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jev_bench.advice import (
    NEUTRAL_MESSAGE,
    OPTION_ORDER,
    PREFER_TOOL,
    AdvisorFormatError,
    AdvisorServiceError,
    RenderClient,
    TypeSafeClient,
    advise,
    choose_hint,
)
from jev_bench.config import load_config, validate

from tests.support import PROJECT, advisor_settings, config


def migration_module(name):
    path = PROJECT.parent/'migration'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('migration_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(names=('a', 'Finish', 'RESPOND'), values=(.4, .35, .25)):
    return dict(top1=names[0], probabilities=[dict(name=n, probability=p) for n, p in zip(names, values, strict=True)])


class MigrationTests(unittest.TestCase):
    def test_code_publication_scans_only_the_explicit_source_branch(self):
        publication = migration_module("publish")
        with tempfile.TemporaryDirectory(dir=PROJECT / "tmp") as directory:
            repository = Path(directory)

            def git(*args, text_input=None):
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "-c",
                        "user.name=Fixture",
                        "-c",
                        "user.email=fixture@example.invalid",
                        *args,
                    ],
                    input=text_input,
                    text=True,
                ).strip()

            git("init", "-q")
            git("symbolic-ref", "HEAD", "refs/heads/development")
            (repository / "old.py").write_text("hf_" + "x" * 35 + "\n")
            git("add", "old.py")
            git("commit", "-q", "-m", "Local development history")
            tree = git("mktree", text_input="")
            clean = git("commit-tree", tree, "-m", "Clean public root")
            git("branch", "mainline", clean)
            publication.public_code_check(repository, "refs/heads/mainline")
            with self.assertRaisesRegex(
                ValueError, "possible credential in reachable history"
            ):
                publication.public_code_check(repository, "refs/heads/development")

        with (
            patch.object(sys, "argv", ["publish.py", "--stage", "code"]),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                publication.main()
            self.assertEqual(error.exception.code, 2)

    def test_adaptive_ranking_gate_and_stable_logical_candidates(self):
        policy = dict(advice='prefer', hint_display=dict(mode='adaptive_top3', threshold=.5))
        prediction = distribution()
        client = SimpleNamespace(predict=lambda *_: prediction, last_response=None, last_usage=None)
        result = advise(policy, client, [], [], [], action_names={'RESPOND': 'Finish'})
        self.assertEqual(result['shown'], ['a', 'Finish', 'RESPOND'])
        self.assertEqual(result['hint_kind'], 'ranked')
        self.assertIn('p=0.35', result['hint'])
        self.assertIn('p=0.25', result['hint'])
        self.assertIn('separate logical candidate', result['hint'])
        self.assertEqual(prediction['top1'], 'a')
        gate = dict(policy, advice='gate', gate=dict(rule='respond_prob_ge', threshold=.1, loop_guard=True))
        blocked = advise(gate, client, [], [], [])
        self.assertEqual((blocked['hint'], blocked['shown'], blocked['hint_kind']), (NEUTRAL_MESSAGE, [], 'neutral'))
        for probability, expected_kind in ((.499, 'ranked'), (.5, 'top1'), (.8, 'top1')):
            prediction = distribution(values=(probability, (1-probability)/2, (1-probability)/2))
            with self.subTest(probability=probability):
                result = advise(policy, client, [], [], [])
                self.assertEqual(result['hint_kind'], expected_kind)
        plain = dict(advice='prefer')
        self.assertEqual(choose_hint(plain, distribution(), [])[0], PREFER_TOOL.format(name='a'))

    def test_renderer_identity_and_non_json_are_not_action_format_fallbacks(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as directory:
            root = Path(directory)
            for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                (root/name).write_text('{}')
            settings = dict(type='jev_render', tokenizer=str(root), base_url='http://127.0.0.1:18768', timeout_seconds=120)
            renderer = RenderClient(settings)
            response = dict(renderer.tokenizer_identity, service_mode='render_only', status='ready',
                            prompt_budget=32704, option_order=OPTION_ORDER, duplicate_candidates='positional',
                            n_prompt_tokens=10, truncated=False, messages_dropped=0,
                            truncation_counts=dict(images_replaced=0, skipped_framework_greeting=0), latency_seconds=.01)
            with patch('jev_bench.advice.json_request', return_value=response):
                renderer.health()
                renderer.render([{'role': 'user', 'content': 'hello'}], [])
            with self.assertRaisesRegex(ValueError, 'cannot predict'):
                renderer.predict([], [])
            wrong = dict(response, service_mode='predict')
            with self.assertRaisesRegex(ValueError, 'identity'):
                renderer.identity(wrong)
            remote = dict(type='typesafe', base_url='https://api.typesafe.ai/v1', model='jev-1.14.0',
                          api_key_env='TYPESAFE_API_KEY', render_advisor='renderer', timeout_seconds=120)
            error = AdvisorFormatError('invalid JSON')
            error.raw_response = 'not-json'
            with patch.dict('os.environ', {'TYPESAFE_API_KEY': 'cpu-test'}):
                client = TypeSafeClient(remote, lambda *_: None, settings)
            with patch('jev_bench.advice.json_request', side_effect=error), self.assertRaises(AdvisorServiceError) as caught:
                advise({'advice': 'prefer'}, client, [{'role': 'user', 'content': 'hi'}], [], [])
            self.assertEqual(caught.exception.record['status'], 'advisor_error')
            self.assertEqual(client.records[-1]['provider_called'], False)
            self.assertEqual(caught.exception.record['raw_response'], 'not-json')

    def test_config_display_and_renderer_are_explicit_and_formal_gate_can_be_off(self):
        value = config()
        executor = value['conditions']['native']['executor']
        value['advisors'] = {'jev': advisor_settings()}
        value['conditions'] = {'ours': dict(executor=executor, advisor='jev', advice='prefer',
                                           hint_display=dict(mode='adaptive_top3', threshold=.5))}
        validate(value)
        for display in ({'mode': 'adaptive_top3'}, {'mode': 'adaptive_top3', 'threshold': .4}):
            bad = deepcopy(value)
            bad['conditions']['ours']['hint_display'] = display
            with self.assertRaises(ValueError):
                validate(bad)
        bad = deepcopy(value)
        bad['advisors']['jev'] = dict(type='jev_render', base_url='http://127.0.0.1:18768', timeout_seconds=120,
                                     tokenizer=value['advisors']['jev']['model'])
        with self.assertRaisesRegex(ValueError, 'cannot be an action advisor'):
            validate(bad)
        spec = importlib.util.spec_from_file_location('formal_migration', PROJECT/'scripts/prepare_formal_config.py')
        formal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formal)
        settings = advisor_settings()
        result = formal.prepare(value, ['ours'], model=settings['model'], temperature=settings['temperature_file'],
                                gate='off', adaptive_top3_threshold=.5)
        self.assertEqual(result['conditions']['ours']['advice'], 'prefer')
        self.assertNotIn('gate', result['conditions']['ours'])
        self.assertIn('hint_display', result['conditions']['ours'])

    def test_six_deployments_change_only_paths_endpoints_and_renderer_binding(self):
        deployment = migration_module('deploy')
        endpoints = dict(qwen='http://127.0.0.1:18001/v1', actionopd='http://127.0.0.1:18002/v1',
                         star='http://127.0.0.1:18003/v1', renderer='http://127.0.0.1:18768')
        for name in ('bfcl_mt', 'tau2', 'toolhop', 'apibank', 'toolsandbox', 'stabletoolbench'):
            original = load_config(PROJECT/f'configs/{name}_main_smoke.json')
            original['conditions'] = {k: v for k, v in original['conditions'].items() if k != 'ours'}
            original['comparisons'] = []
            result = deployment.relocate(original, Path('/target/workspace'), Path('/target/models'), endpoints)
            self.assertEqual(result['task_ids'], original['task_ids'])
            self.assertEqual(result['seed'], original['seed'])
            self.assertEqual(result['conditions'], original['conditions'])
            for key, executor in original['executors'].items():
                self.assertEqual({k: v for k, v in result['executors'][key].items() if k not in ('model_path', 'base_url')},
                                 {k: v for k, v in executor.items() if k not in ('model_path', 'base_url')})
            self.assertEqual(result['advisors']['renderer']['type'], 'jev_render')
            self.assertNotIn('ours_jev', result['advisors'])

    def test_environment_and_service_plans_do_not_execute_or_change_protocol(self):
        environment = migration_module('environments')
        services = migration_module('services')
        args = SimpleNamespace(role='train-runner', prefix=Path('/target/envs/train'),
                               conda_executable=Path('/target/miniconda/bin/conda'), workspace_root=Path('/target/workspace'),
                               causal_mode='source', cuda_home=Path('/usr/local/cuda-12.8'),
                               build_dir=Path('/target/workspace/experiments/jev_train/tmp/build'), causal_wheel=None)
        commands = environment.installation_plan(args)
        self.assertTrue(any('CAUSAL_CONV1D_FORCE_BUILD=TRUE' in c for c in commands))
        build = next(c for c in commands if 'CAUSAL_CONV1D_FORCE_BUILD=TRUE' in c)
        self.assertIn('--no-cache-dir', build)
        joint = [c for c in commands if 'install' in c and '-e' in c]
        self.assertEqual(len(joint), 1)
        self.assertEqual(joint[0].count('-c'), 2)
        server = SimpleNamespace(kind='qwen', python=Path('/target/envs/vllm/bin/python'),
                                 workspace_root=args.workspace_root, model_root=Path('/target/models'),
                                 port=18001, gpu='0', memory_utilization=.8, ours_model=None, ours_temperature=None)
        command = services.service_command(server)
        self.assertEqual(command[command.index('--max-model-len')+1], '131072')
        self.assertEqual(command[command.index('--reasoning-parser')+1], 'qwen3')
        self.assertEqual(command[command.index('--tool-call-parser')+1], 'hermes')
        server.kind, server.gpu = 'renderer', None
        command = services.service_command(server)
        self.assertIn('--render-only', command)
        self.assertNotIn('--model', command)

    def test_training_environment_check_requires_callable_fla_entrypoints(self):
        environment = migration_module('environments')
        with patch.dict('sys.modules', {'environments': environment}):
            checker = migration_module('check_env')
        for case, error in (('valid', None), ('unavailable', ImportError),
                            ('missing', AttributeError), ('not_callable', TypeError)):
            fla = SimpleNamespace(chunk_gated_delta_rule=Mock(), fused_recurrent_gated_delta_rule=Mock())
            if case == 'missing':
                del fla.chunk_gated_delta_rule
            elif case == 'not_callable':
                fla.fused_recurrent_gated_delta_rule = None
            modules = {'torch': SimpleNamespace(version=SimpleNamespace(cuda='12.8')),
                       'causal_conv1d': object(), 'causal_conv1d_cuda': object(),
                       'fla.ops.gated_delta_rule': fla}
            if case == 'unavailable':
                del modules['fla.ops.gated_delta_rule']

            def import_module(name):
                if name not in modules:
                    raise ImportError(name)
                return modules[name]

            args = SimpleNamespace(role='train', workspace_root=Path('/target/workspace'), write_resolved=False)
            output = io.StringIO()
            with self.subTest(case=case), \
                 patch.object(checker.argparse.ArgumentParser, 'parse_args', return_value=args), \
                 patch.object(checker.platform, 'python_version', return_value=environment.PYTHON['train']), \
                 patch.object(checker, 'editable_sources', return_value=[]), \
                 patch.object(checker.importlib, 'import_module', side_effect=import_module), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), redirect_stdout(output):
                if error is not None:
                    with self.assertRaises(error):
                        checker.main()
                else:
                    checker.main()
                    result = json.loads(output.getvalue())
                    self.assertIn('fla.ops.gated_delta_rule', result['binary_imports'])
                    self.assertFalse(result['gpu_inference_checked'])

    def test_publication_rejects_smoke_corrupt_downloads_and_unconfirmed_revisions(self):
        publication = migration_module('publish')
        revision = 'a'*40
        api = SimpleNamespace(dataset_info=Mock(return_value=SimpleNamespace(visibility=1, sha=revision)))
        self.assertEqual(publication.upload_revision(api, 'owner/dataset',
                         [SimpleNamespace(oid='b'*40), SimpleNamespace(oid=revision)]), revision)
        for result in ([], SimpleNamespace(oid=''), SimpleNamespace(oid='not-a-commit')):
            with self.subTest(result=result), self.assertRaises(ValueError):
                publication.upload_revision(api, 'owner/dataset', result)
        for info in (SimpleNamespace(visibility=0, sha=revision), SimpleNamespace(visibility=1, sha='b'*40)):
            api.dataset_info.return_value = info
            with self.assertRaises(ValueError):
                publication.checked_revision(api, 'owner/dataset', revision)
        api.dataset_info.return_value = SimpleNamespace(visibility=1, sha=revision)
        with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as directory:
            root = Path(directory)
            (root/'metadata').mkdir()
            status = dict(status='completed', smoke=False)
            (root/'metadata/run_status.json').write_text(json.dumps(status))
            # A downloader can return zero without creating the expected package.
            argv = ['publish.py', '--stage', 'data-download', '--revision', revision,
                    '--destination', str(root), '--execute']
            with patch('sys.argv', argv), patch.object(publication, 'private_hub', return_value=(api, 'not-logged')), \
                 patch.object(publication.subprocess, 'run'), redirect_stdout(io.StringIO()), \
                 self.assertRaises(FileNotFoundError):
                publication.main()
            files = {}
            from jev_train.portable_pool import file_identity
            for name in publication.POOL_FILES:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n')
                files[name] = file_identity(path)
            manifest = dict(format='jev_selection_pool_v1', dedup_seed=20261015, smoke=False, count=150234,
                            construction_dirty=[], files=files,
                            validation=dict(all_rows_readback_equal=True, delivered_rows_equal=38000,
                                            golden_replays=[8000,38000,76000]))
            (root/'manifest.json').write_text(json.dumps(manifest))
            publication.completed_pool(root)
            manifest['smoke'] = True
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'completed full frozen pool'):
                publication.completed_pool(root)
            manifest['smoke'] = False
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'pool/train.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError):
                publication.completed_pool(root)


    def test_stable_environment_check_rejects_missing_runtime_modules(self):
        environment = migration_module('environments')
        with patch.dict('sys.modules', {'environments': environment}):
            checker = migration_module('check_env')
        required = ('numpy', 'openai', 'httpx', 'pydantic', 'requests', 'yaml', 'tenacity')
        args = SimpleNamespace(role='stabletoolbench', workspace_root=Path('/target/workspace'), write_resolved=False)
        for missing in (None, *required):
            def import_module(name):
                if name == missing:
                    raise ModuleNotFoundError(name=name)
                return SimpleNamespace(__file__=f'/target/env/{name}.py')

            output = io.StringIO()
            with self.subTest(missing=missing), \
                 patch.object(checker.argparse.ArgumentParser, 'parse_args', return_value=args), \
                 patch.object(checker.platform, 'python_version', return_value=environment.PYTHON['stabletoolbench']), \
                 patch.object(checker.importlib, 'import_module', side_effect=import_module), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), redirect_stdout(output):
                if missing is not None:
                    with self.assertRaises(ModuleNotFoundError):
                        checker.main()
                    self.assertEqual(output.getvalue(), '')
                else:
                    checker.main()
                    result = json.loads(output.getvalue())
                    self.assertEqual(set(result['imports']), set(required))
                    self.assertFalse(result['gpu_inference_checked'])

    def test_source_plan_recovers_tau2_from_the_pinned_github_fork(self):
        preparation = migration_module('prepare')
        assets = preparation.validate_assets(json.loads(
            (PROJECT.parent/'migration/assets.json').read_text()))
        workspace = Path('/target/workspace')
        commands = [shlex.split(line) for line in preparation.plan(assets, workspace, None, 'sources')]
        target = str(workspace/'third_party/tau2-bench-verified')
        revision = '3d82db8f3579efe29609ab9daff058a9b7cb187c'
        self.assertIn(['git', 'clone', '--filter=blob:none', '--no-checkout',
                       'https://github.com/DX0369/tau2-bench-verified.git', target], commands)
        self.assertIn(['git', '-C', target, 'fetch', 'origin', revision], commands)
        self.assertIn(['git', '-C', target, 'checkout', '--detach', revision], commands)
        self.assertFalse(any('modelscope' in command or 'bundle' in command for command in commands))

    def test_data_download_requires_only_the_verified_pool(self):
        publication = migration_module('publish')
        revision = 'a'*40
        api = SimpleNamespace(dataset_info=Mock(return_value=SimpleNamespace(visibility=1, sha=revision)))
        files = sorted([
            'README.md', 'manifest.json', 'metadata/command.txt', 'metadata/run_status.json',
            'pool/train.jsonl', 'pool/annotations.jsonl', 'pool/selection_reference.json',
            'eval/val.jsonl', 'eval/test_id.jsonl', 'eval/test_ood.jsonl',
            'selections/task_reply_v2_38k_ids.txt', 'recipes/task_reply_v2_38k.json',
        ])
        with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as directory:
            destination = Path(directory)
            argv = ['publish.py', '--stage', 'data-download', '--revision', revision,
                    '--destination', directory]
            with patch('sys.argv', argv), patch.object(publication, 'private_hub') as hub, \
                 patch.object(publication.subprocess, 'run') as download, redirect_stdout(io.StringIO()):
                publication.main()
                hub.assert_not_called()
                download.assert_not_called()
            for error in (None, ValueError('pool differs')):
                with self.subTest(pool_error=error), patch('sys.argv', [*argv, '--execute']), \
                     patch.object(publication, 'private_hub', return_value=(api, 'not-logged')), \
                     patch.object(publication, 'completed_pool', side_effect=error) as pool, \
                     patch.object(publication.subprocess, 'run') as download, redirect_stdout(io.StringIO()):
                    if error is None:
                        publication.main()
                    else:
                        with self.assertRaisesRegex(ValueError, 'pool differs'):
                            publication.main()
                    pool.assert_called_once_with(destination)
                    download.assert_called_once_with(
                        ['modelscope', 'download', '--dataset', publication.DATASET, '--revision', revision,
                         '--local_dir', directory, '--include', *files], check=True)

    def test_smoke_guidance_overrides_preserve_the_six_benchmark_scopes(self):
        deployment = migration_module('deploy')
        settings = advisor_settings()
        with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as directory:
            root = Path(directory)
            for name in ('bfcl_mt', 'tau2', 'toolhop', 'apibank', 'toolsandbox', 'stabletoolbench'):
                template = PROJECT/f'configs/{name}_main_smoke.json'
                original = load_config(template)
                for gate, display in (('off', True), ('G2', False), (None, True)):
                    output = root/f'{name}_{gate}_{display}.json'
                    argv = ['deploy.py', '--template', str(template), '--workspace-root', str(PROJECT.parents[1]),
                            '--model-root', str(root/'models'), '--qwen-url', 'http://127.0.0.1:18001/v1',
                            '--ours-url', 'http://127.0.0.1:18770', '--ours-model', settings['model'],
                            '--ours-temperature', settings['temperature_file'], '--conditions', 'ours',
                            '--output', str(output)]
                    if gate is not None:
                        argv += ['--ours-gate', gate]
                    if display:
                        argv += ['--ours-adaptive-top3-threshold', '0.5']
                    with self.subTest(benchmark=name, gate=gate, display=display), \
                         patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                        deployment.main()
                        result = load_config(output)
                        for field in ('task_ids', 'output_root', 'trials', 'comparisons'):
                            self.assertEqual(result[field], original[field])
                        if 'mode' in original.get('protocol', {}):
                            self.assertEqual(result['protocol']['mode'], 'smoke')
                        policy = result['conditions']['ours']
                        if gate == 'off':
                            self.assertEqual(policy['advice'], 'prefer')
                            self.assertNotIn('gate', policy)
                        else:
                            self.assertEqual(policy['gate']['rule'], 'top1_prob_lt' if gate == 'G2' else 'respond_prob_ge')
                        self.assertEqual('hint_display' in policy, display)

    def test_smoke_overrides_cannot_modify_a_formal_template(self):
        deployment = migration_module('deploy')
        settings = advisor_settings()
        with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as directory:
            root = Path(directory)
            value = load_config(PROJECT/'configs/bfcl_mt_main_smoke.json')
            value.update(output_root='outputs', task_ids=dict.fromkeys(value['task_ids'], 'all'))
            template, output = root/'formal.json', root/'result.json'
            template.write_text(json.dumps(value))
            argv = ['deploy.py', '--template', str(template), '--workspace-root', str(PROJECT.parents[1]),
                    '--model-root', str(root/'models'), '--qwen-url', 'http://127.0.0.1:18001/v1',
                    '--ours-url', 'http://127.0.0.1:18770', '--ours-model', settings['model'],
                    '--ours-temperature', settings['temperature_file'], '--conditions', 'ours',
                    '--ours-gate', 'off', '--output', str(output)]
            with patch('sys.argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                deployment.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
