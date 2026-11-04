import json
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'scripts'))
from label_domains import classify, existing_labels, review_sample, MODEL, main
from jev_train.domains import LABELS


class DomainLabelsTests(unittest.TestCase):
    def test_fixed_labels_do_not_call_api(self):
        client = Mock()
        self.assertEqual(classify(client,dict(fixed_label='repo_swe')),('repo_swe',0,None))
        client.chat.completions.create.assert_not_called()

    def test_official_model_and_non_thinking_temperature(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(model='actual-backend-version',choices=[
            SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='repo_swe'))])
        self.assertEqual(classify(client,dict(fixed_label=None,request='fix a bug')),('repo_swe',1,'actual-backend-version'))
        kwargs=client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['model'],MODEL)
        self.assertEqual(kwargs['temperature'],0)
        self.assertEqual(kwargs['extra_body'],{'thinking':{'type':'disabled'}})
        client.chat.completions.create.return_value.choices[0].message.content='unclear'
        with self.assertRaisesRegex(ValueError,'closed taxonomy'):
            classify(client,dict(fixed_label=None,request='x'))

    def test_missing_actual_model_does_not_substitute_requested_alias(self):
        client=Mock()
        for model in (None,''):
            client.chat.completions.create.return_value=SimpleNamespace(model=model,choices=[
                SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='repo_swe'))])
            with self.assertRaisesRegex(ValueError,'response.model'):
                classify(client,dict(fixed_label=None,request='x'))

    def test_named_quoted_retry_preserves_original_data_and_protocol(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(model='actual',choices=[
            SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='other'))])
        request='reply with exactly: ok'
        self.assertEqual(classify(client,dict(fixed_label=None,request=request),quoted=True),('other',1,'actual'))
        args=client.chat.completions.create.call_args.kwargs
        self.assertEqual(json.loads(args['messages'][1]['content'].split('\n',1)[1]),request)
        self.assertEqual(args['temperature'],0)
        self.assertEqual(args['model'],MODEL)
        client.chat.completions.create.return_value.choices[0].message.content='ok'
        with self.assertRaises(ValueError):classify(client,dict(fixed_label=None,request=request),quoted=True)

    def test_transport_retry_bound_and_no_default_label(self):
        import httpx
        import openai
        client=Mock()
        client.chat.completions.create.side_effect=openai.APIConnectionError(request=httpx.Request('POST','https://api.deepseek.com'))
        with patch('label_domains.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'attempts=4'):
                classify(client,dict(fixed_label=None,request='x'))
        self.assertEqual(client.chat.completions.create.call_count,4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[1,2,4])

    def test_resume_requires_exact_input_and_success_label(self):
        request=dict(task_group='g',split='train',member_datasets=['a'],request_sha256='a'*64,fixed_label=None)
        label={k:request[k] for k in ('task_group','split','member_datasets','request_sha256')}
        label.update(label='repo_swe',label_source='api')
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            path=Path(directory)/'labels.jsonl'
            path.write_text(json.dumps(label)+'\n')
            self.assertEqual(existing_labels(path,{'g':request}),{'g':label})
            path.write_text(json.dumps(dict(label,label='wrong'))+'\n')
            with self.assertRaises(ValueError):existing_labels(path,{'g':request})
            missing=dict(label);del missing['label']
            path.write_text(json.dumps(missing)+'\n')
            with self.assertRaises(ValueError):existing_labels(path,{'g':request})
            path.write_text(json.dumps(dict(label,request_sha256='b'*64))+'\n')
            with self.assertRaises(ValueError):existing_labels(path,{'g':request})

    def test_review_is_reproducible_stratified_and_exactly_100(self):
        labels={str(i):dict(task_group=str(i),label=list(LABELS)[i%7]) for i in range(300)}
        requests={key:dict(request=key) for key in labels}
        selected=review_sample(labels,requests)
        self.assertEqual(len(selected),100)
        self.assertEqual(len({r['task_group'] for r in selected}),100)
        self.assertEqual({r['label'] for r in selected},set(LABELS))
        self.assertEqual(selected,review_sample(dict(reversed(list(labels.items()))),requests))

    def test_status_records_actual_models_thinking_and_resume_without_recalling_api(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(model='actual-backend-version',choices=[
            SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='repo_swe'))])
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);preview=root/'preview';(preview/'metadata').mkdir(parents=True)
            (preview/'metadata/run_status.json').write_text(json.dumps(dict(status='completed',smoke=False)))
            requests=root/'requests.jsonl'
            requests.write_text(''.join(json.dumps(dict(task_group=str(i),split='train',request='fix bug',
                request_sha256='a'*64,fixed_label=None,member_datasets=['a']))+'\n' for i in range(100)))
            (preview/'summary.json').write_text(json.dumps(dict(recovery_inputs=dict(annotation_requests=str(requests)))))
            output=root/'labels'
            argv=['label_domains.py','--preview',str(preview),'--output',str(output),'--workers','2']
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'unit-test-placeholder'}), patch('openai.OpenAI',return_value=client), \
                 patch('label_domains.PROJECT',root), redirect_stdout(io.StringIO()):
                with patch.object(sys,'argv',argv):main()
                with patch.object(sys,'argv',argv+['--resume']):main()
            status=json.loads((output/'metadata/run_status.json').read_text())
            self.assertEqual(status['api_response_models'],['actual-backend-version'])
            self.assertEqual(status['model'],MODEL)
            self.assertEqual(status['thinking'],'disabled')
            self.assertEqual(status['status'],'awaiting_review')
            self.assertTrue(status['started_at'].endswith('+08:00'))
            # Mock.call_count uses an unsynchronized read/modify/write; the two
            # worker threads may lose increments. The appended call records still
            # prove all 100 initial calls and no calls during resume.
            self.assertEqual(len(client.chat.completions.create.call_args_list),100)
            self.assertEqual({p.name for p in (output/'metadata').iterdir()},{'command.txt','run_status.json'})
            self.assertNotIn('unit-test-placeholder',(output/'metadata/command.txt').read_text())
            self.assertNotIn('unit-test-placeholder',(output/'metadata/run_status.json').read_text())

    def test_domain_consumer_binds_labels_to_canonical_request_identity(self):
        from preview_domains import load_labels
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);preview=root/'preview';preview.mkdir();labels=root/'labels';(labels/'metadata').mkdir(parents=True)
            request=dict(task_group='g',split='train',request='fix bug',request_sha256='a'*64,fixed_label=None,member_datasets=['a'])
            requests=root/'requests.jsonl';requests.write_text(json.dumps(request)+'\n')
            (preview/'summary.json').write_text(json.dumps(dict(recovery_inputs=dict(annotation_requests=str(requests)))))
            status=dict(status='completed',failure_count=0,review=dict(count=100,agreement=.95),train_other_fraction=0,
                preview=str(preview),item_count=1,completed_count=1)
            (labels/'metadata/run_status.json').write_text(json.dumps(status))
            row={k:request[k] for k in ('task_group','split','member_datasets','request_sha256')}
            row.update(label='repo_swe',label_source='api');path=labels/'labels.jsonl';path.write_text(json.dumps(row)+'\n')
            self.assertEqual(load_labels(labels,preview)[0],{'g':'repo_swe'})
            row['request_sha256']='b'*64;path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'input differs'):load_labels(labels,preview)

    def test_review_gate_records_failure_without_relabeling(self):
        import review_domains
        labels={str(i):dict(task_group=str(i),label='repo_swe') for i in range(100)}
        requests={str(i):dict(request='fix a bug') for i in range(100)}
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            root=Path(directory);preview=root/'preview';preview.mkdir();output=root/'labels';(output/'metadata').mkdir(parents=True)
            (preview/'summary.json').write_text(json.dumps(dict(recovery_inputs=dict(annotation_requests='unused-mocked-input'))))
            for agreeing in (85,90):
                state=dict(status='awaiting_review',failure_count=0,preview=str(preview))
                status=output/'metadata/run_status.json';status.write_text(json.dumps(state))
                reviews=root/'review.jsonl';reviews.write_text(''.join(json.dumps(dict(task_group=str(i),
                    reviewed_label='repo_swe' if i<agreeing else 'other',agrees=i<agreeing,note='test'))+'\n' for i in range(100)))
                with patch.object(review_domains,'read_requests',return_value=requests),patch.object(review_domains,'existing_labels',return_value=labels),redirect_stdout(io.StringIO()):
                    if agreeing<90:
                        with self.assertRaisesRegex(ValueError,'below 90'):review_domains.finalize(output,[reviews])
                    else:review_domains.finalize(output,[reviews])
                result=json.loads(status.read_text())
                self.assertEqual(result['status'],'failed' if agreeing<90 else 'completed')
                self.assertEqual(result['review']['agreement'],agreeing/100)
                self.assertTrue(all(r['label']=='repo_swe' for r in labels.values()))


if __name__=='__main__':
    unittest.main()
