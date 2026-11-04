import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'scripts'))
from review_domains import accept_existing
from preview_domains import load_labels
from compare_evals import read_domain_labels
from jev_train.domains import REVIEW_FAILURE,label_acceptance


class LabelAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=P/'tmp');self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name);self.preview=root/'preview';self.preview.mkdir()
        self.directory=root/'labels';(self.directory/'metadata').mkdir(parents=True)
        requests=root/'requests.jsonl'
        records=[dict(task_group=str(i),split='train',request='fix bug',request_sha256='a'*64,
            fixed_label=None,member_datasets=['a']) for i in range(100)]
        requests.write_text(''.join(json.dumps(r)+'\n' for r in records))
        (self.preview/'summary.json').write_text(json.dumps(dict(recovery_inputs=dict(annotation_requests=str(requests)))))
        labels=[dict(task_group=r['task_group'],split='train',label='repo_swe',label_source='api',
            member_datasets=['a'],request_sha256='a'*64) for r in records]
        (self.directory/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
        items=[dict(task_group=str(i),api_label='repo_swe',reviewed_label='repo_swe' if i<85 else 'other',
            agrees=i<85,note='test review') for i in range(100)]
        self.state=dict(status='failed',error=REVIEW_FAILURE,failure_count=0,preview=str(self.preview),
            item_count=100,completed_count=100,train_other_fraction=0,
            review=dict(count=100,agreement=.85,required=.90,items=items))
        self.path=self.directory/'metadata/run_status.json';self.path.write_text(json.dumps(self.state))

    def accept(self):
        with redirect_stdout(io.StringIO()):accept_existing(self.directory,'User accepts this existing 85% batch')
        return json.loads(self.path.read_text())

    def test_explicit_batch_acceptance_preserves_labels_and_original_review(self):
        before=(self.directory/'labels.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError,'no explicit'):load_labels(self.directory,self.preview)
        state=self.accept()
        self.assertEqual({k:v for k,v in state.items() if k!='acceptance'},self.state)
        self.assertEqual((self.directory/'labels.jsonl').read_bytes(),before)
        self.assertTrue(state['acceptance']['accepted_at'].endswith('+08:00'))
        self.assertEqual(len(load_labels(self.directory,self.preview)[0]),100)
        self.assertEqual(len(read_domain_labels(self.directory)),100)
        with self.assertRaisesRegex(ValueError,'already'):self.accept()

    def test_acceptance_cannot_admit_api_failure_or_other_failure(self):
        for key,value in [('failure_count',1),('error','missing input')]:
            state=dict(self.state,**{key:value});self.path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'only the original'):self.accept()
            self.assertNotIn('acceptance',json.loads(self.path.read_text()))

    def test_consumer_rejects_missing_or_mismatched_acceptance_fields(self):
        accepted=self.accept()
        for key,value in [('review_agreement',.86),('review_count',99),('labels_directory','/wrong'),
                          ('authority','model'),('original_required',.85),('reason',''),('accepted_at','2026-10-29T12:00:00')]:
            state=copy.deepcopy(accepted);state['acceptance'][key]=value
            with self.assertRaises(ValueError):label_acceptance(state,self.directory)
        state=copy.deepcopy(accepted);del state['acceptance']['authority']
        with self.assertRaises(KeyError):label_acceptance(state,self.directory)
        state=copy.deepcopy(accepted);state['failure_count']=1
        with self.assertRaises(ValueError):label_acceptance(state,self.directory)

    def test_acceptance_does_not_bypass_label_or_review_content_validation(self):
        state=copy.deepcopy(self.state);state['review']['items'][0]['api_label']='other'
        self.path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'saved review'):self.accept()
        self.path.write_text(json.dumps(self.state))
        path=self.directory/'labels.jsonl';path.write_text(path.read_text().replace('repo_swe','invalid',1))
        with self.assertRaisesRegex(ValueError,'closed taxonomy'):self.accept()


if __name__=='__main__':unittest.main()
