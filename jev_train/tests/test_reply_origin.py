from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from jev_train.reply_origin import HERMES_FINAL, classify_origin


def view(family, content, record, before=None, originals=()):
    raw = ([] if before is None else [before]) + [dict(role='user', content=content), dict(role='assistant', content='done')]
    obj = SimpleNamespace(raw=raw, unit=dict(family=family, record=record, raw_records=originals),
                          capture=dict(context=dict(source='fixture')))
    obj.locator = lambda i: f'fixture:{i+1}'
    prior = dict(call_count_status='known', user_boundary_source=obj.locator(len(raw)-2),
                 framework_user_messages_ignored=0)
    return obj, len(raw)-1, prior


class ReplyOriginTests(unittest.TestCase):
    def test_exact_task_field_is_positive_not_role_or_close_text(self):
        for family, record in [('finance', {'instructions':'task'}),
                               ('toolathlon', {'config':json.dumps({'task_str':'task'})})]:
            v,i,p = view(family, 'task', record)
            self.assertEqual(classify_origin(v,i,p,'missing')['category'], 'verified_reply')
            v.raw[0]['content'] = 'task extra'
            self.assertEqual(classify_origin(v,i,p,'known')['category'], 'unknown')
        v,i,p = view('native_pi', 'a user request', None)
        self.assertEqual(classify_origin(v,i,p,'known')['category'], 'unknown')
        v,i,p = view('toolathlon', 'task', {'config':{}})
        with self.assertRaises(KeyError):
            classify_origin(v,i,p,'missing')

    def test_hermes_needs_both_exact_instruction_and_exhausted_tool_budget(self):
        warning = 'Iteration 15/15. Only 0 iteration(s) left'
        tool = '<tool_response>'+json.dumps({'content':{'_budget_warning':warning}})+'</tool_response>'
        record = dict(conversations=[{'from':'tool','value':tool}, {'from':'human','value':HERMES_FINAL},
                                     {'from':'gpt','value':'done'}])
        v,i,p = view('hermes', HERMES_FINAL, record, dict(role='tool',content=tool))
        actual = classify_origin(v,i,p,'missing')
        self.assertEqual(actual['category'], 'framework_forced')
        self.assertEqual(len(actual['evidence']), 2)
        for text in ('Iteration 14/15. Only 1 iteration(s) left', 'please stop soon'):
            changed = deepcopy(v)
            changed.unit['record']['conversations'][0]['value'] = tool.replace(warning,text)
            self.assertEqual(classify_origin(changed,i,p,'missing')['category'], 'unknown')
        p['call_count_status'] = 'missing_parent'
        self.assertEqual(classify_origin(v,i,p,'missing')['category'], 'unknown')

    def test_codex_requires_exact_adjacent_event_and_timestamp(self):
        text = [dict(type='input_text',text='task')]
        records = [(1,dict(type='response_item',timestamp='T',payload=dict(type='message',role='user',content=text))),
                   (2,dict(type='event_msg',timestamp='T',payload=dict(type='user_message',message='task')))]
        v,i,p = view('native_codex', text, None, originals=records)
        self.assertEqual(classify_origin(v,i,p,'null')['category'], 'verified_reply')
        for field, value in [('message','task extra'), ('type','agent_message')]:
            changed = deepcopy(v)
            changed.unit['raw_records'][1][1]['payload'][field] = value
            self.assertEqual(classify_origin(changed,i,p,'missing')['category'], 'unknown')
        changed = deepcopy(v); changed.unit['raw_records'][1][1]['timestamp']='later'
        self.assertEqual(classify_origin(changed,i,p,'missing')['category'], 'unknown')
        for value in (None,'',1):
            changed = deepcopy(v)
            for _, item in changed.unit['raw_records']:
                item['timestamp'] = value
            self.assertEqual(classify_origin(changed,i,p,'missing')['category'], 'unknown')
        changed = deepcopy(v); del changed.unit['raw_records'][1][1]['timestamp']
        self.assertEqual(classify_origin(changed,i,p,'missing')['category'], 'unknown')
        self.assertEqual(classify_origin(v,i,p,'conflict')['category'], 'unknown')
        p['framework_user_messages_ignored'] = 1
        self.assertEqual(classify_origin(v,i,p,'missing')['category'], 'unknown')


if __name__ == '__main__':
    unittest.main()
