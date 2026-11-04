from collections import Counter
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P/'src'), str(P/'scripts'), str(P.parents[1]/'third_party/decider'),
                str(P.parent/'dataset_tool_inventory/src')]
from jev_train.domains import LABELS
from jev_train.common import digest
from jev_train.task_selection import domain_allocation, domain_order, select, trajectory_order
from test_dedup import plan


def state(sid, task='task', source='a', teacher='teacher', trajectory='trajectory', step=0, reply=False):
    return plan(sid, trajectory, task_group=task, dataset=source, model=teacher,
                turn_index=step, gold_names=['RESPOND'] if reply else ['run'])


class TaskSelectionTests(unittest.TestCase):
    def test_fixed_soft_domain_mix_and_capacity_redistribution(self):
        records, labels, reference = {}, {}, []
        for d, domain in enumerate(LABELS):
            labels[domain] = domain
            for i in range(100):
                sid = f'{domain}/{i}'
                records[sid] = state(sid, task=domain)
                if i < d+1:
                    reference.append(sid)
        reference = Counter(labels[records[sid]['task_group']] for sid in reference)
        allocation = domain_allocation(records, labels, reference, 112)
        for d, domain in enumerate(LABELS):
            self.assertAlmostEqual(allocation[domain]['weight'], .5*(d+1)/28+.5/7)
            self.assertEqual(allocation[domain]['quota'], 2*(d+1)+8)
        scarce = list(LABELS)[0]
        records = {sid: row for sid,row in records.items()
                   if labels[row['task_group']] != scarce or sid == f'{scarce}/0'}
        allocation = domain_allocation(records, labels, reference, 112)
        self.assertEqual(allocation[scarce]['quota'], 1)
        self.assertEqual(sum(v['quota'] for v in allocation.values()), 112)
        self.assertTrue(all(v['quota'] <= v['capacity'] for v in allocation.values()))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            domain_allocation(records, labels, reference, 1000)

    def test_shared_task_across_sources_and_teacher_rotation(self):
        records = [state(f'shared/{s}/{t}/{i}', source=s, teacher=t, step=i)
                   for s in ('a','b') for t in ('x','y') for i in range(10)]
        records += [state(f'{g}/{i}', task=g, step=i) for g in ('second','third') for i in range(5)]
        order = list(domain_order(records, {}, 7))
        self.assertEqual(len({r['task_group'] for r in order[:3]}), 3)
        shared = [r for r in order if r['task_group']=='task']
        self.assertEqual({r['dataset'] for r in shared[:2]}, {'a','b'})
        self.assertEqual({(r['dataset'], r['model']) for r in shared[:4]},
                         {('a','x'), ('a','y'), ('b','x'), ('b','y')})
        self.assertEqual({r['id'] for r in order}, {r['id'] for r in records})
        self.assertEqual(len(order), len(records))
        self.assertEqual(order, list(domain_order(list(reversed(records)), {}, 7)))

    def test_reply_weight_is_per_remaining_state_and_all_states_remain_available(self):
        # One R and three O gives P(R first)=2/5, not a fixed R fraction of 2/3.
        hits = 0
        for t in range(1000):
            records = [state(f'{t}/{i}', trajectory=str(t), step=i, reply=i==0) for i in range(4)]
            classes = {records[0]['id']: 'verified_reply'}
            ordered = list(trajectory_order(records, classes, 7))
            hits += ordered[0]['gold_names'] == ['RESPOND']
            self.assertEqual({r['id'] for r in ordered}, {r['id'] for r in records})
            self.assertEqual(ordered, list(trajectory_order(list(reversed(records)), classes, 7)))
        self.assertTrue(370 <= hits <= 430, hits)
        for classification in ('verified_reply', 'framework_forced', 'unknown'):
            records = [state(str(i), step=i, reply=True) for i in range(12)]
            classes = {r['id']:classification for r in records}
            self.assertEqual(len(list(trajectory_order(records, classes, 7))), 12)

    def test_single_state_per_task_spreads_across_available_trajectory(self):
        records = [state(f'{t}/{i}', task=str(t), trajectory=str(t), step=i)
                   for t in range(1000) for i in range(50)]
        first_pass = list(islice(domain_order(records, {}, 7), 1000))
        bins = Counter(r['turn_index']//5 for r in first_pass)
        self.assertEqual(len({r['task_group'] for r in first_pass}), 1000)
        self.assertTrue(all(70 <= bins[i] <= 130 for i in range(10)), bins)

    def test_new_source_cells_input_immutability_and_invalid_boundaries(self):
        rows = {str(i):state(str(i), task=str(i), source='reference' if i<7 else 'unseen', reply=i==7)
                for i in range(21)}
        labels = {str(i):list(LABELS)[i%7] for i in range(21)}
        classes = {'7':'unknown'}
        reference = dict.fromkeys(LABELS, 1)
        original = deepcopy((rows, labels, classes))
        ids, allocation = select(rows, labels, reference, classes, 21, 7)
        self.assertEqual(set(ids), set(rows))
        self.assertEqual((rows, labels, classes), original)
        self.assertEqual((ids, allocation), select(dict(reversed(list(rows.items()))), labels,
                                                 reference, classes, 21, 7))
        with self.assertRaisesRegex(ValueError, 'exactly'):
            select(rows, labels, reference, {}, 7, 7)
        with self.assertRaisesRegex(ValueError, 'classification'):
            select(rows, labels, reference, {'7':'natural_by_default'}, 7, 7)
        for field in ('turn_index','dataset','model','trajectory_id','gold_names'):
            changed = deepcopy(rows)
            del changed['0'][field]
            with self.assertRaises(KeyError):
                select(changed, labels, reference, classes, 7, 7)
        changed = deepcopy(rows); changed['0']['turn_index'] = None
        with self.assertRaisesRegex(ValueError, 'turn_index'):
            select(changed, labels, reference, classes, 7, 7)

    def test_training_exposure_uses_training_id_order_not_selection_stream(self):
        from preview_task_reply import summarize_selection
        rows = {str(i):state(str(i), task=str(i)) for i in range(9)}
        for i,row in enumerate(rows.values()):
            row.update(strict_loop=False, n_prompt_tokens=8000+1000*i)
        labels = dict.fromkeys(rows,'repo_swe')
        positions = {sid:i for i,sid in enumerate(rows)}
        first = summarize_selection(list(rows),rows,{},labels,positions,[])
        second = summarize_selection(list(reversed(rows)),rows,{},labels,positions,[])
        self.assertEqual(first['schedule'],second['schedule'])
        self.assertEqual(first['planned_linear_loss_shares'],second['planned_linear_loss_shares'])
        self.assertAlmostEqual(sum(first['planned_linear_loss_shares'].values()),1)

    def test_original_source_locator_is_required_to_match(self):
        from preview_task_reply import restore_origins
        capture = dict(dataset='a',trajectory_id='t',messages=[dict(role='user',content='task'),
            dict(role='assistant',content='done')], context=dict(strip_think=False,model='teacher',source='source:1'))
        target = state(digest('a','t',1),trajectory='t',step=1,reply=True)
        target.update(source='source:1',original_gold=['RESPOND'],next_event='trajectory_end')
        unit = dict(capture=capture,targets=[target],family='finance',raw_records=(),prefix=None,
                    record=dict(instructions='task',trace_jsonl=json.dumps(dict(type='model',content='done'))))
        feature = dict(id=target['id'],dataset='a',split='train',next_event='trajectory_end',
                       stop_reason_abnormal=False,turn_calls_before=0)
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            path = Path(directory)/'features.jsonl';path.write_text(json.dumps(feature)+'\n')
            with patch('preview_task_reply.source_units', return_value=[unit]):
                self.assertEqual(restore_origins({target['id']:target},path,{})[target['id']]['category'],
                                 'verified_reply')
                target['source']='wrong:1'
                with self.assertRaisesRegex(ValueError,'locator differs'):
                    restore_origins({target['id']:target},path,{})


if __name__ == '__main__':
    unittest.main()
