import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]
R = P.parents[1]
sys.path[:0] = [str(P/'src'), str(R/'third_party/decider')]
import jev_train.data as data
from jev_train.data import SampleIndex, load_selected, validate
from jev_train.evaluation import training_majority_tools


class SampleIndexTests(unittest.TestCase):
    def setUp(self):
        (P/'tmp').mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=P/'tmp', prefix='sample_index_test_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'samples/train.jsonl'
        self.path.parent.mkdir()
        self.cache = self.root/'index_cache/samples/train.json'
        self.records = [dict(id=f'state_{i}', dataset='甲' if i % 2 else '乙', model='teacher',
                             task_group='g', outcome='success', candidate_level='S', context='上下文',
                             question='q', options=['b (params: path)', 'a: Read', 'RESPOND: reply'],
                             gold=[i % 3], n_prompt_tokens=100+i, messages_dropped=0,
                             truncation_counts={}) for i in range(31)]
        self.write_records(self.records)

    def write_records(self, records):
        self.path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')

    def direct_scan(self):
        entries = []
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    return entries
                row = validate(json.loads(line))
                entries.append(dict(offset=offset, id=row['id'], dataset=row['dataset'],
                                    n_prompt_tokens=row['n_prompt_tokens']))

    def assert_cache_error(self, reason=None):
        before = self.cache.read_bytes()
        with patch.object(SampleIndex, '_write_cache', side_effect=AssertionError('must not rebuild')):
            with self.assertRaises(ValueError) as error:
                SampleIndex(self.path)
        self.assertIn(str(self.cache), str(error.exception))
        self.assertIn('删除后重建', str(error.exception))
        if reason:
            self.assertIn(reason, str(error.exception))
        self.assertEqual(self.cache.read_bytes(), before)

    def test_scan_cache_and_direct_scan_are_identical(self):
        scanned = SampleIndex(self.path)
        self.assertEqual(scanned.source, 'scan')
        payload = json.loads(self.cache.read_text())
        stat = self.path.stat()
        self.assertEqual(payload['source'], dict(path=str(self.path.resolve()), size=stat.st_size,
                                                mtime_ns=stat.st_mtime_ns))
        self.assertEqual(payload['version'], 1)
        self.assertEqual(payload['count'], 31)
        self.assertEqual(set(payload), {'version','source','count','columns'})
        self.assertEqual(list(payload['columns']), ['offset','id','dataset','n_prompt_tokens'])
        self.assertEqual(list(self.cache.parent.iterdir()), [self.cache])
        with patch.object(data, 'validate', wraps=validate) as checked:
            cached = SampleIndex(self.path)
        self.assertEqual(cached.source, 'cache')
        self.assertEqual([call.args[0]['id'] for call in checked.call_args_list],
                         [f'state_{i}' for i in range(0, 31, 2)])
        self.assertEqual(scanned.entries, cached.entries)
        self.assertEqual(cached.entries, self.direct_scan())
        self.assertTrue(all(list(e) == ['offset','id','dataset','n_prompt_tokens'] for e in cached.entries))
        self.assertEqual([cached.read(e['offset']) for e in cached.entries], self.records)
        self.assertEqual(load_selected(cached, 9, 20261015), load_selected(scanned, 9, 20261015))

    def test_empty_and_small_files(self):
        for count in (0, 1, 9, 16):
            with self.subTest(count=count):
                if self.cache.exists():
                    self.cache.unlink()
                self.write_records(self.records[:count])
                scanned = SampleIndex(self.path)
                with patch.object(data, 'validate', wraps=validate) as checked:
                    cached = SampleIndex(self.path)
                self.assertEqual(checked.call_count, count)
                self.assertEqual(scanned.entries, cached.entries)
                self.assertEqual(cached.entries, self.direct_scan())

    def test_source_append_and_mtime_change_fail(self):
        for change in ('append', 'mtime'):
            with self.subTest(change=change):
                if self.cache.exists():
                    self.cache.unlink()
                self.write_records(self.records)
                SampleIndex(self.path)
                if change == 'append':
                    with self.path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(dict(self.records[0], id='extra'))+'\n')
                else:
                    stat = self.path.stat()
                    os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000_000))
                self.assert_cache_error('source')

    def test_source_change_during_scan_does_not_write_cache(self):
        for change in ('size', 'mtime'):
            with self.subTest(change=change):
                self.write_records(self.records)
                before = self.path.stat()
                def change_during_scan(row):
                    validate(row)
                    if row['id'] == self.records[-1]['id']:
                        if change == 'size':
                            with self.path.open('a', encoding='utf-8') as handle:
                                handle.write(json.dumps(dict(row, id='extra'))+'\n')
                            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
                        else:
                            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns+1_000_000_000))
                    return row
                with patch.object(data, 'validate', side_effect=change_during_scan), \
                     patch.object(SampleIndex, '_write_cache') as write_cache:
                    with self.assertRaisesRegex(ValueError, 'changed during scan'):
                        SampleIndex(self.path)
                write_cache.assert_not_called()
                self.assertFalse(self.cache.exists())

    def test_unsampled_offsets_and_ids_checked_for_entire_columns(self):
        SampleIndex(self.path)
        original = json.loads(self.cache.read_text())
        # Odd indices are outside the 16 probes in this 31-row fixture.
        for change in ('equal_offsets', 'reversed_offsets', 'duplicate_ids'):
            with self.subTest(change=change):
                payload = copy.deepcopy(original)
                if change == 'duplicate_ids':
                    payload['columns']['id'][3] = payload['columns']['id'][1]
                    reason = 'id column must be unique'
                else:
                    previous = 2 if change == 'equal_offsets' else 1
                    payload['columns']['offset'][3] = payload['columns']['offset'][previous]
                    reason = 'offset column must be strictly increasing'
                self.cache.write_text(json.dumps(payload))
                self.assert_cache_error(reason)

    def test_sampled_fields_and_offsets_fail(self):
        SampleIndex(self.path)
        original = json.loads(self.cache.read_text())
        for key, value in (('id', 'changed'), ('dataset', 'changed'), ('n_prompt_tokens', 1),
                           ('n_prompt_tokens', 114.0), ('n_prompt_tokens', True),
                           ('offset', original['columns']['offset'][13])):
            with self.subTest(key=key):
                payload = copy.deepcopy(original)
                payload['columns'][key][14] = value
                self.cache.write_text(json.dumps(payload))
                self.assert_cache_error('offset column' if key == 'offset' else 'sampled fields')

    def test_last_offset_and_shortened_cache_fail_eof(self):
        SampleIndex(self.path)
        original = json.loads(self.cache.read_text())
        for shortened in (False, True):
            with self.subTest(shortened=shortened):
                payload = copy.deepcopy(original)
                if shortened:
                    payload['count'] -= 1
                    for column in payload['columns'].values():
                        column.pop()
                else:
                    payload['columns']['offset'][-1] = self.path.stat().st_size+1
                self.cache.write_text(json.dumps(payload))
                self.assert_cache_error('EOF')

    def test_cache_schema_and_source_are_required(self):
        SampleIndex(self.path)
        original = json.loads(self.cache.read_text())
        variants = []
        for key in ('version', 'source', 'count', 'columns'):
            payload = copy.deepcopy(original)
            del payload[key]
            variants.append(payload)
        payload = copy.deepcopy(original); payload['version'] = 2; variants.append(payload)
        payload = copy.deepcopy(original); payload['source']['path'] = 'elsewhere'; variants.append(payload)
        payload = copy.deepcopy(original); payload['count'] = True; variants.append(payload)
        payload = copy.deepcopy(original); payload['count'] -= 1; variants.append(payload)
        payload = copy.deepcopy(original); payload['columns']['id'].pop(); variants.append(payload)
        for i, payload in enumerate(variants):
            with self.subTest(i=i):
                self.cache.write_text(json.dumps(payload))
                self.assert_cache_error()
        self.cache.write_text('{')
        self.assert_cache_error()

    def test_duplicate_id_fails_scan_without_writing_cache(self):
        self.write_records([self.records[0], self.records[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate sample id'):
            SampleIndex(self.path)
        self.assertFalse(self.cache.exists())

    def test_cached_read_still_validates_unsampled_rows(self):
        SampleIndex(self.path)
        stat = self.path.stat()
        self.records[1]['gold'] = [9]
        self.write_records(self.records)
        self.assertEqual(self.path.stat().st_size, stat.st_size)
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        cached = SampleIndex(self.path)
        self.assertEqual(cached.source, 'cache')
        with self.assertRaisesRegex(ValueError, 'gold'):
            cached.read(cached.entries[1]['offset'])

    def test_failed_atomic_replace_leaves_no_partial_cache(self):
        with patch.object(data.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                SampleIndex(self.path)
        self.assertEqual(list(self.cache.parent.iterdir()), [])
        self.assertEqual(SampleIndex(self.path).source, 'scan')

    def test_majority_is_independent_of_read_order(self):
        records = [dict(self.records[i], dataset='d', gold=gold)
                   for i, gold in enumerate(([0], [1], [0], [1], [2], [0,1]))]
        records += [dict(self.records[i+6], dataset='other', gold=[gold])
                    for i, gold in enumerate((0, 0, 1))]
        self.write_records(records)
        index = SampleIndex(self.path)
        selected = load_selected(index)
        random.Random(20261015).shuffle(selected)
        original = list(selected)
        expected = {'d':'a', 'other':'b'}
        for ordered in (selected, sorted(selected, key=lambda e:e['offset']), list(reversed(selected))):
            self.assertEqual(training_majority_tools(index.read(e['offset']) for e in ordered), expected)
        self.assertEqual(selected, original)


if __name__ == '__main__':
    unittest.main()
