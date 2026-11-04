"""Strict JSONL boundary and permutation-aware soft targets."""
import heapq
import json
import os
from pathlib import Path
import random
import tempfile
from collections import Counter, defaultdict
import torch
from decider.model import collate as decider_collate
from .common import digest, rows
from .render import prompt_item

REQUIRED = {'id','dataset','model','task_group','outcome','candidate_level','context','question','options','gold','n_prompt_tokens','messages_dropped','truncation_counts'}


def tool_name(display):
    # Renderer reserves ': ' for the name/description boundary and ' (params:'.
    boundaries=[position for delimiter in (': ',' (params:') if (position:=display.find(delimiter))>=0]
    return display[:min(boundaries)] if boundaries else display


def validate(row):
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f'missing sample fields: {sorted(missing)}')
    if not all(isinstance(row[k], str) and row[k] for k in ('id','dataset','model','task_group','question')):
        raise ValueError('sample identifiers, teacher and question must be nonempty strings')
    if not isinstance(row['context'], str):
        raise ValueError('context must be string')
    options, gold = row['options'], row['gold']
    if not isinstance(options, list) or not 2 <= len(options) <= 255 or not all(isinstance(o,str) and o for o in options):
        raise ValueError('need 2..255 nonempty options')
    if not isinstance(gold, list) or not gold or len(set(gold)) != len(gold) or any(type(i) is not int or not 0 <= i < len(options) for i in gold):
        raise ValueError('gold must contain distinct valid option indices')
    if row['outcome'] not in {'success','failure','unknown'} or row['candidate_level'] not in {'S','N','H','I'}:
        raise ValueError('invalid outcome or candidate level')
    if type(row['n_prompt_tokens']) is not int or row['n_prompt_tokens']<=0:
        raise ValueError('n_prompt_tokens must be a positive integer')
    return row


def make_item(row, tokenizer, max_len=32768, seed=0):
    validate(row)
    rng = random.Random(digest(seed, row['id']))
    item = prompt_item(row['context'], row['question'], row['options'], row['gold'], tokenizer, rng)
    if len(item['ids']) > max_len:
        raise ValueError(f'{row["id"]}: complete prompt {len(item["ids"])} exceeds {max_len}; no silent truncation')
    if len(item['ids']) != row['n_prompt_tokens']:
        raise ValueError(f"{row['id']}: length invariant failed: encoded={len(item['ids'])}, stored={row['n_prompt_tokens']}")
    permutation = item['perms'][0]
    if sorted(permutation) != list(range(len(row['options']))):
        raise ValueError('prompt unexpectedly subsampled candidate options')
    target = torch.zeros(255, dtype=torch.float32)
    for position, original in enumerate(permutation):
        if original in row['gold']:
            target[position] = 1 / len(row['gold'])
    item.update(task=row['dataset'], ex_id=row['id'], targets=target, row=row)
    return item


def collate(items, pad_id):
    batch = decider_collate(items, pad_id)
    batch['targets'] = torch.stack([it['targets'] for it in items])
    return batch


class SampleIndex:
    """Only byte offsets and grouping/length fields stay resident; rows are read on demand."""
    def __init__(self, path):
        self.path = Path(path)
        cache_path = self.path.parent.parent/'index_cache'/self.path.parent.name/(self.path.stem+'.json')
        stat = self.path.stat()
        source = dict(path=str(self.path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if cache_path.exists():
            self.entries = self._load_cache(cache_path, source)
            self.source = 'cache'
            return
        self.entries = []
        seen = set()
        with self.path.open('rb') as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = validate(json.loads(line))
                if row['id'] in seen:
                    raise ValueError(f"{path}: duplicate sample id: {row['id']}")
                seen.add(row['id'])
                self.entries.append(dict(offset=offset, **{k:row[k] for k in ('id','dataset','n_prompt_tokens')}))
        after = self.path.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError(f'{self.path}: source size or mtime_ns changed during scan; cache not written')
        self._write_cache(cache_path, source)
        self.source = 'scan'

    def _load_cache(self, cache_path, source):
        try:
            with cache_path.open(encoding='utf-8') as handle:
                cache = json.load(handle)
            if cache['source'] != source:
                raise ValueError('source path, size or mtime_ns changed')
            if type(cache['version']) is not int or cache['version'] != 1:
                raise ValueError('unsupported cache version')
            count, columns = cache['count'], cache['columns']
            keys = ('offset','id','dataset','n_prompt_tokens')
            if type(count) is not int or count < 0 or set(columns) != set(keys):
                raise ValueError('invalid count or columns')
            if any(not isinstance(columns[k], list) or len(columns[k]) != count for k in keys):
                raise ValueError('count does not match column lengths')
            offsets = columns['offset']
            if any(type(offset) is not int or offset < 0 for offset in offsets):
                raise ValueError('invalid offset column')
            if any(left >= right for left, right in zip(offsets, offsets[1:])):
                raise ValueError('offset column must be strictly increasing')
            if len(set(columns['id'])) != count:
                raise ValueError('id column must be unique')
            entries = [dict(zip(keys, values)) for values in zip(*(columns[k] for k in keys))]
            # Endpoints plus equally spaced positions; small files are checked in full.
            positions = range(count) if count <= 16 else [i*(count-1)//15 for i in range(16)]
            with self.path.open('rb') as handle:
                if count == 0 and source['size'] != 0:
                    raise ValueError('empty cache does not match EOF')
                for i in positions:
                    entry = entries[i]
                    offset = entry['offset']
                    handle.seek(offset)
                    line = handle.readline()
                    if i == count-1 and offset+len(line) != source['size']:
                        raise ValueError('last offset + line length does not match EOF')
                    row = validate(json.loads(line))
                    if any(type(row[k]) is not type(entry[k]) or row[k] != entry[k] for k in keys[1:]):
                        raise ValueError(f'sampled fields differ at entry {i}')
            return entries
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f'{cache_path}: invalid sample index cache ({exc}); 删除后重建') from exc

    def _write_cache(self, cache_path, source):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = dict(version=1, source=source, count=len(self.entries),
                     columns={k:[entry[k] for entry in self.entries]
                              for k in ('offset','id','dataset','n_prompt_tokens')})
        # Unique temporary names also allow simultaneous DDP/evaluation readers.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_path.parent,
                                         prefix=cache_path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(cache, handle, ensure_ascii=False, separators=(',', ':'))
                handle.close()
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, offset):
        with self.path.open('rb') as handle:
            handle.seek(offset)
            return validate(json.loads(handle.readline()))


def load_selected(index, limit=0, seed=0, *, min_per_dataset=False):
    if limit < 0:
        raise ValueError('sample limit must be nonnegative')
    entries = index.entries
    counts = Counter(r['dataset'] for r in entries)
    total = len(entries)
    if limit == 0 or total <= limit:
        return list(entries)
    minimum = int(min_per_dataset)
    if minimum and limit < len(counts):
        raise ValueError('monitor limit must allocate at least one sample per dataset')
    remaining = limit - minimum * len(counts)
    population = total - minimum * len(counts)
    quotas = {d: minimum + remaining * (n-minimum) // population for d,n in counts.items()}
    rest = sorted(counts, key=lambda d: (-(remaining * (counts[d]-minimum) % population), d))
    for d in rest[:limit - sum(quotas.values())]:
        quotas[d] += 1
    heaps = defaultdict(list)
    for r in entries:
        d = r['dataset']; cap = quotas[d]
        if cap == 0:
            continue
        entry = (-int(digest(seed,r['id']),16), r['id'], r)
        if len(heaps[d]) < cap: heapq.heappush(heaps[d], entry)
        elif entry[:2] > heaps[d][0][:2]: heapq.heapreplace(heaps[d], entry)
    return sorted((entry[2] for h in heaps.values() for entry in h), key=lambda r: digest(seed,r['id']))
