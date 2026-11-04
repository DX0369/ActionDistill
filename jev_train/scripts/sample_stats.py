#!/usr/bin/env python3
"""Stream existing samples without reconstruction; write a separate statistics run."""
import argparse
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(REPO/'third_party/decider'), str(REPO/'experiments/dataset_tool_inventory/src')]
from transformers import AutoTokenizer
from decider.prompt import LETTERS, NARROW, _options_ids
from jev_train.common import rows, write_json, now
from jev_train.data import validate, tool_name

SPLITS = ('train', 'val', 'test_id', 'test_ood')


def quantiles(counter):
    """Exact nearest-rank quantiles over a compact frequency table."""
    n = sum(counter.values())
    if not n:
        return {'p50': None, 'p90': None, 'max': None}
    ordered = sorted(counter.items())
    def at(fraction):
        rank = math.ceil(fraction*n)
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative >= rank:
                return value
    return {'p50': at(.5), 'p90': at(.9), 'max': ordered[-1][0]}


def summarize(bucket):
    n = bucket['count']
    return {'count': n, 'option_tokens': quantiles(bucket['tokens']),
            'option_token_fraction': quantiles(bucket['fractions']),
            'history_with_deletion_count': bucket['deleted'],
            'prefix_only_visible_proxy_count': bucket['prefix_proxy'],
            'prefix_only_visible_proxy_fraction': bucket['prefix_proxy']/n if n else None,
            'action_types': dict(bucket['actions'])}


def bucket():
    return {'count': 0, 'tokens': Counter(), 'fractions': Counter(),
            'deleted': 0, 'prefix_proxy': 0, 'actions': Counter()}


def collect(samples, tokenizer, ood_datasets):
    report = json.loads((samples/'build_report.json').read_text())
    harness = report['harness_distributions']
    @lru_cache(maxsize=4096)
    def option_count(question, options):
        if len(options)>NARROW:
            return len(_options_ids(tokenizer, SimpleNamespace(options=options), list(range(len(options)))))
        head = f'\n\nQuestion: {question}\nOptions:'
        body = ''.join(f'\n({LETTERS[j]}) {option}' for j,option in enumerate(options))
        encoded = tokenizer(head+body+'\nAnswer: (', add_special_tokens=False, return_offsets_mapping=True)
        start,end = len(head),len(head)+len(body)
        return sum(a<end and b>start for a,b in encoded['offset_mapping'])
    totals = bucket(); by_split={}; by_dataset=defaultdict(bucket); by_dataset_split={}; id_ood=Counter(); seen=set()
    for split in SPLITS:
        current=bucket(); datasets=defaultdict(bucket)
        for row in rows(samples/(split+'.jsonl')):
            validate(row)
            if row['id'] in seen:raise ValueError(f"duplicate sample id: {row['id']}")
            seen.add(row['id'])
            tokens=option_count(row['question'],tuple(row['options']))
            denominator=row['n_prompt_tokens']
            if type(denominator) is not int or denominator<=0:raise ValueError('n_prompt_tokens must be a positive integer')
            dropped=row['messages_dropped']
            if type(dropped) is not int or dropped<0:raise ValueError('messages_dropped must be a nonnegative integer')
            marker=f'[... {dropped} earlier messages omitted ...]'
            proxy=dropped>0 and row['context'].rstrip().endswith('\n'+marker)
            action='TOOL_SET' if len(row['gold'])>1 else 'RESPOND' if tool_name(row['options'][row['gold'][0]])=='RESPOND' else 'TOOL'
            for stats in (totals,current,datasets[row['dataset']],by_dataset[row['dataset']]):
                stats['count']+=1;stats['tokens'][tokens]+=1;stats['fractions'][tokens/denominator]+=1
                stats['deleted']+=int(dropped>0);stats['prefix_proxy']+=int(proxy);stats['actions'][action]+=1
            if split=='test_ood' and row['dataset'] not in ood_datasets:id_ood[row['dataset']]+=1
            if current['count']%25000==0:print(f'{split}: {current["count"]} rows',flush=True)
        # The report is the only exact harness source: samples have no harness field.
        expected_actions=Counter()
        for entry in harness['splits'][split]:expected_actions.update(entry['action_types'])
        if expected_actions!=current['actions'] or sum(e['states'] for e in harness['splits'][split])!=current['count']:
            raise ValueError(f'{split}: build_report harness coverage differs from samples')
        for dataset,counts in report['final_dataset_counts'].items():
            if counts[split]!=datasets[dataset]['count']:raise ValueError(f'{dataset}/{split}: build_report count mismatch')
        by_split[split]=summarize(current)
        by_dataset_split[split]={d:summarize(v) for d,v in sorted(datasets.items()) if v['count']}
        print(f'{split}: completed {current["count"]} rows',flush=True)
    return {'definitions':{
        'option_tokens':'Canonical option order, labels/newlines/tool text included; excludes Question/Options header and Answer suffix. >10 uses upstream _options_ids; <=10 counts tokens overlapping the option character span in the actual joint question/options/answer encoding (boundary-crossing token assigned to options).',
        'option_token_fraction':'option_tokens / stored n_prompt_tokens; canonical construction view, not training-time option permutations.',
        'quantiles':'Exact nearest-rank p50/p90; max over states, no trajectory weighting.',
        'prefix_only_exact':'Unavailable: flattened samples lack original message boundaries and protected-prefix length.',
        'prefix_only_visible_proxy':'messages_dropped > 0 and context ends with a separate line containing the exact corresponding omission marker. Denominator: all states. This is a user-approved visible-text proxy, not an exact message-level statistic; quoted markers and later protected system messages can affect it.',
        'harness':'Copied from this sample directory build_report.harness_distributions; aggregate actions/counts checked against streamed samples. No inferred dataset-to-harness mapping.',
        'id_pulled_into_test_ood':'Final retained test_ood states whose dataset is outside configured ood_datasets; not the pre-cap grouping population.'},
        'overall':summarize(totals),'splits':by_split,'datasets':{d:summarize(v) for d,v in sorted(by_dataset.items())},
        'dataset_splits':by_dataset_split,'harness_distributions':harness,
        'id_pulled_into_test_ood':{'count':sum(id_ood.values()),'by_dataset':dict(id_ood),'ood_datasets':sorted(ood_datasets)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/build.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=(args.output or PROJECT/'outputs'/datetime.now().strftime('sample_stats_%Y%m%d_%H%M%S')).resolve()
    samples=args.samples.resolve()
    if output==samples or output.is_relative_to(samples):raise ValueError('statistics must not write into source sample directory')
    output.mkdir(parents=True,exist_ok=False);(output/'metadata').mkdir()
    (output/'metadata/command.txt').write_text(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
    status={'status':'running','started_at':now(),'samples':str(samples)}
    write_json(output/'metadata/run_status.json',status)
    print(f'output: {output}',flush=True)
    try:
        config=json.loads(args.config.read_text())
        tokenizer=AutoTokenizer.from_pretrained(config['tokenizer'],local_files_only=True)
        result=collect(samples,tokenizer,set(config['ood_datasets']))
        write_json(output/'summary.json',result)
        status.update(status='completed',ended_at=now(),count=result['overall']['count'])
    except BaseException as exc:
        status.update(status='failed',ended_at=now(),error_type=type(exc).__name__,error=str(exc));raise
    finally:write_json(output/'metadata/run_status.json',status)


if __name__=='__main__':main()
