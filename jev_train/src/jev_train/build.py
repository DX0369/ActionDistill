"""Two source passes with SQLite lightweight indexes; rendered histories never accumulate."""
import hashlib
import json
import multiprocessing
import re
import time
import sqlite3
import shutil
from collections import Counter,defaultdict
from pathlib import Path
from functools import lru_cache
from . import PROJECT,REPO
from .common import digest,rows,write_json
from .split import swe_task_id,source_task_id,assign_groups
from .mix import training_mix,select_hash

SPLITS=('train','val','test_id','test_ood')


def connect(path):
    con=sqlite3.connect(path)
    con.execute('PRAGMA synchronous=OFF');con.execute('PRAGMA journal_mode=OFF')
    return con


def canonical_g3(config):
    old=json.loads((REPO/config['old_g3']).read_text())
    approved={d['dataset']:d['g3'] for d in old['datasets']}
    for d,v in old['post_review_adjudication']['revised_dataset_g3'].items():approved[d]=v['g3']
    new=json.loads((REPO/config['new_g3']).read_text())
    approved.update({d['dataset']:d['g3'] for d in new['datasets']})
    for d in config['datasets']:
        if approved[d] not in {'通过','有条件通过'}:raise ValueError(f'{d}: G3 not admitted')
    return {d:approved[d] for d in config['datasets']}


def setup_entry(entry,run):
    entry=dict(entry)
    if entry.get('candidate_reconstruction')=='harness_union':
        sets=json.loads((run/'review_summary.json').read_text())['harness_tool_sets'][entry['name']]
        entry['harness_tool_sets']={(v['harness'],v['domain']):v['tools'] for v in sets}
    return entry


def swe_repos(root,entry):
    # Projection only: the canonical repo anchors Scale-SWE IDs without guessing
    # how underscores partition owner/repo. No trajectories are loaded here.
    import pyarrow.parquet as pq
    result={}
    for path in sorted(root.glob(entry['glob'])):
        for batch in pq.ParquetFile(path).iter_batches(columns=['instance_id','repo'],batch_size=256):
            for row in batch.to_pylist():
                key,repo=row['instance_id'],row['repo']
                if key in result and result[key]!=repo:raise ValueError('inconsistent canonical SWE repo')
                swe_task_id(key,repo)
                result[key]=repo
    return result


@lru_cache(maxsize=1)
def codex_call_types(path):
    """Recover source types lost by the read-only inventory adapter, one log at a time."""
    result={}
    for row in rows(path):
        if row['type']!='response_item':continue
        payload=row['payload'];kind=payload['type']
        if kind not in {'function_call','custom_tool_call'}:continue
        if kind=='custom_tool_call' and not isinstance(payload['input'],str):raise ValueError('custom tool input must be a string')
        value=payload['input' if kind=='custom_tool_call' else 'arguments']
        key=(payload['call_id'],payload['name'],json.dumps(value,ensure_ascii=False,sort_keys=True))
        if key in result and result[key]!=kind:raise ValueError(f'{path}: ambiguous source call type {key[:2]}')
        result[key]=kind
    return result


def history_features(history,actions,call_types=None):
    from .render import text
    assistants=[(i,m) for i,m in enumerate(history) if m['role']=='assistant']
    previous=[]
    for _,message in reversed(assistants):
        calls=message.get('tool_calls',[])
        if calls:
            names=[call['function']['name'] for call in calls]
            if not all(isinstance(name,str) and name for name in names):raise ValueError('history tool names must be nonempty strings')
            previous=sorted(set(names));break
        if text(message.get('content')).strip():previous=['RESPOND'];break
    transition='first' if not previous else 'repeat' if set(actions)<=set(previous) else 'change'
    loop='no';forms=set()
    recent=assistants[-3:]
    if len(recent)==3 and all(len(m.get('tool_calls',[]))==1 for _,m in recent):
        calls=[m['tool_calls'][0] for _,m in recent]
        if len(actions)==1 and all(call['function']['name']==actions[0] for call in calls) and not any(m['role']=='user' for m in history[recent[0][0]+1:recent[-1][0]]):
            arguments=[]
            for call in calls:
                value=call['function']['arguments'];kind='function_call'
                if isinstance(value,str) and call_types is not None:
                    kind=call_types[(call['id'],call['function']['name'],json.dumps(value,ensure_ascii=False,sort_keys=True))]
                if kind=='custom_tool_call':
                    if not isinstance(value,str):raise ValueError('custom tool input must be a string')
                    normalized=value
                else:
                    # Named, counted normalizations confirmed by the user: an empty argument string is a
                    # no-argument call ({}); unparsable non-empty JSON (e.g. truncated teacher output) is
                    # compared by exact raw text, so only byte-identical broken calls count as identical.
                    if value=='':value='{}';forms.add('empty_arguments')
                    if isinstance(value,str):
                        try:value=json.loads(value)
                        except json.JSONDecodeError:
                            arguments.append(('unparsed_function_call',value));forms.add('unparsed_arguments');continue
                    normalized=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                arguments.append((kind,normalized))
            if len(set(arguments))==1:loop='yes'
    return {'prev_action_names':previous,'transition':transition,'loop_state':loop,'loop_argument_forms':sorted(forms)}


def initial_request(history):
    """The exact request text hashed by the builder; excludes system messages."""
    from .render import text,initial_prefix
    from dataset_tool_inventory.review_aggregate import IMAGE_BLOCK_TYPES,IMAGE_PLACEHOLDER
    requests=[]
    boundary,skipped=initial_prefix(history)
    for message in history[:boundary]:
        if message['role']=='user':
            content=message['content']
            if isinstance(content,list):
                content=[{'type':'text','text':IMAGE_PLACEHOLDER} if b['type'] in IMAGE_BLOCK_TYPES else b for b in content]
            request=re.sub(r'<environment_context>.*?</environment_context>','',text(content),flags=re.DOTALL).strip()
            if request:requests.append(request)
    return '\n'.join(requests),skipped


def light(event,group_config,repo_map,call_types=None):
    meta=event.metadata;task=meta['task_id'];dataset=event.dataset
    if task is not None and (type(task) not in {str,int} or task==''):raise ValueError('invalid canonical task_id')
    if dataset in group_config['source_task_ids']:
        task=source_task_id(event.source,group_config['source_task_ids'][dataset])
    if task is not None and dataset in group_config['swe_datasets']:
        task=swe_task_id(task,repo_map[task] if dataset==group_config['swe_repo_dataset'] else None)
    names=list(event.candidate_tools)
    request,skipped=initial_request(meta['_history'])
    return {'id':digest(dataset,event.trajectory_id,event.turn_index),'dataset':dataset,
            'trajectory_id':event.trajectory_id,'trajectory':json.dumps([dataset,event.trajectory_id]),
            'turn_index':event.turn_index,'model':event.model,'harness':event.harness,'task_id':task,'raw_task_id':meta['task_id'],
            'request_hash':hashlib.sha256(request.encode()).hexdigest() if request else None,'outcome':meta['outcome'],
            'skipped_framework_greeting':skipped,
            'action_kind':event.action_kind,'action_names':list(event.action_names),'candidate_names':names,
            'candidate_level':meta['candidate_level'],'source':event.source,
            **history_features(meta['_history'],['RESPOND'] if event.action_kind=='RESPOND' else list(event.action_names),call_types)}


def first_pass(job):
    from dataset_tool_inventory.adapters.review_formats import scan
    from dataset_tool_inventory.review_policy import eligibility
    from dataset_tool_inventory.schema import DecisionEvent,IssueEvent
    entry,config,run,tmp,limit=job;dataset=entry['name'];root=REPO/config['data_root']/entry['path']
    group_config=json.loads((REPO/config['task_groups']).read_text())
    repo_map=swe_repos(root,entry) if dataset==group_config['swe_repo_dataset'] else {}
    duplicate_doc=json.loads((run/'duplicates.json').read_text())
    duplicate_ids={t for g in duplicate_doc['exact'] for d,t in g['duplicates'] if d==dataset}
    policy=json.loads((REPO/config['review_policy']).read_text());entry=setup_entry(entry,run)
    db=tmp/(dataset+'.sqlite');con=connect(db)
    con.executescript('CREATE TABLE states(id TEXT PRIMARY KEY,trajectory TEXT,payload TEXT,split TEXT,task_group TEXT,selected INT DEFAULT 0,rendered INT DEFAULT 0); CREATE TABLE trajectories(key TEXT PRIMARY KEY,payload TEXT); CREATE INDEX state_trajectory ON states(trajectory);')
    counts=Counter();excluded=Counter();issues=Counter();seen_trajectories={};trajectory_counts=Counter()
    for event in scan(dataset,root,entry,limit):
        if isinstance(event,IssueEvent):issues[event.issue]+=1;continue
        if not isinstance(event,DecisionEvent):continue
        counts['seen_states']+=1;admit=eligibility(event,policy)
        if admit['eligibility'] not in {'confirmed','conditional'}:
            excluded.update(admit['exclusion_reasons']);continue
        counts['eligible_states']+=1
        if event.trajectory_id in duplicate_ids:counts['exact_duplicate_states']+=1;continue
        try:
            call_types=codex_call_types(event.source.rsplit(':',1)[0]) if entry['review_adapter']=='native_codex' else None
            record=light(event,group_config,repo_map,call_types)
        except (ValueError,KeyError,TypeError) as exc:
            raise ValueError(f'{event.source} / decision {event.turn_index}: {exc}') from exc
        counts['skipped_framework_greeting']+=record['skipped_framework_greeting']
        for form in record['loop_argument_forms']:counts['loop_check_'+form+'_states']+=1
        if record['request_hash'] is None:excluded['missing_initial_request']+=1;continue
        trajectory={k:record[k] for k in ('dataset','task_id','request_hash')};trajectory['key']=record['trajectory']
        if trajectory['key'] in seen_trajectories and seen_trajectories[trajectory['key']]!=trajectory:
            raise ValueError(f'{dataset}/{event.trajectory_id} at {event.source}: task identity changed within one trajectory')
        if trajectory['key'] not in seen_trajectories:
            seen_trajectories[trajectory['key']]=trajectory;con.execute('INSERT INTO trajectories VALUES(?,?)',(trajectory['key'],json.dumps(trajectory)))
        con.execute('INSERT INTO states(id,trajectory,payload) VALUES(?,?,?)',(record['id'],record['trajectory'],json.dumps(record,ensure_ascii=False)))
        trajectory_counts[record['trajectory']]+=1
        counts['indexed_states']+=1
        if counts['indexed_states']%1000==0:con.commit()
    for trajectory,n in trajectory_counts.items():
        ids=[sid for (sid,) in con.execute("SELECT id FROM states WHERE trajectory=? ORDER BY json_extract(payload,'$.turn_index'),id",(trajectory,))]
        con.executemany("UPDATE states SET payload=json_set(payload,'$.trajectory_decisions',?,'$.trajectory_position',?) WHERE id=?",((n,i,sid) for i,sid in enumerate(ids)))
    con.commit();con.close()
    return {'dataset':dataset,**dict(counts),'eligibility_exclusions':dict(excluded),'source_issues':dict(issues)}


def group_states(entries,config,tmp):
    trajectories=[]
    for entry in entries:
        con=connect(tmp/(entry['name']+'.sqlite'));trajectories.extend(json.loads(v) for (v,) in con.execute('SELECT payload FROM trajectories'));con.close()
    groups_config=json.loads((REPO/config['task_groups']).read_text())
    assignments,cross=assign_groups(trajectories,groups_config['namespaces'],config['ood_datasets'],config['seed'],groups_config['task_only_datasets'])
    for entry in entries:
        dataset=entry['name'];con=connect(tmp/(dataset+'.sqlite'))
        updates=[(assignments[t['key']][1],assignments[t['key']][0],t['key']) for t in trajectories if t['dataset']==dataset]
        con.executemany('UPDATE states SET split=?,task_group=? WHERE trajectory=?',updates);con.commit()
        con.close()
    counts=defaultdict(Counter);dataset_totals={};members=defaultdict(list)
    for t in trajectories:members[assignments[t['key']][0]].append(t)
    for entry in entries:
        dataset=entry['name'];con=connect(tmp/(dataset+'.sqlite'))
        dataset_totals[dataset]=con.execute('SELECT COUNT(*) FROM states').fetchone()[0]
        for group,count in con.execute('SELECT task_group,COUNT(*) FROM states GROUP BY task_group'):
            counts[group][dataset]=count
        con.close()
    dataset_trajectories=Counter(t['dataset'] for t in trajectories)
    summaries=[];violations=[];warnings=[];multi_task=[]
    for group,datasets in counts.items():
        tasks=sorted({json.dumps(t['task_id'],ensure_ascii=False) for t in members[group] if t['task_id'] is not None})
        summary={'task_group':group,'states':sum(datasets.values()),'trajectories':len(members[group]),
                 'dataset_states':dict(sorted(datasets.items())),'distinct_nonempty_task_ids':len(tasks),
                 'task_id_examples':[json.loads(t) for t in tasks[:5]]}
        summaries.append(summary)
        exceeded={d:n/dataset_totals[d] for d,n in datasets.items() if n/dataset_totals[d]>.05}
        if exceeded:warnings.append({**summary,'dataset_fractions_over_5_percent':exceeded})
        stopped={d:n/dataset_totals[d] for d,n in datasets.items() if dataset_trajectories[d]>=20 and n/dataset_totals[d]>.25}
        reasons=[]
        if stopped:reasons.append('group_exceeds_dataset_25_percent_with_at_least_20_trajectories')
        if len(tasks)>=2:multi_task.append(summary)
        if reasons:violations.append({**summary,'reasons':reasons,'dataset_fractions_over_25_percent':stopped})
    summaries.sort(key=lambda g:(-g['states'],g['task_group']))
    sizes=sorted(g['states'] for g in summaries);n=len(sizes)
    by_dataset={}
    for dataset,total in dataset_totals.items():
        own=[g for g in summaries if dataset in g['dataset_states']]
        largest=max(own,key=lambda g:g['dataset_states'][dataset]) if own else None
        by_dataset[dataset]={'states':total,'trajectories':dataset_trajectories[dataset],'task_groups':len(own),
                             'largest_group':largest,'largest_group_fraction':largest['dataset_states'][dataset]/total if total else None}
    return {'trajectory_count':len(trajectories),'task_group_count':n,'cross_dataset_groups':cross,
            'dataset_state_counts':dataset_totals,'datasets':by_dataset,'top_20_by_states':summaries[:20],
            'size_distribution':{'p50':sizes[max(0,(n+1)//2-1)] if n else None,
                                 'p90':sizes[max(0,(9*n+9)//10-1)] if n else None,'max':sizes[-1] if n else None},
            'threshold_definition':'Stop: dataset has >=20 indexed trajectories and group holds >25% of its indexed states. >5% and >=2 distinct non-null task_id values are report-only. All counts before sampling.',
            'multi_task_id_groups':sorted(multi_task,key=lambda g:(-g['states'],g['task_group'])),
            'groups_over_5_percent':sorted(warnings,key=lambda g:(-g['states'],g['task_group'])),
            'threshold_violations':sorted(violations,key=lambda g:(-g['states'],g['task_group']))}


def select_states(entries,config,tmp):
    reports={}
    for entry in entries:
        dataset=entry['name'];con=connect(tmp/(dataset+'.sqlite'))
        report={'split_before_sampling':{},'selected':{},'task_groups':{}}
        for partition in SPLITS:
            records=[dict(json.loads(v),task_group=g) for v,g in con.execute('SELECT payload,task_group FROM states WHERE split=?',(partition,))]
            report['split_before_sampling'][partition]=len(records);report['task_groups'][partition]=len({r['task_group'] for r in records})
            if partition=='train':
                if config['train_sampling']=='capped_mix':
                    chosen,mix=training_mix(records,config['seed'],config['failure_fraction'],config['tool_fraction'],config['dataset_cap']);report['training_mix']=mix
                else:
                    from .selection import pool_sample,position_summary
                    chosen=pool_sample(records,config['seed'],config['pool_per_trajectory'],config['pool_dataset_cap'])
                    report['pool_sampling']={'before':len(records),'selected':len(chosen),
                                            'population':'Selected pool before representative smoke and rendering.',
                                            'relative_position':position_summary(chosen),
                                            'transition':{v:sum(r['transition']==v for r in chosen) for v in ('first','repeat','change')}}
            else:chosen=select_hash(records,config['eval_cap'],config['seed'])
            con.executemany('UPDATE states SET selected=1 WHERE id=?',[(r['id'],) for r in chosen]);report['selected'][partition]=len(chosen)
        con.commit();con.close();reports[dataset]=report
    return reports


def second_pass(job):
    from transformers import AutoTokenizer
    from dataset_tool_inventory.adapters.review_formats import scan
    from dataset_tool_inventory.schema import DecisionEvent,IssueEvent
    from .render import render_event,HistoryTruncator
    entry,config,run,tmp,limit,shard,shards=job;dataset=entry['name'];root=REPO/config['data_root']/entry['path']
    con=sqlite3.connect(f'file:{tmp/(dataset+".sqlite")}?mode=ro',uri=True)
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer'],local_files_only=True)
    history_truncator=HistoryTruncator(tokenizer,config['head_tokens'],config['tail_tokens'])
    group_config=json.loads((REPO/config['task_groups']).read_text());repo_map=swe_repos(root,entry) if dataset==group_config['swe_repo_dataset'] else {}
    selected={}
    for sid,payload,partition,group in con.execute('SELECT id,payload,split,task_group FROM states WHERE selected=1'):
        original=json.loads(payload)
        if int(digest(original['trajectory_id']),16)%shards==shard:
            selected[sid]=(original,partition,group)
    paths={s:tmp/f'{dataset}.shard_{shard}.{s}.jsonl' for s in SPLITS}
    handles={s:path.open('w') for s,path in paths.items()};visited=set();rendered_ids=set();excluded=Counter();rendered=Counter();render_seconds=0.
    try:
        for event in scan(dataset,root,setup_entry(entry,run),limit):
            if not isinstance(event,DecisionEvent):continue
            sid=digest(dataset,event.trajectory_id,event.turn_index)
            if sid not in selected:continue
            if sid in visited:raise ValueError('duplicate selected decision id on second pass')
            original,partition,group=selected[sid]
            call_types=codex_call_types(event.source.rsplit(':',1)[0]) if entry['review_adapter']=='native_codex' else None
            current=light(event,group_config,repo_map,call_types)
            current['trajectory_decisions']=original['trajectory_decisions']
            current['trajectory_position']=original['trajectory_position']
            if current!=original:raise ValueError(f'{sid}: first/second pass mismatch')
            visited.add(sid)
            started=time.monotonic()
            try:
                content,reason=render_event(event,tokenizer,config['max_len'],config['description_chars'],config['head_tokens'],config['tail_tokens'],history_truncator,length_margin=config['length_margin'])
            except (ValueError,KeyError,TypeError) as exc:
                raise ValueError(f'{event.source} / decision {event.turn_index}: {exc}') from exc
            render_seconds+=time.monotonic()-started
            if reason:
                excluded[reason]+=1;continue
            row={k:current[k] for k in ('id','dataset','model','outcome','candidate_level','trajectory_id','turn_index','harness','action_kind','prev_action_names','transition','loop_state','trajectory_decisions','trajectory_position')};row.update(task_group=group,**content)
            handles[partition].write(json.dumps(row,ensure_ascii=False)+'\n');rendered[partition]+=1
            rendered_ids.add(sid)
        if visited!=selected.keys():raise ValueError(f'{dataset}: {len(selected.keys()-visited)} selected states not found on second pass')

    finally:
        for handle in handles.values():handle.close()
        con.close()
    return {'dataset':dataset,'shard':shard,'visited_ids':visited,'rendered_ids':rendered_ids,'paths':paths,
            'rendered':dict(rendered),'length_exclusions':dict(excluded),'render_seconds':render_seconds}


def merge_render_shards(dataset,results,tmp):
    visited=set();rendered_ids=set();rendered=Counter();excluded=Counter();seconds=0.
    for result in sorted(results,key=lambda r:r['shard']):
        if visited & result['visited_ids']:raise ValueError(f'{dataset}: duplicate visited id across render shards')
        if not result['rendered_ids'] <= result['visited_ids']:raise ValueError(f'{dataset}: rendered id was not visited')
        visited.update(result['visited_ids']);rendered_ids.update(result['rendered_ids'])
        rendered.update(result['rendered']);excluded.update(result['length_exclusions']);seconds+=result['render_seconds']
    con=connect(tmp/(dataset+'.sqlite'))
    try:
        selected={sid for (sid,) in con.execute('SELECT id FROM states WHERE selected=1')}
        if visited!=selected:raise ValueError(f'{dataset}: render shard union differs from selected ids')
        if selected and excluded['over_length_after_drop']/len(selected)>.01:
            raise ValueError(f'{dataset}: over_length_after_drop exceeds 1%; stop per protocol')
        con.executemany('UPDATE states SET rendered=1 WHERE id=?',[(sid,) for sid in rendered_ids])
        for split in SPLITS:
            with (tmp/(dataset+'.'+split+'.jsonl')).open('wb') as target:
                for result in sorted(results,key=lambda r:r['shard']):
                    with result['paths'][split].open('rb') as source:shutil.copyfileobj(source,target)
        con.commit()
        for result in results:
            for path in result['paths'].values():path.unlink()
    finally:con.close()
    return {'dataset':dataset,'visited':len(visited),'rendered':dict(rendered),'length_exclusions':dict(excluded),
            'render_seconds':seconds,'mean_render_seconds':seconds/len(visited) if visited else None}



def representative_states(entries,tmp):
    """Smoke only: keep first/middle/last of the already selected trajectory states."""
    reports={}
    for entry in entries:
        dataset=entry['name'];con=connect(tmp/(dataset+'.sqlite'));by_trajectory=defaultdict(list)
        for sid,trajectory,payload in con.execute('SELECT id,trajectory,payload FROM states WHERE selected=1'):
            by_trajectory[trajectory].append((json.loads(payload)['turn_index'],sid))
        keep=[]
        before=sum(len(v) for v in by_trajectory.values())
        for states in by_trajectory.values():
            states.sort()
            keep.extend(states[i][1] for i in sorted({0,len(states)//2,len(states)-1}))
        con.execute('UPDATE states SET selected=0')
        con.executemany('UPDATE states SET selected=1 WHERE id=?',[(sid,) for sid in keep]);con.commit()
        counts=dict(con.execute('SELECT split,COUNT(*) FROM states WHERE selected=1 GROUP BY split'))
        reports[dataset]={'before':before,'after':len(keep),'trajectories':len(by_trajectory),
                          'split_counts':{s:counts.get(s,0) for s in SPLITS}}
        con.close()
    return reports


def finalize(entries,config,tmp,output,report):
    from .selection import position_summary,POSITION_DEFINITION
    lengths={s:Counter() for s in SPLITS};labels={s:Counter() for s in SPLITS};label_totals={s:Counter() for s in SPLITS};final_counts={}
    label_first={s:{} for s in SPLITS}
    harness_counts={s:defaultdict(lambda: {'states':0,'action_types':Counter(),'candidate_levels':Counter()}) for s in SPLITS}
    feature_counts={}
    handles={s:(output/(s+'.jsonl')).open('w') for s in SPLITS}
    try:
        for dataset_order,entry in enumerate(entries):
            dataset=entry['name'];con=connect(tmp/(dataset+'.sqlite'))
            records=[json.loads(v) for (v,) in con.execute("SELECT payload FROM states WHERE selected=1 AND rendered=1 AND split='train'")]
            if config['train_sampling']=='capped_mix':
                final_train,repair=training_mix(records,config['seed'],config['failure_fraction'],config['tool_fraction'],config['dataset_cap'])
                report['datasets'][dataset]['post_render_cap_enforcement']=repair
            else:final_train=records
            train_ids={r['id'] for r in final_train};final_counts[dataset]={};feature_counts[dataset]={}
            identities={sid:(r['harness'],r['action_kind'],r['candidate_level'],ordinal)
                        for ordinal,sid,payload in con.execute('SELECT rowid,id,payload FROM states WHERE selected=1 AND rendered=1')
                        for r in [json.loads(payload)]}
            for partition in SPLITS:
                count=0;transitions=Counter();loops=Counter();positions=[]
                for row in rows(tmp/(dataset+'.'+partition+'.jsonl')):
                    if partition=='train' and row['id'] not in train_ids:continue
                    harness,kind,level,ordinal=identities[row['id']]
                    stats=harness_counts[partition][harness];stats['states']+=1
                    stats['action_types'][kind]+=1;stats['candidate_levels'][level]+=1
                    transitions[row['transition']]+=1;loops[row['loop_state']]+=1
                    positions.append({k:row[k] for k in ('trajectory_position','trajectory_decisions')})
                    handles[partition].write(json.dumps(row,ensure_ascii=False)+'\n');count+=1;lengths[partition][row['n_prompt_tokens']]+=1
                    label_totals[partition]['multi_label' if len(row['gold'])>1 else 'single_label']+=1
                    for position,g in enumerate(row['gold']):
                        name=row['options'][g];labels[partition][name]+=1
                        order=(dataset_order,ordinal,position)
                        if name not in label_first[partition] or order<label_first[partition][name]:label_first[partition][name]=order
                final_counts[dataset][partition]=count
                feature_counts[dataset][partition]={'transition':{k:transitions[k] for k in ('first','repeat','change')},'loop_state':{k:loops[k] for k in ('yes','no')},'relative_position':position_summary(positions)}
            con.close()
    finally:
        for handle in handles.values():handle.close()
    def quantiles(counter):
        n=sum(counter.values());result={'count':n,'max':max(counter,default=None)}
        for name,fraction in [('p50',.5),('p90',.9)]:
            threshold=max(1,int(n*fraction+.999999));acc=0;result[name]=None
            for value,count in sorted(counter.items()):
                acc+=count
                if acc>=threshold:result[name]=value;break
        return result
    report['final_dataset_counts']=final_counts;report['lengths']={s:quantiles(v) for s,v in lengths.items()}
    report['feature_distributions']={'population':'Final retained states by dataset and split; features computed before rendering from full visible history.','position_definition':POSITION_DEFINITION,'datasets':feature_counts}
    report['label_distributions']={}
    report['harness_distributions']={
        'population':'Final retained samples after rendering and post-render cap enforcement; each state counted once.',
        'splits':{s:[{'harness':h,'states':v['states'],
                      'action_types':{k:v['action_types'][k] for k in ('TOOL','TOOL_SET','RESPOND')},
                      'candidate_levels':{k:v['candidate_levels'][k] for k in ('S','N','H','I')}}
                     for h,v in sorted(counts.items(),key=lambda item:(item[0] is not None,item[0] or ''))]
                  for s,counts in harness_counts.items()}}
    for s,c in labels.items():
        n=sum(final_counts[d][s] for d in final_counts);respond=c.get('RESPOND: reply to the user directly without calling a tool',0)
        tools={k:v for k,v in c.items() if not k.startswith('RESPOND:')}
        report['label_distributions'][s]={'states':n,**dict(label_totals[s]),'respond_count':respond,'respond_fraction':respond/n if n else None,'label_memberships':dict(c),'most_common_tool':min(tools,key=lambda name:(-tools[name],label_first[s][name])) if tools else None,'most_common_tool_fraction':max(tools.values(),default=0)/n if n else None}
    write_json(output/'build_report.json',report)


class GroupingReviewError(ValueError):
    """The user-defined grouping/request thresholds require a decision."""


def build_samples(entries,config,run,tmp,output,workers,limit,representative=False,group_only=False):
    mode=config['train_sampling']
    if mode not in {'capped_mix','pool'}:raise ValueError('train_sampling must be capped_mix or pool')
    if mode=='pool':
        for key in ('pool_per_trajectory','pool_dataset_cap'):
            if type(config[key]) is not int or config[key]<=0:raise ValueError(f'{key} must be a positive integer')
    else:
        for key in ('failure_fraction','tool_fraction','dataset_cap'):config[key]
    shard_counts=config['render_shards']
    if set(shard_counts)!=set(config['datasets']) or any(type(v) is not int or v<1 for v in shard_counts.values()):
        raise ValueError('render_shards must explicitly give a positive integer for every configured dataset')
    jobs=[(e,config,run,tmp,limit) for e in entries]
    with multiprocessing.get_context('spawn').Pool(workers,maxtasksperchild=1) as pool:
        first=[]
        for result in pool.imap(first_pass,jobs):
            first.append(result);print('indexed '+result['dataset']+': '+str(result.get('indexed_states',0)),flush=True)
    groups=group_states(entries,config,tmp)
    report={'bounded':limit is not None,'group_only':group_only,'g3':canonical_g3(config),'groups':groups,'datasets':{r['dataset']:{'first_pass':r} for r in first}}
    missing=[]
    for r in first:
        denominator=r.get('eligible_states',0)-r.get('exact_duplicate_states',0)
        count=r['eligibility_exclusions'].get('missing_initial_request',0)
        fraction=count/denominator if denominator else 0
        r['missing_initial_request_fraction']=fraction
        if fraction>.01:missing.append(r['dataset'])
    write_json(output/'build_report.json',report)
    if (limit is None and groups['threshold_violations']) or missing:
        raise GroupingReviewError(f'Grouping/request stop threshold reached: {len(groups["threshold_violations"])} grouping violations; missing_initial_request >1%: {missing}. See build_report.json.')
    if group_only:return report
    selected=select_states(entries,config,tmp)
    for dataset,selection in selected.items():report['datasets'][dataset].update(selection)
    if representative:
        if limit is None:raise ValueError('representative states require bounded smoke')
        report['representative_smoke']={'selection':'first/middle/last of already selected trajectory states',
                                        'datasets':representative_states(entries,tmp)}
    render_jobs=[(*job,shard,shard_counts[job[0]['name']]) for job in jobs for shard in range(shard_counts[job[0]['name']])]
    results=defaultdict(list)
    with multiprocessing.get_context('spawn').Pool(workers,maxtasksperchild=1) as pool:
        for result in pool.imap(second_pass,render_jobs):results[result['dataset']].append(result)
    for entry in entries:
        dataset=entry['name'];r=merge_render_shards(dataset,results.pop(dataset),tmp)
        report['datasets'][dataset]['second_pass']=r;print('rendered '+dataset+': '+str(r['rendered']),flush=True)
    finalize(entries,config,tmp,output,report)
    return report
