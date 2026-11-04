import sys,unittest,json,sqlite3,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];R=P.parents[1]
sys.path[:0]=[str(P/'src'),str(R/'third_party/decider'),str(R/'experiments/dataset_tool_inventory/src')]
from jev_train.split import assign_groups,swe_task_id,source_task_id
from jev_train.mix import training_mix
from jev_train.render import render_event,render_history,prompt_item,QUESTION,HistoryTruncator,TAU2_GREETING,option
from dataset_tool_inventory.review_aggregate import truncate_history
from jev_train.build import representative_states,light,group_states,merge_render_shards,build_samples,history_features
from jev_train.selection import pool_sample
from jev_train.data import validate,make_item

class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())
    def decode(self,tokens):return bytes(tokens).decode(errors='replace')


def event(history,kind='TOOL_SET',actions=('a','b'),tools=None):
    tools=tools if tools is not None else [{'name':'a','description':'A'},{'name':'b','description':'B'}]
    return SimpleNamespace(harness='fixture',action_kind=kind,action_names=actions,candidate_tools=tuple(t['name'] for t in tools),metadata={'_history':history,'_candidate_definitions':tools})

class BuildTests(unittest.TestCase):
    def test_history_transition_and_exact_loop(self):
        def call(name='a',arguments=None):
            return {'function':{'name':name,'arguments':{'x':1,'y':2} if arguments is None else arguments}}
        def assistant(*calls,content=''):
            return {'role':'assistant','content':content,'tool_calls':list(calls)}
        self.assertEqual(history_features([],['a']),{'prev_action_names':[],'transition':'first','loop_state':'no','loop_argument_forms':[]})
        self.assertEqual(history_features([assistant(content='hello'),assistant()],['RESPOND'])['transition'],'repeat')
        parallel=assistant(call('b'),call('a'),call('b'))
        self.assertEqual(history_features([parallel],['b'])['prev_action_names'],['a','b'])
        self.assertEqual(history_features([parallel],['b'])['transition'],'repeat')
        self.assertEqual(history_features([parallel],['c'])['transition'],'change')
        repeated=[assistant(call()),{'role':'tool','content':'ok'},assistant(call(arguments='{"y":2,"x":1}')),assistant(call())]
        self.assertEqual(history_features(repeated,['a'])['loop_state'],'yes')
        self.assertEqual(history_features(repeated,['b'])['loop_state'],'no')
        self.assertEqual(history_features(repeated+ [assistant(content='next')],['a'])['loop_state'],'no')
        interrupted=repeated[:2]+[{'role':'user','content':'wait'}]+repeated[2:]
        self.assertEqual(history_features(interrupted,['a'])['loop_state'],'no')
        self.assertEqual(history_features([parallel,*repeated[2:]],['a'])['loop_state'],'no')
        empty=[assistant(call(arguments='')),assistant(call(arguments='{}')),assistant(call(arguments={}))]
        self.assertEqual(history_features(empty,['a']),{'prev_action_names':['a'],'transition':'repeat','loop_state':'yes','loop_argument_forms':['empty_arguments']})
        truncated='{"filepath": "/w/a.xlsx", '
        broken=[assistant(call(arguments=truncated)) for _ in range(3)]
        self.assertEqual(history_features(broken,['a'])['loop_state'],'yes')
        self.assertEqual(history_features(broken,['a'])['loop_argument_forms'],['unparsed_arguments'])
        broken[0]['tool_calls'][0]['function']['arguments']=truncated+'"'
        self.assertEqual(history_features(broken,['a'])['loop_state'],'no')
        invalid=[assistant(call(arguments='not JSON')),*repeated[2:]]
        self.assertEqual(history_features(invalid,['a'])['loop_state'],'no')
        del invalid[0]['tool_calls'][0]['function']['arguments']
        with self.assertRaises(KeyError):history_features(invalid,['a'])
        custom=[assistant(dict(call(arguments='*** patch'),id=str(i))) for i in range(3)]
        types={(str(i),'a',json.dumps('*** patch')):'custom_tool_call' for i in range(3)}
        self.assertEqual(history_features(custom,['a'],types)['loop_state'],'yes')
        custom[0]['tool_calls'][0]['function']['arguments']+=' '
        types[('0','a',json.dumps('*** patch '))]='custom_tool_call'
        self.assertEqual(history_features(custom,['a'],types)['loop_state'],'no')
        as_function=history_features(custom,['a'],dict.fromkeys(types,'function_call'))
        self.assertEqual((as_function['loop_state'],as_function['loop_argument_forms']),('no',['unparsed_arguments']))
        with self.assertRaises(KeyError):history_features(custom,['a'],{})

    def test_pool_sampling_is_spread_and_deterministic(self):
        records=[{'id':f'{g}-{t}-{i}','dataset':'d','task_group':str(g),'trajectory_id':f'{g}-{t}','turn_index':i}
                 for g in range(3) for t in range(2) for i in range(40)]
        selected=pool_sample(records,7,16,200)
        from collections import Counter
        self.assertEqual(Counter(r['trajectory_id'] for r in selected),{f'{g}-{t}':16 for g in range(3) for t in range(2)})
        self.assertEqual(selected,pool_sample(list(reversed(records)),7,16,200))
        self.assertEqual(len({r['task_group'] for r in selected[:3]}),3)
        for config,error in (({},KeyError),({'train_sampling':'invalid'},ValueError),({'train_sampling':'pool'},KeyError)):
            with self.assertRaises(error):build_samples([],config,P,P,P,1,1)

    def test_render_shard_union_and_global_length_threshold(self):
        (P/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            folder=Path(directory);con=sqlite3.connect(folder/'d.sqlite')
            con.execute('CREATE TABLE states(id TEXT,selected INT,rendered INT DEFAULT 0)')
            ids={str(i) for i in range(100)}
            con.executemany('INSERT INTO states(id,selected) VALUES(?,1)',[(sid,) for sid in ids]);con.commit();con.close()
            parts=[]
            for shard,visited,rendered in [(0,{'0'},set()),(1,ids-{'0'},ids-{'0'})]:
                paths={split:folder/f'{shard}.{split}.jsonl' for split in ('train','val','test_id','test_ood')}
                for path in paths.values():path.write_text('')
                parts.append({'shard':shard,'visited_ids':visited,'rendered_ids':rendered,'paths':paths,
                              'rendered':{'train':len(rendered)},'length_exclusions':{'over_length_after_drop':1} if shard==0 else {},'render_seconds':0.})
            with self.assertRaisesRegex(ValueError,'union'):merge_render_shards('d',parts[:1],folder)
            with self.assertRaisesRegex(ValueError,'duplicate'):merge_render_shards('d',[parts[0],parts[0]],folder)
            with patch('jev_train.build.shutil.copyfileobj',side_effect=OSError('copy failed')):
                with self.assertRaisesRegex(OSError,'copy failed'):merge_render_shards('d',parts,folder)
            self.assertTrue(all(path.exists() for part in parts for path in part['paths'].values()))
            with sqlite3.connect(folder/'d.sqlite') as con:self.assertEqual(con.execute('SELECT SUM(rendered) FROM states').fetchone()[0],0)
            result=merge_render_shards('d',parts,folder)
            self.assertEqual(result['visited'],100);self.assertEqual(result['length_exclusions']['over_length_after_drop'],1)
            self.assertTrue(all(not path.exists() for part in parts for path in part['paths'].values()))
            with sqlite3.connect(folder/'d.sqlite') as con:self.assertEqual(con.execute('SELECT SUM(rendered) FROM states').fetchone()[0],99)

    def test_render_names_and_shard_configuration(self):
        for name in ('bad: name','bad (params: x'):
            with self.assertRaisesRegex(ValueError,'reserved'):option({'name':name})
        self.assertEqual(option({'name':'valid:namespace'}),'valid:namespace')
        for count in (0,True,1.5):
            with self.assertRaises(ValueError):build_samples([],{'train_sampling':'pool','pool_per_trajectory':16,'pool_dataset_cap':100,'datasets':['d'],'render_shards':{'d':count}},P,P,P,1,1)
        with self.assertRaises(KeyError):build_samples([],{'train_sampling':'pool','pool_per_trajectory':16,'pool_dataset_cap':100,'datasets':['d']},P,P,P,1,1)

    def test_representative_smoke_selects_only_selected_first_middle_last(self):
        (P/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            folder=Path(directory);con=sqlite3.connect(folder/'d.sqlite')
            con.execute('CREATE TABLE states(id TEXT,trajectory TEXT,payload TEXT,selected INT,split TEXT)')
            for i in [8,0,6,2,4]:
                con.execute('INSERT INTO states VALUES(?,?,?,?,?)',(str(i),'a',json.dumps({'turn_index':i}),1,'train'))
            con.execute('INSERT INTO states VALUES(?,?,?,?,?)',('ignored','a',json.dumps({'turn_index':9}),0,'train'))
            con.execute('INSERT INTO states VALUES(?,?,?,?,?)',('single','b',json.dumps({'turn_index':1}),1,'val'))
            con.commit();con.close()
            report=representative_states([{'name':'d'}],folder)
            with sqlite3.connect(folder/'d.sqlite') as con:
                selected={x[0] for x in con.execute('SELECT id FROM states WHERE selected=1')}
            self.assertEqual(selected,{'0','4','8','single'})
            self.assertEqual(report['d']['before'],6);self.assertEqual(report['d']['after'],4)

    def test_prefix_cache_equals_uncached_on_branch_and_mutation(self):
        tok=Tokenizer();cache=HistoryTruncator(tok,8,8)
        history=[{'role':'user','content':'task'}, {'role':'tool','content':'x'*100}]
        for view in (history[:1],history,history+[{'role':'assistant','content':'next'}],
                     history[:1]+[{'role':'tool','content':'branch'*30}]):
            self.assertEqual(cache(view),truncate_history(view,tok,8,8))
        history[1]['content']='changed'*30
        self.assertEqual(cache(history),truncate_history(history,tok,8,8))
    def test_exact_pr_ids_and_required_repo(self):
        self.assertEqual(swe_task_id('yt-project_unyt_pr211','yt-project/unyt'),swe_task_id('yt-project__unyt-211'))
        self.assertNotEqual(swe_task_id('yt-project__unyt-210'),swe_task_id('yt-project__unyt-211'))
        with self.assertRaises(ValueError):swe_task_id('wrong_unyt_pr211','yt-project/unyt')
        with self.assertRaises(ValueError):swe_task_id('bad-id')
    def test_group_namespace_first_request_and_ood(self):
        ts=[{'key':'a','dataset':'A','task_id':'1','request_hash':'h1'},
            {'key':'b','dataset':'B','task_id':'1','request_hash':'h2'},
            {'key':'c','dataset':'OOD','task_id':None,'request_hash':'h2'},
            {'key':'d','dataset':'C','task_id':None,'request_hash':'other'}]
        assignment,cross=assign_groups(ts,{'A':'shared','B':'shared'},['OOD'],3)
        self.assertEqual(assignment['a'],assignment['b']);self.assertEqual(assignment['b'],assignment['c'])
        self.assertEqual(assignment['a'][1],'test_ood');self.assertEqual(cross,1)
        self.assertEqual(assignment,assign_groups(list(reversed(ts)),{'A':'shared','B':'shared'},['OOD'],3)[0])
        with self.assertRaises(ValueError):assign_groups([dict(ts[0],request_hash=None)],{},[],3)
        with self.assertRaises(KeyError):assign_groups([{'key':'a','dataset':'A','task_id':None}],{},[],3)
    def test_initial_request_ignores_environment_and_stops_at_first_assistant(self):
        e=event([{'role':'user','content':'<environment_context>\nshared\n</environment_context>'},
                 {'role':'user','content':[{'type':'text','text':'  actual request  '}]},
                 {'role':'user','content':'constraints'},
                 {'role':'assistant','content':'reply'}, {'role':'user','content':'later request'}])
        e.dataset='d';e.trajectory_id='t';e.turn_index=1;e.model='m';e.source='s'
        e.metadata.update(task_id=None,outcome='success',candidate_level='S')
        config={'swe_datasets':[],'source_task_ids':{}}
        self.assertEqual(light(e,config,{})['request_hash'],hashlib.sha256(b'actual request\nconstraints').hexdigest())
        e.metadata['_history']=e.metadata['_history'][:1]
        self.assertIsNone(light(e,config,{})['request_hash'])
        e.metadata['_history']=[{'role':'user','content':[{'type':'text','text':'request'},{'type':'image','source':{'data':'not read'}}]}]
        self.assertEqual(light(e,config,{})['request_hash'],hashlib.sha256(b'request\n[image omitted]').hexdigest())
        e.metadata['_history']=[{'role':'user'}]
        with self.assertRaises(KeyError):light(e,config,{})
    def test_framework_greeting_request_and_protected_prefix(self):
        history=[{'role':'assistant','content':TAU2_GREETING},
                 {'role':'user','content':'KEEP TAU REQUEST'},
                 {'role':'assistant','content':'old '+('x'*1200)},
                 {'role':'tool','content':'old result'}, {'role':'user','content':'recent'}]
        e=event(history);e.dataset='tau';e.trajectory_id='t';e.turn_index=3;e.model='m';e.source='s'
        e.metadata.update(task_id='1',outcome='success',candidate_level='S')
        record=light(e,{'swe_datasets':[],'source_task_ids':{}},{})
        self.assertEqual(record['request_hash'],hashlib.sha256(b'KEEP TAU REQUEST').hexdigest())
        self.assertEqual(record['skipped_framework_greeting'],1)
        row,reason=render_event(e,Tokenizer(),max_len=450)
        self.assertIsNone(reason);self.assertEqual(row['messages_dropped'],2)
        self.assertIn('KEEP TAU REQUEST',row['context']);self.assertIn(TAU2_GREETING,row['context'])
        self.assertEqual(row['truncation_counts']['skipped_framework_greeting'],1)
        self.assertEqual(render_event(event(history[:2]),Tokenizer(),max_len=10)[1],'over_length_after_drop')
    def test_mle_path_identity_and_task_only_grouping(self):
        config=json.loads((P/'configs/task_groups.json').read_text());rule=config['source_task_ids']['mle-trajectories']
        self.assertEqual(source_task_id('/runs/r/trials/model/cactus/agent/trajectory.json:1',rule),'mle:cactus')
        for bad in ('/runs/r/model/cactus/agent/trajectory.json:1','/trials/model//agent/trajectory.json:1',
                    '/trials/model/cactus/other/trajectory.json:1','/trials/model/cactus/agent/trajectory.json:2',None):
            with self.assertRaises(ValueError):source_task_id(bad,rule)
        ts=[{'key':str(i),'dataset':'mle-trajectories','task_id':t,'request_hash':'same-template'}
            for i,t in enumerate(('mle:cactus','mle:aptos','mle:cactus'))]
        grouped,_=assign_groups(ts,{},[],3,config['task_only_datasets'])
        self.assertEqual(grouped['0'],grouped['2']);self.assertNotEqual(grouped['0'][0],grouped['1'][0])
    def test_group_report_counts_and_thresholds(self):
        (P/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
            folder=Path(directory);con=sqlite3.connect(folder/'d.sqlite')
            con.executescript('CREATE TABLE states(trajectory TEXT,split TEXT,task_group TEXT); CREATE TABLE trajectories(payload TEXT);')
            for i in range(2):
                trajectory={'key':str(i),'dataset':'d','task_id':str(i),'request_hash':'shared'}
                con.execute('INSERT INTO trajectories VALUES(?)',(json.dumps(trajectory),))
                con.execute('INSERT INTO states(trajectory) VALUES(?)',(str(i),))
            con.commit();con.close()
            report=group_states([{'name':'d'}],{'task_groups':str(P/'configs/task_groups.json'),'ood_datasets':[],'seed':3},folder)
            top=report['top_20_by_states'][0]
            self.assertEqual((top['states'],top['trajectories'],top['distinct_nonempty_task_ids']),(2,2,2))
            self.assertEqual(report['size_distribution'],{'p50':2,'p90':2,'max':2})
            self.assertEqual(report['threshold_violations'],[])
            self.assertEqual(len(report['multi_task_id_groups']),1)
            self.assertEqual(len(report['groups_over_5_percent']),1)
            self.assertEqual(report['datasets']['d']['task_groups'],1)
    def test_mix_caps_and_determinism(self):
        rs=[{'id':str(i),'trajectory_id':str(i%7),'outcome':'failure' if i<70 else 'unknown' if i<85 else 'success',
             'action_kind':'TOOL' if i%5 else 'RESPOND','action_names':['a' if i<80 else 'b'] if i%5 else []} for i in range(100)]
        out,report=training_mix(rs,4,cap=17);again,_=training_mix(list(reversed(rs)),4,cap=17)
        self.assertEqual([r['id'] for r in out],[r['id'] for r in again]);self.assertLessEqual(len(out),17)
        self.assertLessEqual(sum(r['outcome']=='failure' for r in out),len(out)*.3)
        for name in ('a','b'):self.assertLessEqual(sum(r['action_names']==[name] for r in out),len(out)*.4)
    def test_group_size_stop_requires_20_trajectories_and_strictly_over_25_percent(self):
        for total,shared,stops in ((19,19,False),(20,5,False),(20,6,True)):
            with self.subTest(total=total,shared=shared),tempfile.TemporaryDirectory(dir=P/'tmp') as directory:
                folder=Path(directory);con=sqlite3.connect(folder/'d.sqlite')
                con.executescript('CREATE TABLE states(trajectory TEXT,split TEXT,task_group TEXT); CREATE TABLE trajectories(payload TEXT);')
                for i in range(total):
                    t={'key':str(i),'dataset':'d','task_id':None,'request_hash':'shared' if i<shared else str(i)}
                    con.execute('INSERT INTO trajectories VALUES(?)',(json.dumps(t),))
                    con.execute('INSERT INTO states(trajectory) VALUES(?)',(str(i),))
                con.commit();con.close()
                report=group_states([{'name':'d'}],{'task_groups':str(P/'configs/task_groups.json'),'ood_datasets':[],'seed':3},folder)
                self.assertEqual(bool(report['threshold_violations']),stops)
    def test_multi_gold_and_respond(self):
        history=[{'role':'user','content':'task'}]
        row,reason=render_event(event(history),Tokenizer());self.assertIsNone(reason);self.assertEqual(row['gold'],[0,1]);self.assertEqual(row['question'],QUESTION)
        row,_=render_event(event(history,'RESPOND',()),Tokenizer());self.assertEqual(row['gold'],[2])
        self.assertIn('a: A',row['options'])
    def test_length_drops_whole_turn_preserves_system_and_first_user(self):
        history=[{'role':'system','content':'KEEP SYSTEM'},{'role':'user','content':'KEEP FIRST'},
                 {'role':'assistant','content':'OLD '+('x'*600),'tool_calls':[{'function':{'name':'a','arguments':{}}}]},
                 {'role':'tool','content':'OLD RETURN'},{'role':'user','content':'LATEST'}]
        row,reason=render_event(event(history),Tokenizer(),max_len=300)
        self.assertIsNone(reason);self.assertEqual(row['messages_dropped'],2)
        for value in ('KEEP SYSTEM','KEEP FIRST','LATEST','2 earlier messages omitted'):self.assertIn(value,row['context'])
        self.assertNotIn('OLD RETURN',row['context']);self.assertNotIn('OLD ',row['context'])
        self.assertEqual(render_event(event(history[:2]),Tokenizer(),max_len=5)[1],'over_length_after_drop')
    def test_invalid_labels_options_and_fields_fail(self):
        h=[{'role':'user','content':'task'}]
        with self.assertRaises(ValueError):render_event(event(h,actions=('missing',)),Tokenizer())
        self.assertEqual(render_event(event(h,tools=[{'name':str(i)} for i in range(255)]),Tokenizer())[1],'too_many_options')
        with self.assertRaises(ValueError):validate({'gold':[0]})
        with self.assertRaises(ValueError):render_event(event([{'role':'user','content':[{'type':'audio','data':'x'}]}]),Tokenizer())
        reference={'type':'tool_reference','tool_name':'TaskCreate','extra':'preserved'}
        row,_=render_event(event(h+[{'role':'tool','content':[reference]}]),Tokenizer())
        self.assertIn('[tool_reference]',row['context']);self.assertIn('preserved',row['context'])
        with self.assertRaises(KeyError):render_event(event(h+[{'role':'tool','content':[{'type':'tool_reference'}]}]),Tokenizer())

    def test_binary_deletion_equals_linear_exact_encoding(self):
        tok=Tokenizer();prefix=[{'role':'system','content':'system'},{'role':'user','content':'environment'},
                                {'role':'user','content':'REAL REQUEST'}]
        history=prefix+sum(([{'role':'assistant','content':str(i)+'x'*100},
                             {'role':'tool','content':'result'+str(i)}] for i in range(20)),[])
        row,reason=render_event(event(history),tok,max_len=700)
        self.assertIsNone(reason)
        for k in range(21):
            context=render_history(prefix+history[len(prefix)+2*k:],2*k,len(prefix))
            n=len(prompt_item(context,QUESTION,row['options'],row['gold'],tok)['ids'])
            if n<=700:break
        self.assertEqual((row['context'],row['n_prompt_tokens']),(context,n))
        self.assertGreater(row['context'].index('earlier messages omitted'),row['context'].index('REAL REQUEST'))

    def test_narrow_punctuation_permutations_fit_margin(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('/data/wangsq/model/Mapika-decider-0.8b',local_files_only=True)
        tools=[{'name':'a','description':'Read files.'},{'name':'b','description':'Write files!'},
               {'name':'c','description':'Search?'}]
        row,reason=render_event(event([{'role':'user','content':'x '*32550}],tools=tools),tok,length_margin=64)
        self.assertIsNone(reason);self.assertLessEqual(row['n_prompt_tokens'],32704)
        row.update(id='punctuation',dataset='d',model='m',task_group='g',outcome='success',candidate_level='S')
        for seed in range(20):self.assertLessEqual(len(make_item(row,tok,32768,seed)['ids']),32768)
        with self.assertRaises(ValueError):render_event(event([]),tok,length_margin=-1)

    def test_candidate_provenance_does_not_change_model_input(self):
        tok = Tokenizer()
        row, reason = render_event(event([{'role': 'user', 'content': 'Read and search'}]), tok)
        self.assertIsNone(reason)
        row.update(id='provenance', dataset='d', model='m', task_group='g',
                   outcome='unknown', candidate_level='S')
        captured = make_item(row, tok, seed=20261015)
        recovered = make_item(dict(row, candidate_level='N'), tok, seed=20261015)
        for key in ('ids', 'slots', 'golds', 'nopts', 'perms'):
            self.assertEqual(captured[key], recovered[key])
        self.assertEqual(captured['targets'].tolist(), recovered['targets'].tolist())
        self.assertEqual(recovered['row']['candidate_level'], 'N')

    def test_list_tool_output_truncates_as_text(self):
        h=[{'role':'user','content':'task'},{'role':'tool','content':[{'type':'text','text':'A'*1000},
              {'type':'image','source':{'data':'placeholder-only'}},
              {'type':'tool_reference','tool_name':'TaskCreate'}]}]
        for cache in (None,HistoryTruncator(Tokenizer(),32,32)):
            row,reason=render_event(event(h),Tokenizer(),head=32,tail=32,history_truncator=cache)
            self.assertIsNone(reason);self.assertNotIn('{"type": "text"',row['context'])
            self.assertEqual(row['truncation_counts']['tool_outputs_truncated'],1)
            self.assertEqual(row['truncation_counts']['images_replaced'],1)

if __name__=='__main__':unittest.main()
