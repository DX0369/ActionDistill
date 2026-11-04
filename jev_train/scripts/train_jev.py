#!/usr/bin/env python3
"""Jev adaptation of decider/train.py@d6fc522: soft targets and auditable resume."""
import argparse,json,math,os,random,resource,shlex,signal,sys,time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];REPO=PROJECT.parents[1]
import decider
if not Path(decider.__file__).resolve().is_relative_to(REPO/'third_party/decider'):
    raise RuntimeError('Wrong decider import; set PYTHONPATH=<repo>/third_party/decider:<repo>/experiments/jev_train/src')
sys.path.insert(0,str(REPO/'experiments/dataset_tool_inventory/src'))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import swanlab
from decider.model import DecisionModel
from decider.train import batches_by_tokens
from jev_train.common import now,write_json,rows
from jev_train.data import SampleIndex,make_item,collate,load_selected
from jev_train.evaluation import evaluate,measured,training_majority_tools
from jev_train.loss import soft_loss
from jev_train.selection import load_train_ids
from jev_train.training_history import read_training_history,resume_boundary


def device_batch(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def distributed_environment(environ):
    keys=('WORLD_SIZE','RANK','LOCAL_RANK')
    if not any(k in environ for k in keys):return 1,0,0
    if not all(k in environ for k in keys):raise ValueError('WORLD_SIZE, RANK and LOCAL_RANK must all be set')
    world_size,rank,local_rank=(int(environ[k]) for k in keys)
    if world_size<1 or not 0<=rank<world_size or not 0<=local_rank<world_size:
        raise ValueError('invalid WORLD_SIZE/RANK/LOCAL_RANK')
    return world_size,rank,local_rank


def check_device_count(device_count,world_size):
    if device_count!=world_size:
        raise ValueError(f'visible CUDA device_count={device_count}, world_size={world_size}; they must match')


def check_resume_configuration(saved,current):
    if 'world_size' not in saved:raise ValueError('checkpoint has no world_size; resume old checkpoints with the old commit')
    if saved['world_size']!=current['world_size']:
        raise ValueError(f"checkpoint world_size={saved['world_size']}, current={current['world_size']}")
    if saved!=current:raise ValueError('checkpoint configuration differs')


def prepare_resume(out,checkpoint,configuration,world_size,smoke):
    checkpoint=Path(checkpoint).resolve()
    if not out.is_dir() or checkpoint.parent!=out/'checkpoints':
        raise ValueError('resume must use a checkpoint from the same output run')
    if (out/'model').exists() or (out/'training_majority_tools.json').exists():
        raise ValueError('final exports already exist; do not overwrite them during resume')
    previous=json.loads((out/'metadata/run_status.json').read_text())
    check_resume_configuration(previous['configuration'],configuration)
    if previous['status'] not in {'failed','running'}:
        raise ValueError('only an interrupted or failed run can resume')
    if not smoke and not previous['swanlab_run_id']:
        raise ValueError('formal resume requires the original SwanLab run id')
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False,mmap=True)
    check_resume_configuration(state['configuration'],configuration)
    if len(state['rank_rng'])!=world_size:
        raise ValueError('checkpoint rank_rng length differs from world_size')
    if any(int(p.name.removeprefix('step_'))>state['step'] for p in (out/'checkpoints').glob('step_*')):
        raise ValueError('later checkpoint directories already exist; do not overwrite')
    boundary=resume_boundary(read_training_history(out/'train.log'),checkpoint,state,previous,now())
    return state,previous,boundary


def next_step_batches(items,max_tokens,seed,epoch,cursor,n,smoke=False,longest=None):
    """Consume n global micro-batches, preserving the original epoch/cursor boundary."""
    selected=[]
    while len(selected)<n:
        batches=batches_by_tokens(items,max_tokens,random.Random(seed+epoch))
        if smoke:
            j=next(j for j,b in enumerate(batches) if longest in b);batches.insert(0,batches.pop(j))
        if cursor>=len(batches):epoch+=1;cursor=0;continue
        take=min(n-len(selected),len(batches)-cursor)
        selected.extend(batches[cursor:cursor+take]);cursor+=take
    return selected,epoch,cursor


def total_steps(n_batches,world_size,accum,epochs,max_steps):
    return max_steps if max_steps else max(1,int(math.ceil(n_batches/(world_size*accum))*epochs))


def rng_state():
    return {'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])


def terminate(signum,frame):
    raise RuntimeError(f'training terminated by signal {signum}')


def log_metrics(prefix,result,step):
    values={prefix+'/overall/'+k:v for k,v in result['overall'].items() if isinstance(v,(int,float))}
    for dataset,metrics_ in result['datasets'].items():
        values.update({prefix+'/'+dataset+'/'+k:v for k,v in metrics_.items() if isinstance(v,(int,float))})
    for label,group in [('overall',result['overall']),*result['datasets'].items()]:
        for field in ('action_kind','outcome','candidate_level','transition'):
            for value,scores in group.get('by_'+field,{}).items():
                values.update({f'{prefix}/{label}/by_{field}/{value}/{k}':v for k,v in scores.items() if v is not None})
    swanlab.log(values,step=step)


def save_checkpoint(model,opt,out,step,epoch,cursor,args,configuration,loss_history,rank_rng,probe):
    path=out/'checkpoints'/f'step_{step:07d}';path.mkdir(parents=True,exist_ok=False)
    model.lm.save_pretrained(path/'model');model.tok.save_pretrained(path/'model')
    (path/'model/decider_config.json').write_text((Path(args.model)/'decider_config.json').read_text())
    state={'configuration':configuration,'step':step,'epoch':epoch,'cursor':cursor,'optimizer':opt.state_dict(),'loss_history':loss_history,
           'rank_rng':rank_rng,'longest_sample_probe':probe}
    torch.save(state,path/'state.pt')
    return path


def learning_rate(step,total,lr,warmup):
    update=step+1
    return lr*update/warmup if update<warmup else lr*.5*(1+math.cos(math.pi*min(1,(update-warmup)/max(1,total-warmup))))


def reject_boolean_config(value):
    if isinstance(value,bool):raise ValueError('boolean training configuration values are not allowed')
    if isinstance(value,dict):
        for child in value.values():reject_boolean_config(child)
    elif isinstance(value,list):
        for child in value:reject_boolean_config(child)


def arguments():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config',default=str(PROJECT/'configs/train_pilot.json'))
    known,_=pre.parse_known_args();cfg=json.loads(Path(known.config).read_text());reject_boolean_config(cfg);ap=argparse.ArgumentParser(parents=[pre])
    ap.add_argument('--samples',required=True);ap.add_argument('--out',required=True);ap.add_argument('--run_name',required=True)
    ap.add_argument('--train_ids');ap.add_argument('--validation_samples',help='Explicit monitor source; training input remains --samples')
    ap.add_argument('--resume');ap.add_argument('--smoke',action='store_true')
    for k,v in cfg.items():ap.add_argument('--'+k,type=type(v),default=v)
    a=ap.parse_args()
    if a.max_options!=255 or a.label_smooth!=0 or a.none_prob!=0 or a.schema_first_prob!=0:ap.error('fixed protocol: 255 options, no label smoothing/none augmentation/schema-first')
    if a.max_len!=32768 or a.max_tokens!=32768:ap.error('protocol fixes max_len=max_tokens=32768')
    if a.accum<=0 or a.save_every<=0 or a.eval_every<=0 or a.train_limit<0 or a.monitor_val_size<=0:ap.error('invalid limits/intervals')
    if a.train_ids and a.train_limit>0:ap.error('--train_ids requires --train_limit 0')
    if a.train_ids:a.train_ids=str(Path(a.train_ids).resolve())
    a.validation_samples=str(Path(a.validation_samples if a.validation_samples is not None else a.samples).resolve())
    if not a.smoke and a.swanlab_mode!='cloud':ap.error('formal training requires SwanLab cloud')
    if a.activation_checkpointing not in {'on','off'}:ap.error('activation_checkpointing must be on or off')
    if a.smoke and (a.swanlab_mode!='disabled' or not Path(a.out).resolve().is_relative_to(PROJECT/'tmp')):ap.error('smoke requires disabled SwanLab and jev_train/tmp output')
    return a


def main():
    a=arguments();out=Path(a.out).resolve();samples=Path(a.samples).resolve()
    world_size,rank,local_rank=distributed_environment(os.environ);distributed=world_size>1
    device=torch.device('cuda',local_rank)
    configuration={k:v for k,v in vars(a).items() if k not in {'resume','config','out'}}
    configuration['world_size']=world_size
    state=previous_status=resume_event=None
    metric_prefix=''
    run=logf=status=None
    old_sigterm=signal.signal(signal.SIGTERM,terminate) if rank==0 else None
    def barrier():
        if distributed:dist.barrier()
    def log(event,**fields):
        if rank!=0:return
        text=json.dumps({'event':event,**fields},ensure_ascii=False);print(text,flush=True);logf.write(text+'\n');logf.flush()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; do not change GPU/length silently')
        check_device_count(torch.cuda.device_count(),world_size)
        torch.cuda.set_device(local_rank)
        if distributed:dist.init_process_group('nccl',timeout=timedelta(hours=2))
        if a.resume:
            state,previous_status,resume_event=prepare_resume(out,Path(a.resume),configuration,world_size,a.smoke)
            metric_prefix=resume_event['metric_prefix']
        barrier()  # Every rank reads the old status/history before rank zero appends.
        if rank==0:
            if not a.resume:out.mkdir(parents=True,exist_ok=False)
            (out/'metadata').mkdir(exist_ok=True)
            logf=(out/'train.log').open('a' if a.resume else 'w')
            if resume_event is not None:
                log(**resume_event)
            with (out/'metadata/command.txt').open('a' if a.resume else 'w') as f:f.write(f'cd {shlex.quote(str(Path.cwd()))}\n{shlex.join([sys.executable,*sys.argv])}\n')
            status={'status':'running','started_at':now(),'smoke':a.smoke,'samples':str(samples),'decider_source':str(Path(decider.__file__).resolve()),'resume':a.resume,'configuration':configuration}
            if resume_event is not None:
                status['resume_attempt']=resume_event['attempt'];status['metric_prefix']=metric_prefix
            if previous_status is not None and 'swanlab_run_id' in previous_status:status['swanlab_run_id']=previous_status['swanlab_run_id']
            write_json(out/'metadata/run_status.json',status)
        barrier()
        if rank==0:
            resume_tracking={'id':previous_status['swanlab_run_id'],'resume':'must'} if a.resume and not a.smoke else {}
            # NVML monitoring has blocked shutdown in DDP and single-card runs.
            # Keep scalar/cloud tracking; GPU peaks are already recorded by the trainer.
            hardware={'settings':swanlab.Settings(hardware_monitor=False)}
            run=swanlab.init(workspace='DX3906',project='action_distill',experiment_name=a.run_name,config={**vars(a),'world_size':world_size},mode=a.swanlab_mode,logdir=str(out/'swanlog'),**resume_tracking,**hardware)
            if not a.smoke:
                if not run.id:raise RuntimeError('SwanLab did not return a run id')
                status['swanlab_run_id']=run.id;write_json(out/'metadata/run_status.json',status)
        barrier()
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.cuda.reset_peak_memory_stats()
        train_index=SampleIndex(samples/'train.jsonl')
        train=load_train_ids(train_index,a.train_ids,a.seed) if a.train_ids else load_selected(train_index,a.train_limit,a.seed)
        if not train:raise ValueError('training set is empty')
        log('data',train_count=len(train),train_limit=a.train_limit)
        model=DecisionModel(str(Path(a.resume)/'model') if a.resume else a.model,grad_ckpt=False).to(device)
        if a.activation_checkpointing=='on':
            model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.lm.config.use_cache=False
        training_model=DDP(model,device_ids=[local_rank],broadcast_buffers=False,gradient_as_bucket_view=True,find_unused_parameters=False) if distributed else model
        items=[{'ids':range(entry['n_prompt_tokens'])} for entry in train]
        if rank==0:
            train_majority=training_majority_tools(train_index.read(entry['offset']) for entry in sorted(train,key=lambda e:e['offset']))
            val_index=SampleIndex(Path(a.validation_samples)/'val.jsonl')
            monitor=load_selected(val_index,a.monitor_val_size,a.seed,min_per_dataset=True)
            log('data_indexed',train_population=len(train_index.entries),val_population=len(val_index.entries),val_monitor=len(monitor),train_index_source=train_index.source,val_index_source=val_index.source,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        barrier()
        longest=max(range(len(items)),key=lambda i:len(items[i]['ids']))
        n_batches=len(batches_by_tokens(items,a.max_tokens,random.Random(a.seed)))
        total=total_steps(n_batches,world_size,a.accum,a.epochs,a.max_steps)
        if total<=0:raise ValueError('training needs positive steps')
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd,betas=(.9,.95))
        step=epoch=cursor=0
        if a.resume:
            opt.load_state_dict(state['optimizer']);step=state['step'];epoch=state['epoch'];cursor=state['cursor']
            restore_rng(state['rank_rng'][rank])
        history=state['loss_history'] if state is not None else []
        probe=state['longest_sample_probe'] if state is not None else None
        del state
        model.train();opt.zero_grad(set_to_none=True);reported_probe=probe is not None;started=time.monotonic()
        while step<total:
            step_started=time.monotonic()
            batches,epoch,cursor=next_step_batches(items,a.max_tokens,a.seed,epoch,cursor,world_size*a.accum,a.smoke,longest)
            batch_losses=[];batch_total_losses=[]
            for local_micro,selected in enumerate(batches[rank::world_size]):
                batch=device_batch(collate([make_item(train_index.read(train[i]['offset']),model.tok,a.max_len,a.seed) for i in selected],model.tok.pad_token_id),device)
                if batch['input_ids'].numel()>a.max_tokens:raise ValueError('padded micro-batch exceeds token budget')
                with training_model.no_sync() if distributed and local_micro<a.accum-1 else nullcontext():
                    logits=training_model(batch);loss,ce,brier=soft_loss(logits,batch['targets'],batch['nopts'],a.brier_w)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    (loss/a.accum).backward()
                batch_losses.append(float(ce.detach()));batch_total_losses.append(float(loss.detach()))
                if probe is None and longest in selected:
                    probe={'prompt_tokens':len(items[longest]['ids']),'padded_sequence_length':batch['input_ids'].shape[1],
                           'sample_id':train[longest]['id'],'max_memory_allocated':torch.cuda.max_memory_allocated(),'rank':rank}
                del batch,logits,loss,ce,brier
            lr=learning_rate(step,total,a.lr,a.warmup)
            for g in opt.param_groups:g['lr']=lr
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));opt.step();opt.zero_grad(set_to_none=True);step+=1
            totals=torch.tensor([sum(batch_losses),sum(batch_total_losses),len(batch_losses)],device=device,dtype=torch.float64)
            memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device,dtype=torch.int64)
            if distributed:dist.all_reduce(totals);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
            mean_loss,mean_total_loss=(totals[:2]/totals[2]).tolist();history.append(mean_loss)
            if not reported_probe:
                probes=[None]*world_size
                if distributed:dist.all_gather_object(probes,probe)
                else:probes[0]=probe
                for entry in probes:
                    if entry is not None:log('longest_sample_memory_probe',**entry);reported_probe=True
                probe=next((entry for entry in probes if entry is not None),None)
            tokens=sum(len(items[i]['ids']) for batch_indices in batches for i in batch_indices)
            values={'loss':mean_loss,'total_loss':mean_total_loss,'lr':lr,'grad_norm':grad,'tokens':tokens,'step_seconds':time.monotonic()-step_started,
                    'max_memory_allocated':int(memory),'seconds':time.monotonic()-started,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            log('train',step=step,total=total,**values)
            if rank==0:swanlab.log({metric_prefix+'train/'+k:v for k,v in values.items()},step=step)
            if step%a.save_every==0 or step==total:
                rank_rng=[None]*world_size
                if distributed:dist.all_gather_object(rank_rng,rng_state())
                else:rank_rng[0]=rng_state()
                if rank==0:
                    checkpoint=save_checkpoint(model,opt,out,step,epoch,cursor,a,configuration,history,rank_rng,probe);log('checkpoint',path=str(checkpoint))
                barrier()
            if step==total:break
            if step%a.eval_every==0:
                if rank==0:
                    result=measured(evaluate(model,val_index,monitor,seed=a.seed,max_tokens=a.max_tokens),train_majority=train_majority);log('validation_monitor',step=step,metrics=result)
                    log_metrics(metric_prefix+'val_monitor',result,step);model.train()
                barrier()
        rank_memory=[None]*world_size;local_memory=torch.cuda.max_memory_allocated()
        if distributed:dist.all_gather_object(rank_memory,local_memory)
        else:rank_memory[0]=local_memory
        if rank==0:
            model.lm.save_pretrained(out/'model');model.tok.save_pretrained(out/'model')
            write_json(out/'training_majority_tools.json',train_majority)
            status.update(status='completed',ended_at=now(),step=step,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          max_memory_allocated=max(rank_memory),max_memory_allocated_by_rank=rank_memory,longest_sample_probe=probe)
            write_json(out/'metadata/run_status.json',status);log('completed',**status)
        barrier()
    except BaseException as exc:
        if rank==0 and status is not None:
            status.update(status='failed',ended_at=now(),error_type=type(exc).__name__,error=str(exc));write_json(out/'metadata/run_status.json',status)
        raise
    finally:
        try:
            if logf is not None:logf.close()
            if run is not None:swanlab.finish()
        finally:
            if dist.is_initialized():dist.destroy_process_group()
            if rank==0:signal.signal(signal.SIGTERM,old_sigterm)

if __name__=='__main__':main()
