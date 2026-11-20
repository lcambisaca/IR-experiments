"""Continue unaffected baseline cases after a fail-closed model response.

Do not repair the response, alter prompts, or retry the rejected cake selection.
Use existing preparations, checkpoints and evaluator; score only real saved states.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import functools
import json
from pathlib import Path
import random
import time
import traceback
import torch
import run_evaluation as runner
from agentir.cache import load_cache, read_json, digest
from agentir.evaluation import matched_queries, evaluate_experiments, save_experiment_results
from agentir.live import generated_query_method, save_json, signature
import agentir.live as live
import agentir.multagent_experiment.multi_agent as multi
from agentir.multagent_experiment import setup
from agentir.retrieval import BlipRetriever

root=Path(__file__).resolve().parent
evidence=Path.home()/'pilot-evidence-20261120'/'continuation'
evidence.mkdir(exist_ok=True)
lock=(evidence.parent/'generation.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
events=(evidence/'observer.jsonl').open('a',buffering=1)
def event(**data):
    events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**data))+'\n')
def observe(label,fn):
    @functools.wraps(fn)
    def call(*args,**kwargs):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();error=None
        detail={k:str(kwargs[k]) for k in ['initial_caption','checkpoint_path','critique'] if k in kwargs}
        event(event='start',method=label,**detail)
        try:return fn(*args,**kwargs)
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            torch.cuda.synchronize()
            event(event='end',method=label,wall_seconds=time.perf_counter()-started,error=error,
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),**detail)
    return call
live.generate_live_dialogue=observe('live_plugir_qwen',live.generate_live_dialogue)
multi.generate_agent_dialogue=observe('multi_agent',multi.generate_agent_dialogue)
event(event='run_start',reason='Preserve invalid cake critique; continue unaffected cases without scientific changes.')
random.seed(runner.SEED);torch.manual_seed(runner.SEED)
config=read_json(root/'experiment.json')
cache=load_cache(root/config['cache'],root/config['manifest'])
pairs=runner.select_target_pairs(matched_queries(read_json(root/config['baseline_queries']),read_json(root/config['supplied_queries']),cache.paths))
retriever=BlipRetriever(cache,device=runner.DEVICE,batch_size=runner.TEXT_BATCH_SIZE,threads=runner.CPU_THREADS,local_files_only=runner.OFFLINE)
bundles={}
def attempt(name,fn):
    try:
        bundle=fn();bundles[name]=bundle
        folder=evidence/'prepared'/name;folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/'metadata.json',bundle.metadata)
        for filename,content in bundle.artifacts.items():save_json(folder/filename,content)
    except BaseException as error:
        if isinstance(error,KeyboardInterrupt):raise
        event(event='case_preparation_failed',name=name,error=repr(error),traceback=traceback.format_exc())
        print(f'PRESERVED FAILURE {name}: {error}',flush=True)

attempt('giraffe_critique',lambda:setup.prepare_three_agents(root,retriever,[pairs[1]],n_states=runner.N_STATES,
    offline=runner.OFFLINE,modes=(True,)))
attempt('qwen_live',lambda:runner.prepare_qwen_live(root,retriever,pairs,n_states=runner.N_STATES,offline=runner.OFFLINE))
# If live stopped on cake, still attempt the independent giraffe using its normal preparation.
if 'qwen_live' not in bundles:
    attempt('qwen_live_giraffe',lambda:runner.prepare_qwen_live(root,retriever,[pairs[1]],n_states=runner.N_STATES,offline=runner.OFFLINE))

agent_folder=root/'runs/three_agents/shared_agents_simple_v1'
conversations={}
for method,subfolder in [('agents_no_critique','agents_no_critique'),('agents_critique','critique_selector')]:
    paths=[agent_folder/subfolder/(Path(raw['img']).stem+'.json') for raw,_ in pairs]
    if all(p.is_file() for p in paths):conversations[method]=[read_json(p) for p in paths]
live_by_target={}
for p in (root/'runs/live_upstream').glob('*/dialogues/*.json'):
    d=read_json(p);live_by_target[d['identity']['target']]=d
if all(raw['img'] in live_by_target for raw,_ in pairs):
    conversations['live_plugir_qwen']=[live_by_target[raw['img']] for raw,_ in pairs]
complete={name:generated_query_method(ds) for name,ds in conversations.items()
          if all(d.get('status')=='complete' and len(d['states'])>=runner.N_STATES for d in ds)}
if 'live_plugir_qwen' in complete:
    complete['live_raw_same_dialogue']=generated_query_method(conversations['live_plugir_qwen'],raw=True)
status={name:[dict(target=d.get('target',d.get('identity',{}).get('target')),status=d.get('status'),
                  states=len(d['states']),error=d.get('last_error')) for d in ds] for name,ds in conversations.items()}
save_json(evidence/'completion.json',status)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
run_metadata=dict(label='initial_pilot_failure_preserving_continuation',seed=runner.SEED,
    selected_target_images=[raw['img'] for raw,_ in pairs],requested_states=runner.N_STATES,
    completion=status,preflight=read_json(evidence.parent/'preflight.json'),
    prepared_experiments={k:v.metadata for k,v in bundles.items()},
    source_changes='None; diagnostic driver isolated remaining cases after original runner stopped on invalid critique slot.',
    runner_sha256=digest(root/'run_evaluation.py'))
def export(methods,states,suffix):
    result=evaluate_experiments(retriever,pairs,methods,rounds=states,ks=runner.K_VALUES,references=())
    result['run']={**run_metadata,'evaluated_states':states,'comparison_scope':suffix}
    destination=save_experiment_results(result,root/'runs/experiments'/(stamp+'_'+suffix))
    (destination/'runner_snapshot.py').write_bytes((root/'run_evaluation.py').read_bytes())
    (destination/'pilot_driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for method,ds in conversations.items():save_json(destination/'prepared'/method/'generated_dialogues.json',ds)
    print('SAVED_COMPARISON',destination,flush=True)
if complete:export(complete,runner.N_STATES,'pilot_completed_methods')
if len(conversations)==3:
    prefix=min(len(d['states']) for ds in conversations.values() for d in ds)
    matched={name:generated_query_method(ds) for name,ds in conversations.items()}
    matched['live_raw_same_dialogue']=generated_query_method(conversations['live_plugir_qwen'],raw=True)
    export(matched,prefix,'pilot_common_prefix')
event(event='run_complete',completion=status)
print('CONTINUATION_FINISHED',json.dumps(status),flush=True)
