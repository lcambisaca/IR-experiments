"""Observation-only launcher for the unchanged run_evaluation runner."""
import datetime,functools,json,os,pathlib,sys,time,traceback
base=pathlib.Path(__file__).resolve().parent;root=base/'IR-experiments';os.chdir(root);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
import torch
from agentir.multagent_experiment import multi_agent as ma,debate
from agentir import live
import run_evaluation as runner
log=base/'history_events.jsonl'
def emit(data):
 with log.open('a') as f:f.write(json.dumps(data,default=str)+'\n')
 print('HISTORY_EVENT',json.dumps(data,default=str),flush=True)
def observe(fn,kind):
 @functools.wraps(fn)
 def wrapped(*args,**kw):
  retriever=args[0]; method='live_plugir_qwen' if kind=='live' else 'agents_debate' if kw.get('method_policy') else 'agents_critique' if kw.get('critique',True) else 'agents_no_critique'
  q,a=kw['questioner'],kw['answerer'];qi,ai=len(q.records),len(a.records);started=time.perf_counter(); start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();error=None
  emit(dict(event='start',method=method,target=retriever.cache.paths[kw['target_id']],utc=start_utc,checkpoint=str(kw['checkpoint_path']),checkpoint_existed=pathlib.Path(kw['checkpoint_path']).exists()))
  torch.cuda.reset_peak_memory_stats()
  try:return fn(*args,**kw)
  except BaseException as e:error=f'{type(e).__name__}: {e}';raise
  finally:
   qr=q.records[qi:];ar=a.records[ai:]
   emit(dict(event='end',method=method,target=retriever.cache.paths[kw['target_id']],start_utc=start_utc,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,error=error,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),qwen_records=len(qr),qwen_actual=sum(not x['reused'] for x in qr),qwen_replays=sum(x['reused'] for x in qr),blip2_records=len(ar),blip2_actual=sum(not x['reused'] for x in ar),blip2_replays=sum(x['reused'] for x in ar),input_tokens=sum(x['input_tokens'] for x in qr),output_tokens=sum(x['output_tokens'] for x in qr),qwen_generation_seconds=sum(x['seconds'] for x in qr if not x['reused'])))
 return wrapped
ma.generate_agent_dialogue=observe(ma.generate_agent_dialogue,'agents');debate.generate_agent_dialogue=ma.generate_agent_dialogue
live.generate_live_dialogue=observe(live.generate_live_dialogue,'live')
if len(sys.argv)>1:
 method,index=sys.argv[1],int(sys.argv[2]);original=runner.select_target_pairs
 runner.select_target_pairs=lambda pairs:[original(pairs)[index]]
 runner.RUN_MULTI_AGENT=method=='agents_critique';runner.RUN_MULTI_AGENT_DEBATE=method=='agents_debate';runner.RUN_LIVE_PLUGIR=method=='live_plugir_qwen'
 emit(dict(event='continuation_dispatch',method=method,target_index=index,reason='Independent history continuation after preserved primary error; runner and scientific source unchanged.'))
runner.main()
