"""Observe unchanged pilot entrypoint timing/resources; preserve every call/return."""
import datetime
import fcntl
import functools
import json
from pathlib import Path
import sys
import time
import traceback
import torch
import run_evaluation as runner
import agentir.live as live
import agentir.multagent_experiment.multi_agent as multi

evidence=Path.home()/'pilot-evidence-20261120'
lock=(evidence/'generation.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
events=(evidence/'observer.jsonl').open('a',buffering=1)
def event(**data):
    events.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data))+'\n')
def observe(label,fn):
    @functools.wraps(fn)
    def call(*args,**kwargs):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        detail={k:str(kwargs[k]) for k in ['initial_caption','checkpoint_path','critique'] if k in kwargs}
        event(event='start',method=label,**detail)
        error=None
        try:
            return fn(*args,**kwargs)
        except BaseException as exc:
            error=repr(exc)
            raise
        finally:
            torch.cuda.synchronize()
            event(event='end',method=label,wall_seconds=time.perf_counter()-started,error=error,
                  peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),**detail)
    return call
live.generate_live_dialogue=observe('live_plugir_qwen',live.generate_live_dialogue)
multi.generate_agent_dialogue=observe('multi_agent',multi.generate_agent_dialogue)
event(event='run_start',argv=sys.argv,checkpoint_inventory=[str(p) for p in Path('runs').rglob('*.json')])
try:
    runner.main()
except BaseException:
    event(event='run_failed',traceback=traceback.format_exc())
    raise
else:
    event(event='run_complete')
