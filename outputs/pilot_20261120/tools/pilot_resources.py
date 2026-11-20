"""Summarize external timings and GPU samples saved by the pilot observer."""
import csv
import datetime as dt
import json
from pathlib import Path
import sys

base=Path(sys.argv[1])
evidence=base/'pilot-evidence-20261120'
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
events=sorted([json.loads(line) for p in evidence.rglob('observer.jsonl') for line in p.read_text().splitlines() if line.strip()],key=lambda e:e['utc'])
samples=[]
for sample_path in evidence.rglob('gpu-samples.csv'):
  with sample_path.open() as f:
    for row in csv.DictReader(f):
        row={k.strip():v.strip() for k,v in row.items()}
        samples.append(dict(utc=dt.datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc),
            memory_mib=int(row['memory.used [MiB]'].split()[0]),utilization_percent=int(row['utilization.gpu [%]'].split()[0])))
pending={};cases=[]
for event in events:
    key=(event.get('method'),event.get('checkpoint_path'))
    if event['event']=='start': pending[key]=event
    elif event['event']=='end':
        start=pending.pop(key)
        begin=dt.datetime.fromisoformat(start['utc']);end=dt.datetime.fromisoformat(event['utc'])
        measured=[s for s in samples if begin<=s['utc']<=end]
        method=event['method']
        if method=='multi_agent': method='agents_critique' if event.get('critique')=='True' else 'agents_no_critique'
        cases.append(dict(method=method,initial_caption=event.get('initial_caption'),start=start['utc'],end=event['utc'],
            wall_seconds=event['wall_seconds'],error=event['error'],peak_cuda_allocated_bytes=event['peak_cuda_allocated'],
            peak_cuda_reserved_bytes=event['peak_cuda_reserved'],
            sampled_peak_gpu_mib=max((s['memory_mib'] for s in measured),default=None),
            gpu_samples=len(measured),checkpoint=event.get('checkpoint_path')))
summary={}
for case in cases:
    item=summary.setdefault(case['method'],dict(wall_seconds=0,peak_cuda_allocated_bytes=0,peak_cuda_reserved_bytes=0,sampled_peak_gpu_mib=0,cases=0))
    item['wall_seconds']+=case['wall_seconds'];item['cases']+=1
    for field in ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','sampled_peak_gpu_mib']:
        item[field]=max(item[field],case[field] or 0)
result=dict(cases=cases,summary=summary,run_events=[e for e in events if e['event'].startswith('run_')],
    unmatched_starts=list(pending.values()),gpu_samples=len(samples),
    sampled_peak_gpu_mib=max((s['memory_mib'] for s in samples),default=None),
    whole_process_resources={str(p.relative_to(evidence)):p.read_text() for p in evidence.rglob('process-resources.txt')})
(out/'resources.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
