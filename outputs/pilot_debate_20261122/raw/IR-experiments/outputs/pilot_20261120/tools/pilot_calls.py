"""Inventory actual persisted generation completions, including invalid responses."""
import collections
import json
from pathlib import Path
import sys

base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
result=[]
for folder in sorted((base/'IR-experiments-pilot/runs').rglob('*_calls')):
    if not folder.is_dir():continue
    records=[json.loads(p.read_text()) for p in folder.glob('*.json')]
    stages=collections.Counter();finish=collections.Counter();seconds=0.;inputs=outputs=0
    for r in records:
        stages[r['request'].get('upstream_function','BLIP-2 answer')]+=1
        seconds+=r.get('seconds',0.);inputs+=r.get('input_tokens',0);outputs+=r.get('output_tokens',0)
        if 'response' in r:finish[r['response']['choices'][0].get('finish_reason','missing')]+=1
    result.append(dict(folder=str(folder.relative_to(base)),persisted_completions=len(records),
        stages=dict(stages),finish_reasons=dict(finish),qwen_decode_seconds=seconds if folder.name=='qwen_calls' else None,
        qwen_input_tokens=inputs if folder.name=='qwen_calls' else None,qwen_output_tokens=outputs if folder.name=='qwen_calls' else None))
(out/'call_inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
