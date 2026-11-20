"""Make byte-identical short checkpoint paths for convenient local review."""
import hashlib
import json
from pathlib import Path
import sys
base=Path(sys.argv[1]);out=Path(sys.argv[2]);(out/'dialogues').mkdir(exist_ok=True)
rows=json.loads((out/'analysis/rounds.json').read_text())
mapping=[]
names={'agents_no_critique':'ma_min_kl','agents_critique':'ma_critique','live_plugir_qwen':'plugir'}
for r in rows:
    if r['round']!=0:continue
    target='cake' if r['target'].endswith('364869.jpg') else 'giraffe'
    name=names[r['method']]+'_'+target+'.json'
    data=(base/r['checkpoint']).read_bytes()
    (out/'dialogues'/name).write_bytes(data)
    mapping.append(dict(alias='dialogues/'+name,source='raw/'+r['checkpoint'].replace('\\','/'),
                        sha256=hashlib.sha256(data).hexdigest(),method=r['method'],target=r['target']))
(out/'dialogue_aliases.json').write_text(json.dumps(mapping,indent=2))
print('Created',len(mapping),'byte-identical checkpoint aliases.')
