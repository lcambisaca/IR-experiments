"""Summarize existing saved pilot evidence; no generation or model imports."""
import collections
import csv
import json
from pathlib import Path
import re
import sys

base=Path(sys.argv[1])
out=Path(sys.argv[2])
out.mkdir(parents=True,exist_ok=True)
repo=base/'IR-experiments-pilot'
evidence=base/'pilot-evidence-20261120'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def key(q): return re.sub(r'[^\w]+',' ',q.casefold()).strip()
dialogues={}
for method,pattern in [('agents_no_critique','runs/three_agents/*/agents_no_critique/*.json'),
                       ('agents_critique','runs/three_agents/*/critique_selector/*.json'),
                       ('live_plugir_qwen','runs/live_upstream/*/dialogues/*.json')]:
    for p in repo.glob(pattern):
        d=read(p)
        target=d.get('target',d.get('identity',{}).get('target'))
        dialogues[(method,target)]=(d,p)
rows=[]; examples=[]; statistics=[]; live_filters=[]
for (method,target),(d,p) in dialogues.items():
    roles=collections.Counter(); contributors=collections.Counter(); selected_keys=collections.Counter(); proposal_keys=collections.defaultdict(list)
    logical=0; reused=0
    for i,s in enumerate(d['states']):
        rank=s['target_rank']; prev=None if i==0 else d['states'][i-1]['target_rank']
        q=s.get('question'); a=s.get('answer'); audit=s.get('audit',{})
        selected=[x['owner'] for x in audit.get('final_proposals',[]) if x['question']==q]
        roles.update(selected)
        co_roles=sorted({x['owner'] for x in audit.get('attempts',[]) if q and key(x['proposal']['question'])==key(q)})
        contributors.update(co_roles)
        if q: selected_keys[key(q)]+=1
        calls=s.get('transport_records',s.get('llm_calls',[]))
        if isinstance(calls,list):
            logical+=len(calls); reused+=sum(c.get('reused',False) for c in calls)
            if method=='live_plugir_qwen':
                for call_index,call in enumerate(calls):
                    request=call.get('request',{})
                    if request.get('upstream_function')=='filter_questions':
                        prompt=request['original_request']['messages'][-1]['content']
                        answer=call['response']['choices'][0]['message']['content']
                        question=prompt.split('[Question]\n',1)[-1].split('\n[Answer]',1)[0].strip()
                        live_filters.append(dict(target=target,round=i,call_index=call_index,question=question,
                            response=answer,uncertain_substring_gate='uncertain' in answer.lower(),reused=call.get('reused')))
        rows.append(dict(target=target,method=method,round=i,question=q,answer=a,query=s['query'],
                         target_rank=rank,rank_improvement=None if prev is None else prev-rank,
                         selected_roles=';'.join(selected),contributing_roles=';'.join(co_roles),window_size=audit.get('shared_retrieval',{}).get('window_size'),
                         checkpoint=str(p.relative_to(base))))
        if i:
            for attempt in audit.get('attempts',[]):
                proposal=attempt['proposal']
                proposal_keys[key(proposal['question'])].append(dict(round=i,owner=attempt['owner'],status=attempt['filter']['status']))
            examples.append(dict(target=target,method=method,round=i,
                representatives=audit.get('shared_retrieval',{}).get('captions'),
                candidate_window_ids=audit.get('shared_retrieval',{}).get('pool_ids'),
                candidate_caption_prompt=s.get('candidate_caption_prompt'),
                attempts=audit.get('attempts'),eligible=audit.get('final_proposals'),
                selection=audit.get('critique_decision'),kl=audit.get('kl'),
                candidate_questions=s.get('candidate_questions'),question=q,answer=a,query=s['query']))
    statistics.append(dict(method=method,target=target,status=d.get('status'),last_error=d.get('last_error'),
        states=len(d['states']),selected_roles=dict(roles),selected_contributing_roles=dict(contributors),logical_qwen_calls_saved_in_completed_states=logical,
        reused_qwen_calls_saved_in_completed_states=reused,
        exact_selected_repeats={k:v for k,v in selected_keys.items() if v>1},
        exact_repeated_proposals={k:v for k,v in proposal_keys.items() if len(v)>1},
        partial_round_audit=d.get('partial_round_audit')))
with (out/'rounds.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['target','method','round'])
    writer.writeheader();writer.writerows(rows)
(out/'rounds.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(out/'proposal_evidence.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
(out/'audit_summary.json').write_text(json.dumps(statistics,indent=2),encoding='utf-8')
(out/'plugir_filter_trace.json').write_text(json.dumps(live_filters,indent=2),encoding='utf-8')
print(json.dumps(statistics,indent=2))
