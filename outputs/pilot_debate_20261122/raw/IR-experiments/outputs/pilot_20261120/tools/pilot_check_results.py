"""Verify saved comparison parity and summarize evidence without new model calls."""
import collections
import hashlib
import json
from pathlib import Path
import sys
base=Path(sys.argv[1]);out=Path(sys.argv[2])
def read(p):return json.loads(p.read_text())
def key(q):
    import re
    return re.sub(r'[^\w]+',' ',q.casefold()).strip()
rows=read(out/'rounds.json');audits=read(out/'proposal_evidence.json')
reports={p.parent.name:read(p) for p in (base/'IR-experiments-pilot/runs/experiments').glob('*/report.json')}
parity=[]
for name,report in reports.items():
    assert report['protocol']['targets']==['unlabeled2017/000000364869.jpg','unlabeled2017/000000562109.jpg']
    assert report['protocol']['pool']==50000 and report['protocol']['saved_references']==[]
    checked=0
    for r in report['per_example']:
        if r['method']=='live_raw_same_dialogue':continue
        source=next(s for s in rows if s['method']==r['method'] and s['target']==r['target'] and s['round']==r['round'])
        assert (r['query'],r['target_rank'])==(source['query'],source['target_rank'])
        checked+=1
    parity.append(dict(report=name,states=report['protocol']['rounds'],matching_query_rank_rows=checked,summary=report['summary']))
for target,rank in [('unlabeled2017/000000364869.jpg',9),('unlabeled2017/000000562109.jpg',89)]:
    initial=[r for r in rows if r['target']==target and r['round']==0]
    assert len(initial)==3 and len({r['query'] for r in initial})==1 and {r['target_rank'] for r in initial}=={rank}
first=[]
for target in ['unlabeled2017/000000364869.jpg','unlabeled2017/000000562109.jpg']:
    a=next(a for a in audits if a['target']==target and a['method']=='agents_no_critique' and a['round']==1)
    b=next(a for a in audits if a['target']==target and a['method']=='agents_critique' and a['round']==1)
    assert a['representatives']==b['representatives'] and a['candidate_window_ids']==b['candidate_window_ids'] and a['eligible']==b['eligible']
    ranking=sorted(a['kl'],key=lambda x:x['kl'])
    first.append(dict(target=target,identical_representatives_window_and_eligible=True,
                      min_kl=a['question'],critique=b['question'],kl_ranking=ranking,critique_decision=b['selection']))
summary=[]
for method,target in sorted({(r['method'],r['target']) for r in rows}):
    group=[a for a in audits if a['method']==method and a['target']==target]
    selected=[r for r in rows if r['method']==method and r['target']==target and r['round']>0]
    statuses=collections.Counter();roles=collections.defaultdict(collections.Counter);keys=collections.defaultdict(list)
    within=0;full_citations=0;proposals=0;no_eligible=[];candidate_duplicates=[]
    for a in group:
        if not a['question']:no_eligible.append(a['round'])
        for p in a['attempts'] or []:
            statuses[p['filter']['status']]+=1;roles[p['owner']][p['filter']['status']]+=1
            keys[key(p['proposal']['question'])].append(a['round']);proposals+=1
            full_citations+=len(set(p['proposal']['evidence_ids']))==len(a['representatives'])
        counts=collections.Counter(key(p['proposal']['question']) for p in a['attempts'] or [])
        within+=sum(n-1 for n in counts.values())
        qs=a.get('candidate_questions') or []
        if qs:candidate_duplicates.append(dict(round=a['round'],count=len(qs),unique=len({key(q) for q in qs})))
    summary.append(dict(method=method,target=target,proposal_attempts=proposals,statuses=dict(statuses),by_role={k:dict(v) for k,v in roles.items()},
        within_round_exact_duplicate_extras=within,distinct_questions_repeated_across_rounds=sum(len(set(v))>1 for v in keys.values()),
        all_representatives_cited=full_citations,no_selected_question_rounds=no_eligible,
        selected_questions=len([r for r in selected if r['question']]),candidate_pool_diversity=candidate_duplicates))
lengths=collections.defaultdict(collections.Counter)
for folder in (base/'IR-experiments-pilot/runs/live_upstream').glob('*/qwen_calls'):
    for p in folder.glob('*.json'):
        r=read(p);lengths[r['request']['upstream_function']][r['response']['choices'][0]['finish_reason']]+=1
fences=[f for f in read(out/'plugir_filter_trace.json') if 'fence' in f['question'] and f['target'].endswith('562109.jpg') and f['round']>=6]
result=dict(query_rank_parity=parity,first_round_selector_comparison=first,proposal_summary=summary,
            live_finish_reasons={k:dict(v) for k,v in lengths.items()},live_late_fence_filter_calls=fences)
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
