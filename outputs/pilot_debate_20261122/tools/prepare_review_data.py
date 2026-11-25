import collections, csv, hashlib, json, pathlib, sys
import torch
from agentir.evaluation import metrics_from_ranks

b = pathlib.Path(__file__).resolve().parent
if b.name == 'tools':
    b = b.parent
raw = pathlib.Path('\\\\?\\' + str((b/'raw').resolve()))
out = b/'analysis'
a = json.loads((out/'analysis.json').read_text())
hist = {(h['method'],pathlib.PurePosixPath(h['target']).stem):h for h in a['histories']}
aliases = out/'checkpoints'; aliases.mkdir(exist_ok=True)
for h in a['histories']:
    name = f"{h['method']}_{pathlib.PurePosixPath(h['target']).stem}.json"
    content = (raw/h['checkpoint']).read_bytes()
    (aliases/name).write_bytes(content)
    assert hashlib.sha256(content).hexdigest() == hashlib.sha256((aliases/name).read_bytes()).hexdigest()
    h['alias'] = 'analysis/checkpoints/'+name

scopes=[]
for label,methods,targets,n in [
    ('cake_all_three_states_0_to_3', ['agents_critique','agents_debate','live_plugir_qwen'],['000000364869'],4),
    ('both_targets_critique_vs_live_states_0_to_9',['agents_critique','live_plugir_qwen'],['000000364869','000000562109'],10),
    ('both_targets_all_three_initial_only',['agents_critique','agents_debate','live_plugir_qwen'],['000000364869','000000562109'],1),
]:
    for method in methods:
        ranks=torch.tensor([hist[method,t]['ranks'][:n] for t in targets]).T
        metrics,bri=metrics_from_ranks(ranks,[1,10,100])
        scopes.append(dict(scope=label,method=method,states=n,targets=targets,BRI=bri,metrics=metrics))

round_rows=[]
for h in a['histories']:
    for row in h['rows']:
        round_rows.append(dict(method=h['method'],target=h['target'],**row))
with (out/'observed_rounds.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(round_rows[0]));w.writeheader();w.writerows(round_rows)
with (out/'observed_rank_comparison.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['target','state','agents_critique','agents_debate','live_plugir_qwen'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for target in ['000000364869','000000562109']:
        for state in range(11):
            row=dict(target=target,state=state)
            for method in fields[2:]:
                ranks=hist[method,target]['ranks'];row[method]=ranks[state] if state<len(ranks) else ''
            w.writerow(row)

checks={}
pre=json.loads((raw/'preflight.json').read_text())
for h in a['histories']:
    image=next(i for i in pre['images'] if i['target']==h['target'])
    assert h['rows'][0]['query']==image['caption']
assert all(not e['checkpoint_existed'] for e in a['events'] if e['event']=='start')
ends=[e for e in a['events'] if e['event']=='end']
assert sum(e['qwen_records'] for e in ends)==470
assert sum(e['blip2_records'] for e in ends)==42
assert not sum(e['qwen_replays']+e['blip2_replays'] for e in ends)
checks['fresh_checkpoints']=6;checks['qwen_records']=470;checks['blip2_records']=42
checks['common_initial_captions_and_ranks']=True

source = raw/'IR-experiments'
for target in ['000000364869','000000562109']:
    critique=json.loads((raw/hist['agents_critique',target]['checkpoint']).read_text())
    debate=json.loads((raw/hist['agents_debate',target]['checkpoint']).read_text())
    c=critique['states'][1]['audit']['shared_retrieval']['captions']
    d=(debate['states'][1]['audit']['shared_retrieval']['captions'] if len(debate['states'])>1
       else debate['partial_round_audit'][0]['payload']['candidates'])
    assert c==d
checks['initial_representatives_identical_between_critique_and_debate']=True
for s in json.loads((raw/hist['agents_debate','000000364869']['checkpoint']).read_text())['states'][1:]:
    assert len(s['audit']['transcript'])==9
    for rnd,n in [(1,3),(2,6)]:
        entries=[x['payload']['transcript'] for x in s['llm_audit'] if x['stage']=='discussion' and x['payload']['discussion_round']==rnd]
        assert len(entries)==3 and all(len(x)==n and x==entries[0] for x in entries)
checks['completed_debate_rounds_with_synchronous_snapshot_verified']=3
for p in (source/'runs/experiments').glob('*/report.json'):
    report=json.loads(p.read_text())
    assert report['protocol']['pool']==50000
    assert report['protocol']['saved_references']==[]
    for row in report['per_example']:
        h=hist[row['method'],pathlib.PurePosixPath(row['target']).stem]
        assert row['target_rank']==h['ranks'][row['round']]
        assert row['query']==h['rows'][row['round']]['query']
checks['standard_live_report_rows_match_checkpoints']=22
checks['max_device_used_MiB']=max(float(row['memory.used [MiB]'].split()[0])
    for p in [raw/'gpu_samples.csv',raw/'gpu_samples_continuations.csv']
    for row in csv.DictReader(p.open(),skipinitialspace=True) if row.get('memory.used [MiB]'))
for scope in scopes:
    assert all(0<=m['recall_percent']<=m['hits_percent']<=100 for m in scope['metrics'])
a['scopes']=scopes;a['checks']=checks
metric_rows=[]
for x in a['aggregate']:
    metric_rows.extend(dict(scope='both_targets_available_prefix',method=x['method'],span_states=x['shared_states'],BRI=x['BRI'],**row) for row in x['metrics'])
for x in scopes:
    metric_rows.extend(dict(scope=x['scope'],method=x['method'],span_states=x['states'],BRI=x['BRI'],**row) for row in x['metrics'])
with (out/'per_round_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
(out/'analysis.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
(out/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
for s in scopes:print(s['scope'],s['method'],'BRI',s['BRI'],'last',s['metrics'][-3:])
print(json.dumps(checks,indent=2))
