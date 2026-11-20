"""Render factual tables from the unchanged pilot's saved files for human review."""
import csv
import json
from pathlib import Path
import sys

base=Path(sys.argv[1]);analysis=Path(sys.argv[2]);dest=Path(sys.argv[3])
repo=base/'IR-experiments-pilot';evidence=base/'pilot-evidence-20261120'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
pre=read(evidence/'preflight.json');resources=read(analysis/'resources.json')
reports=sorted(repo.glob('runs/experiments/*pilot_completed_methods/report.json'))
report=read(reports[-1]) if reports else None
prefix_reports=sorted(repo.glob('runs/experiments/*pilot_common_prefix/report.json'))
prefix=read(prefix_reports[-1]) if prefix_reports else None
rows=read(analysis/'rounds.json');audits=read(analysis/'proposal_evidence.json');stats=read(analysis/'audit_summary.json')
names={'agents_no_critique':'Multi-agent minimum-KL','agents_critique':'Multi-agent critique','live_plugir_qwen':'Live PlugIR minimum-KL','live_raw_same_dialogue':'PlugIR raw-query diagnostic'}
methods=list(names)[:3]
def cell(v):return '—' if v is None else str(v).replace('|','\\|').replace('\n',' ')
def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(cell(x) for x in row)+' |' for row in data]])
text=['# Initial cake/giraffe pilot review','',
      '## 1. Run identity, setup and completion','',
      f"Commit: `{pre['commit']}`. VM: `dl-gpu`, `us-central1-b`, project `astute-solstice-512700-u9`; STANDARD `g2-standard-8`, one NVIDIA L4. Generation used {pre['python']}, Torch {pre['packages']['torch']} / CUDA {pre['cuda']}. See `PILOT_COMMANDS.md` for setup and exact launch commands.",'',
      'The requested runner settings were verified unchanged: both live methods enabled, pilot targets enabled, 11 states, saved references disabled, both devices CUDA, and `AGENT_MODES=(False,True)`. Search was `rewrite`; live selection was `plugir_kl`. Initial caption plus ten interaction rounds were requested, including rounds with no eligible question. No prompts or algorithms were changed. The proposed evidence-analysis/question-synthesis idea was not implemented.','',
      'The local VS Code checkout and VM clone began at the same clean commit. Environment preparation installed the pinned packages in a separate Python environment, transferred the original cache pair after anonymous Drive download failed, and downloaded the two JPEGs and pinned model files. All 126 repository tests passed. Full model loading and cache/JPEG checks passed before generation.','',
      table(['Target','Candidate ID','Starting caption','Initial rank'],[(t['target'],t['candidate_id'],t['initial_caption'],t['initial_rank']) for t in pre['targets']]),'',
      'All methods receive those same selected pairs and initial captions, 11 states, the same pinned retriever, ordered 50,000-image cache, and candidate-caption catalogue. Candidate windows follow 500, 450, …, 50 images. Different dialogue/query histories can produce different window membership and representatives; equal corpus does not imply identical later windows. Within each multi-agent round, all three roles receive exactly the same representatives and dialogue.','',
      'Preflight found no checkpoints. Generation began fresh. Saved request replays occurring later in this run are internal cache reuse, not evidence of an older resumed experiment. The two multi-agent conditions share a call-cache directory. Live PlugIR has its own call caches.','',
      'Pinned models: '+ '; '.join(f"{k}: `{v[0]}` @ `{v[1]}`" for k,v in pre['models'].items())+'.','',
      'Packages: '+', '.join(f"{k} `{v}`" for k,v in pre['packages'].items())+'.','',
      'Existing protocol asymmetries were preserved: live PlugIR gives BLIP-2 the current rewritten caption plus its selected question; multi-agent gives BLIP-2 the selected question alone. Both use the same answerer weights and target image. Live Qwen input limit is 4096 tokens; multi-agent is 8192. Live generates five alternatives sequentially with original filtering/retry/fallback behavior; multi-agent makes three role batches of up to five proposals, applies its shared filter, and selects an existing eligible question. Multi-agent has six Qwen calls per round available; critique can use the sixth call. These differences limit causal attribution of between-system outcomes to question selection alone.','',
      'Completion: partial across the requested three conditions. Cake critique stopped before completing round 6 because the model selected slot 4 when the only eligible slot was 0. Its five completed interaction rounds remain preserved. The validation correctly stopped before asking an unlisted question. A local continuation helper ran the independent remaining cases without repairing the invalid response or changing scientific code. Full-horizon metrics include only methods with 11 real saved states for both images; the separate common-prefix comparison uses the same shorter horizon for every method.','',
      table(['Method','Target','Saved states','Status','Error'],[(names[s['method']],Path(s['target']).stem,s['states'],s['status'],s['last_error']) for s in stats]),'',
      '## 2. Method comparison and resources','']
comparison=[]
for method in names:
    actual=[r for r in (report['per_example'] if report else rows) if r['method']==method]
    finals=[]
    for target in pre['targets']:
        matching=[r for r in actual if r['target']==target['target']]
        if not matching:matching=[r for r in rows if r['target']==target['target'] and r['method']==method]
        if matching:
            last=max(matching,key=lambda r:r['round'])
            finals.append(str(last['target_rank'])+(f" (r{last['round']}, partial)" if last['round']<10 else ''))
        else:finals.append(None)
    summary=report['summary'].get(method,{}) if report else {}
    final={r['k']:r for r in summary.get('final',[])}
    metric=lambda k: f"{final[k]['recall_percent']:.0f}/{final[k]['hits_percent']:.0f}" if k in final else None
    timing=resources['summary'].get(method)
    comparison.append([names[method],*finals,metric(1),metric(10),metric(100),
        f"{summary['BRI']:.6f}" if summary.get('BRI') is not None else None,
        f"{timing['wall_seconds']:.1f}" if timing else None,
        f"{timing['peak_cuda_allocated_bytes']/2**30:.2f}" if timing else None,
        timing['sampled_peak_gpu_mib'] if timing else None])
text += [table(['Method','Cake final','Giraffe final','R/H@1 %','R/H@10 %','R/H@100 %','BRI ↓','Target wall s','CUDA alloc GiB','Sampled GPU MiB'],comparison),'',
    'R/H means per-state Recall / cumulative Hits. BRI is the original best-so-far log-rank integral, not the final rank; earlier success can keep Hits and BRI favorable after later deterioration. All percentages here describe only two images.','',
    'Target wall times include function-level model loading, inference, retrieval, rewrites, checkpoint I/O and request replays; top-level setup/evaluation is excluded. Cake critique reused resident models, while giraffe critique reloaded models in the continuation process. Critique time includes the failed cake attempt and is not the cost of two completed 11-state cases. These are observed execution times with different token budgets and cache/model residency, not a controlled speed comparison. CUDA peaks are PyTorch allocated-memory peaks including the retriever, reset per target; sampled device memory is nvidia-smi at one-second intervals and may miss brief peaks.','',
    table(['Method','Target','Qwen calls in completed rounds','Replayed Qwen calls there','Calls in failed partial audit','Retained selected owners'],[(names[s['method']],Path(s['target']).stem,s['logical_qwen_calls_saved_in_completed_states'],s['reused_qwen_calls_saved_in_completed_states'],len(s.get('partial_round_audit') or []),s['selected_roles']) for s in stats]),'',
    'Partial audits retain logical call payloads/responses but not replay flags. Retained owner counts reflect role-order deduplication; another role may have proposed the same normalized selected question. The derived audit summary also records those co-proposing roles. Exact BLIP retrieval-encoder forward-call counts were not logged.','']
if prefix:
    text += [f"Matched-prefix comparison: {prefix['protocol']['rounds']} states (initial caption plus {prefix['protocol']['rounds']-1} completed rounds) on both targets. These BRI values use this shorter horizon and must not be compared directly with 11-state BRI.",'']
    prefix_rows=[]
    for method in names:
        summary=prefix['summary'][method]
        final={r['k']:r for r in summary['final']}
        last=[r for r in prefix['per_example'] if r['method']==method and r['round']==prefix['protocol']['rounds']-1]
        prefix_rows.append([names[method],*[r['target_rank'] for r in sorted(last,key=lambda r:r['example_index'])],
                           *[f"{final[k]['recall_percent']:.0f}/{final[k]['hits_percent']:.0f}" for k in [1,10,100]],f"{summary['BRI']:.6f}"])
    text += [table(['Method','Cake prefix rank','Giraffe prefix rank','R/H@1 %','R/H@10 %','R/H@100 %','Prefix BRI'],prefix_rows),'']
if report:
    metric_rows=[]
    for rnd in range(11):
        line=[rnd]
        for method in names:
            match=[r for r in report['per_round'] if r['round']==rnd and r['method']==method and r['k']==10]
            if not match and prefix:match=[r for r in prefix['per_round'] if r['round']==rnd and r['method']==method and r['k']==10]
            line.append(f"{match[0]['recall_percent']:.0f}/{match[0]['hits_percent']:.0f}" if match else None)
        metric_rows.append(line)
    text += ['Per-round Recall/Hits@10 (%); critique rounds 0–5 come from the matched prefix and later pooled values are unavailable. @1 and @100 are preserved in each report’s `per_round.csv`:', '',table(['Round',*names.values()],metric_rows),'']
text += ['## 3. Round-by-round generated dialogues','',
         'Rank change is previous rank minus current rank: positive means improvement. A missing question means no eligible question was selected, rather than a BLIP-2 failure.','']
for target in pre['targets']:
    text += ['### '+Path(target['target']).name,'']
    selected=[]
    for method in methods:
        for rnd in range(11):
            matches=[r for r in rows if r['target']==target['target'] and r['method']==method and r['round']==rnd]
            if matches:
                row=matches[0]
                selected.append([names[method],rnd,row['question'],row['answer'],row['query'],row['target_rank'],row['rank_improvement']])
            else:selected.append([names[method],rnd,'Unavailable: generation stopped',None,None,None,None])
    text += [table(['Method','Round','Question','Answer','Rewritten search query','Rank','Change'],selected),'']
text += ['The raw-query diagnostic reuses live PlugIR’s generated Q&A and searches their concatenation. It is not an additional questioner or an independently generated dialogue. Its queries/ranks are in the common report and CSV files.','',
         '## 4. Representative captions, proposals and role distinctions','',
         'IDs below are zero-based manifest caption-evidence IDs. The multi-agent code maps a representative caption string to the highest-ranked matching image in the current window; duplicate caption strings do not establish the original clustering representative’s exact image index. These are public candidate possibilities, not confirmed target facts.','']
for target in pre['targets']:
    examples=[a for a in audits if a['target']==target['target'] and a['method']=='agents_no_critique' and a['round']==1]
    if not examples:continue
    a=examples[0]
    text += ['### '+Path(target['target']).name+' — first minimum-KL round','',table(['Evidence ID','Representative candidate caption'],[(c['id'],c['caption']) for c in a['representatives']]),'']
    proposal_rows=[]
    for attempt in a['attempts']:
        p=attempt['proposal'];f=attempt['filter']
        proposal_rows.append([attempt['owner'],p['question'],','.join(map(str,p['evidence_ids'])),f['status'],f['reason'],'yes' if p['question']==a['question'] else ''])
    text += [table(['Role','Proposal','Evidence IDs','Filter','Reason','Selected'],proposal_rows),'']
text += ['## 5. Selection, answer quality, repetition and rewrite faithfulness','',
         'Qualitative review to be completed against the saved dialogues and audits before final delivery.','',
         '## 6. Raw evidence and reproducibility','',
         f"VM root: `{repo.name}` under `/home/lrc87/`. External evidence: `/home/lrc87/pilot-evidence-20261120/`. Source/configuration hashes and target/cache checksums are recorded in `preflight.json`.",'',
         'Raw files include `runs/three_agents/shared_agents_simple_v1/agents_no_critique/`, `critique_selector/`, `qwen_calls/`, `blip2_calls/`; `runs/live_upstream/<identity>/dialogues/`, `identity.json` and call caches; and `runs/experiments/<timestamp>/report.json`, CSV metrics, runner snapshot and prepared dialogue/prompt artifacts. The local archive preserves these paths.','',
         'Derived read-only analyses: `analysis/rounds.csv`, `rounds.json`, `proposal_evidence.json`, `audit_summary.json`, and `resources.json`. These summarize the original outputs; raw audit records remain authoritative.','',
         '## 7. Limitations and next experiment','',
         'Only two deliberately selected images are tested. Model answers are simulated-user responses, not human ground truth. Multiple internal proposal/filter/selector calls do not imply more interactions with the answerer. Exact normalized duplicate checks exist; semantic repetitions require qualitative review.','',
         'Live PlugIR saves candidate-caption prompt strings, final candidate questions and model request/response traces, but does not explicitly save the representative manifest IDs, every window member, per-question KL scores or a structured filtering-decision audit. Mark these unavailable unless directly supported by the saved traces. Multi-agent rounds retain these structured audits. No additional logging or generation calls were added to fill these scientific gaps.','']
dest.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(dest)
