"""Build the requested review from immutable saved evidence; no model calls."""
import collections, json, pathlib

b=pathlib.Path(__file__).resolve().parent
if b.name == 'tools':
    b=b.parent
raw=pathlib.Path('\\\\?\\'+str((b/'raw').resolve()))
a=json.loads((b/'analysis/analysis.json').read_text())
pre=json.loads((raw/'preflight.json').read_text())
H={(h['method'],pathlib.PurePosixPath(h['target']).stem):h for h in a['histories']}
E={(e['method'],pathlib.PurePosixPath(e['target']).stem):e for e in a['events'] if e['event']=='end'}
methods=['agents_critique','agents_debate','live_plugir_qwen']
targets=['000000364869','000000562109']
names={'agents_critique':'Critique','agents_debate':'Debate','live_plugir_qwen':'Live PlugIR'}
prefix='outputs/pilot_debate_20261122/'
def link(label,path): return f'[{label}]({prefix}{path})'
def esc(v): return str(v if v is not None else '—').replace('|','\\|').replace('\n','<br>')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(esc(x) for x in r)+' |' for r in rows])+'\n'
def load(method,target):return json.loads((raw/H[method,target]['checkpoint']).read_text())
def fmt(x,n=4):return 'N/A' if x is None else f'{x:.{n}f}'
parts=[]
def add(s):parts.append(s.strip()+'\n')

add('''# Cake/giraffe critique–debate–PlugIR pilot review

**Run: 22–23 November 2026. Three of six histories completed all 11 states.** Critique completed cake but failed on giraffe’s tenth interaction. Debate stopped after three answered cake interactions and before its first answered giraffe interaction. Both live PlugIR histories completed. The failures, completed prefixes, raw model responses, and checkpoints are preserved. **This version is not ready for a 32-target run. No 32-target run was launched.**

The strongest finding is operational and qualitative: discussion recognized candidate-caption differences, but cake debate repeatedly asked the target-only answerer to identify or inspect retrieval candidates. Structured-output validation then stopped both debate histories. These two development images do not establish a general retrieval winner.

## 1. Run identity, setup, and reproduction

The GCP machine was `dl-gpu` in project `astute-solstice-512700-u9`, zone `us-west4-a`: standard/on-demand `g2-standard-8`, one NVIDIA L4 (23,034 MiB visible), 150 GB boot disk, approximately 31 GiB host RAM. Driver: `580.178.04`; PyTorch CUDA build: **12.8** (the driver’s supported CUDA banner is 13.0). Existing IAP SSH access was tested before setup. No new firewall rule was necessary.

The exact local checkout was **`82ed6186d4c5b98d32dff4e949e32d45804b023f` plus uncommitted changes**, including the new `debate.py` and runner updates. A Git pull alone would not reproduce it. The local working tree had changes to `EXPERIMENTS.md`, `README.md`, `run_evaluation.py`, `scripts/run_plugir_live.py`, `multi_agent.py`, `setup.py`, and three existing test files, plus untracked `debate.py` and `tests/test_debate.py`.

It was transferred into a fresh checkout at `/home/lrc87/pilot_debate_20261122/IR-experiments`. All **63 transferred source/input files** matched their hashes before generation and again after it. No scientific source, prompt, algorithm, filter, token ceiling, or answer/rewrite protocol was changed during the pilot. The earlier `outputs/pilot_20261120` evidence was retained and was not reused for generation or merged into these histories.

Fresh generation ran from **23:33:03 EST on November 22 to 00:17:31 EST on November 23** (04:33:03–05:17:31 UTC), including the pause to inspect the first failure and launch independent continuations. Preparation/download/test time is additional. Method–target timings below come from the function-level event log, not this whole dispatch interval.

The requested runner settings were verified before generation:

```python
RUN_LIVE_PLUGIR = True
RUN_MULTI_AGENT = True
RUN_MULTI_AGENT_DEBATE = True
USE_PILOT_TARGETS = True
N_STATES = 11
INCLUDE_SAVED_REFERENCES = False
DEVICE = "cuda"
LIVE_DEVICE = "cuda"
SEARCH = "rewrite"
SELECTION = "plugir_kl"
MAX_INPUT_TOKENS = 8192
# src/agentir/multagent_experiment/setup.py
AGENT_MODES = (True,)
```

`N_TARGETS=32` and `START_TARGET=0` remained in the source but were bypassed by pilot selection. The same two selected pairs, exact initial captions, 50,000-image vector pool, candidate catalogue, model revisions, and requested state count were used throughout. The initial ranks were **cake 9; giraffe 89** in all three methods. Initial representative-caption lists were also verified identical between critique and debate for both targets. Subsequent retrieved windows follow each method’s own rewritten query; they are not identical after histories diverge. PlugIR uses its original clustering/RNG path, so identical representative samples across PlugIR and the multi-agent methods were not assumed.
''')
add(table(['Target','Initial caption','JPEG SHA-256'],[[x['target'],x['caption'],x['sha256']] for x in pre['images']]))
add('''The vector cache was validated as `(50000, 256)` normalized float32 vectors with exact manifest order. Cache SHA-256: `8afcfb6393d71261c18938acd9a4db0f9e74439da8852594b8c983846a12a34f`; manifest SHA-256: `8a32638192419131513f2e5a6e5159d191de6789678fb3b1faa18e1ab46b44af`. Candidate captions covered exactly all 50,000 filenames. Both JPEGs were downloaded through the existing resolver, verified, and matched the earlier local image bytes.

The fresh VM had no project or generation caches. Necessary environment work was an isolated Python environment, dependency installation, model downloads, and input transfer—not scientific fixes. System Python 3.10 was left alone. Python **3.11.18**, `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `transformers 4.51.3`, `tokenizers 0.21.4`, `huggingface-hub 0.30.2`, Pillow `11.3.0`, NumPy `2.2.6`, Accelerate `1.6.0`, SentencePiece `0.2.1`, fast-pytorch-kmeans `0.2.2`, and psutil `7.1.0` were verified. `pip check` passed; **161 tests passed in 22.18 s**. Both generation models loaded on the GPU during preflight without generating extra completions.
''')
add(table(['Purpose','Pinned model','Revision'],[['Retrieval',*pre['models'][0]],['Questioning/rewrite',*pre['models'][1]],['Target-image answers',*pre['models'][2]]]))
add('''Qwen used BF16, BLIP-2 FP16, and retrieval BLIP float32. BLIP-2 alone received target pixels, with the shared current-description-plus-question input. Qwen rewrites used the original PlugIR reconstruction and paraphrase fallback. Question agents/coordinator received the initial caption, actual generated Q&A, representative captions/IDs, and avoidance history; measured ranks and target identity were bookkeeping outside their prompts. No saved human answers or future dialogue were supplied for generation.

The current runner initially executed critique on both targets in its normal order. Its exception on giraffe prevented the remaining preparers from running. After preserving that error, the same runner was invoked in four separate processes, enabling one remaining method and selecting one of the original pilot pairs per invocation. These **dispatch-only overrides** isolated failures; they did not change a method’s scientific settings. The frozen runner snapshot still contains all three enabled flags; the launcher and event log document the effective continuation overrides.

The removed older one-off pilot helpers were not used. Newly written verification, observation, dispatch, and export scripts live outside the GCP checkout and are preserved with this run. The observation wrapper only measured function duration, allocator peaks, and existing transport records; it made no extra model calls.

Reproducible commands (in a fresh destination; source archive is required for the uncommitted files):

```bash
gcloud compute ssh dl-gpu --zone=us-west4-a \\
  --project=astute-solstice-512700-u9 --tunnel-through-iap
# Restore debate_pilot_source.tar.gz into a fresh checkout at the recorded commit.
# The preserved setup script installs the exact project pins:
python3 -m pip install --user uv==0.13.7
~/.local/bin/uv venv --python 3.11 --seed ~/.venvs/agentir-debate
source ~/.venvs/agentir-debate/bin/activate
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \\
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[live,test]'
python -m pip check
python -m pytest -q
cd /home/lrc87/pilot_debate_20261122/IR-experiments
python -u ../verify_session.py
python -u ../execute_session.py  # calls current run_evaluation.main(); primary attempt
# Actual independent continuations after the preserved primary failure:
python -u ../execute_session.py agents_debate 0
python -u ../execute_session.py agents_debate 1
python -u ../execute_session.py live_plugir_qwen 0
python -u ../execute_session.py live_plugir_qwen 1
```

All six histories started with **no existing checkpoint**. There were **zero Qwen or BLIP-2 replays**. Model-weight caching after preflight is distinct from generation/checkpoint reuse. Interrupted histories were not resumed or repaired. Source snapshot archive SHA-256: `8871188e6616dc3610add8261df0a6f397cac371f62da7ac3dcb186e54d1148f`. Raw delivery archive SHA-256: `b85698e39334f71abd218e3ca089168f1d6864e50876751ee0416e2e49e80756`; all 769 files listed in its artifact manifest passed local readback verification. Historical files included in the checkout are excluded from the current-run analysis, which reads only its top-level `IR-experiments/runs`.

## 2. Completion, ranks, metrics, computation, and resources

Ranks are one-based, lower is better. **Last observed rank is not an 11-state final rank for an interrupted history.** BRI below uses only each row’s observed span; compare BRI across methods only at the matched spans shown afterward. A state includes the initial caption; all 42 completed interactions had an answer, and there were no completed skipped interactions.
''')
rows=[]
for method in methods:
 for target in targets:
  h=H[method,target];e=E[method,target]
  rows.append([names[method],'Cake' if target==targets[0] else 'Giraffe',h['status'],f"{h['states']}/11",h['final_rank'] if h['status']=='complete' else 'N/A',h['final_rank'],h['best_rank'],fmt(h['BRI']),f"{e['wall_seconds']/60:.2f}",f"{e['qwen_actual']} / {e['blip2_actual']}"])
add(table(['Method','Target','Status','States','11-state final','Last observed','Best','Observed-span BRI','Wall min','Qwen / BLIP-2 calls'],rows))
add('''There are 48 observed states out of 66 planned, comprising six initial states and 42 answered interactions. The missing 18 interactions were not imputed. Both standard live reports were exported by the current runner. The aborted critique/debate preparation did not reach its standard report export; its metrics here are calculated from saved checkpoint ranks using the repository’s existing `metrics_from_ranks`, with the same checksum-verified upstream metric implementation. All 22 live report rows matched their checkpoint ranks and query strings exactly.

Matched comparisons:
''')
rows=[]
for s in a['scopes']:
 if s['scope']=='both_targets_all_three_initial_only':continue
 last={m['k']:m for m in s['metrics'][-3:]}
 rows.append(['Cake, states 0–3' if s['states']==4 else 'Both targets, states 0–9',names[s['method']],fmt(s['BRI']),last[10]['recall_percent'],last[10]['hits_percent'],last[100]['recall_percent'],last[100]['hits_percent']])
add(table(['Matched scope','Method','BRI','End Recall@10 %','End Hits@10 %','End Recall@100 %','End Hits@100 %'],rows))
add('''Across **both images and all three methods**, only state 0 is shared: Recall/Hits@1 = 0%, @10 = 50%, @100 = 100%; one-state BRI is undefined. Consequently, there is **no completed three-method longitudinal comparison on both images**. Cake alone supports a shared four-state comparison; all three have the same best-so-far BRI there despite different last ranks (39 critique, 48 debate, 23 live PlugIR).

Live PlugIR’s completed 11-state, two-target BRI is **2.750661**; final Recall@10 is 0%, Hits@10 50%, Recall/Hits@100 100%. Its state-9 comparison with critique uses BRI **2.792202 versus 3.324607** over the same two targets and ten states. That is an observation on these two development images, not evidence of general superiority. Do not compare the full 11-state BRI directly with a shorter interrupted span.

Per-round two-target Recall percentages are below. In every available two-target span, Hits@1 stays 0%, Hits@10 stays 50%, and Hits@100 stays 100%, because cake started at rank 9 and both images started inside 100. Recall@1 is 0% throughout. “—” is unavailable, not zero. Complete per-case and aggregate metric arrays are in `analysis/analysis.json`.
''')
ag={x['method']:x for x in a['aggregate']}
rows=[]
for state in range(11):
 row=[state]
 for method in methods:
  x=ag[method]
  if state>=x['shared_states']:row.append('—');continue
  v={m['k']:m['recall_percent'] for m in x['metrics'] if m['round']==state}
  row.append(f"{v[10]:g} / {v[100]:g}")
 rows.append(row)
add(table(['State','Critique R@10 / R@100 %','Debate R@10 / R@100 %','Live PlugIR R@10 / R@100 %'],rows))
add('''Actual transport costs, including completions that subsequently failed validation:
''')
rows=[]
for method in methods:
 for target in targets:
  e=E[method,target]
  rows.append([names[method],'Cake' if target==targets[0] else 'Giraffe',f"{e['input_tokens']:,} / {e['output_tokens']:,}",f"{e['qwen_generation_seconds']:.2f}",f"{e['peak_allocated_bytes']/2**30:.3f} / {e['peak_reserved_bytes']/2**30:.3f}"])
add(table(['Method','Target','Qwen input / output tokens','Qwen generation s','Peak allocated / reserved GiB'],rows))
add('''Totals: **470 actual Qwen calls and 42 actual BLIP-2 answer calls**, with zero replays. The three answered debate rounds used exactly **12 Qwen calls each**; cake’s failed next opening added one call, and giraffe’s failed opening used one. Completed critique rounds used six calls normally, with one extra paraphrase in cake round 2 and giraffe round 4. There were no proposal regenerations, reviewer retries, coordinator revisions, or skipped completed rounds. Those bounded recovery paths therefore were not exercised by real generation in this run.

This is **not a matched-cost comparison**. Debate roughly doubled Qwen call count per answered interaction relative to critique, but original PlugIR itself used 171 cake and 141 giraffe Qwen calls because of its own generation/filter/fallback procedure. Calls also have very different token lengths; fewer calls did not guarantee less runtime. Qwen generation seconds exclude model loading and most transport/parking work. Wall time includes loading and answering; critique’s second target reused warm model objects, whereas the continuation histories ran in separate processes. These wall times are not a controlled speed benchmark.

One-second `nvidia-smi` sampling peaked at **10,692 MiB device memory used**; per-history allocator peaks are above. Allocator peaks exclude some driver allocations, while device sampling can miss subsecond peaks. No out-of-memory failure occurred. BLIP-2 token counts and isolated answer latency, retrieval-encoder call counts, and peak host RSS were not instrumented and are unavailable. Recorded BLIP-2 generation configurations and all model requests/responses remain in the raw cache.

## 3. Round-by-round questions, answers, rewrites, and ranks

State 0 is the starting caption. Positive Δ means improvement (`previous rank − current rank`). Questions, answers, and rewritten queries below reproduce the saved strings, including formatting prefixes and unsupported content; they have not been editorially corrected. Failed and unattempted states carry no invented query, answer, or rank.
''')
for target,label in zip(targets,['Cake — 000000364869.jpg','Giraffe — 000000562109.jpg']):
 add('### '+label)
 rows=[]
 for method in methods:
  h=H[method,target]
  for r in h['rows']:
   rows.append([names[method],r['state'],r['question'] or '(initial caption)',r['answer'],r['query'],r['rank'],r['improvement']])
  if h['status']!='complete':
   n=h['states'];rows.append([names[method],n,'FAILED before selecting/answering next question',None,None,None,None])
   if n<10:rows.append([names[method],f'{n+1}–10','Not attempted after interruption',None,None,None,None])
 add(table(['Method','State','Question','Answer','Rewritten query','Rank','Δ'],rows))

add('''## 4. Representative evidence, roles, and full debate examples

The multi-agent candidate window was 500, 450, 400, …, 50 entries across the ten planned interactions, with ten representative captions per completed interaction. The audit preserves every window’s IDs and representative text. Representative IDs identify a matching catalogue caption in the window; when caption strings duplicate, the implementation chooses the highest-ranked matching entry. They should not be interpreted as guaranteed recovery of the upstream cluster representative’s exact image index. Caption counts are not cluster population proportions.

### Critique cake, interaction 1: useful distinctions with overlapping proposals
''')
cake=load('agents_critique',targets[0]);ca=cake['states'][1]['audit']
add(table(['Representative ID','Caption'],[[x['id'],x['caption']] for x in ca['shared_retrieval']['captions']]))
add(table(['Role','Proposal','Evidence IDs','Decision and reason'],[[x['owner'],x['proposal']['question'],x['proposal']['evidence_ids'],x['filter']['status']+': '+x['filter']['reason']] for x in ca['attempts']]))
add('''The captions exposed occasion/action distinctions: birthday candles, wedding pairs, cutting versus posing, and table placement. The roles proposed different topics, but also repeated cake type and candles. Generalist and relationships produced the same candle question; it was deduplicated. Attributes’ “Are there candles…” paraphrase remained a separate eligible option. The selected question was the generalist candle question, even though BLIP-2’s answer did not confirm candles.

The attributes role also proposed cake colour, but its cited captions do not explicitly give that colour. Citation validity alone does not establish that a proposed distinction is well supported by the compressed evidence.

Across 19 answered critique interactions, selected eligible-question ownership was **generalist 16, attributes 1, relationships 2** (cake 8/0/2; giraffe 8/1/0). This credits the owner of the deduplicated eligible proposal, not every role that suggested a similar question. The generalist often supplied relationship-themed questions, so ownership does not by itself prove specialist expertise was necessary.

Completed critique audits contain 192 proposal-level gates: 97 `unknown`, 18 `already_known`, 7 exact within-pool `duplicate`, 4 `ambiguous`, 9 `unsupported`, 13 `already_asked`, and 44 `previously_rejected`. These counts exclude the failed giraffe pool, whose model filter omitted one decision. Exact repetition was often blocked at proposal time; semantic overlap remained in selected questions.

### Critique giraffe, interaction 1: representative examples and role proposals
''')
gir=load('agents_critique',targets[1]);ga=gir['states'][1]['audit']
add(table(['Representative ID','Caption'],[[x['id'],x['caption']] for x in ga['shared_retrieval']['captions']]))
add(table(['Role','First two proposals','Evidence and filtering'],[[role,'<br>'.join(x['proposal']['question'] for x in ga['attempts'] if x['owner']==role and x['slot']<2),'<br>'.join(str(x['proposal']['evidence_ids'])+' '+x['filter']['status']+': '+x['filter']['reason'] for x in ga['attempts'] if x['owner']==role and x['slot']<2)] for role in ['full','attributes','relations']]))
add('''The selected attributes question, “Is the field shaded or open?”, largely re-asked the initial caption’s “open field.” Its answer “open” then became the stronger rewrite “sunlit.” This is an example of a validity/faithfulness problem despite shared candidate evidence, not evidence that more specialist prompts necessarily produce new information.

### Full cake debate examples

The following tables preserve every opening/discussion contribution in two completed interactions, including its observation/comment, proposed question, citations, and limitation. Full payloads, coordinator attempts, and filtering are also saved in `analysis/selected_debate_examples.json` and the byte-identical debate checkpoint. Actual call payloads verified three independent openings, then identical three-message snapshots for all speakers in discussion round 1 and identical six-message snapshots in discussion round 2; each coordinator read all nine contributions. Thus the observed failures below are not an accidental within-round ordering leak.
''')
db=load('agents_debate',targets[0]);examples=[]
for state_index in [1,3]:
 s=db['states'][state_index];audit=s['audit'];examples.append(s)
 add(f'#### Cake interaction {state_index} — complete transcript')
 rows=[]
 for t in audit['transcript']:
  c=t['content'];text='<br>'.join(o['text']+' [IDs '+','.join(map(str,o['evidence_ids']))+']' for o in c.get('observations',[])) if t['stage']=='opening' else c['comment']
  rows.append([t['stage']+' '+str(t['discussion_round']),t['role'],text,c['question'],c['evidence_ids'],c['limitation']])
 add(table(['Stage / round','Role','Observation or discussion comment','Proposed question','Question evidence IDs','Limitation'],rows))
 co=audit['coordinator_attempts'][0]
 add(table(['Coordinator field','Saved value'],[['Question',co['proposal']['question']],['Evidence IDs',co['proposal']['evidence_ids']],['Reason',co['reason']],['Unresolved disagreement',co['unresolved_disagreement']],['Filter',co['validation']],['Answer',s['answer']],['Rewrite',s['query']],['Rank',s['target_rank']]]))
add('''In interaction 1, all three roles had already framed the task as **which candidate matches the initial caption**. Discussion converged on the absent cake detail in the group-posing caption. The coordinator copied the generalist/relationships discussion-round-2 question verbatim. The filter justified acceptance because candidate captions did not explicitly confirm the scene—even though the initial caption already did. Candidate-caption uncertainty was substituted for an unresolved target fact.

Across the three answered cake debate interactions, the coordinator adopted an existing transcript question exactly in **2/3** cases. Interaction 3 introduced new wording relative to all nine suggestions, combining static posing with the bride/groom/cake alternative; it still asked about a retrieval candidate rather than the target image. “New wording” is not evidence of a new useful distinction. There were no final-question revisions. The disagreement field often described incomplete captions rather than genuine disagreement between specialists.

All three selected cake debate questions referred to candidates; interactions 2 and 3 named IDs **46619 and 21232**. Those are different images from the cake target (target manifest ID 31281). BLIP-2 received only the target JPEG and current-description-plus-question, not those candidate photographs or the candidate-ID catalogue. Its “yes” answers therefore cannot establish facts about the referenced candidates, or safely transfer their wedding/posing attributes to the target. This is an answer-interface failure, not leakage of the target label to question agents.

Giraffe debate never reached discussion or coordination. Its first generalist opening did identify potentially useful count/species differences in the representative captions (multiple versus single giraffes and an elephant alternative), but failed the observation schema. There is no evidence here about how three roles would debate or synthesize a giraffe question.

## 5. Selection, repetition, answer quality, rewrite faithfulness, and failures

### Exact failure diagnoses
''')
add(table(['History','Failure point','Observed response problem','Preserved outcome'],[
['Critique giraffe','Interaction 10, filter_batch','Expected slots 0–5; returned only 0–4. Valid JSON, missing slot 5.','10 states / 9 answers; last rank 96. No selection, answer, or rewrite for state 10.'],
['Debate cake','Interaction 4, generalist opening','observations was a list of strings, not objects containing text and evidence_ids.','4 states / 3 answers; last rank 48.'],
['Debate giraffe','Interaction 1, generalist opening','Same list-of-strings observation schema violation.','Initial state only; no selected question, answer, or rewrite.'],
]))
add('''Both debate responses were syntactically valid JSON and their top-level question evidence IDs were in the supplied set. The validator’s generic message (“Opening observations need valid supplied evidence IDs”) also covers the incorrect **per-observation structure**; it does not mean the top-level IDs were fabricated. Neither output hit its token cap. Critique’s new reviewer-slot retry could not help a malformed batch filter; regeneration is for a valid but empty eligible pool, not an invalid filter response. No hidden repair loop was added.

The failed giraffe filter additionally labelled a proposal `already_known` with “Field is open, not an enclosure,” despite the updated prompt’s explicit correction. This shows model reasoning did not consistently follow that distinction; it does not show that the new prompt was absent. The frozen sources and actual prompts/responses are preserved.

### Repetition and role usefulness

- **Literal selected-question repeats:** live cake repeated its colour question at interactions 7 and 8; live giraffe repeated the fence question at 5 and 6 and the coat-colour question at 2 and 7. A formatting-prefixed version of the same giraffe fence question appeared again at 10 after 9. The first coat-colour attempt was unanswered, so its later repetition did obtain a new answer; the repeated fence answers “a fence” did not initially resolve fence type.
- **No literal or surface-normalized selected-question repeats** occurred in the completed critique/debate prefixes. This does not establish semantic novelty. Critique cake’s interaction/relationship/action questions overlap (4, 5, 8); its proximity question at 9 adds little to the starting relation. Giraffe’s 3→4 repetition is a defensible follow-up to a nonanswer, whereas asking standing versus moving at 5 largely repeats the starting caption. Cake debate’s first two questions repeat the same group/cake-position matching criterion in different forms.
- **Within-pool overlap:** the cake candle example above shows both exact duplicates and unmerged paraphrases. Specialists did expose different topics, but selected ownership was heavily generalist and role-labelled suggestions often overlapped in meaning.
- **The candidate-interface problem is not unique to debate:** live PlugIR cake’s final question also asks which retrieval candidate best describes the scene and presupposes children/diverse ages. All three debate cake questions showed this problem; it also appeared in the live baseline. A successful filter call is not proof of target-only answerability.

### Answers and rewrites: separate evidence problems

BLIP-2 repeatedly returned the current description instead of the requested attribute. Examples include critique cake’s candle, blowing-candles, decoration, gaze, and surface questions; critique giraffe’s first other-giraffes and dry/green questions; live cake’s gender/age-group and colour questions; and live giraffe’s early coat-colour, fence-type, and vegetation questions. “A cake” to cake type and “a fence” to fence type were also unhelpfully generic. Some direct answers were useful (giraffe tree presence, coat tan, zoo setting), while compound questions and candidate-ID questions made other answers hard to interpret.

Both multi-agent methods and PlugIR used the same pinned answerer and aligned input protocol. Repeating the description is consistent with the answer input encouraging caption-like responses, but this pilot did not isolate that cause. The earlier pilot had a different multi-agent rewrite/answer-input protocol and is historical diagnostic evidence, not a controlled before/after ablation.

The following are **unsupported by the authoritative caption plus actual Q&A**, even when an added fact might happen to be visible in the JPEG. A question’s premise and a representative caption are not a confirmed target answer.
''')
add(table(['Method / state','What the answer established','Unsupported or over-strong rewrite content'],[
['Critique cake 1','Repeated the starting group/cake description; did not answer candles.','decorated cake with candles'],
['Critique cake 3','yes to birthday cake; earlier blowing-candles question was unanswered.','some of them blowing out the candles'],
['Critique cake 4 / 7','Repeated a description; did not deny interaction or confirm gaze.','no one ... interacting directly; later all looking at the cake directly'],
['Critique giraffe 1','open, which was already in the caption.','sunlit; later sunny/unshaded'],
['Critique giraffe 3','Did not answer whether other giraffes were present.','alone, before the direct no at interaction 4'],
['Critique giraffe 7','no to a conjunction: directly visible AND in foreground.','a tree in the background and ambiguous not in the foreground; negating a conjunction does not identify which condition fails'],
['Critique giraffe 9','Repeated a giraffe/field/trees description; no dry/green answer.','green field'],
['Debate cake 3','yes to a question about candidate 21232, not an identified target fact.','wedding cake ... static, posed arrangement transferred into the target search'],
['Live cake 6','a young woman; previous demographic questions were unanswered.','diverse age range and mixed genders'],
['Live cake 7 / 8','Repeated birthday-cake description; no colour answer.','vibrant colours, then red, yellow, and blue'],
['Live cake 9 / 10','Repeated a description; no new expression or children answer.','celebration and joy; then children and joyful posing'],
['Live giraffe 3 / 4 / 5','Early fence/vegetation questions were unanswered; later a fence gave no vegetation type.','fence/enclosure presence first came from the question premise; vegetation/typical grassy background were added without that answer'],
['Live giraffe 8','a grassy area describes surface/area, not necessarily a path.','a grassy path nearby'],
]))
add('''The cake JPEG shows a green decorated cake and a person holding a cutting knife; no candles are visible. The candles/blowing-candles story is therefore particularly questionable. Birthday and young-woman labels were model answers rather than facts independently established by this review. The giraffe image has vegetation and a background fence, but a “no” answer to being **near** a fence is spatially ambiguous; it should not be silently converted to total fence absence. Image inspection was post-hoc review only and was not fed to any question agent.

Useful retrieval changes can coexist with unsupported rewrites: live cake reached rank 4 while its rewrite included unsupported colour embellishment, and live giraffe reached 29 while its query contained inferred background detail. These ranks are measured observations, not evidence those embellishments were faithful or causally responsible for improvement. The standard rewrite has no separate factual-verification stage.

## 6. Evidence, reports, checkpoints, and source paths

All current-run results remain on GCP under `/home/lrc87/pilot_debate_20261122`. The local evidence directory below contains the original archive, exact source/input snapshot, extracted raw files, derived tables, validation results, and short checkpoint aliases. The aliases are byte-identical copies of the long original checkpoint paths, making them practical to open on Windows. Nothing was substituted into the original checkpoints.
''')
items=[('Raw delivery archive','pilot_debate_delivery.tar.gz'),('Exact source/input archive','debate_pilot_source.tar.gz'),('Original transfer manifest','TRANSFER_MANIFEST.json'),('Original local diff','local_changes.patch'),('Raw artifact manifest','raw/ARTIFACT_MANIFEST.json'),('Preflight/settings/model identities','raw/preflight.json'),('Final source verification','raw/final_source_verification.json'),('Installed package versions','raw/packages.txt'),('GPU/driver identity','raw/gpu-identity.txt'),('161-test output','raw/pytest.log'),('Actual per-history timing/call/resource events','raw/history_events.jsonl'),('Primary runner failure and progress','raw/primary_runner.log'),('Continuation dispatch/exit history','raw/continuation_dispatch.log'),('GPU samples, primary','raw/gpu_samples.csv'),('GPU samples, continuations','raw/gpu_samples_continuations.csv'),('Derived metrics and audit summary','analysis/analysis.json'),('Observed questions/answers/rewrites CSV','analysis/observed_rounds.csv'),('Wide ranks CSV; blanks mean unavailable','analysis/observed_rank_comparison.csv'),('Validation checks','analysis/validation.json'),('Selected full debate examples','analysis/selected_debate_examples.json'),('Observed settings and runner source','raw/IR-experiments/run_evaluation.py'),('Debate source and prompts','raw/IR-experiments/src/agentir/multagent_experiment/debate.py'),('Independent-agent source','raw/IR-experiments/src/agentir/multagent_experiment/multi_agent.py'),('Shared filter prompt source','raw/IR-experiments/src/agentir/multagent_experiment/agent_prompts.py'),('Observer/continuation launcher','raw/execute_session.py'),('Environment setup commands','raw/setup_debate_pilot.sh'),('Primary dispatch commands','raw/run_pilot.sh'),('Independent dispatch commands','raw/continue_independent.sh')]
items.extend([('Per-round Recall/Hits by explicit comparison scope','analysis/per_round_metrics.csv'),('Checkpoint analysis command source','tools/analyze_results.py'),('Matched-scope validation command source','tools/prepare_review_data.py'),('Review generation command source','tools/build_review.py')])
add(table(['Evidence','Local link'],[[label,link('open',p)] for label,p in items]))
add(table(['Method','Target','Full dialogue/audit checkpoint'],[[names[m],'Cake' if t==targets[0] else 'Giraffe',link('checkpoint',H[m,t]['alias'])] for m in methods for t in targets]))
add(table(['Run log','Local link'],[[m+' / '+('Cake' if i==0 else 'Giraffe'),link('log',f'raw/{m}_{i}.log')] for m in ['agents_debate','live_plugir_qwen'] for i in [0,1]]))
reports=sorted((raw/'IR-experiments/runs/experiments').glob('*/report.json'))
add(table(['Standard runner output','Report'],[[p.parent.name,link('report.json','raw/'+str(p.relative_to(raw)).replace('\\','/'))] for p in reports]))
add('''Raw call caches live under the **current** `raw/IR-experiments/runs/three_agents/`, `three_agents_debate/`, and `live_upstream/` method folders (`qwen_calls/*.json`, `blip2_calls/*.json`). Checkpoint identities and the event log retain exact GCP paths. Failure-time copies are in `raw/errors/`. Completed live exports include the standard CSVs, runner snapshot, and `prepared/qwen_live/generated_dialogues.json`. Failed critique/debate preparers did not produce their usual final prompt-pack/export files; actual prompts remain in their raw request records and the frozen source files linked above.

Each multi-agent completed state preserves representative IDs/captions, all candidate-window IDs, proposal/filter decisions, avoidance history, selected question, answer, rewritten query, and rank; debate states additionally preserve all nine contributions, coordinator reasons, disagreements, and attempts. Partial failure audits preserve the calls completed before the exception. There is no invented final decision for an omitted filter slot.

PlugIR preserves representative-caption prompt text, candidate questions, selected question, answer, rewritten/raw queries, ranks, and Qwen request/response traces. Its original representative image IDs, complete per-round candidate-window ID list, explicit KL values, and a normalized per-proposal filter-decision table are **not logged**. Its 500→50 window size is derived from recorded configuration/source, not recovered IDs. Raw traces can be inspected but were not promoted into invented normalized decisions.

Raw query strings/dialogue are preserved for all methods. Saved references and PlugIR’s extra raw-same-dialogue scoring pass were disabled. A raw query derived from one method’s own generated dialogue is a reformulation diagnostic, not a separate independently generated conversation; no such extra diagnostic column is presented here.

## 7. Limitations and the next experiment

**Do not scale this version to 32 targets yet.** Three of six independent histories interrupted, including both debate histories. Giraffe debate has no answered interaction, and the only three-method matched trajectory is three interactions on cake. These are two development images with one deterministic attempt per history, not a representative test set, a repeated-seed study, or a matched-cost comparison.

Before another fresh two-target pilot, the specific blockers are:

1. Make observation-object and batch-slot compliance reliable under an explicitly bounded, audited policy. The current run has no evidence that reviewer retry or empty-pool regeneration repairs these different failure types. Preserve the present failures when changing validation/decoding or adding any repair attempt; use a new source/protocol identity and fresh checkpoints.
2. Make the question’s audience explicit and validate it: the answerer sees the target image, not candidate IDs/photos. Reject candidate-identification questions and avoid merely re-testing facts in the starting caption. Test that this holds for both debate and the live baseline.
3. Address semantic repetition and the distinction between a real answer, a nonanswer, a question premise, and candidate-caption possibilities. The false open-field/enclosure inference and the candle/colour/children additions are concrete regression cases. Any rewrite-fidelity or answer-input change belongs in a separate controlled protocol; do not quietly swap BLIP-2 or claim the earlier pilot is its control.
4. Re-run the two-image pilot with completion/validity gates, comparable observation overhead, full candidate-selection diagnostics where available, and explicit compute budgets. Then use held-out targets and more than one seed before making retrieval-performance claims. An agreement/difference/uncertainty analysis is useful only if it leads to a new, target-answerable distinction; the first cake debate demonstrates that recognizing caption uncertainty alone is insufficient.

No proposed follow-up algorithm, prompt repair, answerer replacement, or 32-target experiment was implemented in this baseline run. The VM was left running; stopping generation does not stop VM billing.
''')
(b/'analysis/selected_debate_examples.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False),encoding='utf-8')
(b/'PILOT_DEBATE_RESULTS_REVIEW.md').write_text('\n'.join(parts),encoding='utf-8')
print('REPORT',len('\n'.join(parts)),'characters',len('\n'.join(parts).split()),'words')
