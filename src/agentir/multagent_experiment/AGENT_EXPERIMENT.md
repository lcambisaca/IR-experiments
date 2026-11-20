# Specialist questions with shared retrieval evidence

The next development experiment uses one neutral query, one retrieved candidate
pool and the same ten representative captions for three question generators.
Their specialities affect question generation, not which confirmed facts survive
or which search produces the official rank. `AGENT_MODES` selects minimum-KL and/or critique selection.

## Selected cases and rounds

`run_evaluation.py` selects exactly these two image identities, in this order:

| Initial caption | Image | Verified CPU starting rank / 50,000 |
|---|---|---:|
| A group of people posing in front of a cake | unlabeled2017/000000364869.jpg | 9 |
| A giraffe is standing in an open field | unlabeled2017/000000562109.jpg | 89 |

`N_STATES` in `run_evaluation.py` controls both live methods; 11 means the caption
plus ten interaction rounds. `USE_PILOT_TARGETS=True` selects these two identities.
With `USE_PILOT_TARGETS=False`, `START_TARGET` and `N_TARGETS` select a shared batch
(32 targets by default). The multi-agent allowance follows the selected batch.

These are deliberately selected development cases, not an unbiased benchmark.
Cake starts inside the top ten; it can show rank/BRI changes but initial Recall@10
is already successful. The giraffe starts outside the top ten. GPU ranks should
be recorded rather than assumed identical to CPU ranks.

## One interaction

Original caption + confirmed, actually asked Q&A
→ current neutral description (the original caption at round zero)
→ retrieve the candidate pool and cluster ONCE
→ select ten representatives using the original entropy/KMeans helper
→ give the SAME description, record and ten captions to each agent
→ each generates up to FIVE questions in ONE response
→ omit malformed, unsupported-ID, previously asked/rejected and exact duplicate questions
→ one combined neutral filter checks all remaining proposals
→ minimum-KL or Qwen critique selects ONE eligible question
→ obtain ONE BLIP-2 answer using current description + selected question
→ append actual Q&A and use the unchanged PlugIR reconstruction and paraphrase fallback
→ measure the official target rank against the full 50k database.

The original shrinking candidate-window schedule is retained: 500, 450, 400,
350 and 300 across the five interactions. All agents share the current round's
pool, not a permanently frozen set. Target inclusion in that pool is not guaranteed.
There is no specialist reranking, new clustering, rank fusion or image re-encoding.
If no question survives, log a no-question state without calling the answerer or
inventing Q&A; the six states do not guarantee five valid answered questions.

## Prompts and evidence

The common prompt follows PlugIR's description/dialogue/candidate distinction,
with synthetic worked examples and compact structured output. This is a local
adaptation, not an exact paper reproduction or weight training.

| Stable branch ID | Question preference |
|---|---|
| full | Generalist: any useful unresolved distinction |
| attributes | Colours, counts, materials, clothing and object types |
| relations | Actions, interactions, spatial relationships and settings |

Every agent receives all confirmed facts. Candidate captions are possibilities,
not facts about the target. Missing details remain unknown. Questions already
answered by the original caption or a confirmed answer must be omitted even when
candidate captions omit or contradict that detail. Specialists may return fewer
questions or an empty list rather than force an unsupported question.

Live proposal output is only `question` and `evidence_ids`, inside `proposals`.
The worked examples include short explanations, while live generation does not
request extended reasoning. Cited captions are retained in the audit and can be
looked up without asking the model to restate them. The combined filter uses
globally unique slots, short reasons, and receives no author-role labels.

Reformulation now uses checksum-verified original PlugIR `reconstruct_dialog`.
Both pipelines format the input as the original caption followed by actual
question/answer strings, using the original prompt, exemplar, temperature zero,
512-token output request and seed. If the result exactly equals the previous
search description, both invoke the original `paraphrase` helper (temperature 0.7,
top_p 0.8, 512 tokens). This is stochastic, so matching procedures do not guarantee
identical text for different histories or model-call slots.

The custom multi-agent `reformulate` prompt and its JSON/format checks are removed.
The original rewrite procedure has no answer-responsiveness or fidelity gate;
this known limitation is deliberately retained to compare question generation.
BLIP-2 receives the current description plus the selected question in both paths.
No stronger visual answerer or critique-index recovery is added in this change.
Earlier pilot results used a custom agent rewrite and question-only answer input;
they must not be described as results from this aligned protocol.

## Runtime bounds

With critique off, a normal answered round uses three proposal calls, one filter,
and one original reconstruction call. Critique adds one selector call. The
unchanged-query fallback can add one paraphrase call beyond the ordinary
`max_calls_per_round` ceiling. All calls are counted and saved; original PlugIR
rewrite retry behavior remains unchanged. If no question is eligible, the round
makes no answer or rewrite call. Proposal/filter/critique output still uses its
structured JSON validation and stops on invalid output.

The agent transport explicitly allows up to 1024 output tokens, accommodating
the combined filter's maximum 960-token request. Individual proposal requests
remain capped at 512; the original live PlugIR transport keeps its default
512-token ceiling. These limits are ceilings, not mandatory output lengths.

For two cases and five answered interactions per condition, the usual totals are
50 Qwen requests without critique or 60 with critique, plus up to ten paraphrase
calls per condition, and ten BLIP-2 answers per condition. Request count is not a runtime guarantee: encoding candidate
captions for entropy, model transfers and longer histories also take time.
Cases run sequentially, using the same model instances in preparation.

The existing CUDA environment, pinned Qwen/BLIP-2 weights and saved image vectors
are reused. Opening/importing the runner does not launch models. Running the
entry point does; selected target JPEGs may be downloaded if missing and allowed
by `DOWNLOAD_MISSING_TARGETS`. Windows CPU dependencies are not a GPU environment.

## Outputs, memory and interpretation

The normal evaluator exports per-round ranks, Recall/Hits and BRI under
`runs/experiments/<timestamp>/`, along with the exact runner snapshot. Saved
human/supplied dialogues remain comparison references and never enter generation.

`prepared/three_agents/agent_dialogues.json` contains original/shared descriptions,
all generated questions, filters, evidence IDs/captions, shared pool IDs, selected
questions/answers and target ranks. `prompt_pack.json` includes the actual role,
shared, task and worked-example instructions plus the pinned upstream rewrite provenance.

Checkpoints are saved under `runs/three_agents/<run_name>/plugir_original_rewrite_v1/`, in separate
directories for each condition. The default run name is `shared_agents_simple_v1`.
The new policy directory separates these runs from the earlier custom rewrite.
Resume rejects checkpoints without the matching rewrite-policy marker, but does not
otherwise check source, image, or configuration fingerprints. Use a new run name when changing inputs, settings, prompts, or code.
Sampling seeds and the shared model-response caches retain their own keys.

`question_memory` records proposals separately from confirmed facts. Asked and
rejected questions form avoidance history; normalised exact repeats are blocked.
Useful unasked proposals may remain eligible later. Semantic duplicates can still
slip through. Candidate IDs never identify the target to the agents.

Diagnostic CSVs include per-role proposal/retained counts, selected roles, batch
call counts and actual token/generation timings. Shared query/Top-10 overlap is
expected to be one; it is no longer evidence of failed agent diversification.
Inspect question content to judge whether specialities actually differ. Duplicate
questions have one selection slot and gain no voting weight.

Minimum KL measures distribution stability, not expected information gain.
Neither grounded questions nor a generalist proposal guarantees improved ranks.
Rank improvement and real GPU runtime must be measured in the development run.

## Phase two and provenance

Critique remains off by default. Use `AGENT_MODES=(False, True)` to compare
both selectors, or `(True,)` for critique alone. With critique off, minimum KL
selects a question. With critique on, one neutral review call compares the
filtered questions and returns an eligible slot and a reason; KL is skipped.
This adds at most one call per round. The review cannot revise or invent questions.

Vendored PlugIR files remain unchanged and checksum verified. Original text
encoding, candidate clustering/entropy, KL selection and answer helpers are
reused. Specialist prompts, combined filtering, memory and orchestration are ours.
The caption selector returns strings; duplicate catalogue captions are linked to
the highest-ranked matching image in the current window, not necessarily the
internal representative's exact image identity.
