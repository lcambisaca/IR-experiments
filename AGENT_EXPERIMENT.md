# Specialist questions with shared retrieval evidence

The next development experiment uses one neutral query, one retrieved candidate
pool and the same ten representative captions for three question generators.
Their specialities affect question generation, not which confirmed facts survive
or which search produces the official rank. Critique is off for this test.

## Selected cases and rounds

`run_evaluation.py` selects exactly these two image identities, in this order:

| Initial caption | Image | Verified CPU starting rank / 50,000 |
|---|---|---:|
| A group of people posing in front of a cake | unlabeled2017/000000364869.jpg | 9 |
| A giraffe is standing in an open field | unlabeled2017/000000562109.jpg | 89 |

`N_STATES=6` records the initial caption plus five interaction rounds per case.
`TARGET_IMAGES` selects identities explicitly, avoiding a contiguous slice that
would accidentally include other targets. Missing or duplicate names fail before
model preparation. Set `TARGET_IMAGES=None` to restore contiguous selection with
`START_TARGET` and `N_TARGETS` for other experiments.

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
→ original minimum-KL selection chooses ONE question against the neutral query
→ obtain ONE BLIP-2 answer
→ append actual Q&A, create one neutral rewrite and audit its faithfulness
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

Reformulation happens once after a shared answer. It rebuilds from the original
caption and actual Q&A, preserving counts, negations, uncertainty, photographic
properties and who does what. A separate fidelity audit may still miss mistakes;
these are fallible language-model checks, not semantic guarantees.

## Runtime bounds

At most six Qwen requests per interaction with critique off:
three proposal batches, one combined filter, one rewrite and one fidelity audit.
The runner sets `max_calls_per_round=6`, `questions_per_agent=5`,
`replacement_attempts=0` and `AGENT_MODES=(False,)`. There are no replacement
loops. Proposal responses are capped at 512 tokens; the combined filter has a
bounded budget proportional to its proposal count. Invalid/truncated responses
stop with an interrupted checkpoint instead of a hidden repair loop.

The agent transport explicitly allows up to 1024 output tokens, accommodating
the combined filter's maximum 960-token request. Individual proposal requests
remain capped at 512; the original live PlugIR transport keeps its default
512-token ceiling. These limits are ceilings, not mandatory output lengths.

For two cases and five interactions, this is at most 60 Qwen requests and ten
BLIP-2 answers. Request count is not a runtime guarantee: encoding candidate
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
shared, task, worked-example and neutral-rewrite instructions.

Resumable checkpoints are under `runs/three_agents/<identity>/`. The prompt
version `three_roles_v5_shared_evidence_specialists` and source hashes distinguish
this experiment from the earlier per-agent searches. Old results are preserved
and are not silently reused as new-protocol checkpoints.

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

Optional critique/revision code is retained for phase two but is not enabled by
this runner. Its additional calls require an explicitly larger budget. Specialist
caption selection and larger per-cluster shortlists remain unimplemented ideas.

Vendored PlugIR files remain unchanged and checksum verified. Original text
encoding, candidate clustering/entropy, KL selection and answer helpers are
reused. Specialist prompts, combined filtering, memory and orchestration are ours.
The caption selector returns strings; duplicate catalogue captions are linked to
the highest-ranked matching image in the current window, not necessarily the
internal representative's exact image identity.
