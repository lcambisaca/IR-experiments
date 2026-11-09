# Three-agent question experiment

This implements the agreed first experiment: three searches generate questions,
optional peer critique revises them, one common KL selector chooses a question,
and one BLIP-2 answer updates the shared conversation. The official rank at every
state comes from a **neutral description of the original caption and shared QA**.
Agent ranks are diagnostics. There is no rank fusion or spectral clustering yet.

## Run through the existing template

Use `run_evaluation.py`, the same entry point as saved baselines and the original
PlugIR + Qwen adaptation. Normal defaults remain saved-only, CPU, 32 targets and
two states. Merely importing the agent module does not load generation models.

For the first GPU pilot, change these settings in that file:

```python
N_TARGETS = 1
N_STATES = 2                  # caption plus ONE shared-answer interaction
DEVICE = "cuda"              # BLIP retrieval
LIVE_DEVICE = "cuda"         # Qwen and BLIP-2, activated in separate phases
RUN_LABEL = "three_agents_pilot"
RUN_THREE_AGENTS = True
RUN_LIVE_PLUGIR = False       # enable separately to also generate the original baseline
AGENT_MODES = (False, True)   # no critique and critique
```

The existing validated 50k image-vector cache, catalogue and selected target JPEG
are reused. No new image encoding is performed. Do not copy the Windows `.venv`
to Linux. On the GCP machine use a working NVIDIA driver and CUDA-enabled PyTorch
matching the project's requirements; then install the project live/test extras
in its Linux environment (`python -m pip install -e ".[live,test]"`). The existing
CPU lockfile is not the GCP GPU setup. We must verify the actual VM/environment
before starting a pretrained pilot; this change has not started a VM or installed
GPU packages.

Run `python run_evaluation.py` from the project, or use VS Code's Run action with
the correct environment. No API key or paid LLM service is used. Cloud GPU runtime
and first-use model downloads still consume time, disk and the VM budget.

The pilot guard allows at most two targets (`AGENTS.max_targets`). Leaving the
saved baseline's 32-target default while enabling agents fails before BLIP loads.
After inspecting the one-interaction pilot, use three states, then up to eleven.
State count can be extended without regenerating completed compatible prefixes.

## Roles and example prompts

`src/agentir/agent_prompts.py` contains editable shared instructions, stage schemas,
role cards and varied examples: a kitchen/cake, jacket/suitcase, dogs/collars, and
a child/bench. It includes redundancy, unsupported-premise, abstention, and keeping
a good question outside one's specialty. These are few-shot prompts, not training.
The exact pack is exported with every completed experiment.

- Full context: initial caption unchanged, then faithful neutral descriptions.
- Attributes: confirmed colours, counts, clothing and object details.
- Relationships: confirmed actions, locations and spatial relationships.

All queries rebuild from the authoritative initial caption and **actually asked**
QA, never from the previous role rewrite. Each generated query has a separate
LLM fidelity check. This check can miss errors: it is an audited heuristic, not a
semantic guarantee. A rejected query or malformed response stops with a checkpoint
instead of silently using an invented description.

## One interaction

1. Independently rewrite and retrieve for each role. Agent 1 uses the original
   caption unchanged before any answers. Each branch gets original PlugIR's
   shrinking 500-to-50 window, ten cosine KMeans clusters and entropy-selected
   captions. These clusters do not become attribute-only or relation-only clusters.
2. Each agent proposes at most one question using all its representative captions
   together. Invalid/redundant proposals may be replaced once. Attempts and reasons remain internal, separate from asked QA. Prior rejected attempts are supplied as avoidance history; exact previously asked questions are blocked.
3. A neutral shared-context filter checks redundancy, unsupported assumptions and
   candidate evidence. Candidate IDs must belong to the actual branch evidence.
   A caption missing an attribute does not establish its absence.
4. Exact normalized duplicate questions get one selection slot. Other proposing
   roles are recorded; semantically equivalent but differently worded duplicates
   may remain. No voting/fusion weight is assigned here.
5. In critique mode, Agent 1 reviews Agent 2, Agent 2 reviews Agent 3, Agent 3
   reviews Agent 1. Each author gets one opportunity to retain, revise or withdraw.
   Authors receive the critic's comments and their own cited evidence. Revised
   questions go through the same neutral filter and deduplication again.
6. Original `select_question` chooses minimum KL using **one neutral description
   and the same candidate slice** for every surviving question. With default K,
   the top ten are excluded, matching the original helper. The unanswered question
   is appended to the neutral description for this comparison. No target label,
   target rank or hypothetical target answer is used to select it.
7. Only the selected question is sent to BLIP-2, using the original answer helper
   and neutral-description-plus-question prompt. Qwen is parked first. BLIP-2 can
   inspect the target pixels; agents cannot. Its answer may be wrong or uncertain.
8. Append the one asked QA to the shared record, neutrally reformulate with a
   fidelity audit, retrieve, and measure the official rank. If no question survives
   the bounded attempts, carry the description/rank forward and log no question.

Human answering is a future transport extension; the GPU builder currently wires
the existing BLIP-2 answerer. It never asks all proposals to choose the best answer.

## Comparison and interpretation

Two new method names enter the common evaluator:

- `agents_no_critique`: three-role proposals, filter, common KL, answer, neutral rank.
- `agents_critique`: the same pipeline plus one peer-critique/revision round.

Proposal sampling seeds depend on the confirmed context, role, stage and request,
not on the condition name. At an identical starting context, initial proposals
match; cached calls can be reused. Later histories may diverge because different
questions produce different answers. This is a full-rollout comparison, not a
claim that every later round has matched context or that a question revision is
automatically beneficial. The logs include the pre-critique KL choice on each
critique context, but that counterfactual question is NOT also answered.

The three saved references remain visible. Their existing human/supplied dialogue
histories are evaluation references, never prompts for generation. Original live
PlugIR is still available separately and executes the authors' original loop. Its
default five proposals and these three initial agent proposals are different
budgets; the existing original run is not an equal-compute control. A later study
should also compare a single questioner with equal proposal/search budgets.

## Outputs and resume

The existing `runs/experiments/<timestamp>/` contains per-round Recall/Hits,
per-example target ranks, BRI and side-by-side rank comparison. Under
`prepared/three_agents/` it also contains:

- `agent_dialogues.json`: shared QA, official neutral queries/ranks, branch queries,
  observation-only branch ranks, representative captions, all proposals/filters,
  peer critiques, revisions, common KL scores and model-call records.
- `diagnostics.json`: compact tables for role diversity, critique changes and rounds.
- `prompt_pack.json`: exact shared/role/task instructions and few-shot examples.

Resumable checkpoints and model call caches live under
`runs/three_agents/<identity>/`. That folder also gets `diversity.csv`,
`critique_changes.csv` and `round_diagnostics.csv` after preparation completes.
Interrupted work remains in checkpoints even if common evaluator export has not
yet occurred. Use only one process/writer per identity folder.

Diversity records include query-embedding cosine similarity, exact normalized
query equality and Jaccard overlap of the agents' top-ten result sets. Critique
records include before/after text, verdict, action, reason and final filter status.
Round diagnostics include selected-question changes, retained proposal counts,
official rank, Qwen calls and new versus reused generation tokens/time. These
measure behavior, not semantic correctness or calibrated confidence.

Qwen generation seconds exclude BLIP-2, retrieval and memory transfers. Metadata
also records total preparation wall time by condition, including startup/cache
effects; it is not a controlled latency comparison. Qwen request caches retain
actual input/output token counts and response time for new generations.

Both modes share pinned Qwen/BLIP-2 weights, alternating GPU phases rather than
loading three LLM copies. Each interaction allows at most 40 Qwen requests,
including validation and critique, with bounded token outputs. This is an upper
bound, not a prediction of GPU cost. Model semantic behavior, full-model memory
fit, speed and retrieval improvement require the real GCP pilot.

## Provenance

The vendored PlugIR files remain unchanged and checksum verified. The experiment
directly calls original retrieval-context extraction (KMeans/entropy), BLIP feature
method, question-selection KL and answer generation helpers. Its role prompts,
structured filters, discussion, canonical reformulation and round orchestration
are new. This is an agent extension, not an exact paper reproduction.

The original caption selector returns strings rather than representative IDs.
For identical catalogue captions, evidence IDs use the highest-ranked matching
member of that branch's window. They identify support for the returned caption;
they do not claim to recover the internal representative's exact index.
