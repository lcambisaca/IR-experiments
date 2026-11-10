# Experiments and evaluation

Use [run_evaluation.py](run_evaluation.py) as the main entry point.
[README.md](README.md) covers installation, cache downloads and GPU settings.
Our research control is original PlugIR + Qwen. Independent agent methods
produce their own conversations and search states, then enter the same evaluator
under separate names. They need not call PlugIR's generation or selection logic;
keep retrieval/answering models, targets and interaction budgets consistent.

## Methods in the comparison

| Method | Search input |
|---|---|
| `caption_only` | Initial caption, unchanged at every state |
| `raw_dialogue` | Caption plus saved VisDial dialogue so far |
| `supplied_plugir` | Authors' saved query for that state |
| `live_plugir_qwen` | Description from newly generated PlugIR/Qwen Q&A; enabled by the live switch |
| `live_raw_same_dialogue` | Raw text from that same generated conversation |

Saved VisDial and supplied PlugIR can have different conversations/models; they
are historical references, not controls isolating an agent contribution.
`live_raw_same_dialogue` diagnoses reformulation, but its questions followed the
reformulated path, so it is not an independent live questioner.

Compare matched targets, image pools, starting information and interaction
rounds. Record additional model/search calls. Target rank is measured **after**
search; never use it to choose a winning query or branch. Label oracle analyses
as upper bounds. Higher target cosine alone does not guarantee a better rank.

## Add a per-state method

Edit the template and register it in `EXPERIMENTS`:

```python
def my_first_experiment(context, retriever):
    """Build this state's query from information available so far."""
    return context.current_query  # Identity control: should match raw_dialogue.

EXPERIMENTS = {"my_first_experiment": my_first_experiment}
```

`context` contains the initial caption, past raw `dialogue`, joined `current_query`,
round/example indices and this method's `previous_query` / `previous_state`.
It excludes future turns, target labels/ranks and supplied PlugIR text. Example
indices are sample positions, not image IDs. Do not recover hidden labels through
external lookups. Keep state 0 unchanged unless a rewrite there is intentional.

| Return value | How to use it |
|---|---|
| Description string | Evaluator encodes it with BLIP and scores every image |
| `vector_output(retriever, vector, query=...)` | Normalizes a custom vector and computes cosine scores |
| `SearchOutput(scores, "fusion", notes=...)` | Returns fused scores, such as RRF, in manifest order |

Import the helpers from `agentir.evaluation`. Scores must be one finite value
per image, with higher meaning better. RRF scores are not cosine similarities;
convert rankings to per-image scores before returning them. For normalized
embedding batches use `retriever.score_vectors()`; it handles device transfer.

Optional returned `state` persists separately per method/example within a run;
it is not serialized. Put reproducibility details in JSON-compatible `notes`.
Load expensive models once and reuse them.

## Add a method that needs preparation

Use `PREPARED_EXPERIMENTS` for generated conversations or reusable agent outputs:

```python
from agentir.evaluation import PreparedExperiment

def prepare_control(root, retriever, pairs, *, n_states, offline):
    """Prepare reusable data; this example is an unchanged-caption control."""
    def search_state(context, model):
        """Return this method's text at the requested state."""
        return context.initial_caption

    return PreparedExperiment(
        methods={"prepared_caption": search_state},
        metadata={"states": n_states},
        artifacts={"setup.json": {"targets": len(pairs)}},
    )

PREPARED_EXPERIMENTS = {"control": prepare_control}
```

The built-in live switch uses this interface. Preparation receives pairs to
locate target images for answering; keep those labels out of question generation
and selection. Generated-state callbacks should return their own history rather
than consume saved human answers. Method names must be unique; artifacts use
plain `.json` filenames. Preserve the original baseline when adding experiments.

## Outputs and metrics

Each run creates `runs/experiments/<timestamp>/`:

| Output | Contents |
|---|---|
| `rank_comparison.csv` | Target ranks side by side for each example/state |
| `per_example.csv` | Queries, ranks, scores, top image IDs, notes and paired rank changes |
| `per_round.csv` | Recall/Hits at each state and K; differences in percentage points |
| `final_metrics.csv` | Final Recall/Hits and overall BRI |
| `report.json` | Full results, selected targets, settings and cache/software provenance |
| `runner_snapshot.py` | Editable runner used for this run |
| `prepared/<name>/` | Preparation artifacts; live output includes `generated_dialogues.json` |

- **Recall@K:** percentage of targets currently ranked within the top K.
- **Hits@K:** percentage that reached the top K at least once up to this state.
- **BRI:** average trapezoidal area of log best-so-far rank across states; lower
  rewards reaching good ranks earlier. Undefined for a single state.
- **Rank improvement:** previous/reference rank minus current rank; 20 → 5 is +15.

State 0 is the initial caption. Caption-only repeats it, so its Recall/Hits stay
constant. At state 0, Recall equals Hits. The evaluator calls the authors' metric
code; use its CSVs for reported results. Keep development and final-evaluation
targets separate, and record external method code, model revisions, prompts,
seeds and costs in addition to the runner snapshot.

## Code reading order

1. [run_evaluation.py](run_evaluation.py): settings and `main()`.
2. [Original generator](third_party/PlugIR/generate_dialog.py): interaction loop and research functions.
3. [live.py](src/agentir/live.py): preparation, generated states and checkpoints.
4. [live_models.py](src/agentir/live_models.py): local Qwen/BLIP-2 connections.
5. [plugir_upstream.py](src/agentir/plugir_upstream.py): source loading and device adapters.
6. [evaluation.py](src/agentir/evaluation.py): scoring, metrics and exports.

The [comparison](PLUGIR_COMPARISON.md) explains source fidelity and runtime
differences.
# Optional three-agent questions

Enable `RUN_THREE_AGENTS` in the main template only after setting the GPU pilot
to one target/two states. This adds `agents_no_critique` and `agents_critique` to
the common rank/Recall/Hits/BRI comparison. Their official rankings use neutral
shared-context descriptions, while agent searches supply question evidence.
See [AGENT_EXPERIMENT.md](AGENT_EXPERIMENT.md) for prompt contracts, budgets,
resume and audit outputs. This is a new experiment, separate from the original
PlugIR-loop baseline. Existing saved settings are unchanged.
