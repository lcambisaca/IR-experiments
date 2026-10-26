# Run and extend the evaluation in VS Code

Open `run_evaluation.py`, select the project's `.venv` interpreter, and click
**Run Python File**. No terminal commands are required. Output appears in VS
Code's run terminal. The first run may download BLIP's checkpoint if needed;
this does not encode images again. No API key or new question generation is used.

## Start with the three references

For a first check, set `N_TARGETS = 4` and `N_STATES = 3`. Every run searches
**all 50,000 images**; the current runner settings may select more targets.
State 0 is the caption; states 1 and 2 add one saved dialogue turn each.
Change `N_TARGETS`, `N_STATES`, and `K_VALUES` at the top of the runner.
`N_TARGETS = None` selects all 2,064 matched targets. Increasing target count
does not change the image pool or require encoding it again.

Every run includes:

| Method | What gets searched |
|---|---|
| `caption_only` | The original description, repeated unchanged at every state |
| `raw_dialogue` | Original description plus saved VisDial dialogue so far |
| `supplied_plugir` | Authors' saved PlugIR query for the corresponding state |

Supplied PlugIR queries may have different question/answer histories from raw
VisDial. Treat them as a reference system, not proof that any difference comes
from reformulation alone. Nothing in this runner generates a new PlugIR dialogue.
Colab reproduction remains available separately; new experiments don't need it.

## Add a new text experiment

Replace the body of `my_first_experiment` with your own query construction, then
uncomment its entry in `EXPERIMENTS`:

```python
def my_first_experiment(context, retriever):
    """Build a search description from the caption and information known so far.
    Return text; the evaluator handles BLIP encoding, scoring, and target ranks.
    """
    return context.current_query  # Placeholder: identical to raw_dialogue.

EXPERIMENTS = {"my_first_experiment": my_first_experiment}
```

The unchanged template is a control, not an improvement method. Enabling it should
match `raw_dialogue`. Add more named functions to the dictionary to compare them
in the same run. No new experiment is implemented by default.

Each callback gets only the current example's available information:

- `context.initial_caption`: original description.
- `context.dialogue`: tuple of saved raw dialogue turns so far, excluding caption.
- `context.current_query`: original caption joined with those turns.
- `context.round_index`: 0-based search state.
- `context.example_index`: position in this evaluation sample, not an image ID.
- `context.previous_query`: this method's prior returned text, or None initially.
- `context.previous_state`: optional state returned by this method last round.

There are no future turns, target filenames, target embeddings, measured ranks,
or supplied PlugIR descriptions in this context. The evaluator measures ranks
only after the method returns. The retriever exposes the whole image pool for
retrieval; callbacks must not look up target labels elsewhere or use example
indices to recover them. This interface prevents accidental feedback, not
deliberate access to public benchmark answers.

If the method needs a separate language model, load it once outside the callback,
then reuse it. No Qwen/OpenRouter backend is added here. Keep round 0 unchanged
if you want a shared initial state, or clearly document a rewrite at state 0.

## Return a custom search vector instead

Import `vector_output` from `agentir.evaluation`. It normalizes your vector,
compares it against the same saved image vectors, and packages the scores.

```python
from agentir.evaluation import vector_output

def my_vector_experiment(context, retriever):
    """Build a query vector and return its cosine scores for shared evaluation.
    This identity example uses the raw query vector; replace the construction
    with your actual vector-combination idea when ready.
    """
    vector = retriever.embed_texts([context.current_query])[0]
    return vector_output(retriever, vector, query=context.current_query)
```

## Return combined agent scores or rankings

Import `SearchOutput` from `agentir.evaluation`. Return exactly one finite score
per candidate image, in **manifest order**. Higher means a better match.

For RRF or another fusion method, return
`SearchOutput(scores=fused_scores, score_kind="fusion", notes={...})`.
These scores are not cosine similarities; the report leaves `target_cosine`
empty instead of falsely labeling them cosine. A ranked list of image IDs is
not directly accepted: convert a complete ranking to per-image scores, or
construct scores from your ranking fusion algorithm before returning it.

To explore ten agent queries, score/combine their results inside one callback
and return one final score vector. Store their descriptions in JSON-compatible
`notes` for inspection. All ten internal searches still represent one search
state/user-information budget. Never select the winning branch using the hidden
target's rank. A separately labeled oracle analysis could inspect that afterward.

For method-specific history, use `text_output(..., state=...)`,
`vector_output(..., state=...)`, or `SearchOutput(..., state=...)`.
State is kept separately for each method/example, reset each run, and not saved
in JSON (it can contain tensors). Log reproducibility details in `notes` as well.
Do not put API keys or other secrets in notes.

This framework evaluates changes under **saved dialogue**. Testing which new
question to ask requires a later live answerer adapter and matched interaction
budgets. It is not implemented by replaying answers to unrelated questions.

## Read the results

Each click Run creates a new folder under `runs/experiments/`:

| File | What it tells you |
|---|---|
| `rank_comparison.csv` | One row per target/state, one rank column per method |
| `per_example.csv` | Query, previous query, target rank/score, best rank, top IDs, notes and rank changes |
| `per_round.csv` | Recall/Hits for every method, state and K, with changes against references |
| `final_metrics.csv` | Final Recall/Hits and overall BRI, suitable for side-by-side comparison |
| `report.json` | Complete structured results, selected targets, settings and cache/software provenance |
| `runner_snapshot.py` | Copy of the editable runner used for this run |

`rank_improvement_vs_previous = previous rank - current rank`.
For example, 20 -> 5 gives +15; positive is better. State 0 has no previous rank.
The same sign convention applies to changes versus the reference methods.
Recall/Hits changes are **percentage points**, not relative percentages.
BRI is one summary across states; lower is better. It is undefined for one state.
An unchanged caption repeated over states can accumulate no new Hits successes.

The pipeline uses target **rank among all candidates** as the main measurement.
Higher target cosine alone need not mean a better rank. Keep pool, targets,
initial information and interaction budget the same across methods. Four targets
are a debugging sample, not evidence of general performance. Develop ideas on a
development subset and reserve separate targets for final evaluation.

The editable runner seeds Python/PyTorch and records its source checksum/snapshot.
If you later import custom methods from other files or use stochastic LLMs,
record those code versions, model revisions, prompts, generation settings,
agent/API costs and generated outputs too; the runner snapshot alone won't
capture externally defined code or guarantee deterministic LLM outputs.
