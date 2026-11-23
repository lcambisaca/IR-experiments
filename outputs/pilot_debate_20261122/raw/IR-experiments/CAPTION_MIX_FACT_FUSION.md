# Caption mix and fact fusion

This guide covers our query-construction methods: what they are, where they sit in the code, how to
run them, and the results. The dialogue is never changed. Only the **search vector built from the
dialogue** changes.

## 1. The idea

PlugIR searches with one sentence, $T'_t$, that Qwen rewrites from the caption $T_0$ and all Q&A so
far. With Qwen3-4B this rewrite drifts: it drops confirmed details and invents new ones. As a result,
Recall@10 falls from 73.5 (caption only) to 64.4 after ten rounds.

$E_t$ is BLIP's text encoder, and every embedding is L2-normalised.

| Method | Search vector at round t | Where in the code |
|---|---|---|
| PlugIR (rewrite) | $E_t(T'_t)$ | `get_text_features` in [generate_dialog.py](third_party/PlugIR/generate_dialog.py) |
| **Caption mix** | $\mathrm{norm}(0.7\,E_t(T'_t) + 0.3\,E_t(T_0))$ | [fact_fusion.py](fact_fusion.py) `evaluate()`, the `mix = ...` line |
| **Fact fusion** | $\mathrm{norm}\big(\mathrm{mean}[E_t(T_0), E_t(T_0\text{, }\phi_1), \dots, E_t(T_0\text{, }\phi_k)]\big)$ | `facts_only` in [fact_fusion.py](fact_fusion.py) (post-hoc); `facts_vector` in [facts_e2e.py](src/agentir/facts_e2e.py) (live) |

**What a fact is.** After every round, one greedy Qwen call (`FACT_PROMPT`) turns question + answer
into a short positive visual phrase $\phi$, such as "red skis".
- If the answer is a non-answer, repeats the caption, says "not visible", or is unsure, the call
  returns `NONE` and no fact is kept.
- Phrases containing a negation are dropped by `clean_fact`. BLIP reads "no red car" as a red car.

## 2. Two ways to evaluate

1. **Post-hoc** ([fact_fusion.py](fact_fusion.py)). Re-rank saved live PlugIR conversations with each
   search vector.
   - Questions and answers are identical across methods, so any difference comes from the query.
   - Check: re-searching the saved rewrites reproduces every live rank. `fact_fusion.py` prints the
     agreement, and [compare_e2e.py](scripts/compare_e2e.py) asserts it.
2. **End to end** (`SEARCH = "facts"` in [run_evaluation.py](run_evaluation.py)). The fact-fusion
   vector replaces the rewrite *inside* the live PlugIR loop.
   - [facts_e2e.py](src/agentir/facts_e2e.py) `install()` wraps PlugIR's `get_text_features`:
     - the description → the fact-fusion vector;
     - "description, question", used by PlugIR's KL question choice → the fact-fusion vector plus
       that question;
     - anything else → unchanged.
   - Fact fusion therefore drives the shown ranking, the candidate images that seed questions, and
     the question selection.
   - In [live.py](src/agentir/live.py), `checkpoint_round` extracts a fact after every answer. It
     saves `fact`, `facts` and `rewrite_only_rank`, the rank plain PlugIR would show, for comparison.

## 3. Running

```bash
# Live PlugIR conversations (GPU). Rerunning resumes from checkpoints.
PYTHONPATH=src python scripts/run_plugir_live.py --start 150 --targets 100                  # PlugIR
PYTHONPATH=src python scripts/run_plugir_live.py --start 150 --targets 100 --search facts   # end to end

# Post-hoc scoring: rewrite, caption mix, fact fusion, plus secondary variants
python scripts/collect_confirm.py
PYTHONPATH=src python fact_fusion.py confirm_151_250=/abs/path/runs/confirm_151_250

# End to end vs PlugIR on the same targets, plus conversation diagnostics
PYTHONPATH=src python scripts/compare_e2e.py runs/live_upstream/<plugir_id> runs/live_upstream/<facts_id> results/fact_fusion/ranks_confirm_151_250.csv
python scripts/diagnose_e2e.py runs/live_upstream/<plugir_id> runs/live_upstream/<facts_id> results/fact_fusion/facts_qwen3_4b_instruct_2507.json

# Tests: scripted models, CPU only
PYTHONPATH=src python -m pytest tests/test_facts_e2e.py
```

**Notes on `fact_fusion.py`:**
- Its settings, prompt and methods were fixed before any result was seen, and the file is frozen. Its
  SHA-256 is in [fact_fusion.sha256](results/fact_fusion_confirm/fact_fusion.sha256).
- Its docstring and default folders still refer to the development runs it was first written for.
- [facts_e2e.py](src/agentir/facts_e2e.py) keeps an identical copy of the prompt, and a test checks
  that they match.

## 4. Results

All runs use Qwen3-4B-Instruct-2507 (questioner, rewriter, fact extractor), BLIP-2 flan-t5-xl as the
simulated user, BLIP over 50,000 images, and 10 rounds. The values are Recall@10 at round 10; the
last column is the paired bootstrap 95% CI.

| Targets | PlugIR | Caption mix | Fact fusion | Fact fusion − mix |
|---|---:|---:|---:|---|
| Primary 31–149 (119) | 62.2 | 68.1 | **73.9** | +5.9 [+1.7, +10.9] |
| Held out 151–250 (100) | 67.0 | 76.0 | **81.0** | +5.0 [−1.0, +11.0] |
| Pooled (219) | 64.4 | 71.7 | **77.2** | +5.5 [+1.8, +9.1] |
| End to end, 151–250 (100) | 67.0 | 76.0 | **79.0** | +3.0 [−3.0, +10.0] |

- **Vs PlugIR:** fact fusion gains +12.8 [+7.8, +17.8] pooled and +12.0 [+4.0, +20.0] end to end.
- **Vs caption mix:** fact fusion is ahead on every block. The gap is significant on the primary and
  pooled sets but not on the held-out block alone, so it is not yet confirmed.
- **Trade-off:** Hits@10 drops (84.0 → 79.0 vs the mix, pooled). Fact fusion keeps found targets
  near the top rather than finding more of them.
- **End to end ≈ post-hoc** (−2.0 [−8, +4]). Question quality does not change: about 23% of
  questions are repeats and about 30% of answers echo the caption in both runs. The gain comes from
  the query, not from better questions.

## 5. Files

| Path | Contents |
|---|---|
| [src/agentir/facts_e2e.py](src/agentir/facts_e2e.py) | Live fact extraction and the fact-fusion search hook |
| [src/agentir/live.py](src/agentir/live.py) | Live PlugIR wrapper; the `search="facts"` parts are marked `FACTS` |
| [fact_fusion.py](fact_fusion.py) | Post-hoc scoring of rewrite, caption mix, fact fusion and variants |
| [scripts/](scripts/) | GPU launcher, collection, end-to-end comparison and diagnostics |
| [tests/test_facts_e2e.py](tests/test_facts_e2e.py) | Prompt identity, ranks recomputed independently, the conversation differs from plain PlugIR, resume is identical |
| [results/fact_fusion/](results/fact_fusion/) | `report.json` (all metrics), per-target ranks per method and round, and the fact cache |
| [results/fact_fusion_confirm/](results/fact_fusion_confirm/) | Protocols fixed before each run, and the end-to-end report and ranks |
| [results/conversations/](results/conversations/) | Compact live conversations: PlugIR 31–150, PlugIR 151–250, and end-to-end fact fusion 151–250 |
| [assets/protocol/confirm_targets_151_250.txt](assets/protocol/confirm_targets_151_250.txt) | The held-out target list |
