# Fusing caption rewrites and vision re-ranking

This guide covers two round-0 experiments from the paper: what they test, where they sit in the
code, how to run them, and the results. Both use only the caption $T_0$ (no dialogue yet), BLIP
over the same 50,000 images, and the same random sample of 200 VisDial captions (seed 42).

- **Fusing caption rewrites** supports the query-construction results (caption mix and fact fusion,
  [CAPTION_MIX_FACT_FUSION.md](CAPTION_MIX_FACT_FUSION.md)). It fuses several queries that add **no**
  new information, to check whether multiple queries help on their own.
- **Vision re-ranking** works after the query: a vision-language model reorders BLIP's top 30.
  It can follow any query, including fact fusion.

Every number in the paper's sections on these experiments is printed by
[rewrite_vision_numbers.py](rewrite_vision_numbers.py) from the committed results, and
[tests/test_rewrite_vision_numbers.py](tests/test_rewrite_vision_numbers.py) checks them.

## 1. Fusing caption rewrites

$E_t$ is BLIP's text encoder, and every embedding is L2-normalised. Qwen3-4B (via Ollama) writes
5 rewrites $T_0^{(1)}, \dots, T_0^{(5)}$ of the caption under each of four few-shot strategies. Every
prompt forbids adding objects.

| Strategy | What changes | Where in the code |
|---|---|---|
| paraphrase | Wording only | Reused from [paraphrase_experiment.py](paraphrase_experiment.py) (`results/paraphrase/qwen3_4b_rewrites.json`) |
| detail | Shorter or longer versions | `STRATEGIES["detail"]` in [strategy_experiment.py](strategy_experiment.py) |
| refocus | A different stated element as the main subject | `STRATEGIES["refocus"]` |
| decompose | One object, action or setting per query | `STRATEGIES["decompose"]` |

The caption and its rewrites are fused **without knowing the target** (`methods_for` in
[fusion_experiment.py](fusion_experiment.py)):

| Fusion | Score or search vector | Code |
|---|---|---|
| Mean embedding | $\mathrm{norm}(\sum_i w_i E_t(T_0^{(i)}))$, caption weight $w_0 \in \{1,2,3\}$ | `mean embedding (orig xW)` |
| Reciprocal rank fusion | $\sum_i w_i / (60 + \mathrm{rank}_i(I))$ | `rrf (orig xW)` |
| z-max | Standardise each query's scores over the 50K images, keep the maximum | `z-max` |
| Interleave | Round-robin over the result lists, skipping duplicates | `interleave_rank` |
| Oracle (upper bound) | Best rank over all queries; uses the target, so not a method | `best of all queries (oracle*)` |

## 2. Vision re-ranking

All in [vision_rerank_experiment.py](vision_rerank_experiment.py).

1. **Shortlist** (`build_shortlists`, CPU): BLIP's top $K = 30$ for each caption.
2. **Score** (`ask_vlm`, `parse_yes_prob`): Qwen2.5-VL-7B (via Ollama) sees the photo and the
   caption and is asked "Is this the photo described? Answer yes or no." We read the next-token
   probabilities, not the generated word: $v = P(\text{yes}) / (P(\text{yes}) + P(\text{no}))$.
   It is stored as `score` $= 10v$.
3. **Re-rank**, with two rules fixed before the run:
   - **Rule 1** (`rerank`): sort the 30 by $v$; ties keep BLIP's order.
   - **Rule 2** (`split_rerank`): photos with $v \ge 0.5$ first, then the rest, **each in BLIP's
     order**. The model can only demote photos it rejects.
4. **Stratified pilot** (`STRATA`, `weighted_recall`). Of the 200 sample targets, 176 are in BLIP's
   top 30, and scoring them all costs 5,280 model calls. The pilot therefore scores 10 of the 92
   targets at rank 1, 10 of the 57 at ranks 2–10, and all 27 at ranks 11–30, and weights each group
   by its share of the 200. `shortlist --all` builds shortlists for all 176 for an exact run.

Every answer is saved and reused, so an interrupted run resumes without repeating work.
`compare` tries prompts on 6 targets before the pilot.

## 3. Running

```bash
# Fusing caption rewrites. CPU only; the saved rewrites are committed, so Ollama is not needed.
PYTHONPATH=src python strategy_experiment.py   # rewrites per strategy -> runs/strategies/<time>/
PYTHONPATH=src python fusion_experiment.py     # fusion rules, reusing that run -> runs/fusion/<time>/

# Vision re-ranking
PYTHONPATH=src python vision_rerank_experiment.py shortlist          # CPU: BLIP top 30, 47 pilot targets
python vision_rerank_experiment.py rerank runs/vision_rerank/shortlists_top30.json  # GPU + Ollama
python vision_rerank_experiment.py compare runs/vision_rerank/shortlists_top30.json --prompts yes_prob

# Reproduce the paper's numbers without a GPU (reads results/ only)
python rewrite_vision_numbers.py

# Tests: CPU only, the vision model is faked
PYTHONPATH=src python -m pytest tests/test_vision_rerank.py tests/test_rewrite_vision_numbers.py
```

**GPU runs.** The `rerank` and `compare` phases need only the Python standard library, Pillow and an
Ollama server with `qwen2.5vl:7b`. They take about 2 s per photo on a T4.
[notebooks/vision_rerank_kaggle.ipynb](notebooks/vision_rerank_kaggle.ipynb) runs them on a free
Kaggle GPU. To rebuild the pilot report without a GPU, copy
`results/vision_rerank/qwen2.5vl_7b_yes_prob_*.json` to `data/vlm_scores/` and run `rerank` on
`results/vision_rerank/shortlists_top30.json`: every answer is reused.

## 4. Results

**Fusing caption rewrites** (Recall@10 at round 0, 200 targets; caption alone: 74.5).

| Strategy | One rewrite (avg) | Oracle |
|---|---:|---:|
| paraphrase | 73.7 | 80.0 |
| detail | 68.0 | 82.5 |
| refocus | 71.0 | 81.0 |
| decompose | 35.8 | 80.5 |
| all 21 queries | – | 86.5 |

- **The queries find different targets:** the oracle over all 21 gains +12.0 [+7.5, +16.5].
- **Fusion without the target recovers almost none of it:**
  - all 21 fused: +0.5 [−2.0, +3.0];
  - the best rule on the sample, paraphrase z-max: +3.5 [+1.0, +6.0] on 200 targets, but
    −0.3 [−1.3, +0.6] on all 2,064.
- **Fusion did best with the caption weighted 2–3×.** Several queries help only when they add
  information and stay anchored to $T_0$, as fact fusion does.

**Vision re-ranking** (round 0, 200 targets, pilot estimate; Δ = Recall@10 change vs BLIP, 95% CI).

| Method | R@1 | R@10 | Δ R@10 |
|---|---:|---:|---|
| BLIP | 46.0 | 74.5 | |
| Rule 1 (vision order) | 40.5 | 72.6 | −1.9 [−13.7, +6.5] |
| Rule 2 (vision filter) | 50.0 | **78.5** | +4.0 [+2.0, +6.5] |
| Target in BLIP's top 30 | 88.0 | 88.0 | |

- **Rule 1** rescues 11 of the 27 targets at ranks 11–30, but drops 1 of 10 out of the top 10 in each
  group already there. The model accepts many lookalikes: in 12 of 47 shortlists, 10 or more of the
  30 photos scored above 0.96.
- **Rule 2** lost none of the 20 top-10 targets checked and rescued 8 of 27.
- **Caveats:** 47 targets. The interval is narrow partly because no top-10 target was lost, and a
  bootstrap cannot show losses absent from the sample. Round 0 only.
- **Prompt choice:** the yes probability separated the target from the other photos best (6 targets:
  0.93 vs 0.41). Asking for a 0–10 score gave most photos 9 or 10. Those answers came from an earlier
  Kaggle session and were not kept, so only the yes-probability answers are committed.

## 5. Files

| Path | Contents |
|---|---|
| [strategy_experiment.py](strategy_experiment.py) | Rewrite strategies (prompts and examples), single-rewrite and oracle ranks |
| [fusion_experiment.py](fusion_experiment.py) | Label-free fusion rules on the saved rewrites, plus paraphrases on all 2,064 targets |
| [vision_rerank_experiment.py](vision_rerank_experiment.py) | Shortlists, vision scoring, rules 1 and 2, stratified estimate |
| [rewrite_vision_numbers.py](rewrite_vision_numbers.py) | Every number in the paper, with bootstrap CIs, from `results/` |
| [notebooks/vision_rerank_kaggle.ipynb](notebooks/vision_rerank_kaggle.ipynb) | Runs the vision phases on a Kaggle GPU |
| [results/rewrite_fusion/](results/rewrite_fusion/) | `rewrites/` (detail, refocus, decompose), summaries, and per-target ranks for every method |
| [results/vision_rerank/](results/vision_rerank/) | The 47 shortlists, all 1,410 model answers, and per-target ranks under BLIP and rules 1 and 2 |
| [tests/](tests/) | `test_vision_rerank.py` (ordering, weighting, a full run with a fake model); `test_rewrite_vision_numbers.py` (the paper's numbers) |
