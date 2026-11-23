# Fact fusion confirmation — protocol (fixed 2026-11-14, before any data)

**Question.** Does `facts_only` beat `caption_mix` on targets never used before?

**Targets.** Matched targets 151–250 (indices 150–249, `START_TARGET=150`, `N_TARGETS=100`). Indices 0–149 were used
by earlier runs or analyses; none of 150–249 have been.

**Conversations.** Live PlugIR, unchanged: `SELECTION="plugir_kl"`, Qwen/Qwen3-4B-Instruct-2507 questioner and
rewriter, BLIP-2 flan-t5-xl answerer, BLIP blip-itm-large-coco retriever, `LiveSettings` exactly as in
`run_evaluation.py` (seed 1021), 11 states. Same settings as the 119-target run.

**Scoring.** `fact_fusion.py` frozen at SHA-256 recorded in `fact_fusion.sha256` (no change to prompt, weights,
methods or fact model). All six methods are reported.

**Primary test (one).** `facts_only` minus `caption_mix`, Recall@10 at round 10, paired bootstrap over targets
(5,000 resamples, seed 42). **Confirmed** if the 95% CI lower bound is > 0.

**Also reported (not part of the pass decision).** R@10 per round, R@10 averaged over rounds 1–10, R@1, Hits@10,
better/worse/same counts, and the pooled result with the 119 original targets.

**Stopping.** Targets that do not finish all 11 states within the time cap are excluded and listed; analysis uses
only complete conversations. No target is rerun or dropped for any other reason.
