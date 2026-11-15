# Results

Small copies of our main runs, so everyone can see the numbers without rerunning.
Full raw outputs (report.json, per-example baseline ranks) stay local in `runs/`, which git ignores.

## baseline/ (run 20261029T235137036757Z)
All 2,064 VisDial targets, 50,000 candidate images, 11 states (caption + 10 rounds), K = 1, 10, 100.
Settings are the ones now in `run_evaluation.py`.

| Method | Recall@10 (round 10) | Hits@10 (round 10) |
|---|---|---|
| caption_only | 71.1 | 71.1 |
| raw_dialogue (ChatIR-style, human VisDial dialogue) | 59.4 | 85.0 |
| supplied_plugir (authors' saved queries) | 74.3 | 91.5 |

- `final_metrics.csv`, `per_round.csv`: Recall@K and Hits@K for every round.
- `rank_comparison.csv`: rank of the target photo for every example, round and method.

## paraphrase/ (run 20261030T020707Z, `paraphrase_experiment.py`)
Caption only (round 0), all 2,064 targets, 5 rewrites per caption from Qwen3 4B (local, via Ollama).
At round 0, Recall@K equals Hits@K.

| Method | R@1 | R@10 | R@100 |
|---|---|---|---|
| original caption | 41.9 | 71.1 | 93.8 |
| single rewrite (average) | 38.5 | 69.0 | 92.7 |
| mean embedding of 6 | 40.4 | 71.2 | 93.5 |
| RRF of 6 | 40.2 | 70.7 | 93.6 |
| best of 6 (oracle, peeks at answer) | 48.2 | 77.4 | 95.7 |

- `summary.csv`: the table above.
- `per_example.csv`: target rank for the original and each rewrite, per example.
- `qwen3_4b_rewrites.json`: all 10,320 rewrites. Reuse these instead of regenerating.

## vision_rerank/ (Kaggle run 20261115T030727Z, `vision_rerank_experiment.py`)
Round 0 only (original caption, no dialogue). BLIP keeps its top 30; a vision-language model
(`qwen2.5vl:7b` via Ollama 0.40.16 on a Kaggle T4, digest `5ced39df…`) looks at each of the 30 photos
with the caption and is asked "Is this the photo described? Answer yes or no." The score is
10 × P(yes) / (P(yes) + P(no)), read from token log-probabilities (prompt `yes_prob`).

Targets: the same 200-caption sample as the strategy experiment (seed 42). Re-ranking can only help
targets in BLIP's top 30 (176 of 200), so the pilot scored 47 of them by BLIP rank: 10 of the 92 at
rank 1, 10 of the 57 at ranks 2–10, and all 27 at ranks 11–30. Each group is weighted by its share
of the 200, so the recall below is an **estimate for the whole sample**, not the hand-picked mix.

Two ways to use the scores, both fixed before the run:
- **Rule 1:** sort the 30 photos by score.
- **Rule 2:** photos scoring ≥ 5 (the model leans "yes") go ahead of the rest; inside each group BLIP's order is kept.
  The model only demotes photos it rejects and never reorders photos it accepts.

| Method (200 targets) | R@1 | R@5 | R@10 |
|---|---|---|---|
| BLIP, original caption | 46.0 | 63.1 | 74.5 |
| Rule 1: sort by vision score | 40.5 | 67.7 | 72.5 |
| **Rule 2: yes/no at 5, BLIP order inside** | **50.0** | **67.0** | **78.5** |
| Ceiling: target in BLIP's top 30 | 88.0 | 88.0 | 88.0 |

Top 10 per group, BLIP → rule 1 / rule 2: rank 1: 10 → 9 / 10 (of 10); ranks 2–10: 10 → 9 / 10 (of 10);
ranks 11–30: 0 → 11 / 8 (of 27).

- Rule 1 rescues more targets from ranks 11–30 but pushes some out of the top 10. The model says "yes"
  to many lookalikes (a dozen photos above 9.6 for one caption), so its order among them is close to noise.
  Rule 2 lets BLIP order those and only uses the model to drop clear mismatches.
- **Caveats.** 47 targets. Rule 2 lost none of the 20 top-10 targets checked; a small loss rate on the
  other 129 would not show here and would cut the gain. Bootstrap 90% interval for rule 2's R@10 gain:
  +2.0 to +6.0 points, which is optimistic for that reason. Variants chosen after seeing the data
  (cut-off at 8: 79.0; sum of BLIP and vision ranks: 79.0) need new targets before they count.
  Cost: 30 vision calls per search, about 65 s on a T4.
- Next: score all 176 reachable targets (`shortlist --all`) for an exact number, then apply rule 2
  inside the live 10-round PlugIR + Qwen loop.

Files:
- `summary.csv`: the table above. `per_target.csv`: BLIP rank, rule 1 rank (`vlm_rank`), rule 2 rank
  (`split_rank`) and the 30 scores for every target.
- `shortlists_top30.json`: the 47 targets and their BLIP top 30.
- `qwen2.5vl_7b_yes_prob_854fcf36e17e.json`: all 1,410 model answers. Copy it to `data/vlm_scores/` and
  `python vision_rerank_experiment.py rerank results/vision_rerank/shortlists_top30.json` rebuilds the report without a GPU.
