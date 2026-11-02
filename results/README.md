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
