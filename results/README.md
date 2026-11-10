# Historical results

These are saved-query baseline results, not results from the new
live Qwen pipeline. New runs go to ignored `runs/` folders; the files here are
curated shared records. Metrics are percentages unless stated otherwise.

## Saved-query baseline

Run `20261029T235137036757Z`: **2,064 targets, 50,000 candidate images, 11 states,
K = 1, 10, 100**. These are the recorded run settings, not current runner defaults.

| Method | Final Recall@10 | Final Hits@10 |
|---|---:|---:|
| Caption only | 71.1 | 71.1 |
| Raw VisDial dialogue | 59.4 | 85.0 |
| Authors' supplied PlugIR queries | 74.3 | 91.5 |

[Final metrics](baseline/final_metrics.csv), [per-round metrics](baseline/per_round.csv)
and [per-target ranks](baseline/rank_comparison.csv) contain the full figures.
