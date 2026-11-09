# Historical results

> **Live experiments (Nov 2026):** see [live_experiments/README.md](live_experiments/README.md) for
> agent search vs PlugIR, Matan's extra-search strategies and the caption mix, all on live conversations.

These are saved-query and initial-caption experiments, not results from the new
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

## Paraphrase study

Run `20261030T020707Z`: initial captions only, all 2,064 targets, five Qwen3 4B
rewrites per caption through Ollama. Original caption + five rewrites = six queries.

| Method | Recall@1 | Recall@10 | Recall@100 |
|---|---:|---:|---:|
| Original caption | 41.9 | 71.1 | 93.8 |
| Single rewrite, average | 38.5 | 69.0 | 92.7 |
| Mean embedding of six | 40.4 | 71.2 | 93.5 |
| RRF of six | 40.2 | 70.7 | 93.6 |
| Best of six, oracle | 48.2 | 77.4 | 95.7 |

At this single state, Recall equals Hits. The oracle uses the target answer and
is an upper bound, not a usable query-selection rule. Details:
[summary](paraphrase/summary.csv), [per-example results](paraphrase/per_example.csv),
[10,320 saved rewrites](paraphrase/qwen3_4b_rewrites.json).

To replay the historical experiment, open [paraphrase_experiment.py](../paraphrase_experiment.py):

```python
N_EXAMPLES = None
PARAPHRASE_SOURCE = "saved"
FILTER_FLAGGED = False
DEDUPLICATE = False
```

Current defaults use 100 examples, warning filtering and deduplication, which
define a different experiment. Heuristic count/color/negation checks can reject
good rewrites or miss bad ones; the original always remains. Saved mode reuses
the bank without Ollama; explicit `"generate"` mode runs local Ollama, records
new provenance and unloads Qwen before BLIP. `DEVICE` controls BLIP, not Ollama.
Historical model digest, quantization and server version were not recorded, so
this bank is not a controlled model match to the live Transformers Qwen baseline.
