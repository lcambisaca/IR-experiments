# Verified local baseline — 2026-10-25

This project was executed in its own Windows Python 3.11 environment using
**CPU-only PyTorch 2.11.0**, torchvision 0.26.0, Transformers 4.51.3 and the exact
BLIP checkpoint recorded by the completed Colab image cache. No GPU, image
archive, image encoding, OpenRouter request or Qwen download was used.

## Checks completed

- `pip check`: no broken requirements.
- `pytest -q`: **10 passed**.
- Actual Drive cache downloaded and validated: **50,000 x 256**, float32,
  sequential IDs, finite values, unit lengths and exact manifest/cache hashes.
- Real BLIP search using the bedroom description: target
  `unlabeled2017/000000185565.jpg` ranked **1/50,000**, cosine **0.4834548831**.
- Full supplied-query reference: **32 matched targets x 11 states x 2 methods**.
- All 11 Recall@10/Hits@10 rows and BRI matched the original full-pool Colab
  reference (`matches_colab: true`, no differences). Saved Colab percentages were
  rounded to two decimals; comparison tolerances are 0.011 percentage points and
  0.0001 BRI. The two reported BRI values were identical in this execution.

| Method | Final Recall@10 | Final Hits@10 | BRI |
|---|---:|---:|---:|
| VisDial raw dialogue | 59.375% | 75.0% | 1.2532765865325928 |
| Supplied PlugIR queries | 71.875% | 90.625% | 1.0645685195922852 |

## Evidence in the local project

- `runs/cache_verification.json`
- `runs/smoke_search.json`
- `runs/colab_reproduction/report.json`
- `runs/colab_reproduction/per_round.csv`
- `runs/colab_reproduction/per_example.csv`

The reference query hashes and the original run's target order were checked
before running the comparison. `assets/colab_reference` holds the source CSVs
and run configuration fetched from the existing shared Drive folder. `runs` is
ignored by Git; export/copy results deliberately when sharing experiment evidence.

## What these checks do not show

They establish that ordinary Python can load your real 50k cache and reproduce
the existing supplied-query evaluation locally. They do not demonstrate a new
agent method, new dialogue generation, Qwen integration or retrieval improvement.
Only 32 of the available 2,064 matched targets were evaluated in this check.
The larger candidate pool is the full 50k, not the former 5k development pool.

Do not re-encode images when increasing the query count. Add new strategies
against the same loader/retriever/metrics, record runtime and model settings,
and compare methods on the same target examples and confirmed information.

## Later experiment-runner and interface checks

The reusable experiment evaluator added eight synthetic cases, bringing the
suite to 18 passing tests. It checks parity with the existing metrics, an
identity control, past-only callback inputs, separate per-method/example state,
custom vector normalization, non-cosine fusion labels, and aligned CSV exports.

The terminal interface has now been removed. Open `run_evaluation.py` and use
Run Python File, or its Run and Debug preset. The earlier real search/Colab
evidence above is preserved as historical verification; it was not rerun merely
to remove the interface. Existing result files and research functions remain.
