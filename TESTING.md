# Verification

## Run checks

In VS Code's **Testing** panel, configure **pytest → tests** and run the suite.
Alternatively, use `python -m pytest -q` in the project environment. Tests use
synthetic inputs or tiny random models; no pretrained downloads/API calls are
needed. CUDA checks skip without NVIDIA CUDA; the real-KMeans test skips if the
optional live dependency is absent. Run `python -m pip check` after installation.

Then run the small saved-query example in [README.md](README.md#3-run-a-small-saved-query-check)
to exercise your actual downloaded cache and BLIP weights.

## Latest verified checks — November 3, 2026

| Check | Result |
|---|---|
| Windows CPU suite, with optional clustering dependencies available | **45 passed; 6 CUDA checks skipped** |
| Original ten-round loop versus wrapper | Matching chat requests, selected questions, answers, captions and ranks under controlled responses; original fallbacks/switches and real CPU KMeans covered |
| Original versus shared pretrained BLIP text features | Maximum absolute difference **1.49e-8** across three descriptions; tolerance 1e-6 |
| Saved-baseline regression | **72/72 ranks matched**: 8 targets starting at index 64, 3 states, 3 methods, full 50k pool; 38 ranks were below first place |
| Original generator/evaluator bytes | SHA-256 checks passed |

These checks used the existing CPU environment and cached BLIP weights. Extra
clustering packages were isolated for testing, not installed into that environment.
The teammate also reported matching Windows/Mac saved-baseline results on the
shared cache; their full environment was not independently verified here.

**Pending:** full pretrained Qwen + BLIP-2 GPU inference, CUDA test execution,
peak host/GPU memory, and retrieval quality of live Q&A or future agents.
Source parity and small regression checks do not establish benchmark improvement.

## Historical Colab reproduction — October 25, 2026

The Windows CPU setup validated the real `(50000, 256)` float32 cache and
reproduced the saved **32-target, 11-state** Colab reference, including query
hashes, target order, Recall/Hits and BRI. Tolerances: 0.011 percentage points
and 0.0001 BRI. The bedroom target ranked 1/50,000, cosine 0.4834548831.

| Method | Final Recall@10 | Final Hits@10 | BRI |
|---|---:|---:|---:|
| Raw VisDial | 59.375% | 75.0% | 1.253277 |
| Supplied PlugIR | 71.875% | 90.625% | 1.064569 |

Reference files are in [assets/colab_reference](assets/colab_reference). Original
local evidence paths were `runs/cache_verification.json`, `runs/smoke_search.json`
and `runs/colab_reproduction/`; ignored `runs/` files are not supplied by cloning.
Full 2,064-target historical results are documented separately in
[results/README.md](results/README.md).
