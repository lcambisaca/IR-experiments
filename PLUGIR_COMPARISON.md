# Original PlugIR and our Qwen baseline

**We execute the authors' original generation functions and interaction loop.
Qwen replaces GPT for chat responses.** This document describes the current
baseline implementation. Our independent multi-agent system will generate its
own conversations and be compared through the same evaluator; it need not use
PlugIR's clustering, question-selection or reformulation logic.

## Research pipeline

| Stage | Original PlugIR | Current setup |
|---|---|---|
| Retrieval | Salesforce/blip-itm-large-coco | Same model family; original feature function on our pinned weights/cache |
| Candidate captions | Shrinking window, ten cosine KMeans clusters, lowest-entropy representatives | Original functions and fast-pytorch-kmeans |
| Questions and filtering | Original few-shot prompts, sequential alternatives, retries and fallbacks | Same functions/messages; Qwen supplies responses |
| Question selection | Minimum KL over the original candidate slice | Original `select_question` |
| Image answer | Salesforce/blip2-flan-t5-xl, original prompt and greedy generation | Same model family and original `generate_answer` |
| Caption update | Reconstruction and unchanged-caption paraphrase fallback | Same functions/messages; Qwen supplies responses |
| Next interaction | Updated caption and generated dialogue | Original loop |
| Metrics | Recall/Hits functions and BRI calculation | Actual original metric code |

Only BLIP-2 sees target pixels. Questions use public candidate captions and
generated history. Measured ranks never select questions or stop the loop.

| Model | Pinned revision |
|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Salesforce/blip2-flan-t5-xl | `0eb0d3b46c14c1f8c7680bca2693baafdb90bb28` |
| BLIP retriever | Revision recorded in the validated image-cache metadata |

The original code uses GPT-3.5-turbo variants. We map its options to local
Transformers inference: questions temperature 0.5 / 32 tokens; filtering 0 / 10;
reconstruction 0 / 512; fallback paraphrase 0.7 / top_p 0.8 / 512.
BLIP-2 keeps checkpoint generation defaults. Qwen's tokenizer/template and
responses differ from GPT's, so identical paper scores are not expected.

The runner uses `q_n=5`, `recall_hitting=10`, `threshold_low=500` and all four
feature switches enabled, matching the authors' supplied-query filename.
Original CLI defaults are `q_n=1` with switches off. Ten clusters and three
filter attempts remain source constants; original fallback quirks are preserved.

## What the wrapper changes

The original script starts model loading and reads hard-coded ChatIR paths at
import. [plugir_upstream.py](src/agentir/plugir_upstream.py) selects its original
definitions/loop through Python's syntax tree while skipping that startup.
Original function bodies and vendored file bytes remain unchanged and verified.
The adapter reverses its declared loop edits to check structural equality.

| Adaptation | Current behavior |
|---|---|
| Model connection | Original chat messages/options go to Qwen; GPT model-name/network-timeout arguments are logged, not executed |
| Devices | Inputs move to the selected device; BLIP-2 pixels are FP16, IDs remain integer; Qwen/BLIP-2 alternate between GPU and RAM |
| Paths and loop hooks | Supply our cache/JPEGs, bound/resume rounds, restore a prefix and save begin/end checkpoints |
| Randomness | Explicitly seed/save Torch's KMeans RNG; isolate Qwen sampling RNG and cache calls by reproducible slots |
| Failure handling | Save progress and stop on local errors; guard overlong Qwen input rather than silently truncating |
| Dependencies | Pin fast-pytorch-kmeans 0.2.2 and psutil 7.1.0; upstream did not specify the KMeans version |
| Metric edges | Convert one-based ranks to original indexing; deterministic manifest-index ties; null BRI for one state |

Four categories of loop edits exist: target path, round bounds/resume, prefix
restoration and begin/end hooks. Invalid/empty candidate windows and the broken
original non-referring `q_n>1` combination are rejected before inference.

Original generation does not truncate text; original evaluation does. We preserve
both, so a long-query generation checkpoint may show a different observation-only
rank. Report common-evaluator CSV metrics. Exact ties, device rounding, explicit
seeding and our pinned environment can also differ from an upstream historical run.

## Evidence and limits

[Parity tests](tests/test_live.py) independently execute the untouched ten-round
loop with controlled responses, then compare requests, selected questions,
answers, captions and ranks with the wrapper. They cover original fallbacks,
feature switches, real CPU KMeans and checkpoint recovery.

[TESTING.md](TESTING.md) records measured checks. They establish matching decisions
for tested inputs, not correct model answers or identical benchmark scores.
Full pretrained Qwen/BLIP-2 GPU inference and peak memory remain unverified.
Use [README.md](README.md#4-run-the-live-qwen-baseline-on-an-nvidia-gpu) for the pilot
and [EXPERIMENTS.md](EXPERIMENTS.md) for fair agent comparisons and metrics.

## Original source

Commit: `ce79c9ff834aabc9de698efb706c6cb9427ecf35`.

- [Generator](https://github.com/Saehyung-Lee/PlugIR/blob/ce79c9ff834aabc9de698efb706c6cb9427ecf35/generate_dialog.py)
- [Evaluator](https://github.com/Saehyung-Lee/PlugIR/blob/ce79c9ff834aabc9de698efb706c6cb9427ecf35/eval.py)
- [Checksums and provenance](third_party/PlugIR/PROVENANCE.json), [MIT license](third_party/PlugIR/LICENSE)
