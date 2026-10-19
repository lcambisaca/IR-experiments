# PlugIR: VisDial Retrieval Experiments

Team project for understanding and evaluating **PlugIR**, from [Interactive Text-to-Image Retrieval with Large Language Models: A Plug-and-Play Approach](https://arxiv.org/abs/2406.03411) (ACL 2024).

We use the [authors' official implementation](https://github.com/Saehyung-Lee/PlugIR), with adjustments for Google Colab and a smaller image pool to keep computation manageable.

## What this project currently does

This is **text-to-image retrieval**, not image classification: given a description and dialogue, we rank candidate images to find the intended target.

The quick-start experiment compares two query sources using the same pretrained BLIP retriever:

| Method | Input to the retriever |
|---|---|
| VisDial dialogue baseline | Initial caption plus the provided question–answer turns accumulated so far |
| Supplied PlugIR queries | The authors' saved, LLM-reformulated search descriptions for each round |

**This notebook makes no LLM or OpenRouter API calls and performs no training.** It evaluates previously generated queries. The baseline is the supplied VisDial conversation data; we are not regenerating the paper's zero-shot questioner experiment.

## Default experiment

| Setting | Default |
|---|---|
| Benchmark scope | VisDial only |
| Candidate image pool | 5,000 images, selected from the official 50,000-image pool |
| Evaluation targets | 32 matched examples |
| Pool construction | Retain every selected target and sample the remaining distractors |
| Sampling seed | 42 |
| Query states | 11: initial caption plus ten dialogue rounds |
| Retriever | `Salesforce/blip-itm-large-coco` |
| Image / query batch size | 8 / 8 |
| Metrics | Recall@10, Hits@10, BRI |
| Execution environment | Google Colab with an NVIDIA GPU |

Both methods use the **same target examples, candidate pool, model, and number of rounds**. Targets are matched by image path, and initial captions are checked for agreement.

The reduced pool makes retrieval easier. These are **development experiments, not a reproduction of the paper's full benchmark scores**. Comparisons are meaningful within the same pool and settings.

## Notebooks

### `01_visdial_quickstart.ipynb`

Main team entry point. It:

1. Installs dependencies and checks GPU access.
2. Fetches the original PlugIR repository at the configured commit.
3. Creates the matched query subset and deterministic image pool.
4. Restores a compatible image-vector cache when available.
5. Otherwise prepares the selected images and encodes them with BLIP.
6. Runs the adapted official evaluator on both query sources.
7. Displays metrics and saves results and completed caches to Drive.

The notebook embeds the helpers it needs; opening it does not require separately copying Python files from the repository into Colab.

### Optional learning notebook

`02_learning_and_generation.ipynb`, if included in the repository, contains the embedding walkthrough, 16-image demonstration, and an optional simplified LLM rewrite. Paid execution is disabled by default. It is **not** the full PlugIR dialogue generator.

## Start in Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Open notebook → GitHub**, paste this repository's URL, and select `01_visdial_quickstart.ipynb`.
3. Select **Runtime → Change runtime type → GPU** and connect.
4. Read the configuration cell and set `SHARED_DIR` to the team's Drive folder as it appears in your account.
5. Run the notebook sections in order. If package installation requests a session restart, restart the session and rerun configuration before continuing.

If the dependencies are already installed in the current runtime, skip installation. Do not run multiple encoding jobs concurrently in the same runtime.

## Shared data and access

GitHub contains notebooks and code. **Google Drive contains the image archives, image embeddings, and experiment outputs.** Repository access does not automatically grant Drive access.

Before running, ask the dataset owner to share the team's `PlugIR` Drive folder with your Google account. Add a shortcut under **My Drive**, or adjust the configured path to match your actual folder location.

Default configuration:

```python
SHARED_DIR = Path('/content/drive/MyDrive/PlugIR')
```

The shared folder contains or will receive:

```text
PlugIR/
├── plugir_corpus_50k.tar   # Existing full image backup, approximately 8.22 GB
├── subsets/               # Small image archives and their exact path lists
├── cache/                 # Completed BLIP vectors and compatibility metadata
├── results/               # Logs, run settings, and metric tables
└── generated/             # Optional saved learning-demo LLM outputs
```

Not every folder exists before its first successful run. The code creates the necessary output folders.

No new COCO or Flickr30k benchmark download is needed. Although our VisDial images originate from COCO, the separate COCO benchmark is outside the current scope.

## First-run time and cache reuse

Encoding images is a one-time preparation step for a particular image pool, model, and compatible setup. BLIP converts each image into a 256-dimensional vector. Later queries compare their text vectors against those saved image vectors.

At the observed T4 speed of roughly 1.1 seconds per batch of eight images, encoding the default 5,000-image pool is approximately **11–12 minutes**. This excludes installation, model downloads, archive transfer, extraction, and backups; actual time varies.

- Existing local images are reused.
- If a matching cache is available, these text-query evaluations skip raw image restoration and encoding.
- Without a cache, the first preparer saves a smaller image archive so teammates can avoid repeatedly copying the full archive.
- Completed vectors are saved locally and backed up to Drive with metadata and a checksum.
- Pool, model revision, adapter, or software changes can require a different cache. Never rename an unrelated cache to force reuse.

**Wait for `Image embeddings backed up to Drive` before treating the vector cache as persistent.** An unfinished indexing pass is not checkpointed. Colab runtime storage is temporary; the shared Drive copies persist.

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| Recall@10 | Fraction of targets appearing in the top ten at the current round | Higher |
| Hits@10 | Fraction appearing in the top ten at least once by the current round | Higher |
| BRI | Average area under the log of the best target rank achieved so far; rewards earlier improvement | Lower |

The notebook reads these results from the authors' evaluator logs. Raw cosine similarity scores from the learning demo are not substitutes for these metrics.

## Changes to the official evaluator

The notebook creates `eval_colab.py` and leaves the upstream `eval.py` intact. The adapter:

- Uses the selected image-pool manifest.
- Reduces batches and data-loader workers for Colab.
- Moves text tensors to the GPU and sets models to evaluation mode.
- Avoids retaining unused full ranking tensors.
- Writes completed caches atomically and makes logging/progress visible.

The original metric formulas are retained. Local synthetic checks verified target matching, deterministic pool selection, cache dimensions, archive restoration, and agreement between original and adapted metric outputs. **A completed end-to-end Colab run still needs to be verified before reporting results.**

Record the upstream commit, model revision, query and image manifests, software versions, seed, and run settings with every experiment. Use a distinct `RUN_TAG` for separate team runs so results are not overwritten.

## API budget and later work

The team shares a limited OpenRouter credit balance. The quick-start evaluation uses none of it.

Planned next steps:

1. Complete and inspect the small-pool baseline and supplied-query comparison.
2. Integrate the original live PlugIR generation loop through OpenRouter, starting with one target and one or two interaction rounds.
3. Explore the proposed multi-agent extension: internal agents investigate hypothetical question–answer branches and select one informative question for the user.

The goal of the extension is better retrieval with fewer user questions. Internal calls, cost, and latency must be measured separately. Hypothetical answers must remain distinct from user-confirmed information.

## Credentials and publishing

Never commit API keys or paste them into notebook Markdown, code, or outputs. Use Colab Secrets or hidden input prompts for optional API experiments. The exposed key from the original learning notebook must be revoked; removing it from cleaned copies does not invalidate it.

Before saving to GitHub, omit notebook outputs. Keep image archives, model weights, caches, and credentials out of Git commits. Colab's **Save a copy in GitHub** saves the notebook, not arbitrary files in its runtime.

## References and attribution

- [PlugIR paper](https://arxiv.org/abs/2406.03411)
- [Official PlugIR implementation](https://github.com/Saehyung-Lee/PlugIR)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

This repository contains team experiment notebooks and adaptations. The original PlugIR code remains subject to its upstream MIT license and attribution requirements.
