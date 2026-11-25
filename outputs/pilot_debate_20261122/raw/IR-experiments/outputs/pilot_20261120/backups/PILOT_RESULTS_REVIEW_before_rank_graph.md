# Initial cake/giraffe pilot review

**Five of six requested histories completed.** Critique reached cake rank 1 after the attributes agent asked its colour, then stopped on an invalid selection in round 6. At the full 11-state horizon, multi-agent minimum-KL finished at cake/giraffe ranks **29/93**, versus live PlugIR **35/155**. Both had final Recall@10 of 0%; live PlugIR had the lower BRI because its earlier giraffe ranks were better. These two images do not establish a general winner.

## 1. Run identity, setup and completion

Commit: [`0855fb8b55236098565c466567d96419e551825d`](https://github.com/lcambisaca/IR-experiments/commit/0855fb8b55236098565c466567d96419e551825d). VM: `dl-gpu`, `us-central1-b`, project `astute-solstice-512700-u9`; STANDARD `g2-standard-8`, one NVIDIA L4, driver **580.178.04**, 23,034 MiB reported device memory, 150 GB boot disk. Generation used Python 3.11.18, Torch 2.11.0+cu128 / CUDA 12.8. See [PILOT_COMMANDS.md](PILOT_COMMANDS.md) for setup and exact launch commands.

Generation phases were **17:47:59–18:13:02 UTC** and **18:16:47–18:41:25 UTC**, 20 November 2026. The first process exited 1 on the preserved critique failure; continuation and metric export exited 0. The final VM check showed no GPU memory in use and no tracked-source diff. The VM was left running.

The requested runner settings were verified unchanged: both live methods enabled, pilot targets enabled, 11 states, saved references disabled, both devices CUDA, and `AGENT_MODES=(False,True)`. Search was `rewrite`; live selection was `plugir_kl`. Initial caption plus ten interaction rounds were requested, including rounds with no eligible question. No prompts or algorithms were changed. The proposed evidence-analysis/question-synthesis idea was not implemented.

The local VS Code checkout and VM clone began at the same clean commit. Environment preparation installed the pinned packages in a separate Python environment, transferred the original cache pair after anonymous Drive download failed, and downloaded the two JPEGs and pinned model files. All 126 repository tests passed. Full model loading and cache/JPEG checks passed before generation.

| Target | Candidate ID | Starting caption | Initial rank |
|---|---|---|---|
| unlabeled2017/000000364869.jpg | 31281 | a group of people posing in front of a cake | 9 |
| unlabeled2017/000000562109.jpg | 48282 | a giraffe is standing in an open field | 89 |

All methods were configured with those same selected pairs and initial captions, 11 states, the same pinned retriever, ordered 50,000-image cache, and candidate-caption catalogue. Candidate windows follow 500, 450, …, 50 images. Different dialogue/query histories can produce different window membership and representatives; equal corpus does not imply identical later windows. Within each multi-agent round, all three roles receive exactly the same representatives and dialogue. Initial captions/ranks matched across all three conditions. All 44 generated-query/rank rows in the full-horizon comparison and all 36 in the matched-prefix comparison exactly matched their saved dialogue states.

Effective budgets: runner seed 42; live/agent seed 1021; five live alternatives; up to five proposals per role; agent call budget six per round; threshold 500; KL excludes the current top 10; retrieval text batch eight; four CPU threads; `OFFLINE=False`; prompt version `three_roles_v8_critique_selector`. Question agents receive text, not target pixels or measured target ranks. BLIP-2 alone receives target JPEGs. `N_TARGETS` and `START_TARGET` were bypassed by pilot selection.

Role IDs `full`, `attributes` and `relations` mean generalist, attributes and relationships. All three use the same Qwen weights in sequential role calls and receive identical representative captions/dialogue within a round.

Preflight found no checkpoints. Generation began fresh. Saved request replays occurring later in this run are internal cache reuse, not evidence of an older resumed experiment. The two multi-agent conditions share a call-cache directory. Live PlugIR has its own call caches.

Pinned models: qwen: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`; answerer: `Salesforce/blip2-flan-t5-xl` @ `0eb0d3b46c14c1f8c7680bca2693baafdb90bb28`; retriever: `Salesforce/blip-itm-large-coco` @ `19502f1e215844f7e48bd48473f86932486d3441`.

Packages: torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, transformers `4.51.3`, tokenizers `0.21.4`, huggingface-hub `0.30.2`, pillow `11.3.0`, numpy `2.2.6`, accelerate `1.6.0`, sentencepiece `0.2.1`, fast-pytorch-kmeans `0.2.2`, psutil `7.1.0`.

Existing protocol asymmetries were preserved: live PlugIR gives BLIP-2 the current rewritten caption plus its selected question; multi-agent gives BLIP-2 the selected question alone. Both use the same answerer weights and target image. Live Qwen input limit is 4096 tokens; multi-agent is 8192. Live generates five alternatives sequentially with original filtering/retry/fallback behavior; multi-agent makes three role batches of up to five proposals, applies its shared filter, and selects an existing eligible question. Multi-agent has six Qwen calls per round available; critique can use the sixth call. These differences limit causal attribution of between-system outcomes to question selection alone.

Completion: partial across the requested three conditions. Cake critique stopped before completing round 6 because the model selected slot 4 when the only eligible slot was 0. Its five completed interaction rounds remain preserved. The validation correctly stopped before asking an unlisted question. A local continuation helper ran the independent remaining cases without repairing the invalid response or changing scientific code. Full-horizon metrics include only methods with 11 real saved states for both images; the separate common-prefix comparison uses the same shorter horizon for every method.

| Method | Target | Saved states | Status | Error |
|---|---|---|---|---|
| Multi-agent minimum-KL | 000000364869 | 11 | complete | — |
| Multi-agent minimum-KL | 000000562109 | 11 | complete | — |
| Multi-agent critique | 000000364869 | 6 | interrupted | Critique must choose one eligible proposal slot and give a reason. |
| Multi-agent critique | 000000562109 | 11 | complete | — |
| Live PlugIR minimum-KL | 000000562109 | 11 | complete | — |
| Live PlugIR minimum-KL | 000000364869 | 11 | complete | — |

## 2. Method comparison and resources

Per-state Recall/Hits figure: [PNG](outputs/pilot_20261120/figures/recall_hits_11_states.png), [vector PDF](outputs/pilot_20261120/figures/recall_hits_11_states.pdf), [SVG](outputs/pilot_20261120/figures/recall_hits_11_states.svg), [plotted data](outputs/pilot_20261120/figures/recall_hits_plot_data.csv). Panels cover @1, @10 and @100; critique ends at state 5, with no later-value imputation.

| Method | Cake final | Giraffe final | R/H@1 % | R/H@10 % | R/H@100 % | BRI ↓ | Target wall s | CUDA alloc GiB | Sampled GPU MiB |
|---|---|---|---|---|---|---|---|---|---|
| Multi-agent minimum-KL | 29 | 93 | 0/0 | 0/50 | 100/100 | 3.257655 | 1187.5 | 9.76 | 10646 |
| Multi-agent critique | 1 (r5, partial) | 95 | — | — | — | — | 923.3 | 9.69 | 10536 |
| Live PlugIR minimum-KL | 35 | 155 | 0/0 | 0/50 | 50/100 | 3.092526 | 835.5 | 9.55 | 10550 |
| PlugIR raw-query diagnostic | 97 | 292 | 0/0 | 0/50 | 50/100 | 3.270193 | — | — | — |

R/H means per-state Recall / cumulative Hits. BRI is the original best-so-far log-rank integral, not the final rank; earlier success can keep Hits and BRI favorable after later deterioration. All percentages here describe only two images.

Target wall times include function-level model loading, inference, retrieval, rewrites, checkpoint I/O and request replays; top-level setup/evaluation is excluded. Cake critique reused resident models, while giraffe critique reloaded models in the continuation process. Critique time includes the failed cake attempt and is not the cost of two completed 11-state cases. These are observed execution times with different token budgets and cache/model residency, not a controlled speed comparison. CUDA peaks are PyTorch allocated-memory peaks including the retriever, reset per target; sampled device memory is nvidia-smi at one-second intervals and may miss brief peaks.

Whole-process peak host RSS was **29.65 GiB** in the original process and **29.73 GiB** in continuation; neither reported swaps. The largest sampled GPU usage was **10,646 MiB (10.40 GiB)**, and peak PyTorch reserved memory was **10.15 GiB**. GPU inference fit on the L4, while host RAM use approached the machine's approximately 31 GiB usable memory. No CUDA OOM was observed. The process timings were 25:10.65 and 24:45.51, including setup/evaluation within those processes; model downloads and local review/transfer time are excluded.

### Generation call counts and reuse

| Condition | Logical Qwen calls | BLIP-2 answers | Notes |
|---|---:|---:|---|
| Multi-agent minimum-KL | 92 | 14 | Six cake and eight giraffe answers; six rounds had no eligible question |
| Multi-agent critique | 89 | 12 | Includes five calls in failed cake round 6; eight recorded Qwen cache replays |
| Live PlugIR | 410 | 20 | 202 cake and 208 giraffe Qwen calls; no recorded Qwen replay |
| Raw-query diagnostic / metric export | 0 | 0 | Reuses saved dialogue; retrieval encoding only |

The shared multi-agent caches contain **173 fresh Qwen completions and 26 fresh BLIP-2 answers**, including the rejected critique completion. Live caches contain **410 fresh Qwen completions and 20 BLIP-2 answers**. Thus there were **583 unique Qwen generations and 46 BLIP-2 generations**; retrieval-encoder forward-call counts are unavailable. Eight identical first-round proposal/filter requests were replayed across the two multi-agent conditions. No prior dialogue checkpoint was resumed, and the failed cake selection was not retried or repaired.

Live Qwen stages were 184 referring-question calls, 192 filter calls, eight fallback question-generation calls, 20 reconstructions and six paraphrases. All 184 referring calls reached their original 32-token cap; 90 filter calls reached their 10-token cap. These are saved `finish_reason="length"` outcomes, not 274 execution exceptions. The original parser uses the question before `?`; explanations or filter answers can still be cut off. The limits were preserved. All 173 fresh multi-agent completions ended with `stop`.

Fresh Qwen decoding produced 27,506 output tokens across the two multi-agent conditions, versus 7,821 for live PlugIR, despite fewer multi-agent calls. Decoding seconds were 1,524.42 versus 458.84 respectively. These totals cover unequal conditions/completion horizons and exclude model movement, retrieval and BLIP-2 time; they explain why call count alone is an inadequate cost measure.

| Method | Target | Qwen calls in completed rounds | Replayed Qwen calls there | Calls in failed partial audit | Retained selected owners |
|---|---|---|---|---|---|
| Multi-agent minimum-KL | 000000364869 | 44 | 0 | 0 | {'attributes': 1, 'full': 1, 'relations': 4} |
| Multi-agent minimum-KL | 000000562109 | 48 | 0 | 0 | {'full': 5, 'attributes': 3} |
| Multi-agent critique | 000000364869 | 30 | 4 | 5 | {'full': 1, 'attributes': 4} |
| Multi-agent critique | 000000562109 | 54 | 4 | 0 | {'full': 5, 'attributes': 2} |
| Live PlugIR minimum-KL | 000000562109 | 208 | 0 | 0 | {} |
| Live PlugIR minimum-KL | 000000364869 | 202 | 0 | 0 | {} |

Partial audits retain logical call payloads/responses but not replay flags. Retained owner counts reflect role-order deduplication; another role may have proposed the same normalized selected question. The derived audit summary also records those co-proposing roles. Exact BLIP retrieval-encoder forward-call counts were not logged.

Matched-prefix comparison: 6 states (initial caption plus 5 completed rounds) on both targets. These BRI values use this shorter horizon and must not be compared directly with 11-state BRI.

| Method | Cake prefix rank | Giraffe prefix rank | R/H@1 % | R/H@10 % | R/H@100 % | Prefix BRI |
|---|---|---|---|---|---|---|
| Multi-agent minimum-KL | 18 | 103 | 0/0 | 0/50 | 50/100 | 3.285357 |
| Multi-agent critique | 1 | 84 | 50/50 | 50/50 | 100/100 | 2.916032 |
| Live PlugIR minimum-KL | 21 | 48 | 0/0 | 0/50 | 100/100 | 3.150841 |
| PlugIR raw-query diagnostic | 43 | 248 | 0/0 | 0/50 | 50/100 | 3.283029 |

Per-round Recall/Hits@10 (%); critique rounds 0–5 come from the matched prefix and later pooled values are unavailable. @1 and @100 are preserved in each report’s `per_round.csv`:

| Round | Multi-agent minimum-KL | Multi-agent critique | Live PlugIR minimum-KL | PlugIR raw-query diagnostic |
|---|---|---|---|---|
| 0 | 50/50 | 50/50 | 50/50 | 50/50 |
| 1 | 0/50 | 0/50 | 0/50 | 0/50 |
| 2 | 0/50 | 0/50 | 0/50 | 0/50 |
| 3 | 0/50 | 0/50 | 50/50 | 0/50 |
| 4 | 0/50 | 0/50 | 0/50 | 0/50 |
| 5 | 0/50 | 50/50 | 0/50 | 0/50 |
| 6 | 0/50 | — | 0/50 | 0/50 |
| 7 | 0/50 | — | 0/50 | 0/50 |
| 8 | 0/50 | — | 0/50 | 0/50 |
| 9 | 0/50 | — | 0/50 | 0/50 |
| 10 | 0/50 | — | 0/50 | 0/50 |

## 3. Round-by-round generated dialogues

Ranks are one-based, with lower values better. Rank change is previous rank minus current rank: positive means improvement. A missing question means no eligible question was selected, rather than a BLIP-2 failure; rows explicitly marked unavailable were not generated.

### 000000364869.jpg

| Method | Round | Question | Answer | Rewritten search query | Rank | Change |
|---|---|---|---|---|---|---|
| Multi-agent minimum-KL | 0 | — | — | a group of people posing in front of a cake | 9 | — |
| Multi-agent minimum-KL | 1 | What type of cake is it? | a birthday cake | a group of people posing in front of a birthday cake | 21 | -12 |
| Multi-agent minimum-KL | 2 | What activity are the people engaged in besides posing? | a group of people are posing for a picture | a group of people posing in front of a birthday cake | 21 | 0 |
| Multi-agent minimum-KL | 3 | Where is the group posing in front of the cake? | a table | a group of people are posing for a picture in front of a birthday cake on a table | 24 | -3 |
| Multi-agent minimum-KL | 4 | What is the relationship between the people and the table? | a group of people standing in front of a table | a group of people standing in front of a table, posing for a picture, in front of a birthday cake | 21 | 3 |
| Multi-agent minimum-KL | 5 | Is the cake visible to the people posing? | no | a group of people standing in front of a table, posing for a picture, in front of a birthday cake that is not visible to the people | 18 | 3 |
| Multi-agent minimum-KL | 6 | Are the people standing or sitting in relation to the cake? | are standing or sitting | a group of people standing or sitting in front of a table, posing for a picture, with a birthday cake that is not visible to them | 29 | -11 |
| Multi-agent minimum-KL | 7 | — | — | a group of people standing or sitting in front of a table, posing for a picture, with a birthday cake that is not visible to them | 29 | 0 |
| Multi-agent minimum-KL | 8 | — | — | a group of people standing or sitting in front of a table, posing for a picture, with a birthday cake that is not visible to them | 29 | 0 |
| Multi-agent minimum-KL | 9 | — | — | a group of people standing or sitting in front of a table, posing for a picture, with a birthday cake that is not visible to them | 29 | 0 |
| Multi-agent minimum-KL | 10 | — | — | a group of people standing or sitting in front of a table, posing for a picture, with a birthday cake that is not visible to them | 29 | 0 |
| Multi-agent critique | 0 | — | — | a group of people posing in front of a cake | 9 | — |
| Multi-agent critique | 1 | Is the cake decorated with candles? | yes | a group of people are posing in front of a cake that is decorated with candles | 15 | -6 |
| Multi-agent critique | 2 | How many candles are on the cake? | a few | a group of people are posing in front of a cake that is decorated with a few candles | 12 | 3 |
| Multi-agent critique | 3 | Are there any other decorations on the cake besides candles? | yes | a group of people posing in front of a cake that is decorated with candles and a few candles, and has other decorations besides candles | 21 | -9 |
| Multi-agent critique | 4 | Are the people posing in front of a birthday cake? | yes | a group of people are posing in front of a birthday cake that is decorated with candles and a few other decorations | 25 | -4 |
| Multi-agent critique | 5 | What color is the cake? | green | a group of people are posing in front of a green birthday cake that is decorated with candles and a few other decorations | 1 | 24 |
| Multi-agent critique | 6 | Unavailable: generation stopped | — | — | — | — |
| Multi-agent critique | 7 | Unavailable: generation stopped | — | — | — | — |
| Multi-agent critique | 8 | Unavailable: generation stopped | — | — | — | — |
| Multi-agent critique | 9 | Unavailable: generation stopped | — | — | — | — |
| Multi-agent critique | 10 | Unavailable: generation stopped | — | — | — | — |
| Live PlugIR minimum-KL | 0 | — | — | a group of people posing in front of a cake | 9 | — |
| Live PlugIR minimum-KL | 1 | What is the gender composition of the group posing in front of the cake? | a group of people posing in front of a cake | A crowd of individuals standing in front of a cake. | 16 | -7 |
| Live PlugIR minimum-KL | 2 | What is the occasion associated with the cake in the photo? | birthday | a group of people posing in front of a birthday cake | 21 | -5 |
| Live PlugIR minimum-KL | 3 | What type of clothing are the people wearing when posing in front of the cake? | a group of people posing in front of a birthday cake | A group of individuals standing in front of a birthday cake. | 10 | 11 |
| Live PlugIR minimum-KL | 4 | What type of cake is being posed in front of? | a birthday cake | a group of people posing in front of a birthday cake | 21 | -11 |
| Live PlugIR minimum-KL | 5 | What are the people wearing when posing in front of the birthday cake? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, wearing various clothing styles, celebrating a birthday occasion | 21 | 0 |
| Live PlugIR minimum-KL | 6 | What colors are dominant in the cake and the people's clothing during the pose? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, wearing colorful clothing, with vibrant colors dominating both the cake and the attire | 13 | 8 |
| Live PlugIR minimum-KL | 7 | What specific clothing items do the people wear when posing in front of the birthday cake? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, wearing colorful clothing, with vibrant hues dominating both the cake and the attire, no specific clothing details provided, but overall festive and celebratory appearance | 27 | -14 |
| Live PlugIR minimum-KL | 8 | What specific items are on the birthday cake? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, wearing colorful clothing, with festive elements on the cake, in a lively and celebratory setting | 18 | 9 |
| Live PlugIR minimum-KL | 9 | what is the other object in the image? | a group of people posing in front of a cake | A group of people posing in front of a birthday cake, wearing colorful clothing, with festive decorations on the cake and no other prominent objects in the scene. | 20 | -2 |
| Live PlugIR minimum-KL | 10 | What specific clothing items do the people wear when posing in front of the birthday cake? | a group of people posing in front of a cake | a group of people posing in front of a birthday cake, wearing various clothing colors, with no specific clothing items mentioned, and the cake featuring no specific decorations or additional items visible | 35 | -15 |

### 000000562109.jpg

| Method | Round | Question | Answer | Rewritten search query | Rank | Change |
|---|---|---|---|---|---|---|
| Multi-agent minimum-KL | 0 | — | — | a giraffe is standing in an open field | 89 | — |
| Multi-agent minimum-KL | 1 | What type of environment surrounds the giraffe? | grassy | a giraffe is standing in a grassy environment | 85 | 4 |
| Multi-agent minimum-KL | 2 | Are there any trees nearby? | yes | a giraffe is standing in a grassy environment with trees nearby | 96 | -11 |
| Multi-agent minimum-KL | 3 | What type of grass is present? | a grass | a giraffe is standing in a grassy environment with trees nearby and a grass present | 77 | 19 |
| Multi-agent minimum-KL | 4 | Are there any rocks or stones present in the environment? | yes | a giraffe is standing in a grassy environment with trees nearby and a grass present, and rocks or stones are also present | 71 | 6 |
| Multi-agent minimum-KL | 5 | Are the trees tall or short? | tall | a giraffe is standing in an open field surrounded by grassy environment with tall trees and rocks | 103 | -32 |
| Multi-agent minimum-KL | 6 | Is the field surrounded by rocks or not? | no | a giraffe is standing in an open field surrounded by grassy terrain with tall trees and rocks, but not surrounded by rocks | 102 | 1 |
| Multi-agent minimum-KL | 7 | Are the trees located near the giraffe or in a distant area? | distant | a giraffe is standing in an open field surrounded by grassy terrain with tall trees in a distant area, and rocks or stones are present, but the field is not surrounded by rocks | 85 | 17 |
| Multi-agent minimum-KL | 8 | Are the trees visible in the background or in the foreground? | a giraffe in a zoo | a giraffe is standing in an open field surrounded by grassy environment with tall trees in a distant area, which are visible in the background; the grass present is a grass; there are rocks or stones in the environment; the field is not surrounded by rocks | 93 | -8 |
| Multi-agent minimum-KL | 9 | — | — | a giraffe is standing in an open field surrounded by grassy environment with tall trees in a distant area, which are visible in the background; the grass present is a grass; there are rocks or stones in the environment; the field is not surrounded by rocks | 93 | 0 |
| Multi-agent minimum-KL | 10 | — | — | a giraffe is standing in an open field surrounded by grassy environment with tall trees in a distant area, which are visible in the background; the grass present is a grass; there are rocks or stones in the environment; the field is not surrounded by rocks | 93 | 0 |
| Multi-agent critique | 0 | — | — | a giraffe is standing in an open field | 89 | — |
| Multi-agent critique | 1 | Is there a tree, rock, or bridge in the scene? | no | a giraffe is standing in an open field with no tree, rock, or bridge in the scene | 81 | 8 |
| Multi-agent critique | 2 | Are there any other animals present besides the giraffe? | yes | a giraffe is standing in an open field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe | 105 | -24 |
| Multi-agent critique | 3 | Is the field surrounded by a fence or boundary? | no | a giraffe is standing in an open field; there are no trees, rocks, or bridges in the scene; there are other animals present besides the giraffe; the field is not surrounded by a fence or boundary | 27 | 78 |
| Multi-agent critique | 4 | Are the other animals in the field or within an enclosure? | a zebra in a pen | a giraffe is standing in an open field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary | 72 | -45 |
| Multi-agent critique | 5 | What is the surface type of the field? | grass | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary | 84 | -12 |
| Multi-agent critique | 6 | — | — | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary | 84 | 0 |
| Multi-agent critique | 7 | What is the exact location of the zebra relative to the giraffe? | zebra | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the zebra is in the field and within the pen; the field is not surrounded by a fence or boundary | 94 | -10 |
| Multi-agent critique | 8 | Is the zebra in the pen located near the giraffe? | yes | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary; the zebra in the pen is located near the giraffe | 95 | -1 |
| Multi-agent critique | 9 | — | — | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary; the zebra in the pen is located near the giraffe | 95 | 0 |
| Multi-agent critique | 10 | — | — | a giraffe is standing in an open grass field; there is no tree, rock, or bridge in the scene; there are other animals present besides the giraffe, including a zebra in a pen; the field is not surrounded by a fence or boundary; the zebra in the pen is located near the giraffe | 95 | 0 |
| Live PlugIR minimum-KL | 0 | — | — | a giraffe is standing in an open field | 89 | — |
| Live PlugIR minimum-KL | 1 | What is the activity of the giraffe while standing in the open field? | a giraffe is standing in an open field | A giraffe is standing in a wide, open field. | 150 | -61 |
| Live PlugIR minimum-KL | 2 | What type of enclosure or environment surrounds the giraffe in the open field? | a zoo | a giraffe standing in an open field within a zoo enclosure | 52 | 98 |
| Live PlugIR minimum-KL | 3 | What color is the giraffe's coat in the open field? | a giraffe standing in an open field within a zoo enclosure | A giraffe standing in a spacious field inside a zoo's fenced area. | 57 | -5 |
| Live PlugIR minimum-KL | 4 | What color is the giraffe's coat when standing in the open field of a zoo? | a tan | a tan giraffe standing in an open field within a zoo enclosure | 57 | 0 |
| Live PlugIR minimum-KL | 5 | What is the giraffe doing besides standing in the open field of a zoo? | a tan giraffe standing in an open field within a zo | A light-colored giraffe standing in an open field inside a zoo's fenced area. | 48 | 9 |
| Live PlugIR minimum-KL | 6 | What color is the fence surrounding the giraffe in the open field of the zoo? | white | a tan giraffe standing in an open field within a zoo, surrounded by a white fence | 96 | -48 |
| Live PlugIR minimum-KL | 7 | What color is the fence surrounding the giraffe in the open field of the zoo? | white | a tan giraffe stands in an open field within a zoo, surrounded by a white fence | 92 | 4 |
| Live PlugIR minimum-KL | 8 | What color is the fence surrounding the giraffe in the open field of the zoo? | white | A light-brown giraffe stands in an open field inside a zoo, bordered by a white fence. | 85 | 7 |
| Live PlugIR minimum-KL | 9 | What color is the fence surrounding the giraffe in the open field of the zoo? | white | a tan giraffe stands in an open field within a zoo, surrounded by a white fence | 92 | -7 |
| Live PlugIR minimum-KL | 10 | What animal is standing with the giraffe in the open field of the zoo? | a zebra | a tan giraffe stands in an open field within a zoo, surrounded by a white fence, with a zebra standing nearby | 155 | -63 |

The raw-query diagnostic reuses live PlugIR’s generated Q&A and searches their concatenation. It is not an additional questioner or an independently generated dialogue. Its queries/ranks are in the common report and CSV files.

## 4. Representative captions, proposals and role distinctions

IDs below are zero-based manifest caption-evidence IDs. The multi-agent code maps a representative caption string to the highest-ranked matching image in the current window; duplicate caption strings do not establish the original clustering representative’s exact image index. These are public candidate possibilities, not confirmed target facts.

### 000000364869.jpg — first minimum-KL round

| Evidence ID | Representative candidate caption |
|---|---|
| 28475 | a birthday cake on a table with candles on it |
| 44888 | a woman slicing a cake at a party |
| 21232 | bride and groom slicing a cake together |
| 17567 | a young girl is looking at her own cake |
| 25887 | a bride is wearing a dress and a wedding cake |
| 18211 | a group of boys with tennis rackets posing for a picture |
| 32174 | a man in military clothing cuts a cake |
| 10025 | five men at a table sitting together |
| 46619 | group of people posing together for a picture |
| 39505 | a woman cuts a chocolate cake on a picnic table |

| Role | Proposal | Evidence IDs | Filter | Reason | Selected |
|---|---|---|---|---|---|
| full | What type of cake is present? | 28475,39505 | unknown | question not answered in dialogue or initial caption |  |
| full | Is the cake decorated with candles? | 28475 | unknown | no confirmation of candles in dialogue |  |
| full | Who is posing with the cake? | 28475,44888,21232,25887,32174,46619 | already_known | initial caption states group posing in front of cake |  |
| full | Is the cake being cut or sliced? | 44888,21232,32174,39505 | unsupported | no evidence of cutting in initial or dialogue |  |
| full | Is the setting indoors or outdoors? | 28475,39505 | unknown | setting not specified in dialogue or caption |  |
| attributes | What type of cake is it? | 28475,39505 | unknown | type of cake not confirmed | yes |
| attributes | Is the cake chocolate? | 39505 | unknown | chocolate not mentioned in dialogue |  |
| attributes | Are there candles on the cake? | 28475 | unknown | candles not confirmed in dialogue |  |
| attributes | Who is posing in front of the cake? | 28475,44888,21232,25887,32174,46619 | already_known | initial caption states group posing in front of cake |  |
| attributes | Is the cake being sliced? | 44888,21232,32174,39505 | unsupported | no evidence of slicing in dialogue or initial caption |  |
| relations | What is the cake being used for? | 28475,44888,21232,17567,25887,32174,39505 | unknown | purpose of cake not specified |  |
| relations | Are the people posing in front of the cake or cutting it? | 28475,44888,21232,17567,25887,32174,39505 | already_known | people are posing, not cutting, per initial caption |  |
| relations | Is the cake on a table or outdoors? | 28475,39505 | unknown | location not confirmed in dialogue or caption |  |
| relations | Is the cake decorated with candles? | 28475 | duplicate | Question already proposed this round. |  |
| relations | Is the cake chocolate? | 39505 | duplicate | Question already proposed this round. |  |

### 000000562109.jpg — first minimum-KL round

| Evidence ID | Representative candidate caption |
|---|---|
| 26575 | a giraffe standing in a shady area with green leaves |
| 43487 | two giraffes graze on grass next to a tree |
| 9158 | a giraffe standing next to a tree in a zoo |
| 18601 | three giraffes walking across the open plains |
| 48324 | giraffe in a zoo with rocks on a dirt path |
| 17090 | two giraffes walking through a small trail of a dry desert |
| 22645 | two giraffes in a field of tall grass |
| 42751 | a large elephant standing in a tall grassy plain |
| 45162 | a giraffe in an enclosure with a large rock |
| 19606 | a giraffe standing in front of a wooden bridge |

| Role | Proposal | Evidence IDs | Filter | Reason | Selected |
|---|---|---|---|---|---|
| full | How many giraffes are present? | 43487,17090,22645,45162,19606 | unknown | No answer to count of giraffes confirmed |  |
| full | Is the scene set in a natural open field or a man-made enclosure? | 26575,9158,48324,45162,19606 | already_known | Original says open field, not enclosure |  |
| full | What type of environment surrounds the giraffe? | 26575,43487,9158,22645,48324 | unknown | Environment not confirmed in dialogue | yes |
| full | Is there a tree, rock, or bridge in the scene? | 43487,9158,48324,45162,19606 | unknown | No confirmation of tree, rock, or bridge |  |
| full | Are the giraffes grazing or standing still? | 43487,22645,17090,18601 | unknown | Grazing or standing not addressed in dialogue |  |
| attributes | What type of grass is in the field? | 26575,22645 | unknown | Grass type not confirmed |  |
| attributes | Are there any trees present? | 43487,9158,19606 | unknown | Presence of trees not confirmed |  |
| attributes | Is the giraffe standing alone or with others? | 26575,43487,18601,22645,17090 | unknown | Alone or with others not confirmed |  |
| attributes | Is the scene set in a zoo or an open field? | 9158,48324,45162 | already_known | Original specifies open field, not zoo |  |
| attributes | What is the background landscape like? | 26575,43487,18601,22645,17090,45162 | unknown | Background landscape not confirmed |  |
| relations | What is adjacent to the giraffe? | 26575,43487,9158,18601,48324,17090,22645,45162,19606 | unknown | Adjacent objects not confirmed |  |
| relations | Are there other animals present? | 43487,17090,22645,42751 | unknown | Other animals not confirmed |  |
| relations | Is the field surrounded by structures? | 9158,45162,19606 | unknown | Surrounding structures not confirmed |  |
| relations | Is the grass tall or short? | 22645,43487,18601 | unknown | Grass height not confirmed |  |
| relations | Is the setting natural or enclosed? | 26575,43487,9158,18601,48324,17090,22645,45162,19606 | already_known | Setting is natural open field, not enclosed |  |

## 5. Selection, answer quality, repetition and rewrite faithfulness

### What the representative captions actually exposed

Cake's first representatives expose **cake occasion/type, candles, cutting versus posing, table/setting, and group composition**, but none explicitly describes cake colour. The generalist and attributes roles both asked about cake type; all three roles proposed candle-related questions, and relationships repeated the attributes role's chocolate question. The roles therefore supplied some distinct angles, but their question sets were not cleanly separated by speciality. “Birthday” and “chocolate” are also compatible categories, rather than mutually exclusive answers to a precise attribute question.

Giraffe representatives expose **one/two/three giraffes, grazing/walking/standing, tall grass, trees, rocks, bridge and enclosure**. Generalist proposed counts/environment, attributes proposed grass/trees/grouping, and relationships proposed adjacency/other animals/structures. This is useful candidate evidence, but many proposals overlap or conflate compatible categories. The filter marked “zoo or open field” / “natural or enclosed” as already known, although an open field can lie within a zoo or enclosure. It also retained “grazing or standing” despite standing being in the initial caption. Saved filter judgments should not be treated as correct labels.

The most useful observed specialist distinction was **colour**, even though the representative captions did not expose explicit colour alternatives:

| Case | Proposal and cited examples | Filter/selection outcome |
|---|---|---|
| Cake minimum-KL, r2 | Attributes: “What colour is the cake?” IDs 22363 (“two people are standing near a cake”), 25887 (“a bride is wearing a dress and a wedding cake”), 30788 (school-uniform group), 17567 (girl looking at cake) | Rejected: “Color not mentioned in dialogue or captions.” The same normalized question stayed blocked in r3–10. |
| Cake critique, r2 | Generalist/attributes/relationships proposed colour variants citing 19755 (birthday cake/blowing candles), 35524 (cake/candle/plates), 36949 (older woman/40th birthday cake) | Generalist “color” and attributes “colour” were both eligible; the relationships exact duplicate was removed. These captions also do not specify colour. Critique chose candle count instead. |
| Cake critique, r5 | Attributes: “What color is the cake?” IDs 35524 and 46493 (“cake and cupcakes set up for a birthday”) | Eligible; critique selected it. BLIP-2 answered “green”; rank improved **25 → 1**. Green is visibly supported by the target JPEG. |

This is evidence of a useful attribute question and inconsistent filtering, not evidence that the captions displayed a green-versus-other-colours contrast. Caption silence left an uncertainty worth probing. The minimum-KL trajectory lost that opportunity after one rejection was carried forward as memory; critique's later success therefore combines selector choice, different history/evidence, and different filtering outcomes.

### Critique versus minimum-KL selection

The first-round 500-image windows, representative-caption lists and eligible pools were **exactly equal** across the multi-agent conditions for each target. Only this shared context supports a direct selector comparison; later histories diverge.

| Target | Minimum-KL choice / score | Critique choice / its KL position | First resulting ranks |
|---|---|---|---|
| Cake | “What type of cake is it?” / 0.0000849261 | “Is the cake decorated with candles?” / 5th of 8, KL 0.0002398705 | Initial 9; minimum-KL 21; critique 15 |
| Giraffe | “What type of environment surrounds the giraffe?” / 0.0000582172 | “Is there a tree, rock, or bridge in the scene?” / 12th of 12, KL 0.0013172373 | Initial 89; minimum-KL 85; critique 81 |

Critique selected a higher-KL eligible question, as intended; it did not synthesize one. Its cake rationale used the candle-bearing caption 28475, but described other candidates as including or excluding candles even though silence does not establish absence. Its giraffe choice bundled three scene elements into one yes/no question, making the answer less diagnostic about which distinction was resolved.

At the common five-interaction horizon, critique reached cake/giraffe **1/84**, versus minimum-KL **18/103** and live PlugIR **21/48**. Critique had the best pooled prefix BRI (2.916032) and Recall@1 (50%), but live had the better giraffe rank. At 11 states, minimum-KL had better final ranks than live on both images, while live had better pooled BRI (3.092526 versus 3.257655), reflecting its earlier giraffe rank 48 versus minimum-KL's best 71. These metrics describe different aspects of these two trajectories.

Cake critique then failed before completing r6: its only eligible question was slot 0, “Where is the cake located relative to the people?” The response returned slot 4 and justified the already-asked candle-count question. Validation stopped before BLIP-2 answering or rewriting. The failed round has no selected question, answer, rewrite or rank; it is not a no-eligible-question round. The cached response, eligible payload and traceback remain preserved.

### Contributions and repeated questions

Retained selected owners were **generalist 12, attributes 10, relationships 4** across the 26 multi-agent Q&As. All four relationships-owned selections occurred in cake minimum-KL. Relationships also co-proposed the candle question retained under generalist in cake critique r1; role-order deduplication means retained ownership is not proof of exclusive contribution. Some selected relationship questions were proposed under attributes/generalist, so the role labels do not establish speciality adherence.

| Multi-agent history | Proposal records in completed rounds | Within-round exact duplicate extras | Distinct normalized questions repeated across rounds | Rounds with no selected question |
|---|---:|---:|---:|---|
| Cake minimum-KL | 95 | 13 | 17 | 7–10 |
| Giraffe minimum-KL | 112 | 2 | 19 | 9–10 |
| Cake critique, completed r1–5 | 63 | 7 | 9 | None; r6 failed separately |
| Giraffe critique | 102 | 10 | 14 | 6, 9, 10 |

Exact keys lowercase and remove punctuation/spacing. Duplicate extras count additional occurrences of a key in a round regardless of its eventual gate; they are not the same as the filter's `duplicate` status, because already-asked/rejected gates can take precedence. These counts exclude the failed r6 proposal audit and count logical trajectories, including reused initial responses. All four multi-agent histories had **zero exact repeated selected questions**.

Semantic overlap remains visible: cake type (“What type of cake is present?” / “What type of cake is it?”), candles (“Are there candles…?” / “Is the cake decorated with candles?”), and `color` versus `colour` survived exact normalization as separate questions. In critique r2, the latter two spelling variants both received eligible slots. Semantic overlap was reviewed qualitatively; no exhaustive semantic-similarity score was logged or invented.

Live PlugIR repeated the exact cake clothing question at **r7/r10**, and the exact giraffe fence-colour question at **r6–9**, receiving “white” four times. Its five final alternatives collapsed to one unique normalized question in **six of twenty rounds**: cake r3/r8 and giraffe r3/r4/r6/r10. Clothing questions at cake r3/r5/r7/r10 and coat-colour questions at giraffe r3/r4 also overlap semantically. Retrying an unanswered clothing/coat question differs from repeating fence colour after “white” was already supplied; neither produced new information in the repeated-answer examples.

The saved filter traces explain a relevant upstream behavior. Later fence filter responses were often truncated fragments without `uncertain`, so the original string gate treated those attempts as context-answerable/rejected. Yet the final candidate pools still included the question. The unchanged upstream loop appends a candidate after up to three attempts, and its generic fallback tests cumulative `len(ques_prior) == 3`, rather than a per-candidate exhausted-attempt condition. This can leave an answered question in a later pool. The review records this original behavior; it was not patched into a different baseline.

### Answer quality and visual checks

All 46 selected questions received nonempty BLIP-2 strings. Nonempty does not mean responsive. Clear nonresponses or unresolved alternatives include cake minimum-KL r2 (posing repeated), r6 (“are standing or sitting”); giraffe minimum-KL r3 (“a grass”) and r8 (“a giraffe in a zoo” to tree placement); giraffe critique r7 (“zebra” to exact location); live cake r1/r3/r5–10 (group/cake description repeated); and live giraffe r1/r3/r5 (caption-style restatements). “A few” candles is a vague count, rather than an exact count. Nine completed multi-agent rounds had no eligible question; these were skipped interactions, not unanswered BLIP-2 calls.

Manual inspection of the two preserved JPEGs is limited visual corroboration, not a human-annotated answer dataset. Cake is clearly green, supporting the r5 colour answer. The giraffe scene visibly contains trees and wire fencing with reddish/brown posts; a blanket “no tree” reading of critique's combined question is inconsistent with the image, and the white-fence answer is not visually supported. I cannot corroborate a zebra from the visible background animals at this resolution. Candle presence/count is not confidently established from the cake JPEG alone. Image-answer errors must be distinguished from rewrites that faithfully propagate those answers.

### Rewrite faithfulness to the actual record

| History / round | Saved rewrite issue | Evidence-based interpretation |
|---|---|---|
| Cake minimum-KL r3 | Adds cake “on a table” after “Where is the group…?” → “a table” | Plausible attachment, but the answer does not clearly establish cake-on-table rather than group-at-table. |
| Cake minimum-KL r6 | Replaces confirmed standing with “standing or sitting” | An echo of the alternatives weakens the earlier standing fact; it resolves nothing. |
| Cake critique r4–5 | “a few other decorations” | “A few” answered candle count; other decorations were confirmed only by yes. The quantifier migrated to a different object. Green at r5 is an independently supported addition. |
| Giraffe minimum-KL r1–4 | Drops “open field” for grassy environment | Omission of an original constraint; later r5 restores it. |
| Giraffe minimum-KL r8 | Adds trees “visible in the background” | The placement question was not answered; “distant” from r7 does not directly confirm foreground/background. Mark as an unconfirmed inference. |
| Giraffe critique r7 | Adds “the zebra is in the field and within the pen” | The answer “zebra” did not establish the zebra's location; earlier “a zebra in a pen” established pen, not the same field. R8 subsequently removes this extra field assertion. |
| Live cake r5–8 | Adds varied/colourful clothing, vibrant colours, festive cake elements and a lively celebratory setting | The corresponding answers repeated the original group/cake description; no clothing colours or decorations were supplied. |
| Live cake r9–10 | “no other prominent objects”; “no specific decorations or additional items visible” | An unanswered question does not confirm absence. These negative scene claims are unsupported. |
| Live giraffe r1/r3/r5 | Wide/spacious field and “fenced area” expansions | “Open” does not explicitly specify size; “a zoo” does not by itself confirm a physical fence. Some expansions arise in paraphrase fallback, not a new answer. |

These are textual record checks, not claims that every added phrase caused the observed rank change. Faithfulness and image truth are separate: the white fence and zebra can be propagated from BLIP-2 while still being questionable image facts. Conversely, unsupported negatives can be unfaithful even if an answerer returned a nonempty sentence. The multi-agent `coverage_issue` self-report is not an independent fidelity audit.

## 6. Raw evidence and reproducibility

VM checkout: `/home/lrc87/IR-experiments-pilot`. External evidence: `/home/lrc87/pilot-evidence-20261120/`, including its `continuation/` directory. The original checkpoint paths remain intact. All local links below are relative to this review's repository folder.

| Evidence | Local artifact |
|---|---|
| Complete preserved bundle | [pilot-delivery-20261120.tar.gz](outputs/pilot_20261120/pilot-delivery-20261120.tar.gz) |
| 11-state completed-method report | [report.json](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_completed_methods/report.json) |
| 11-state final metrics | [final_metrics.csv](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_completed_methods/final_metrics.csv) |
| 11-state per-round Recall/Hits | [per_round.csv](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_completed_methods/per_round.csv) |
| 11-state per-target queries/ranks | [per_example.csv](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_completed_methods/per_example.csv) |
| Matched six-state report | [prefix report.json](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_common_prefix/report.json) |
| Matched six-state metrics | [prefix per_round.csv](outputs/pilot_20261120/raw/IR-experiments-pilot/runs/experiments/20261120T184123036436Z_pilot_common_prefix/per_round.csv) |
| Cake / giraffe minimum-KL dialogue and audits | [cake](outputs/pilot_20261120/dialogues/ma_min_kl_cake.json), [giraffe](outputs/pilot_20261120/dialogues/ma_min_kl_giraffe.json) |
| Cake / giraffe critique dialogue and audits | [cake, including failure](outputs/pilot_20261120/dialogues/ma_critique_cake.json), [giraffe](outputs/pilot_20261120/dialogues/ma_critique_giraffe.json) |
| Cake / giraffe live PlugIR dialogue and call traces | [cake](outputs/pilot_20261120/dialogues/plugir_cake.json), [giraffe](outputs/pilot_20261120/dialogues/plugir_giraffe.json) |
| Checkpoint-alias hashes/original paths | [dialogue_aliases.json](outputs/pilot_20261120/dialogue_aliases.json) |
| Shared prompt snapshot | [prompt_pack.json](outputs/pilot_20261120/raw/pilot-evidence-20261120/continuation/prepared/giraffe_critique/prompt_pack.json) |
| Effective settings, packages, models, source/cache/JPEG hashes | [preflight.json](outputs/pilot_20261120/raw/pilot-evidence-20261120/preflight.json) |
| Environment/source post-run check | [post-run.txt](outputs/pilot_20261120/raw/pilot-evidence-20261120/post-run.txt), [installed packages](outputs/pilot_20261120/raw/pilot-evidence-20261120/installed-packages.txt), [VM description](outputs/pilot_20261120/raw/pilot-vm-description.json) |
| Original / continuation stdout and errors | [generation.log](outputs/pilot_20261120/raw/pilot-evidence-20261120/generation.log), [continuation log](outputs/pilot_20261120/raw/pilot-evidence-20261120/continuation/generation.log) |
| Timings and resource samples | [resources.json](outputs/pilot_20261120/analysis/resources.json), [original GPU samples](outputs/pilot_20261120/raw/pilot-evidence-20261120/gpu-samples.csv), [continuation GPU samples](outputs/pilot_20261120/raw/pilot-evidence-20261120/continuation/gpu-samples.csv) |
| All round-level Q/A/query/ranks | [rounds.csv](outputs/pilot_20261120/analysis/rounds.csv) |
| Candidate captions/IDs, window IDs, proposals, filters and choices | [proposal_evidence.json](outputs/pilot_20261120/analysis/proposal_evidence.json) |
| Role/repetition statistics | [audit_summary.json](outputs/pilot_20261120/analysis/audit_summary.json), [verification.json](outputs/pilot_20261120/analysis/verification.json) |
| Derived live filter-call trace | [plugir_filter_trace.json](outputs/pilot_20261120/analysis/plugir_filter_trace.json) |
| Fresh completion counts/tokens/stages | [call_inventory.json](outputs/pilot_20261120/analysis/call_inventory.json) |
| Preserved target JPEGs | [cake](outputs/pilot_20261120/raw/IR-experiments-pilot/data/images/unlabeled2017/000000364869.jpg), [giraffe](outputs/pilot_20261120/raw/IR-experiments-pilot/data/images/unlabeled2017/000000562109.jpg) |
| Final VM/GPU status proof | [run-status.png](outputs/pilot_20261120/run-status.png) |

The six short dialogue links are **byte-identical copies** of the raw checkpoints, supplied to avoid long Windows hash-based paths; their hashes and original locations are in the alias manifest. Raw completion files remain under `raw/IR-experiments-pilot/runs/three_agents/shared_agents_simple_v1/{qwen_calls,blip2_calls}/` and `raw/IR-experiments-pilot/runs/live_upstream/aaec02ad94051834aeaf/{qwen_calls,blip2_calls}/`. The full and prefix report directories contain both the original runner snapshot and the actual continuation-driver snapshot.

Archive SHA-256: `0acf872597784e83bca14d37f587facfd8bfdb95fb895aff94231de7b2d0a802`. Its inner VM archive SHA-256 is `425e68bdecaa2840dc0de74d423bd52e99dd621b563a7fb2113aeb63fb01ca93`. Both were verified before safe extraction of 693 original files. Image-cache SHA-256 is `8afcfb6393d71261c18938acd9a4db0f9e74439da8852594b8c983846a12a34f`; ordered-manifest SHA-256 `8a32638192419131513f2e5a6e5159d191de6789678fb3b1faa18e1ab46b44af`; caption catalogue SHA-256 `df5fbf1b5ee0856b89b3a96a9094f245cc5db6eb17a1d3fff0604c06109dd5cd`. The existing local VS Code cache also matched the VM cache checksum.

To reproduce, checkout the pinned commit, restore the same cache/metadata and target JPEGs, install the pinned environment, and run the validation and observation helpers:

```bash
git checkout 0855fb8b55236098565c466567d96419e551825d
~/.venvs/agentir-pilot/bin/python -u pilot_preflight.py
~/.venvs/agentir-pilot/bin/python -u pilot_observe.py
# If the same critique fails, preserve it and continue independent cases:
~/.venvs/agentir-pilot/bin/python -u pilot_continue.py
```

Use [PILOT_COMMANDS.md](PILOT_COMMANDS.md) for full install, GPU sampling and process-resource commands. Use fresh experiment directories for a fresh rerun; running against these preserved directories reuses checkpoints/calls. Model decoding can vary across hardware/software environments even with pinned seeds. The preflight loads models but generates no Q&A; the reporting scripts make no new Qwen/BLIP-2 calls.

Environment/execution accommodations were isolated Python/CUDA-package installation, authenticated cache transfer, external resource observers and failure-isolating continuation. Anonymous Drive download and an early copy attempt failed during setup; no corpus was re-encoded. A read-only status display was corrected for an initial checkpoint without `status`. Browser download handling required an archive-only authenticated text transfer and checksum reconstruction; its temporary server was stopped. Windows extraction used long-path support. None changed prompts, candidate selection, filters, answers, rewrites or ranking algorithms.

## 7. Limitations and next experiment

Only two deliberately selected images are tested. Model answers are simulated-user responses, not human ground truth. Multiple internal proposal/filter/selector calls do not imply more interactions with the answerer. Exact normalized duplicate checks exist; semantic repetitions require qualitative review.

Live PlugIR saves candidate-caption prompt strings, final candidate questions and model request/response traces, but does not explicitly save the representative manifest IDs, every window member, per-question KL scores or a structured filtering-decision audit. Mark these unavailable unless directly supported by the saved traces. Multi-agent rounds retain these structured audits. No additional logging or generation calls were added to fill these scientific gaps.

The derived PlugIR filter trace exposes each saved response and its actual `uncertain` substring gate; it does not invent a final acceptance-decision log. Live candidate-window size is derivable from the frozen schedule, while actual membership IDs and exact representative IDs remain unavailable. Multi-agent IDs are caption-evidence mappings, as noted above. Failed cake r6 retains candidate/proposal/filter/critique payloads in its partial audit, but no completed-round audit or target rank. Per-method BLIP retrieval forward-call counts and per-answer BLIP-2 latency were not saved. Semantic-repetition analysis is qualitative, not exhaustive. No full-horizon pooled critique metric exists.

Other limits are unequal answer prompts, different Qwen input/output budgets, different filtering/retry/memory rules, model parking/reloading and cached calls. The synthetic prompt examples themselves mention cake/giraffe-style scenes, and these are development pilot targets. A broad performance claim would require a larger held-out cohort and repeated seeds. Lower final rank and lower BRI did not agree on a winner here.

For the next experiment:

1. **Keep this baseline frozen and use a new run identity/folder.** Multi-agent's fixed `shared_agents_simple_v1` checkpoint folder does not validate a source/settings fingerprint on resume. Reusing it after editing prompts could silently reuse old histories; retain these checkpoints and isolate new conditions.
2. **Test robustness separately:** constrain critique to eligible slot IDs, preserve fail-closed behavior, and isolate target failures in orchestration. Treat any one-eligible-question shortcut or changed retry policy as a newly labeled condition. Add explicit live representative/window IDs, KL values and structured filter decisions to future audits.
3. **Then test the proposed evidence-analysis stage as one controlled change:** explicitly separate candidate agreement, compatible differences, genuine alternatives and caption silence. Require citations for positive evidence; do not convert missing detail into absence. The cake-colour case argues for distinguishing an unsupported object premise from an unresolved attribute of an already-confirmed object. First select from existing questions; test synthesis in a separate later condition.
4. **Audit the answer/rewrite bottlenecks independently.** Compare matched BLIP-2 input prompts as an explicit ablation, use a small human-checked answer set, and test preservation of counts, negation, subject/object attachment and uncertainty. Do not bundle those changes into the evidence-analysis comparison.
5. **Expand the evaluation with fixed, preselected targets and seeds**, the same initial captions/50k pool/11-state request, and transparent call/token/runtime budgets. Report missing histories and questionless rounds alongside Recall/Hits/BRI. Two images, one green-cake success and one invalid critique response are useful debugging observations, not a reliable estimate of general retrieval quality.

No evidence-analysis stage, question synthesis, changed prompt, changed filter, or revised selector was implemented in this run.

