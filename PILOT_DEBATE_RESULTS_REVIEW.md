# Cake/giraffe critique–debate–PlugIR pilot review

**Run: 22–23 November 2026. Three of six histories completed all 11 states.** Critique completed cake but failed on giraffe’s tenth interaction. Debate stopped after three answered cake interactions and before its first answered giraffe interaction. Both live PlugIR histories completed. The failures, completed prefixes, raw model responses, and checkpoints are preserved. **This version is not ready for a 32-target run. No 32-target run was launched.**

The strongest finding is operational and qualitative: discussion recognized candidate-caption differences, but cake debate repeatedly asked the target-only answerer to identify or inspect retrieval candidates. Structured-output validation then stopped both debate histories. These two development images do not establish a general retrieval winner.

## 1. Run identity, setup, and reproduction

The GCP machine was `dl-gpu` in project `astute-solstice-512700-u9`, zone `us-west4-a`: standard/on-demand `g2-standard-8`, one NVIDIA L4 (23,034 MiB visible), 150 GB boot disk, approximately 31 GiB host RAM. Driver: `580.178.04`; PyTorch CUDA build: **12.8** (the driver’s supported CUDA banner is 13.0). Existing IAP SSH access was tested before setup. No new firewall rule was necessary.

The exact local checkout was **`82ed6186d4c5b98d32dff4e949e32d45804b023f` plus uncommitted changes**, including the new `debate.py` and runner updates. A Git pull alone would not reproduce it. The local working tree had changes to `EXPERIMENTS.md`, `README.md`, `run_evaluation.py`, `scripts/run_plugir_live.py`, `multi_agent.py`, `setup.py`, and three existing test files, plus untracked `debate.py` and `tests/test_debate.py`.

It was transferred into a fresh checkout at `/home/lrc87/pilot_debate_20261122/IR-experiments`. All **63 transferred source/input files** matched their hashes before generation and again after it. No scientific source, prompt, algorithm, filter, token ceiling, or answer/rewrite protocol was changed during the pilot. The earlier `outputs/pilot_20261120` evidence was retained and was not reused for generation or merged into these histories.

Fresh generation ran from **23:33:03 EST on November 22 to 00:17:31 EST on November 23** (04:33:03–05:17:31 UTC), including the pause to inspect the first failure and launch independent continuations. Preparation/download/test time is additional. Method–target timings below come from the function-level event log, not this whole dispatch interval.

The requested runner settings were verified before generation:

```python
RUN_LIVE_PLUGIR = True
RUN_MULTI_AGENT = True
RUN_MULTI_AGENT_DEBATE = True
USE_PILOT_TARGETS = True
N_STATES = 11
INCLUDE_SAVED_REFERENCES = False
DEVICE = "cuda"
LIVE_DEVICE = "cuda"
SEARCH = "rewrite"
SELECTION = "plugir_kl"
MAX_INPUT_TOKENS = 8192
# src/agentir/multagent_experiment/setup.py
AGENT_MODES = (True,)
```

`N_TARGETS=32` and `START_TARGET=0` remained in the source but were bypassed by pilot selection. The same two selected pairs, exact initial captions, 50,000-image vector pool, candidate catalogue, model revisions, and requested state count were used throughout. The initial ranks were **cake 9; giraffe 89** in all three methods. Initial representative-caption lists were also verified identical between critique and debate for both targets. Subsequent retrieved windows follow each method’s own rewritten query; they are not identical after histories diverge. PlugIR uses its original clustering/RNG path, so identical representative samples across PlugIR and the multi-agent methods were not assumed.

| Target | Initial caption | JPEG SHA-256 |
|---|---|---|
| unlabeled2017/000000364869.jpg | a group of people posing in front of a cake | a7425cfc97a00bf87a30fbb26f1b41cd4b76114f401aba2cb2cb2158102b177f |
| unlabeled2017/000000562109.jpg | a giraffe is standing in an open field | 8d445abe263c554fefd86068fe619780dff5db017ce514208ef862bf04c8e1d9 |

The vector cache was validated as `(50000, 256)` normalized float32 vectors with exact manifest order. Cache SHA-256: `8afcfb6393d71261c18938acd9a4db0f9e74439da8852594b8c983846a12a34f`; manifest SHA-256: `8a32638192419131513f2e5a6e5159d191de6789678fb3b1faa18e1ab46b44af`. Candidate captions covered exactly all 50,000 filenames. Both JPEGs were downloaded through the existing resolver, verified, and matched the earlier local image bytes.

The fresh VM had no project or generation caches. Necessary environment work was an isolated Python environment, dependency installation, model downloads, and input transfer—not scientific fixes. System Python 3.10 was left alone. Python **3.11.18**, `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `transformers 4.51.3`, `tokenizers 0.21.4`, `huggingface-hub 0.30.2`, Pillow `11.3.0`, NumPy `2.2.6`, Accelerate `1.6.0`, SentencePiece `0.2.1`, fast-pytorch-kmeans `0.2.2`, and psutil `7.1.0` were verified. `pip check` passed; **161 tests passed in 22.18 s**. Both generation models loaded on the GPU during preflight without generating extra completions.

| Purpose | Pinned model | Revision |
|---|---|---|
| Retrieval | Salesforce/blip-itm-large-coco | 19502f1e215844f7e48bd48473f86932486d3441 |
| Questioning/rewrite | Qwen/Qwen3-4B-Instruct-2507 | cdbee75f17c01a7cc42f958dc650907174af0554 |
| Target-image answers | Salesforce/blip2-flan-t5-xl | 0eb0d3b46c14c1f8c7680bca2693baafdb90bb28 |

Qwen used BF16, BLIP-2 FP16, and retrieval BLIP float32. BLIP-2 alone received target pixels, with the shared current-description-plus-question input. Qwen rewrites used the original PlugIR reconstruction and paraphrase fallback. Question agents/coordinator received the initial caption, actual generated Q&A, representative captions/IDs, and avoidance history; measured ranks and target identity were bookkeeping outside their prompts. No saved human answers or future dialogue were supplied for generation.

The current runner initially executed critique on both targets in its normal order. Its exception on giraffe prevented the remaining preparers from running. After preserving that error, the same runner was invoked in four separate processes, enabling one remaining method and selecting one of the original pilot pairs per invocation. These **dispatch-only overrides** isolated failures; they did not change a method’s scientific settings. The frozen runner snapshot still contains all three enabled flags; the launcher and event log document the effective continuation overrides.

The removed older one-off pilot helpers were not used. Newly written verification, observation, dispatch, and export scripts live outside the GCP checkout and are preserved with this run. The observation wrapper only measured function duration, allocator peaks, and existing transport records; it made no extra model calls.

Reproducible commands (in a fresh destination; source archive is required for the uncommitted files):

```bash
gcloud compute ssh dl-gpu --zone=us-west4-a \
  --project=astute-solstice-512700-u9 --tunnel-through-iap
# Restore debate_pilot_source.tar.gz into a fresh checkout at the recorded commit.
# The preserved setup script installs the exact project pins:
python3 -m pip install --user uv==0.13.7
~/.local/bin/uv venv --python 3.11 --seed ~/.venvs/agentir-debate
source ~/.venvs/agentir-debate/bin/activate
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[live,test]'
python -m pip check
python -m pytest -q
cd /home/lrc87/pilot_debate_20261122/IR-experiments
python -u ../verify_session.py
python -u ../execute_session.py  # calls current run_evaluation.main(); primary attempt
# Actual independent continuations after the preserved primary failure:
python -u ../execute_session.py agents_debate 0
python -u ../execute_session.py agents_debate 1
python -u ../execute_session.py live_plugir_qwen 0
python -u ../execute_session.py live_plugir_qwen 1
```

All six histories started with **no existing checkpoint**. There were **zero Qwen or BLIP-2 replays**. Model-weight caching after preflight is distinct from generation/checkpoint reuse. Interrupted histories were not resumed or repaired. Source snapshot archive SHA-256: `8871188e6616dc3610add8261df0a6f397cac371f62da7ac3dcb186e54d1148f`. Raw delivery archive SHA-256: `b85698e39334f71abd218e3ca089168f1d6864e50876751ee0416e2e49e80756`; all 769 files listed in its artifact manifest passed local readback verification. Historical files included in the checkout are excluded from the current-run analysis, which reads only its top-level `IR-experiments/runs`.

## 2. Completion, ranks, metrics, computation, and resources

Ranks are one-based, lower is better. **Last observed rank is not an 11-state final rank for an interrupted history.** BRI below uses only each row’s observed span; compare BRI across methods only at the matched spans shown afterward. A state includes the initial caption; all 42 completed interactions had an answer, and there were no completed skipped interactions.

| Method | Target | Status | States | 11-state final | Last observed | Best | Observed-span BRI | Wall min | Qwen / BLIP-2 calls |
|---|---|---|---|---|---|---|---|---|---|
| Critique | Cake | complete | 11/11 | 19 | 19 | 9 | 2.1972 | 9.61 | 61 / 10 |
| Critique | Giraffe | interrupted | 10/11 | N/A | 96 | 78 | 4.4520 | 9.16 | 59 / 9 |
| Debate | Cake | interrupted | 4/11 | N/A | 48 | 9 | 2.1972 | 6.60 | 37 / 3 |
| Debate | Giraffe | interrupted | 1/11 | N/A | 89 | 89 | N/A | 1.07 | 1 / 0 |
| Live PlugIR | Cake | complete | 11/11 | 13 | 13 | 4 | 1.9016 | 7.11 | 171 / 10 |
| Live PlugIR | Giraffe | complete | 11/11 | 82 | 82 | 29 | 3.5997 | 7.20 | 141 / 10 |

There are 48 observed states out of 66 planned, comprising six initial states and 42 answered interactions. The missing 18 interactions were not imputed. Both standard live reports were exported by the current runner. The aborted critique/debate preparation did not reach its standard report export; its metrics here are calculated from saved checkpoint ranks using the repository’s existing `metrics_from_ranks`, with the same checksum-verified upstream metric implementation. All 22 live report rows matched their checkpoint ranks and query strings exactly.

Matched comparisons:

| Matched scope | Method | BRI | End Recall@10 % | End Hits@10 % | End Recall@100 % | End Hits@100 % |
|---|---|---|---|---|---|---|
| Cake, states 0–3 | Critique | 2.1972 | 0.0 | 100.0 | 100.0 | 100.0 |
| Cake, states 0–3 | Debate | 2.1972 | 0.0 | 100.0 | 100.0 | 100.0 |
| Cake, states 0–3 | Live PlugIR | 2.1972 | 0.0 | 100.0 | 100.0 | 100.0 |
| Both targets, states 0–9 | Critique | 3.3246 | 0.0 | 50.0 | 100.0 | 100.0 |
| Both targets, states 0–9 | Live PlugIR | 2.7922 | 50.0 | 50.0 | 100.0 | 100.0 |

Across **both images and all three methods**, only state 0 is shared: Recall/Hits@1 = 0%, @10 = 50%, @100 = 100%; one-state BRI is undefined. Consequently, there is **no completed three-method longitudinal comparison on both images**. Cake alone supports a shared four-state comparison; all three have the same best-so-far BRI there despite different last ranks (39 critique, 48 debate, 23 live PlugIR).

Live PlugIR’s completed 11-state, two-target BRI is **2.750661**; final Recall@10 is 0%, Hits@10 50%, Recall/Hits@100 100%. Its state-9 comparison with critique uses BRI **2.792202 versus 3.324607** over the same two targets and ten states. That is an observation on these two development images, not evidence of general superiority. Do not compare the full 11-state BRI directly with a shorter interrupted span.

Per-round two-target Recall percentages are below. In every available two-target span, Hits@1 stays 0%, Hits@10 stays 50%, and Hits@100 stays 100%, because cake started at rank 9 and both images started inside 100. Recall@1 is 0% throughout. “—” is unavailable, not zero. Complete per-case and aggregate metric arrays are in `analysis/analysis.json`.

| State | Critique R@10 / R@100 % | Debate R@10 / R@100 % | Live PlugIR R@10 / R@100 % |
|---|---|---|---|
| 0 | 50 / 100 | 50 / 100 | 50 / 100 |
| 1 | 0 / 50 | — | 0 / 100 |
| 2 | 0 / 50 | — | 0 / 100 |
| 3 | 0 / 50 | — | 0 / 100 |
| 4 | 0 / 50 | — | 0 / 100 |
| 5 | 0 / 50 | — | 0 / 100 |
| 6 | 0 / 50 | — | 50 / 100 |
| 7 | 0 / 100 | — | 50 / 100 |
| 8 | 0 / 50 | — | 0 / 100 |
| 9 | 0 / 100 | — | 50 / 100 |
| 10 | — | — | 0 / 100 |

Actual transport costs, including completions that subsequently failed validation:

| Method | Target | Qwen input / output tokens | Qwen generation s | Peak allocated / reserved GiB |
|---|---|---|---|---|
| Critique | Cake | 71,523 / 6,614 | 361.09 | 9.653 / 10.055 |
| Critique | Giraffe | 70,848 / 7,949 | 434.43 | 9.657 / 10.055 |
| Debate | Cake | 44,961 / 3,932 | 216.88 | 9.771 / 10.197 |
| Debate | Giraffe | 685 / 166 | 9.23 | 9.438 / 9.471 |
| Live PlugIR | Cake | 78,807 / 3,157 | 179.88 | 9.539 / 10.061 |
| Live PlugIR | Giraffe | 67,401 / 2,639 | 146.95 | 9.550 / 10.059 |

Totals: **470 actual Qwen calls and 42 actual BLIP-2 answer calls**, with zero replays. The three answered debate rounds used exactly **12 Qwen calls each**; cake’s failed next opening added one call, and giraffe’s failed opening used one. Completed critique rounds used six calls normally, with one extra paraphrase in cake round 2 and giraffe round 4. There were no proposal regenerations, reviewer retries, coordinator revisions, or skipped completed rounds. Those bounded recovery paths therefore were not exercised by real generation in this run.

This is **not a matched-cost comparison**. Debate roughly doubled Qwen call count per answered interaction relative to critique, but original PlugIR itself used 171 cake and 141 giraffe Qwen calls because of its own generation/filter/fallback procedure. Calls also have very different token lengths; fewer calls did not guarantee less runtime. Qwen generation seconds exclude model loading and most transport/parking work. Wall time includes loading and answering; critique’s second target reused warm model objects, whereas the continuation histories ran in separate processes. These wall times are not a controlled speed benchmark.

One-second `nvidia-smi` sampling peaked at **10,692 MiB device memory used**; per-history allocator peaks are above. Allocator peaks exclude some driver allocations, while device sampling can miss subsecond peaks. No out-of-memory failure occurred. BLIP-2 token counts and isolated answer latency, retrieval-encoder call counts, and peak host RSS were not instrumented and are unavailable. Recorded BLIP-2 generation configurations and all model requests/responses remain in the raw cache.

## 3. Round-by-round questions, answers, rewrites, and ranks

State 0 is the starting caption. Positive Δ means improvement (`previous rank − current rank`). Questions, answers, and rewritten queries below reproduce the saved strings, including formatting prefixes and unsupported content; they have not been editorially corrected. Failed and unattempted states carry no invented query, answer, or rank.

### Cake — 000000364869.jpg

| Method | State | Question | Answer | Rewritten query | Rank | Δ |
|---|---|---|---|---|---|---|
| Critique | 0 | (initial caption) | — | a group of people posing in front of a cake | 9 | — |
| Critique | 1 | Is the cake decorated with candles? | a group of people posing in front of a cake | a group of people posing in front of a decorated cake with candles | 15 | -6 |
| Critique | 2 | Are any of the people blowing out candles on the cake? | a group of people posing in front of a decorated cake with candles | A group of people standing in front of a beautifully decorated cake with candles. | 18 | -3 |
| Critique | 3 | Is the cake a birthday cake? | yes | a group of people posing in front of a birthday cake decorated with candles, some of them blowing out the candles | 39 | -21 |
| Critique | 4 | Are any of the people interacting with the cake directly? | a group of people posing in front of a cake | a group of people posing in front of a birthday cake decorated with candles, with no one blowing out the candles or interacting directly with the cake | 22 | 17 |
| Critique | 5 | What is the relationship between the people and the cake? | a group of people posing in front of a cake | a group of people posing in front of a birthday cake decorated with candles, with no direct interaction, simply celebrating in the background | 36 | -14 |
| Critique | 6 | What is the cake's decoration? | a group of people posing in front of a cake | a group of people posing in front of a birthday cake decorated with candles, with no direct interaction, simply celebrating the occasion | 29 | 7 |
| Critique | 7 | Are any of the people looking at the cake directly? | a group of people posing in front of a cake | a group of people posing in front of a birthday cake decorated with candles, with no direct interaction, all looking at the cake directly | 27 | 2 |
| Critique | 8 | What action is the group of people performing? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake decorated with candles, with no direct interaction, all looking at the cake | 30 | -3 |
| Critique | 9 | Are any of the people standing near the cake? | a group of people standing in front of a cake | a group of people standing in front of a birthday cake decorated with candles, posing and looking at the cake, with no direct interaction | 26 | 4 |
| Critique | 10 | Is the cake on a table or on a surface? | a group of people standing in front of a birthday cake decorated with candles, posing | a group of people standing in front of a birthday cake decorated with candles, posing, with the cake on a surface, but no one interacting with it directly | 19 | 7 |
| Debate | 0 | (initial caption) | — | a group of people posing in front of a cake | 9 | — |
| Debate | 1 | Which candidate shows a group of people posing together with a cake directly in front of them? | a group of people posing together with a cake | a group of people posing together in front of a cake | 14 | -5 |
| Debate | 2 | Does candidate 46619 show a cake directly in front of the group during the pose? | yes | a group of people posing together with a cake directly in front of them | 9 | 5 |
| Debate | 3 | Does candidate 21232 show the bride and groom in a static, posed arrangement with the wedding cake directly in front of them, as opposed to simply slicing it together? | yes | a group of people posing together with a wedding cake directly in front of them in a static, posed arrangement | 48 | -39 |
| Debate | 4 | FAILED before selecting/answering next question | — | — | — | — |
| Debate | 5–10 | Not attempted after interruption | — | — | — | — |
| Live PlugIR | 0 | (initial caption) | — | a group of people posing in front of a cake | 9 | — |
| Live PlugIR | 1 | What type of cake are they posing in front of? | a cake | A crowd of individuals standing in front of a cake. | 16 | -7 |
| Live PlugIR | 2 | What is the occasion associated with the cake they are posing in front of? | a birthday | a group of people posing in front of a birthday cake | 21 | -5 |
| Live PlugIR | 3 | What is the gender composition of the group posing in front of the cake? | a group of people posing in front of a birthday cake | A crowd of individuals standing in front of a birthday cake. | 23 | -2 |
| Live PlugIR | 4 | What is the age range or demographic of the people in the group posing in front of the cake? | a crowd of individuals standing in front of a birthday cake | a group of people posing in front of a birthday cake | 21 | 2 |
| Live PlugIR | 5 | What are the genders of the individuals in the group posing in front of the cake? | a group of people posing in front of a birthday cake | A crowd of individuals standing in front of a birthday cake. | 23 | -2 |
| Live PlugIR | 6 | What is the age of the woman cutting the cake in the scene? | a young woman | A group of people posing in front of a birthday cake, including a young woman cutting the cake, with a diverse age range and mixed genders, standing in a crowd setting. | 8 | 15 |
| Live PlugIR | 7 | **Question:** What colors are the birthday cake and the clothing of the people posing in front of it? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, featuring a young woman cutting the cake, with vibrant colors on both the cake and the clothing of the individuals | 4 | 4 |
| Live PlugIR | 8 | **Question:** What colors are the birthday cake and the clothing of the people posing in front of it? | a group of people posing in front of a birthday cake | a group of people posing in front of a birthday cake, featuring a young woman cutting the cake, with vibrant colors on both the cake and the clothing of the individuals, including bright hues like red, yellow, and blue | 16 | -12 |
| Live PlugIR | 9 | What types of poses or expressions are the people making while posing in front of the birthday cake? | a group of people posing in front of a birthday cake | A group of people posing in front of a birthday cake, featuring a young woman cutting the cake, with vibrant colors on both the cake and their clothing, and expressions of celebration and joy. | 8 | 8 |
| Live PlugIR | 10 | **Question:** Which of the retrieval candidates best describes a group of people posing in front of a birthday cake, including the presence of children and a diverse age range? | a group of people posing in front of a birthday cake | a group of people of diverse ages, including children, posing joyfully in front of a birthday cake, with a young woman cutting the cake, featuring vibrant colors in both the cake and the clothing of the individuals | 13 | -5 |

### Giraffe — 000000562109.jpg

| Method | State | Question | Answer | Rewritten query | Rank | Δ |
|---|---|---|---|---|---|---|
| Critique | 0 | (initial caption) | — | a giraffe is standing in an open field | 89 | — |
| Critique | 1 | Is the field shaded or open? | open | a giraffe is standing in an open, sunlit field | 191 | -102 |
| Critique | 2 | Is there a tree nearby? | yes | a giraffe stands in an open field with a tree nearby | 143 | 48 |
| Critique | 3 | Is the giraffe standing alone or with other giraffes? | a giraffe stands in an open field with a tree nearby | a giraffe stands alone in an open field with a tree nearby | 151 | -8 |
| Critique | 4 | Are there other giraffes nearby? | no | A lone giraffe stands in an open field, near a tree. | 165 | -14 |
| Critique | 5 | Is the giraffe standing or moving? | standing | a giraffe stands alone in an open field with a tree nearby, standing still | 167 | -2 |
| Critique | 6 | Is the giraffe near a fence? | no | a giraffe stands alone in an open field with a tree nearby, not near a fence, in a sunny, unshaded area | 124 | 43 |
| Critique | 7 | Is the tree directly visible and in the foreground? | no | a giraffe stands alone in an open field with a tree nearby in the background, not near a fence, and not in the foreground | 78 | 46 |
| Critique | 8 | Are there rocks near the giraffe? | no | a giraffe stands alone in an open field with a tree nearby, not near a fence or rocks, the tree is not in the foreground, and the scene is unshaded | 102 | -24 |
| Critique | 9 | Is the field dry or green? | a giraffe stands alone in an open field with trees nearby | a giraffe stands alone in an open, green field with trees nearby, not near a fence or rocks, and not in the foreground of the tree | 96 | 6 |
| Critique | 10 | FAILED before selecting/answering next question | — | — | — | — |
| Debate | 0 | (initial caption) | — | a giraffe is standing in an open field | 89 | — |
| Debate | 1 | FAILED before selecting/answering next question | — | — | — | — |
| Debate | 2–10 | Not attempted after interruption | — | — | — | — |
| Live PlugIR | 0 | (initial caption) | — | a giraffe is standing in an open field | 89 | — |
| Live PlugIR | 1 | What is the exact location or setting of the open field where the giraffe is standing? | a zoo | a giraffe is standing in an open field at a zoo | 50 | 39 |
| Live PlugIR | 2 | What color is the giraffe's coat in the open field at the zoo? | a giraffe is standing in an open field at a zoo. | a giraffe standing in an open field at a zoo | 50 | 0 |
| Live PlugIR | 3 | What type of fence or enclosure is present near the giraffe in the open field at the zoo? | a giraffe standing in an open field at a zoo | a giraffe standing in an open field at a zoo, with a fence or enclosure nearby, coat color not specified | 73 | -23 |
| Live PlugIR | 4 | What type of vegetation or landscape features are visible in the open field where the giraffe is standing at the zoo? | a giraffe standing in an open field at a zoo | a giraffe standing in an open field at a zoo, with a fence enclosure nearby, surrounded by vegetation, coat color typical of a giraffe | 33 | 40 |
| Live PlugIR | 5 | What type of fence is surrounding the open field where the giraffe is standing at the zoo? | a fence | a giraffe standing in an open field at a zoo, surrounded by a fence, with typical grassy vegetation in the background | 29 | 4 |
| Live PlugIR | 6 | What type of fence is surrounding the open field where the giraffe is standing at the zoo? | a fence | A giraffe standing in an open field within a zoo, enclosed by a fence, with grassy plants in the background. | 33 | -4 |
| Live PlugIR | 7 | What color is the giraffe's coat in the open field at the zoo? | a tan | a giraffe standing in an open field at a zoo, surrounded by a fence, with a tan coat and typical zoo landscape features visible in the background | 68 | -35 |
| Live PlugIR | 8 | What type of path or surface is visible in the open field at the zoo where the giraffe is standing? | a grassy area | a giraffe standing in a grassy open field at a zoo, surrounded by a fence, with a tan coat and a grassy path nearby | 86 | -18 |
| Live PlugIR | 9 | What type of fence is the giraffe near in the open field at the zoo? | a wooden fence | a giraffe with a tan coat stands in a grassy open field at a zoo, near a wooden fence | 63 | 23 |
| Live PlugIR | 10 | **Question:** What type of fence is the giraffe near in the open field at the zoo? | a wooden fence | a giraffe standing in a grassy open field at a zoo, near a wooden fence, with a tan coat | 82 | -19 |

## 4. Representative evidence, roles, and full debate examples

The multi-agent candidate window was 500, 450, 400, …, 50 entries across the ten planned interactions, with ten representative captions per completed interaction. The audit preserves every window’s IDs and representative text. Representative IDs identify a matching catalogue caption in the window; when caption strings duplicate, the implementation chooses the highest-ranked matching entry. They should not be interpreted as guaranteed recovery of the upstream cluster representative’s exact image index. Caption counts are not cluster population proportions.

### Critique cake, interaction 1: useful distinctions with overlapping proposals

| Representative ID | Caption |
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

| Role | Proposal | Evidence IDs | Decision and reason |
|---|---|---|---|
| full | What type of cake is present? | [28475, 39505] | unknown: no evidence confirms cake type |
| full | Is the cake decorated with candles? | [28475] | unknown: no mention of candles in dialogue |
| full | Who is performing the action of cutting the cake? | [44888, 32174, 39505] | unknown: no action of cutting attributed to people |
| full | Is the cake being viewed or observed? | [17567, 28475] | already_known: people posing in front of cake |
| full | Is the setting indoors or outdoors? | [39505, 44888, 21232] | unknown: no setting specified in initial caption |
| attributes | What colour is the cake? | [28475, 39505] | unknown: no color mentioned in dialogue |
| attributes | What type of cake is it? | [28475, 39505] | unknown: no type of cake confirmed |
| attributes | Are there candles on the cake? | [28475] | unknown: candles not explicitly confirmed |
| attributes | Is the cake chocolate? | [39505] | unknown: chocolate not confirmed in dialogue |
| attributes | Is the cake being cut? | [44888, 21232, 32174, 39505] | unknown: cutting not confirmed in dialogue |
| relations | What is the cake being used for? | [28475, 44888, 21232, 17567, 25887, 32174, 39505] | unknown: purpose not specified in dialogue |
| relations | Is the cake on a table or elsewhere? | [28475, 39505] | unknown: location not confirmed in dialogue |
| relations | Are the people posing in front of the cake or cutting it? | [28475, 44888, 21232, 32174, 39505] | already_known: people posing in front of cake |
| relations | Is the cake decorated with candles? | [28475] | duplicate: Question already proposed this round. |
| relations | Is the cake being viewed or consumed? | [17567, 44888, 21232, 32174, 39505] | ambiguous: viewing vs consumption not distinguished |

The captions exposed occasion/action distinctions: birthday candles, wedding pairs, cutting versus posing, and table placement. The roles proposed different topics, but also repeated cake type and candles. Generalist and relationships produced the same candle question; it was deduplicated. Attributes’ “Are there candles…” paraphrase remained a separate eligible option. The selected question was the generalist candle question, even though BLIP-2’s answer did not confirm candles.

The attributes role also proposed cake colour, but its cited captions do not explicitly give that colour. Citation validity alone does not establish that a proposed distinction is well supported by the compressed evidence.

Across 19 answered critique interactions, selected eligible-question ownership was **generalist 16, attributes 1, relationships 2** (cake 8/0/2; giraffe 8/1/0). This credits the owner of the deduplicated eligible proposal, not every role that suggested a similar question. The generalist often supplied relationship-themed questions, so ownership does not by itself prove specialist expertise was necessary.

Completed critique audits contain 192 proposal-level gates: 97 `unknown`, 18 `already_known`, 7 exact within-pool `duplicate`, 4 `ambiguous`, 9 `unsupported`, 13 `already_asked`, and 44 `previously_rejected`. These counts exclude the failed giraffe pool, whose model filter omitted one decision. Exact repetition was often blocked at proposal time; semantic overlap remained in selected questions.

### Critique giraffe, interaction 1: representative examples and role proposals

| Representative ID | Caption |
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

| Role | First two proposals | Evidence and filtering |
|---|---|---|
| full | What, if anything, is near the giraffe?<br>How many giraffes are present? | [26575, 43487, 9158, 18601, 48324, 17090, 22645, 45162, 19606] unknown: No evidence confirms or denies proximity details<br>[43487, 9158, 18601, 22645, 45162, 19606] unknown: Original mentions one giraffe, not count |
| attributes | What colour is the giraffe's coat?<br>What type of grass is in the field? | [26575, 43487, 9158, 18601, 48324, 17090, 22645, 45162, 19606] unknown: Coat color not mentioned in any evidence<br>[26575, 22645, 18601] unknown: Grass type not specified in any evidence |
| relations | What is near the giraffe? | [26575, 43487, 9158, 18601, 48324, 17090, 22645, 45162, 19606] ambiguous: Repeats question without new detail |

The selected attributes question, “Is the field shaded or open?”, largely re-asked the initial caption’s “open field.” Its answer “open” then became the stronger rewrite “sunlit.” This is an example of a validity/faithfulness problem despite shared candidate evidence, not evidence that more specialist prompts necessarily produce new information.

### Full cake debate examples

The following tables preserve every opening/discussion contribution in two completed interactions, including its observation/comment, proposed question, citations, and limitation. Full payloads, coordinator attempts, and filtering are also saved in `analysis/selected_debate_examples.json` and the byte-identical debate checkpoint. Actual call payloads verified three independent openings, then identical three-message snapshots for all speakers in discussion round 1 and identical six-message snapshots in discussion round 2; each coordinator read all nine contributions. Thus the observed failures below are not an accidental within-round ordering leak.

#### Cake interaction 1 — complete transcript

| Stage / round | Role | Observation or discussion comment | Proposed question | Question evidence IDs | Limitation |
|---|---|---|---|---|---|
| opening 0 | full | The query mentions people posing in front of a cake, but most candidate captions describe actions like slicing or looking at a cake, not posing in front of it. [IDs 28475,44888,21232,17567,25887,32174,39505] | Which candidate shows a group of people posing in front of a cake, as opposed to interacting with or near a cake in another way? | [28475, 44888, 21232, 17567, 25887, 32174, 39505] | Some candidates involve a cake but lack explicit mention of people posing, making it hard to confirm alignment with 'posing in front of a cake'. |
| opening 0 | attributes | No candidate mentions people posing in front of a cake, only cake-related actions or individuals interacting with cakes. [IDs 28475,44888,21232,17567,25887,32174,39505] | Which candidate shows a group of people posing in front of a cake with visible candles? | [28475, 44888, 21232, 17567, 25887, 32174, 39505] | Several captions describe cake interactions without specifying posing or a group in front of a cake, making it unclear if any match the query. |
| opening 0 | relations | The initial caption mentions a group of people posing in front of a cake, which is not explicitly present in most candidate captions except one that describes a group posing together. [IDs 46619] | Which candidate shows a group of people posing in front of a cake, as described in the initial caption? | [46619, 28475, 21232, 25887, 32174] | Some candidates describe people near or interacting with a cake, but none explicitly state 'posing in front of a cake' with a group. |
| discussion 1 | full | The only candidate with a group posing together lacks explicit mention of a cake, while others involve cake interactions without posing. | Which candidate shows a group of people posing together in front of a cake, with no indication of cutting, looking at, or interacting with the cake? | [46619] | Candidate 46619 mentions a group posing but does not specify the presence of a cake, making it ambiguous whether they are in front of one. |
| discussion 1 | attributes | No candidate explicitly shows a group posing in front of a cake. | Which candidate shows a group of people posing together with a cake in the background? | [46619] | Only one candidate mentions a group posing together, but it does not specify the cake's presence or position in the background. |
| discussion 1 | relations | The only candidate with a group posing together lacks explicit mention of a cake in the foreground. | Which candidate shows a group of people posing together with a cake visible in the background or directly in front of them? | [46619, 28475, 21232, 25887, 32174] | No candidate explicitly describes the group posing in front of a cake, only near or interacting with it. |
| discussion 2 | full | No candidate explicitly describes a group posing in front of a cake. | Which candidate shows a group of people posing together with a cake directly in front of them? | [46619, 28475, 21232, 25887, 32174] | The presence of a cake in front of the group is not explicitly stated in any caption. |
| discussion 2 | attributes | Only candidate 46619 mentions a group posing together, but cake presence is unspecified. | Which candidate shows a group of people posing together with a cake clearly visible in the foreground? | [46619, 28475, 21232, 25887, 32174] | No candidate explicitly states the cake is in the foreground during posing. |
| discussion 2 | relations | No candidate clearly shows a group posing in front of a cake. | Which candidate shows a group of people posing together with a cake directly in front of them? | [46619, 28475, 21232, 25887, 32174] | The presence of a cake in front of the group is not explicitly stated in any caption. |

| Coordinator field | Saved value |
|---|---|
| Question | Which candidate shows a group of people posing together with a cake directly in front of them? |
| Evidence IDs | [46619, 28475, 21232, 25887, 32174] |
| Reason | This question targets a clear visual distinctionâ€”group posing with a cake directly in frontâ€”while acknowledging the lack of explicit confirmation in any caption. |
| Unresolved disagreement | Candidate 46619 mentions a group posing but does not specify the cake's position, while others describe cake interactions without posing, creating ambiguity about whether any candidate meets the exact requirement. |
| Filter | {'status': 'unknown', 'reason': 'No candidate explicitly shows group posing with cake directly in front'} |
| Answer | a group of people posing together with a cake |
| Rewrite | a group of people posing together in front of a cake |
| Rank | 14 |

#### Cake interaction 3 — complete transcript

| Stage / round | Role | Observation or discussion comment | Proposed question | Question evidence IDs | Limitation |
|---|---|---|---|---|---|
| opening 0 | full | Candidate 39741 shows a person cutting into a cake, but no group is posing with it directly in front; candidate 21232 shows bride and groom slicing a cake together, implying interaction but not group posing with cake directly in front [IDs 39741,21232] | Which candidate shows a group of people posing with a cake directly in front of them, not just interacting with or cutting it? | [30788, 18211, 39741, 10025, 21232, 46769, 17567, 25887, 40193, 32174] | No candidate explicitly mentions 'posing together' with a cake directly in front, making it difficult to verify the exact match |
| opening 0 | attributes | Candidate 39741 shows a person cutting into a chocolate birthday cake, but no group is posed with the cake directly in front. [IDs 39741] | Does candidate 21232 show a group of people posing with a wedding cake directly in front of them? | [21232] | No candidate explicitly describes a group posing together with a cake directly in front, making it difficult to confirm the exact scene. |
| opening 0 | relations | Candidate 39741 shows a person cutting into a chocolate birthday cake, but no group is posing with it. [IDs 39741] | Does candidate 21232 show a group of people posing together with a wedding cake directly in front of them? | [21232] | No candidate explicitly mentions a group posing with a cake in front of them, making it difficult to confirm the exact setting. |
| discussion 1 | full | The original query emphasizes 'posing together' with a cake directly in front, not just interacting with it. | Which candidate shows a group of people in a posed arrangement with a cake clearly visible and directly in front of them, not just cutting or viewing it? | [30788, 18211, 39741, 10025, 21232, 46769, 17567, 25887, 40193, 32174] | No candidate's caption explicitly describes a group posing with a cake directly in front, leaving ambiguity about the spatial relationship. |
| discussion 1 | attributes | Only candidate 21232 involves a group interacting with a cake, but posing is not explicitly stated. | Does candidate 21232 show a group of people posing together with a wedding cake directly in front of them? | [21232] | The caption mentions 'slicing a cake together' but does not confirm if they are posing with the cake directly in front of them. |
| discussion 1 | relations | No candidate clearly shows a group posing with a cake directly in front. | Which candidate shows a group of people posing together with a cake directly in front of them, as opposed to interacting with or cutting it? | [30788, 18211, 39741, 10025, 21232, 46769, 17567, 25887, 40193, 32174] | Only a few candidates mention a cake, and none explicitly describe a group posing with it directly in front. |
| discussion 2 | full | The key distinction remains the presence of a group in a posed arrangement with the cake directly in front. | Which candidate shows a group of people in a formal pose with a cake clearly and directly in front of them, as opposed to cutting, viewing, or surrounding it? | [30788, 18211, 39741, 10025, 21232, 46769, 17567, 25887, 40193, 32174] | No candidate's caption explicitly describes a group in a formal pose with the cake directly in front, making it ambiguous which, if any, meets the exact criteria. |
| discussion 2 | attributes | Candidate 21232 is the only one with a group and a cake, but posing is not confirmed. | Does candidate 21232 show the bride and groom in a posed arrangement with the wedding cake directly in front of them? | [21232] | The caption mentions slicing the cake together but does not specify if they are posing with it directly in front. |
| discussion 2 | relations | The key distinction is between posing with a cake directly in front versus interacting with it. | Which candidate shows a group of people in a static, posed arrangement with a cake clearly and directly in front of them, not engaged in cutting or viewing? | [30788, 18211, 39741, 10025, 21232, 46769, 17567, 25887, 40193, 32174] | No candidate explicitly describes a static pose with the cake directly in front, making it hard to confirm the spatial relationship. |

| Coordinator field | Saved value |
|---|---|
| Question | Does candidate 21232 show the bride and groom in a static, posed arrangement with the wedding cake directly in front of them, as opposed to simply slicing it together? |
| Evidence IDs | [21232] |
| Reason | The caption mentions slicing the cake together but does not confirm a static pose with the cake directly in front. |
| Unresolved disagreement | Some argue that slicing together implies posing, while others maintain that no explicit pose is described, leaving the spatial relationship ambiguous. |
| Filter | {'status': 'unknown', 'reason': 'No explicit confirmation of posed arrangement or cake position'} |
| Answer | yes |
| Rewrite | a group of people posing together with a wedding cake directly in front of them in a static, posed arrangement |
| Rank | 48 |

In interaction 1, all three roles had already framed the task as **which candidate matches the initial caption**. Discussion converged on the absent cake detail in the group-posing caption. The coordinator copied the generalist/relationships discussion-round-2 question verbatim. The filter justified acceptance because candidate captions did not explicitly confirm the scene—even though the initial caption already did. Candidate-caption uncertainty was substituted for an unresolved target fact.

Across the three answered cake debate interactions, the coordinator adopted an existing transcript question exactly in **2/3** cases. Interaction 3 introduced new wording relative to all nine suggestions, combining static posing with the bride/groom/cake alternative; it still asked about a retrieval candidate rather than the target image. “New wording” is not evidence of a new useful distinction. There were no final-question revisions. The disagreement field often described incomplete captions rather than genuine disagreement between specialists.

All three selected cake debate questions referred to candidates; interactions 2 and 3 named IDs **46619 and 21232**. Those are different images from the cake target (target manifest ID 31281). BLIP-2 received only the target JPEG and current-description-plus-question, not those candidate photographs or the candidate-ID catalogue. Its “yes” answers therefore cannot establish facts about the referenced candidates, or safely transfer their wedding/posing attributes to the target. This is an answer-interface failure, not leakage of the target label to question agents.

Giraffe debate never reached discussion or coordination. Its first generalist opening did identify potentially useful count/species differences in the representative captions (multiple versus single giraffes and an elephant alternative), but failed the observation schema. There is no evidence here about how three roles would debate or synthesize a giraffe question.

## 5. Selection, repetition, answer quality, rewrite faithfulness, and failures

### Exact failure diagnoses

| History | Failure point | Observed response problem | Preserved outcome |
|---|---|---|---|
| Critique giraffe | Interaction 10, filter_batch | Expected slots 0–5; returned only 0–4. Valid JSON, missing slot 5. | 10 states / 9 answers; last rank 96. No selection, answer, or rewrite for state 10. |
| Debate cake | Interaction 4, generalist opening | observations was a list of strings, not objects containing text and evidence_ids. | 4 states / 3 answers; last rank 48. |
| Debate giraffe | Interaction 1, generalist opening | Same list-of-strings observation schema violation. | Initial state only; no selected question, answer, or rewrite. |

Both debate responses were syntactically valid JSON and their top-level question evidence IDs were in the supplied set. The validator’s generic message (“Opening observations need valid supplied evidence IDs”) also covers the incorrect **per-observation structure**; it does not mean the top-level IDs were fabricated. Neither output hit its token cap. Critique’s new reviewer-slot retry could not help a malformed batch filter; regeneration is for a valid but empty eligible pool, not an invalid filter response. No hidden repair loop was added.

The failed giraffe filter additionally labelled a proposal `already_known` with “Field is open, not an enclosure,” despite the updated prompt’s explicit correction. This shows model reasoning did not consistently follow that distinction; it does not show that the new prompt was absent. The frozen sources and actual prompts/responses are preserved.

### Repetition and role usefulness

- **Literal selected-question repeats:** live cake repeated its colour question at interactions 7 and 8; live giraffe repeated the fence question at 5 and 6 and the coat-colour question at 2 and 7. A formatting-prefixed version of the same giraffe fence question appeared again at 10 after 9. The first coat-colour attempt was unanswered, so its later repetition did obtain a new answer; the repeated fence answers “a fence” did not initially resolve fence type.
- **No literal or surface-normalized selected-question repeats** occurred in the completed critique/debate prefixes. This does not establish semantic novelty. Critique cake’s interaction/relationship/action questions overlap (4, 5, 8); its proximity question at 9 adds little to the starting relation. Giraffe’s 3→4 repetition is a defensible follow-up to a nonanswer, whereas asking standing versus moving at 5 largely repeats the starting caption. Cake debate’s first two questions repeat the same group/cake-position matching criterion in different forms.
- **Within-pool overlap:** the cake candle example above shows both exact duplicates and unmerged paraphrases. Specialists did expose different topics, but selected ownership was heavily generalist and role-labelled suggestions often overlapped in meaning.
- **The candidate-interface problem is not unique to debate:** live PlugIR cake’s final question also asks which retrieval candidate best describes the scene and presupposes children/diverse ages. All three debate cake questions showed this problem; it also appeared in the live baseline. A successful filter call is not proof of target-only answerability.

### Answers and rewrites: separate evidence problems

BLIP-2 repeatedly returned the current description instead of the requested attribute. Examples include critique cake’s candle, blowing-candles, decoration, gaze, and surface questions; critique giraffe’s first other-giraffes and dry/green questions; live cake’s gender/age-group and colour questions; and live giraffe’s early coat-colour, fence-type, and vegetation questions. “A cake” to cake type and “a fence” to fence type were also unhelpfully generic. Some direct answers were useful (giraffe tree presence, coat tan, zoo setting), while compound questions and candidate-ID questions made other answers hard to interpret.

Both multi-agent methods and PlugIR used the same pinned answerer and aligned input protocol. Repeating the description is consistent with the answer input encouraging caption-like responses, but this pilot did not isolate that cause. The earlier pilot had a different multi-agent rewrite/answer-input protocol and is historical diagnostic evidence, not a controlled before/after ablation.

The following are **unsupported by the authoritative caption plus actual Q&A**, even when an added fact might happen to be visible in the JPEG. A question’s premise and a representative caption are not a confirmed target answer.

| Method / state | What the answer established | Unsupported or over-strong rewrite content |
|---|---|---|
| Critique cake 1 | Repeated the starting group/cake description; did not answer candles. | decorated cake with candles |
| Critique cake 3 | yes to birthday cake; earlier blowing-candles question was unanswered. | some of them blowing out the candles |
| Critique cake 4 / 7 | Repeated a description; did not deny interaction or confirm gaze. | no one ... interacting directly; later all looking at the cake directly |
| Critique giraffe 1 | open, which was already in the caption. | sunlit; later sunny/unshaded |
| Critique giraffe 3 | Did not answer whether other giraffes were present. | alone, before the direct no at interaction 4 |
| Critique giraffe 7 | no to a conjunction: directly visible AND in foreground. | a tree in the background and ambiguous not in the foreground; negating a conjunction does not identify which condition fails |
| Critique giraffe 9 | Repeated a giraffe/field/trees description; no dry/green answer. | green field |
| Debate cake 3 | yes to a question about candidate 21232, not an identified target fact. | wedding cake ... static, posed arrangement transferred into the target search |
| Live cake 6 | a young woman; previous demographic questions were unanswered. | diverse age range and mixed genders |
| Live cake 7 / 8 | Repeated birthday-cake description; no colour answer. | vibrant colours, then red, yellow, and blue |
| Live cake 9 / 10 | Repeated a description; no new expression or children answer. | celebration and joy; then children and joyful posing |
| Live giraffe 3 / 4 / 5 | Early fence/vegetation questions were unanswered; later a fence gave no vegetation type. | fence/enclosure presence first came from the question premise; vegetation/typical grassy background were added without that answer |
| Live giraffe 8 | a grassy area describes surface/area, not necessarily a path. | a grassy path nearby |

The cake JPEG shows a green decorated cake and a person holding a cutting knife; no candles are visible. The candles/blowing-candles story is therefore particularly questionable. Birthday and young-woman labels were model answers rather than facts independently established by this review. The giraffe image has vegetation and a background fence, but a “no” answer to being **near** a fence is spatially ambiguous; it should not be silently converted to total fence absence. Image inspection was post-hoc review only and was not fed to any question agent.

Useful retrieval changes can coexist with unsupported rewrites: live cake reached rank 4 while its rewrite included unsupported colour embellishment, and live giraffe reached 29 while its query contained inferred background detail. These ranks are measured observations, not evidence those embellishments were faithful or causally responsible for improvement. The standard rewrite has no separate factual-verification stage.

## 6. Evidence, reports, checkpoints, and source paths

All current-run results remain on GCP under `/home/lrc87/pilot_debate_20261122`. The local evidence directory below contains the original archive, exact source/input snapshot, extracted raw files, derived tables, validation results, and short checkpoint aliases. The aliases are byte-identical copies of the long original checkpoint paths, making them practical to open on Windows. Nothing was substituted into the original checkpoints.

| Evidence | Local link |
|---|---|
| Raw delivery archive | [open](outputs/pilot_debate_20261122/pilot_debate_delivery.tar.gz) |
| Exact source/input archive | [open](outputs/pilot_debate_20261122/debate_pilot_source.tar.gz) |
| Original transfer manifest | [open](outputs/pilot_debate_20261122/TRANSFER_MANIFEST.json) |
| Original local diff | [open](outputs/pilot_debate_20261122/local_changes.patch) |
| Raw artifact manifest | [open](outputs/pilot_debate_20261122/raw/ARTIFACT_MANIFEST.json) |
| Preflight/settings/model identities | [open](outputs/pilot_debate_20261122/raw/preflight.json) |
| Final source verification | [open](outputs/pilot_debate_20261122/raw/final_source_verification.json) |
| Installed package versions | [open](outputs/pilot_debate_20261122/raw/packages.txt) |
| GPU/driver identity | [open](outputs/pilot_debate_20261122/raw/gpu-identity.txt) |
| 161-test output | [open](outputs/pilot_debate_20261122/raw/pytest.log) |
| Actual per-history timing/call/resource events | [open](outputs/pilot_debate_20261122/raw/history_events.jsonl) |
| Primary runner failure and progress | [open](outputs/pilot_debate_20261122/raw/primary_runner.log) |
| Continuation dispatch/exit history | [open](outputs/pilot_debate_20261122/raw/continuation_dispatch.log) |
| GPU samples, primary | [open](outputs/pilot_debate_20261122/raw/gpu_samples.csv) |
| GPU samples, continuations | [open](outputs/pilot_debate_20261122/raw/gpu_samples_continuations.csv) |
| Derived metrics and audit summary | [open](outputs/pilot_debate_20261122/analysis/analysis.json) |
| Observed questions/answers/rewrites CSV | [open](outputs/pilot_debate_20261122/analysis/observed_rounds.csv) |
| Wide ranks CSV; blanks mean unavailable | [open](outputs/pilot_debate_20261122/analysis/observed_rank_comparison.csv) |
| Validation checks | [open](outputs/pilot_debate_20261122/analysis/validation.json) |
| Selected full debate examples | [open](outputs/pilot_debate_20261122/analysis/selected_debate_examples.json) |
| Observed settings and runner source | [open](outputs/pilot_debate_20261122/raw/IR-experiments/run_evaluation.py) |
| Debate source and prompts | [open](outputs/pilot_debate_20261122/raw/IR-experiments/src/agentir/multagent_experiment/debate.py) |
| Independent-agent source | [open](outputs/pilot_debate_20261122/raw/IR-experiments/src/agentir/multagent_experiment/multi_agent.py) |
| Shared filter prompt source | [open](outputs/pilot_debate_20261122/raw/IR-experiments/src/agentir/multagent_experiment/agent_prompts.py) |
| Observer/continuation launcher | [open](outputs/pilot_debate_20261122/raw/execute_session.py) |
| Environment setup commands | [open](outputs/pilot_debate_20261122/raw/setup_debate_pilot.sh) |
| Primary dispatch commands | [open](outputs/pilot_debate_20261122/raw/run_pilot.sh) |
| Independent dispatch commands | [open](outputs/pilot_debate_20261122/raw/continue_independent.sh) |
| Per-round Recall/Hits by explicit comparison scope | [open](outputs/pilot_debate_20261122/analysis/per_round_metrics.csv) |
| Checkpoint analysis command source | [open](outputs/pilot_debate_20261122/tools/analyze_results.py) |
| Matched-scope validation command source | [open](outputs/pilot_debate_20261122/tools/prepare_review_data.py) |
| Review generation command source | [open](outputs/pilot_debate_20261122/tools/build_review.py) |

| Method | Target | Full dialogue/audit checkpoint |
|---|---|---|
| Critique | Cake | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/agents_critique_000000364869.json) |
| Critique | Giraffe | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/agents_critique_000000562109.json) |
| Debate | Cake | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/agents_debate_000000364869.json) |
| Debate | Giraffe | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/agents_debate_000000562109.json) |
| Live PlugIR | Cake | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/live_plugir_qwen_000000364869.json) |
| Live PlugIR | Giraffe | [checkpoint](outputs/pilot_debate_20261122/analysis/checkpoints/live_plugir_qwen_000000562109.json) |

| Run log | Local link |
|---|---|
| agents_debate / Cake | [log](outputs/pilot_debate_20261122/raw/agents_debate_0.log) |
| agents_debate / Giraffe | [log](outputs/pilot_debate_20261122/raw/agents_debate_1.log) |
| live_plugir_qwen / Cake | [log](outputs/pilot_debate_20261122/raw/live_plugir_qwen_0.log) |
| live_plugir_qwen / Giraffe | [log](outputs/pilot_debate_20261122/raw/live_plugir_qwen_1.log) |

| Standard runner output | Report |
|---|---|
| 20261123T050953285296Z | [report.json](outputs/pilot_debate_20261122/raw/IR-experiments/runs/experiments/20261123T050953285296Z/report.json) |
| 20261123T051728439643Z | [report.json](outputs/pilot_debate_20261122/raw/IR-experiments/runs/experiments/20261123T051728439643Z/report.json) |

Raw call caches live under the **current** `raw/IR-experiments/runs/three_agents/`, `three_agents_debate/`, and `live_upstream/` method folders (`qwen_calls/*.json`, `blip2_calls/*.json`). Checkpoint identities and the event log retain exact GCP paths. Failure-time copies are in `raw/errors/`. Completed live exports include the standard CSVs, runner snapshot, and `prepared/qwen_live/generated_dialogues.json`. Failed critique/debate preparers did not produce their usual final prompt-pack/export files; actual prompts remain in their raw request records and the frozen source files linked above.

Each multi-agent completed state preserves representative IDs/captions, all candidate-window IDs, proposal/filter decisions, avoidance history, selected question, answer, rewritten query, and rank; debate states additionally preserve all nine contributions, coordinator reasons, disagreements, and attempts. Partial failure audits preserve the calls completed before the exception. There is no invented final decision for an omitted filter slot.

PlugIR preserves representative-caption prompt text, candidate questions, selected question, answer, rewritten/raw queries, ranks, and Qwen request/response traces. Its original representative image IDs, complete per-round candidate-window ID list, explicit KL values, and a normalized per-proposal filter-decision table are **not logged**. Its 500→50 window size is derived from recorded configuration/source, not recovered IDs. Raw traces can be inspected but were not promoted into invented normalized decisions.

Raw query strings/dialogue are preserved for all methods. Saved references and PlugIR’s extra raw-same-dialogue scoring pass were disabled. A raw query derived from one method’s own generated dialogue is a reformulation diagnostic, not a separate independently generated conversation; no such extra diagnostic column is presented here.

## 7. Limitations and the next experiment

**Do not scale this version to 32 targets yet.** Three of six independent histories interrupted, including both debate histories. Giraffe debate has no answered interaction, and the only three-method matched trajectory is three interactions on cake. These are two development images with one deterministic attempt per history, not a representative test set, a repeated-seed study, or a matched-cost comparison.

Before another fresh two-target pilot, the specific blockers are:

1. Make observation-object and batch-slot compliance reliable under an explicitly bounded, audited policy. The current run has no evidence that reviewer retry or empty-pool regeneration repairs these different failure types. Preserve the present failures when changing validation/decoding or adding any repair attempt; use a new source/protocol identity and fresh checkpoints.
2. Make the question’s audience explicit and validate it: the answerer sees the target image, not candidate IDs/photos. Reject candidate-identification questions and avoid merely re-testing facts in the starting caption. Test that this holds for both debate and the live baseline.
3. Address semantic repetition and the distinction between a real answer, a nonanswer, a question premise, and candidate-caption possibilities. The false open-field/enclosure inference and the candle/colour/children additions are concrete regression cases. Any rewrite-fidelity or answer-input change belongs in a separate controlled protocol; do not quietly swap BLIP-2 or claim the earlier pilot is its control.
4. Re-run the two-image pilot with completion/validity gates, comparable observation overhead, full candidate-selection diagnostics where available, and explicit compute budgets. Then use held-out targets and more than one seed before making retrieval-performance claims. An agreement/difference/uncertainty analysis is useful only if it leads to a new, target-answerable distinction; the first cake debate demonstrates that recognizing caption uncertainty alone is insufficient.

No proposed follow-up algorithm, prompt repair, answerer replacement, or 32-target experiment was implemented in this baseline run. The VM was left running; stopping generation does not stop VM billing.
