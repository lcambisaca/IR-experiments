# Debate v8: two-image Qwen-VL pilot review

**November 24, 2026 | cake and giraffe | same 50,000-image retrieval pool | 11 states.**

## Executive summary

V8 is a more productive questioner on these two examples: **16/20 answered interactions versus 12/20 for v7**. It reduced skipped rounds from eight to four, reduced repeated/rejected attempts, maintained cake in the top 10, and improved giraffe's best rank from 89 to 28 and final rank from 165 to 93. There is direct transcript evidence of a useful discussion-induced change (grass height → enclosure/fence) and a minority opening recovered by coordinator revision (fence color). However, final-round specialists actually converged more often on identical wording, and Hits@10 did not improve. This is evidence of better exploration across interactions, not proof that debate generally increases independent specialist diversity or improves retrieval.

**Leakage audit: no evidence of target image, target identity, target catalogue caption or target rank being provided to the debating agents.** We checked every saved v8 text-model request and image-answer request, not just the prompt instructions. There are still semantic grounding problems unrelated to direct image leakage (see below).

## 1. Controlled comparison and run identity

The source of truth was the local `experiment` branch at `f59bc3e7b367da7d261dc2414db807eb022361ae`, with the uncommitted `src/agentir/multagent_experiment/debate.py` changes. The local v8 file and executed archived file both had SHA-256 `adaa21bee379a005a34effc08c3528ee62af6eafd380831ef6839613b7966cbc`. The previous 67-file snapshot was verified against every current local runtime file; only this debate file differed. The exact uncommitted patch was transferred to an isolated directory and checked by hash. The only execution overrides were `RUN_LIVE_PLUGIR=False` and `RUN_MULTI_AGENT=False` in the fresh run copy of `run_evaluation.py`; the local working tree was not changed. Effective runner hash `f9887dacd4dfcdf7c49d9d69d39e2f1bc78008bd09aee3ecdbcf08da8a998bf1`. Final source hash verification passed all 67 files.

The scientific changes are limited to prompts and the policy identifier `specialists_target_questions_v8`: avoid repeated detail; challenge weak peer ideas; preserve useful independent distinctions; and consider minority/opening suggestions irrespective of agreement. Two discussion rounds, coordinator revision, final-round fallback and filtering logic are unchanged. Note that subsequent candidate windows and deterministic prompt/context-derived sampling seeds can differ when generated dialogues diverge; this is a method outcome, not input leakage or a changed starting pool.

- Targets: `unlabeled2017/000000364869.jpg` (cake) and `unlabeled2017/000000562109.jpg` (giraffe). Captions: “a group of people posing in front of a cake” and “a giraffe is standing in an open field”. Initial ranks 9 and 89.
- Enabled only debate; `USE_PILOT_TARGETS=True`, `ANSWERER="qwen_vl"`, `N_STATES=11`, `SEARCH="rewrite"`, `SELECTION="plugir_kl"`, `MAX_INPUT_TOKENS=8192`, saved references off, CUDA devices.
- Preflight compared both versions' image hashes, captions, initial ranks, exact retrieval cache and manifest hashes, candidate catalogue count, resolved model revisions, packages, CUDA, and GPU identity. All matched. NVIDIA L4 (`dl-gpu`, `us-east4-c`), Python 3.11.18, torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 4.51.3. 194 applicable tests passed; the same absent historical-results fixture test was excluded, as in v7.
- Models: Salesforce/blip-itm-large-coco `19502f1e215844f7e48bd48473f86932486d3441`, Qwen/Qwen3-4B-Instruct-2507 `cdbee75f17c01a7cc42f958dc650907174af0554`, Qwen/Qwen2.5-VL-3B-Instruct `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Both histories started with `checkpoint_existed=false`; zero cached text or vision completions replayed. Generation 18:44:33–19:13:22 UTC, 28m49s, primary exit 0, all 11 states per target. Old v7, critique and PlugIR results reused **as comparison artifacts**, not regenerated.

## 2. Retrieval results

| Method | Cake best/final | Giraffe best/final | Final mean rank | BRI (lower better) | Final Recall@10 / Hits@10 | Final Recall@100 | Runtime (two targets) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Live PlugIR | 1 / 1 | 50 / 127 | 64 | 2.1352 | 50% / 50% | 50% | 12m34s |
| Critique | 1 / 1 | 43 / 87 | 44 | 2.9427 | 50% / 50% | 100% | 13m14s |
| Debate v7 | 1 / 1 | 89 / 165 | 83 | 2.2992 | 50% / 50% | 50% | 25m10s |
| **Debate v8** | **1 / 1** | **28 / 93** | **47** | **2.2580** | **50% / 50%** | **100%** | **28m49s elapsed** |

BRI improved slightly from v7 (about 1.8% reduction), but PlugIR retained the lower BRI and critique retained the better final mean rank. All methods' cumulative Hits@10 remained 50% throughout because cake was already in the top 10 and giraffe never entered it. V8's improvement is in rank and dialogue productivity, not Hits@10. No reranking was performed.

| State | Cake PlugIR | Cake critique | Cake v7 | Cake v8 | Giraffe PlugIR | Giraffe critique | Giraffe v7 | Giraffe v8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 9 | 9 | 9 | 89 | 89 | 89 | 89 |
| 1 | 21 | 17 | 1 | 7 | 50 | 68 | 172 | 88 |
| 2 | 1 | 21 | 5 | 7 | 98 | 109 | 151 | 28 |
| 3 | 2 | 10 | 2 | 10 | 120 | 110 | 160 | 54 |
| 4 | 1 | 15 | 2 | 3 | 260 | 86 | 149 | 40 |
| 5 | 1 | 8 | 2 | 2 | 155 | 76 | 149 | 31 |
| 6 | 1 | 8 | 2 | 2 | 194 | 63 | 149 | 69 |
| 7 | 1 | 16 | 2 | 2 | 150 | 43 | 145 | 48 |
| 8 | 1 | 9 | 2 | 1 | 124 | 87 | 171 | 93 |
| 9 | 1 | 1 | 1 | 1 | 114 | 134 | 318 | 93 |
| 10 | 1 | 1 | 1 | 1 | 127 | 87 | 165 | 93 |

![Rank progression: all four methods](outputs/pilot_debate_v8_20261124/pilot_rank_v8_comparison.png)

## 3. Did discussion improve diversity?

The counts below use all 20 opportunities per version; exact question matching normalizes case and punctuation, but does not collapse semantic paraphrases. Answered interactions exclude saved no-question states. An opening-only match means the selected question exactly matched at least one independent opening but no final-round question.

| Audit measure | v7 | v8 |
|---|---:|---:|
| Answered questions (cake / giraffe) | 4 / 8 | 8 / 8 |
| Skipped opportunities | 8/20 (40%) | 4/20 (20%) |
| Mean unique normalized opening questions per interaction (max 3) | 2.25 | 2.50 |
| Mean unique normalized final discussion questions (max 3) | 1.65 | 1.40 |
| All three specialists give same normalized opening | 2/20 | 1/20 |
| All three give same normalized final discussion question | 9/20 | 12/20 |
| Coordinator-selected question matches an opening exactly | 1/12 | 3/16 |
| Coordinator-selected question matches an opening but not final discussion | 1/12 | 1/16 |
| Coordinator-selected question matches final discussion exactly | 6/12 | 15/16 |
| Coordinator revisions (two attempts) | 14 | 6 |
| Specialist fallback attempts | 24 | 12 |
| Selected specialist fallbacks | 0 | 0 |
| Recorded partial-format recoveries | 112 | 0 |
| Recorded JSON/schema validation error entries | 132 | 0 |

The reduction in failure/rejection is encouraging, but final-stage convergence did not decrease. Independent openings are slightly more diverse by wording; discussion frequently converges to one question. V8's greater rate of adopting the final proposal is evidence of alignment, not proof that minority suggestions were generally preserved. Format recovery also improved markedly, a meaningful confound/mediator of the prompt-only effect because v7 often had to recover incomplete specialist responses. We cannot isolate the contribution of novelty instructions from the changed examples and improved format adherence in this paired pilot.

### Positive example — giraffe round 2: peer challenge changed the question

All three openings asked about short grass, after round 1 had already answered `no` to tall grass. The discussion explicitly recognized that none of the representative captions specified grass height. The generalist suggested enclosure versus open area, the relationships agent proposed a fence/barrier, and the attributes agent retained a separate rocks question in both discussion rounds. Candidate evidence included ID 15696 (giraffe in enclosure with stones and trees), 7283 (giraffe by a chain-link fence), 4103 (giraffe next to a tree in a zoo pen), and 48324 (giraffe in a zoo with rocks on a dirt path). These support meaningful alternative setting/terrain distinctions without proving the target has those details.

Coordinator selected the relationships agent's question, “Is the field enclosed by a fence or barrier?” Qwen-VL answered `yes`; the rewrite was `a giraffe is standing in an open field enclosed by a fence, with no tall grass present`. Rank went **88 → 28**. This is the clearest example of a peer challenge yielding an independently useful distinction; it did not need target images or target rank to select the question.

### Minority opening recovered — giraffe round 3

Independent attributes proposal: “What color is the fence surrounding the giraffe?” (IDs 23841, `the fence is green`, and 3089, `a metal fence surrounds a grass area`). The other openings asked about posture; after discussion all three settled on “Is the giraffe standing or lying down?” The coordinator initially selected the consensus posture question, correctly rejected as `already_known` because the original caption says standing. On revision, it returned to the attributes opening, which was absent from the final discussion. Answer `red`; rewrite appended the red fence, rank **28 → 54**.

This demonstrates the intended opening/minority reconsideration behavior, but its retrieval outcome was negative. The coordinator also misleadingly framed green vs. metal as a color contrast, although a material does not imply a color. In v7 cake round 7, a similarly promising attributes opening about how many people sliced the cake was lost without ever reaching the filter. The revised coordinator can now recover an earlier idea, yet did so on only one exact opening-only match in this run.

## 4. Rejection audit

| Rejection status | v7, all attempted proposals | v8, all attempted proposals | v7, skipped rounds only | v8, skipped rounds only |
|---|---:|---:|---:|---:|
| Previously rejected | 24 | 11 | 24 | 11 |
| Already known | 16 | 11 | 10 | 9 |
| Already asked | 5 | 0 | 5 | 0 |
| Duplicate | 1 | 0 | 1 | 0 |
| **Total rejected** | **46** | **22** | **40** | **20** |

Each rejected record is counted separately, not a unique question. V8 no longer repeatedly asked the identical accepted questions, but its remaining four skipped rounds still exhibit repetition and some overbroad filtering:

| Target/round | Rejected theme | Assessment |
|---|---|---|
| Cake 2 | Cake being interacted with or static background | Related to round 1 central/background, but not logically equivalent. Filter cited interaction in candidate captions, not an established target fact. Potential false rejection. |
| Cake 7 | Woman cutting in a formal setting vs. picnic table | Already answered `formal setting` at round 6. Rejection justified; all final fallbacks repeated it. |
| Giraffe 9 | Stones in the field or on a separate path | Round 7 answered `no rocks`; near-repeat. Rejection reasonable, although exact location cannot always be inferred. |
| Giraffe 10 | Giraffe standing directly at fence or away from it | All coordinator attempts rejected because fence presence was confirmed. Presence does **not** establish proximity; plausible novel question rejected by filter. The exact fallbacks were then rejected by memory. |

Thus we should not conclude that the filter is always correct. The v7 eight-round audit primarily implicated repetition, but v8 now provides plausible examples of the filter conflating candidate evidence/presence with knowledge of the target. Review these decisions manually before modifying filter policy.

## 5. Answer quality and query faithfulness

| State | Cake question topic → answer | Rank | Giraffe question topic → answer | Rank |
|---:|---|---:|---|---:|
| 1 | Central element/background → central element | 7 | Tall grass → no | 88 |
| 2 | Skipped | 7 | Fence or barrier → yes | 28 |
| 3 | Wedding/birthday → birthday | 10 | Fence color → red | 54 |
| 4 | Who interacts/how relates to pose → woman in center holding knife, smiling (truncated phrase) | 3 | Trees around field → yes | 40 |
| 5 | Chocolate cake/picnic table event and pose → separate event | 2 | Zoo/natural habitat → zoo | 31 |
| 6 | Picnic table/formal setting → formal setting | 2 | Fence material → metal | 69 |
| 7 | Skipped | 2 | Rocks in field or nearby path → no rocks | 48 |
| 8 | Cake type → green | 1 | Grass color → green | 93 |
| 9 | Woman cutting alone/with another → with another person | 1 | Skipped | 93 |
| 10 | Relationship between cutters → mother | 1 | Skipped | 93 |

Cake round 5 is particularly concerning: “Does the woman cutting the chocolate cake on a picnic table occur during the group's pose in front of the birthday cake, or is it a separate event?” combines descriptions of different candidate images while asking about a single target photo. The answer `separate event` produced a rewrite claiming cake cutting occurs separately from the pose, an unsupported interpretation of event timing. Cake round 10's answer `mother` is an unverified relationship, not normally inferable from appearance; rewrite asserted “her mother”. Giraffe round 1 turned `no` to tall grass into `short grass`, implying grass exists. Later color/material claims were incorporated into the query and sometimes made retrieval worse. These are grounding/faithfulness problems, **not evidence of the debaters seeing the target photo**. No independent human image annotation was used here. Rank improvement alone cannot validate the truth of an answer.

## 6. Target-image and information-leakage audit

Verified source data flow in `src/agentir/multagent_experiment/{debate,multi_agent}.py`, `src/agentir/live_models.py` and `src/agentir/live.py`, and cross-checked actual saved requests:

1. `generate_agent_dialogue` constructs the questioner record using only initial caption and earlier Q/A dialogue. `RetrievalContext.candidates(query, interaction, role, record)` uses the current text query against the common precomputed retrieval vectors and selects representative **caption strings and catalogue IDs**. It has no `target_id` or `image_path` parameter; selection is not driven by target rank. `debate_round` passes record, current query, representative captions, rejection feedback and prior discussion to the text model. `debate_messages` serializes these as JSON inside string messages; `Qwen3._create()` uses a text tokenizer and a causal language model.
2. Inspected all **249** saved `qwen_calls` requests (206 debate completion calls, 27 shared completion/filter calls, 16 reconstruction calls). Every original message content was a string. Neither target filename nor target SHA-256 appeared in the messages. Every saved opening, discussion, coordinator and filter model payload was checked for target identity, image/image path and target-rank keys in its top-level data and dialogue record; none were present. The initial-caption record matched the intended target in each history.
3. `target_id` is used by the outer evaluation loop for checkpoint metadata and scoring, after selection/answer/rewrite, not as questioner input. Rank is measured by `ranks_for_targets` and saved, not passed to `debate_round`. The target naturally appeared in some candidate retrieval windows (cake all ten for both versions; giraffe v7 states 1–8, v8 states 1–9). But **neither target's catalogue index (cake 31281, giraffe 48282) nor exact first catalogue-caption text appeared in any of the 40 saved representative-caption sets** across v7 and v8. The questioner sees the prescribed starting caption and unlabelled candidate evidence, not a privileged target caption. The full shared embedding pool is used for retrieval, not provided to text generation as an image or identified target vector.
4. Only after selecting a question does the outer loop park the text model and call `QwenVLAnswerer.answer_original(..., image_path, neutral + ". " + question)`. The Qwen2.5-VL processor loads the actual JPEG. We compared all **16** archived visual-answer cache requests against the expected multiset of (correct target JPEG SHA-256, previous search description + selected question). All 16 matched, with zero extra visual answer requests or reused completions. Cake JPEG SHA-256 `a7425cfc97a00bf87a30fbb26f1b41cd4b76114f401aba2cb2cb2158102b177f`; giraffe `8d445abe263c554fefd86068fe619780dff5db017ce514208ef862bf04c8e1d9`.

Conclusion: the saved run and code support proper separation of the target-image answerer from the caption-only debating agents. This is a targeted audit of the actual inputs and source, not a claim that all possible data contamination (e.g. pretraining memorization or semantically similar candidate captions) can be ruled out. An independent guardrail test could assert the text request schema and no target/rank fields in future experiments.

## 7. Runtime and next steps

| Method and target | History seconds | Text calls (actual) | Visual answers | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| v7 cake | 755.22 | 136 | 4 | 9.74 GiB |
| v7 giraffe | 754.99 | 138 | 8 | 9.76 GiB |
| v8 cake | 874.41 | 123 | 8 | 9.82 GiB |
| v8 giraffe | 845.61 | 126 | 8 | 9.79 GiB |

V8 used fewer text calls (249 vs 274) but more visual answers (16 vs 12) and took approximately 14% longer per two histories. A simple extrapolation of 28m49s for two targets gives 7.68 hours for 32 v8 targets on the same GPU if all states are retained; not a reliable benchmark prediction with only two images. No 32-target run was started.

Next: (1) inspect and correct the difference between already established target facts and candidate-only evidence; (2) preserve alternative evidence-grounded questions when the dominant topic is already covered; (3) collect semantic topic-diversity rather than only exact question diversity, and compare against an independent/no-discussion controller; (4) verify visual answers with human annotation and explicitly distinguish visible facts from candidate assumptions. The experiment supports the revised prompt direction, but not a general performance claim.

## 8. Reproduction and evidence paths

- VM run directory: `/home/lrc87/pilot_debate_v8_20261124`. `bash run.sh` activated the existing Python environment by absolute interpreter path, ran tests/preflight, then `python -u execute.py` (observation wrapper around `run_evaluation.main()`). `dispatch.log`, `primary.log`, `history_events.jsonl`, `preflight.json`, `gpu_samples.csv`, `final_source_verification.json` and `IR-experiments/{TRANSFER_MANIFEST.json,local_changes.patch}` preserve run evidence and exact changes.
- Official artifacts: `IR-experiments/runs/experiments/20261124T191319766491Z/{report.json,rank_comparison.csv,prepared/agent_debate/debate_dialogues.json}`; fresh histories under `IR-experiments/runs/three_agents_debate/shared_agents_debate_v1/d8a17f183217bbfed53a/dialogues/`, with original text/answer caches in the new runs tree.
- Archive: `/home/lrc87/debate_v8_pilot_delivery.tar.gz` on GPU and copied to authenticated Cloud Shell home, matching SHA-256 `a2460f4e920ff1bc6933f5dc630f72fcb148eb6fc6bbb76db95913ef432c9632`. The complete archive has not been downloaded to the local repo; retain Cloud Shell copy before deleting resources. Prior baseline archive `qwen_pilot_delivery.tar.gz` remains in Cloud Shell; comparison details in `PILOT_DEBATE_QWEN_REVIEW.md`.
- The local companion graph is `outputs/pilot_debate_v8_20261124/pilot_rank_v8_comparison.png`; no scientific source changes, re-runs or additional GPU compute were made during this review.
