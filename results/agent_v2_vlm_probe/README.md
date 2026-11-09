# Agent v2 (vision-model probing) vs live PlugIR, and the caption mix

**Final run: 119 fresh targets, live conversations, 50,000 candidate photos, 10 question rounds.**
Both methods use the same pipeline: Qwen3-4B-Instruct writes the questions and rewrites,
BLIP-2 (flan-t5-xl) plays the user, and BLIP (blip-itm-large-coco) is the search engine.

## Summary

1. **Agent v2 does not beat PlugIR.** At round 10, Recall@10 is 59.7 for the agent and 62.2 for
   PlugIR, a difference of −2.5 (95% range −10.1 to +5.0). Recall@1 leans slightly toward the agent
   (+5.9, range −0.8 to +13.4) and Hits@10 slightly toward PlugIR (−2.5, range −7.6 to +1.7), but
   every range includes zero. The agent also takes about 1.5× as long per target.
2. **The caption mix helps both methods, and the gain is real.** Searching with
   0.7 × rewrite + 0.3 × original caption raises Recall@10 at round 10 by **+5.9** for PlugIR
   (range +1.7 to +10.1) and **+6.7** for the agent (range +2.5 to +11.8), with no extra model calls.
   It works by stopping the drift: without it, Recall@10 falls from 71 at round 0 to about 60–62 at
   round 10; with it, it ends at about 66–68.
3. **With the mix on both sides, the agent and PlugIR are still tied** (Recall@10 −1.7, range
   −8.4 to +5.0).

## What was compared

**PlugIR (live, Qwen).** Leo's wrapper runs the authors' unchanged `generate_dialog.py` each round.
Qwen writes 5 candidate questions, and PlugIR keeps the one whose wording changes the scores of the
current top candidates **least** (minimum KL), which keeps questions on topic. BLIP-2 answers by
looking at the target photo, and Qwen rewrites the description.

**Agent v2 (`probe_vlm`).** Only the question-choice step changes. For each candidate question, the
agent's own vision model (Qwen2.5-VL-3B) answers it on the **current top 20 results**. The agent is
never told which photo, if any, is the target. The agent asks the question whose answers vary most
across those 20 photos (highest answer entropy), because its answer should split the candidates.
Opinion questions ("what might this symbolize?") are skipped, since they have no factual answer.
If no question gets different answers, the agent falls back to PlugIR's choice.

**Caption mix.** A display change only: each round, search with the normalized blend
0.7 × (rewrite embedding) + 0.3 × (original caption embedding). Questions are chosen from PlugIR's
own search, so the mix does not change the conversation. Applying it to the finished conversations
therefore gives exactly what a live run with the mix would show. The 0.3 share was fixed earlier on
the other half of the saved GPT-3.5 data and was not tuned here.

## Protocol

- **Fresh targets.** Targets 31–149 in the matched-target order; none were used to design
  the agent. Both methods ran every target with the same seed and settings.
- **Fixed in advance.** The plan was 120 targets (31–150) and the decision rule: the agent wins if
  the Recall@10 range at round 10 is above zero and Hits@10 is not clearly lower. There was no early
  stopping. The run stopped at **119 of 120 targets** (target 150 was in progress) because the cloud
  account's billing was suspended overnight, not because of the results.
- **Check.** Searching each saved rewrite again reproduces every rank recorded during the live runs
  (0 mismatches out of 2,618). So the with-mix and without-mix numbers come from the same searches.
- **Metrics.** Recall@K is whether the target is in the top K at that round. Hits@10 is whether it
  was in the top 10 at any round so far. Ranges are paired bootstrap 95% intervals (5,000
  resamples) over targets.

## Results at round 10 (119 targets)

| Method | Recall@1 | Recall@10 | Recall@100 | Hits@10 | Median rank |
|---|---:|---:|---:|---:|---:|
| PlugIR | 28.6 | 62.2 | 82.4 | 80.7 | 5 |
| Agent v2 | 34.5 | 59.7 | 82.4 | 78.2 | 5 |
| PlugIR + mix | 37.8 | **68.1** | 88.2 | 79.8 | 3 |
| Agent v2 + mix | **41.2** | 66.4 | 87.4 | 78.2 | 3 |

Paired differences (95% range):

| Comparison | Recall@1 | Recall@10 | Hits@10 | Recall@10, mean of rounds 1–10 | Final rank better / worse / same |
|---|---|---|---|---|---|
| Agent v2 − PlugIR | +5.9 [−0.8, +13.4] | −2.5 [−10.1, +5.0] | −2.5 [−7.6, +1.7] | −1.5 [−6.6, +3.5] | 49 / 37 / 33 |
| PlugIR: mix − no mix | +9.2 [+3.4, +15.1] | **+5.9 [+1.7, +10.1]** | −0.8 [−2.5, 0.0] | +3.8 [+1.8, +6.2] | 69 / 4 / 46 |
| Agent v2: mix − no mix | +6.7 [+2.5, +11.8] | **+6.7 [+2.5, +11.8]** | 0.0 [−2.5, +2.5] | +6.6 [+3.7, +9.8] | 59 / 7 / 53 |
| With mix: agent − PlugIR | +3.4 [−2.5, +9.2] | −1.7 [−8.4, +5.0] | −1.7 [−5.9, +2.5] | +1.3 [−2.5, +5.2] | 33 / 40 / 46 |

Recall@10 by round (round 0 is the caption alone, identical for all four):

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PlugIR | 71.4 | 73.1 | 69.7 | 68.9 | 66.4 | 68.1 | 64.7 | 63.0 | 63.0 | 61.3 | 62.2 |
| PlugIR + mix | 71.4 | 72.3 | 70.6 | 72.3 | 68.9 | 69.7 | 70.6 | 70.6 | 68.1 | 67.2 | 68.1 |
| Agent v2 | 71.4 | 68.9 | 69.7 | 68.1 | 66.4 | 64.7 | 63.0 | 63.9 | 60.5 | 60.5 | 59.7 |
| Agent v2 + mix | 71.4 | 73.9 | 71.4 | 73.1 | 72.3 | 70.6 | 72.3 | 72.3 | 71.4 | 67.2 | 66.4 |

## Interpretation

**The agent's early lead was luck.** On the first 30 fresh targets, the agent led Recall@10 by +6.7
(range −10.0 to +23.3). On the next 89 targets, the difference was −5.6 (range −13.5 to +3.4). This
is why the target count was fixed before looking: 30 targets cannot separate a few points of real
difference from noise.

**Why the agent does not help.** It does what it was designed to do: in 1,190 rounds it fell back to
PlugIR only 8 times, and the question it chose split its 20 photos into varied answers (2.6 bits on
average). But a question that splits the agent's guesses about the top 20 is not necessarily one
whose answer moves the target up. The answer comes from BLIP-2 looking at the target, then passes
through Qwen's rewrite, and both steps lose detail. Also, each round offers few real choices: after
filtering, there are only about 2.4 distinct factual candidate questions per round.

**Why the mix helps.** Each rewrite adds the latest answer, sometimes with wrong or overemphasized
detail, and the search drifts away from what the user first said. Keeping 30% of the original
caption anchors the search. The gain grows over rounds as drift accumulates, and Hits@10 barely
changes. The mix keeps the target in view rather than finding new targets.

## Cost

On one NVIDIA L4: about 8 minutes per target for PlugIR and about 12 for the agent. The agent made
57,460 vision-model answers over the run. The caption mix costs one cached text embedding per
target.

## Limitations

- 119 targets, one short of the planned 120. The ranges support "no large agent gain (more than
  about 5 points of Recall@10)", not "exactly zero difference".
- One simulated user (BLIP-2) and one search model (BLIP). Results with a human or GPT-3.5 user may
  differ.
- The caption mix was tested by recomputing the display on finished conversations, which is exact
  only because the display does not feed back into question selection.

## Files

- [summary.json](summary.json): every number above, including the agent's behaviour counts.
- [per_round.csv](per_round.csv): Recall@1/10/100, Hits@10 and median rank per round, per method.
- [per_target_ranks.csv](per_target_ranks.csv): the target's rank at every round, per method.
- Code: [final_results.py](../../final_results.py) (this analysis);
  [probe_agent.py](../../src/agentir/probe_agent.py) (agent v2);
  [run_evaluation.py](../../run_evaluation.py) (live runs, `SELECTION = "probe_vlm"`).
- Related: [agent v1 results](../agent_probe_split/README.md). Matan's extra-search strategies
  (details, focus, split, with max/mean merging) were also tested on live conversations and did not
  beat the original search once compared at the same number of photos shown.
