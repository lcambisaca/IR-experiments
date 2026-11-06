# Agent searches for choosing PlugIR's question: first live test

**Question:** PlugIR chooses its next question using only the results of the current search. If the agent also runs its own searches before asking, does it choose more useful questions and find the target sooner?

**Short answer, based on 30 targets:** no, not with this first version. After 10 rounds the agent finds the target in the top 10 for 50% of targets, against 63% for PlugIR. With only 30 targets this gap is not significant (95% range −30 to +3 points). The conversations show why it fails: the agent prefers questions whose answers would split the candidates well, but the simulated user often answers those questions wrongly.

## Setup

This is the full interactive pipeline, as in ChatIR and PlugIR. It starts from the VisDial caption only; no human-written dialogue is used.

| Part | Model |
|---|---|
| Image search | BLIP (`Salesforce/blip-itm-large-coco`) over 50,000 photos |
| Questions and rewrites | Qwen3-4B-Instruct-2507, running locally, the same model for both methods |
| Simulated user (sees the target photo) | BLIP-2 (`Salesforce/blip2-flan-t5-xl`), PlugIR's original prompt |
| PlugIR code | The authors' original `generate_dialog.py`, unchanged, run through Leo's wrapper |
| Targets | The first 30 matched VisDial targets, 10 rounds each |
| Hardware | Google Cloud, one NVIDIA L4 GPU |

Qwen replaces PlugIR's GPT-3.5 in both methods, so any difference comes from the method and not from the model.

## The two methods

Both run PlugIR's loop: search with the current description, cluster the top candidates and caption one per cluster, have Qwen write 5 candidate questions, and filter out questions the description already answers. They differ only in **which of the remaining questions gets asked**.

| | How the question is chosen |
|---|---|
| **PlugIR** | Adds each question's text to the search and keeps the question that changes the candidates' scores **least** (minimum KL divergence). In practice this picks safe, on-topic questions. |
| **Agent (probe split)** | For each question, Qwen guesses up to 3 possible answers (for example "white / dark wood / gray"). The agent then searches once per guess (*"…, what color are the cabinets? white"*). Each of the ~500 current candidates "votes" for the guess whose search fits it best. The question whose votes come closest to an even split is asked: whatever the user answers, it should rule out the most candidates. The score is the mutual information between the answer and the candidate, in bits. |

The simulated user's answer, Qwen's rewrite and the next round then proceed exactly as in PlugIR.

**Fairness:**
- The agent never sees the target photo, its name or its rank. We checked all 727 guessing prompts.
- Probes are not shown to the user and not scored.
- Only the ranking after each user answer counts.
- One setting (how strictly a vote counts) was fixed before any test.

Code: [`src/agentir/probe_agent.py`](../../src/agentir/probe_agent.py), switched on with `SELECTION = "probe_split"` in `run_evaluation.py`. The default remains original PlugIR.

## Results (30 targets)

| Round | Recall@10 PlugIR + Qwen | Recall@10 Agent | Hits@10 PlugIR + Qwen | Hits@10 Agent | Recall@10 saved PlugIR (GPT-3.5) |
|---|---|---|---|---|---|
| 0 (caption) | 63 | 63 | 63 | 63 | 63 |
| 5 | 67 | 53 | 80 | 77 | 70 |
| 10 | **63** | **50** | **83** | **80** | 73 |

Recall@10 is the share of targets in the top 10 at that round. Hits@10 is the share that reached the top 10 at least once by that round. Full tables: [`summary.csv`](summary.csv) and [`per_target_ranks.csv`](per_target_ranks.csv).

- **Agent − PlugIR, Recall@10 at round 10:** −13 points, 95% paired bootstrap range −30 to +3.
- **Target by target at round 10:** the agent ranked the target better on 12, worse on 10, and the same on 8.
- **Recall@1 at round 10:** agent 33%, PlugIR 27%.
- **Easy sample:** 19 of the 30 targets were already in the top 10 from the caption alone.

## What we learned

1. **PlugIR with Qwen barely improves over 10 rounds.** On the same targets its Recall@10 stays at 63%, while the authors' saved GPT-3.5 conversations reach 73%. Qwen3-4B is a weaker question writer than GPT-3.5, which is why both methods must use the same model.
2. **The agent picks questions BLIP-2 answers badly.** Colour questions split the candidates well, but BLIP-2 often answers "white". In one bedroom conversation the search became *"a white bedroom filled with white posters … white chair … white pillows on a white bed"*, and the rank fell from 1 to 15. "What is the gender composition?" was answered "male", turning *"6 people"* into *"Six men"*, and the rank fell from 7 to 73. See [`agent_worse_conversations.md`](agent_worse_conversations.md).
3. **The split score ignores whether the user can answer correctly.** A question with a big split but an unreliable answer does more harm than good. PlugIR's minimum-KL rule avoids this by accident: its safe questions often get BLIP-2 to repeat the description, which keeps the search close to the caption.
4. **The agent often has little to choose from.** PlugIR produced on average only 2.4 distinct questions per round out of 5 attempts, and in 28% of rounds there was only one.
5. **It is slow.** About 7.5 minutes per target per method on an L4.

## Limitations and next steps

- **30 targets is a small sample**, and these 30 have now been examined closely. Any revised agent must be tested on **different targets**, for example targets 31 to 330.
- **Next version:** weigh how reliably the user can answer a question, not just how well its answers split the candidates. For example, skip a question if BLIP-2 gives the same default answer whatever the photo. Also avoid questions about what is absent.
- **The full PlugIR + Qwen baseline on 330 targets** is running on the GPU now ([`scripts/gcp/baseline.sh`](../../scripts/gcp/baseline.sh)).
- [`conversations.json`](conversations.json) holds every question, answer, rewrite, rank and agent decision for both methods.
