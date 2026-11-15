# End-to-end facts-only search — protocol (fixed 2026-11-15, before the run)

**Question.** When the whole live loop searches with the facts-only query (ranking shown, candidate images
that seed questions, and the minimum-KL question choice), does it beat live PlugIR end to end?

**Targets.** The same matched targets 151–250 (`--start 150 --targets 100`) as the PlugIR confirmation run.

**Pipeline.** Live PlugIR with one change (`SEARCH="facts"`, `src/agentir/facts_e2e.py`): every search of
the description inside the loop uses normalize(mean(BLIP(caption), BLIP("caption, fact_i")...)). After each
answer, the same Qwen3-4B-Instruct-2507 makes one greedy fact call with the prompt frozen in
`fact_fusion.py` (a test checks it is identical). Everything else is unchanged: questioner, filter, KL rule,
BLIP-2 answerer and its prompt, rewriter (still used as text context), seed 1021, 11 states.

**Baseline.** Live PlugIR on the same targets (run `0c32e6de265e727ba961`), scored two ways: rewrite
(what PlugIR shows) and caption mix (normalize(0.7 rewrite + 0.3 caption)).

**Primary test (one).** End-to-end facts (its own `target_rank`) minus live PlugIR caption mix, Recall@10 at
round 10, paired bootstrap over targets (5,000 resamples, seed 42). **Passes** if the 95% CI lower bound > 0.

**Also reported.** vs live PlugIR rewrite; vs the post-hoc facts-only score of the PlugIR conversations;
R@10 per round, averaged over rounds 1–10, R@1, Hits@10, better/worse/same; fact rate per round; seconds per
target. Incomplete targets at the time cap are excluded and listed; nothing else is dropped or rerun.
