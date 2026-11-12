# Shared representative captions: development protocol

All conditions use one shared neutral search over the same 50,000 cached images.
At interaction zero the original PlugIR helper clusters the top 500 and returns
up to ten representative captions (duplicate IDs are collapsed). Later windows
shrink by 50 per interaction. This test uses captions only, with no new image encoder.

Conditions:

1. `shared_neutral_three`: one unrestricted questioner, three sequential proposals.
2. `shared_roles`: full-context, attributes and relationships roles, one proposal each.
3. `shared_roles_critique`: the same specialist proposals, then one peer review and
   one retain/revise/withdraw response per eligible proposal.

Each slot has at most one replacement proposal. Specialists have explicit allowed
question categories. Each question names its unknown detail and quotes the shared
record. A separate record-only answerability call rejects known questions; another
record-only call checks factual presuppositions. Candidate evidence cannot enter
these two calls. Exact previously asked questions are rejected even if the earlier
answer was uncertain. LLM judgments and semantic duplicate detection remain imperfect.

Eligible questions partition all shared caption IDs into explicit value groups and
an unknown bucket. Each value requires a literal caption quote. Missing information
is never absence. An all-unknown grouping is allowed: asking an unknown property
of a confirmed object can still be valid. Quote checks prove textual provenance,
not correct object/property binding; review this manually. Group sizes are diagnostic,
not information gain or posterior probabilities. Cluster representatives are not
equally likely target images.

All final questions use the same neutral-query reference, common candidate slice,
original top-10 exclusion and original minimum-KL selector. Critique does not change
the selector. One chosen question gets one BLIP-2 target-image answer. Other proposed
questions and candidate details never enter the authoritative record. Target IDs,
ranks, target pixels and saved future answers never enter questioner messages.

Official query: original caption, followed by each actual `Question: ... Answer: ...`
verbatim. There is no generative summary or fusion. Token limits are checked before
retrieval; an overlong query stops rather than silently truncating confirmed context.
This deliberately changes the earlier lossy summary policy: comparisons to earlier
runs are not isolated role ablations. BLIP-2 observations can themselves be wrong.

Run on the existing GPU environment:

```bash
python run_shared_development.py --start-target 1 --targets 3 --states 2
```

This selects the existing guitar, mirror and bird development examples, with one
interaction per condition. It is a pipeline/behavior check, not a benchmark or
statistical evidence. Their initial ranks were already 1 in prior runs, so they
cannot demonstrate a rank improvement. Do not tune on these and call them held out.

Each round is capped at 40 calls. Critique adds compute; comparisons are proposal-
budget matched, not compute matched. Checkpoints, raw requests/responses, rejection
reasons, group tables, revisions, selected questions, KL values, tokens and timings
are saved under `runs/shared_evidence/`. The common evaluator exports the same
rank and retrieval metric formats under `runs/experiments/`. Use the existing
`scripts/report_results.py` on the exported result for plots.

After this check, inspect question-category diversity, known-question rejection,
grounded alternatives, quote binding, abstentions, actual critique changes and
record retention. Compare against the neutral three-proposal control before
claiming roles or debate help. Test image inspection as a separate later ablation.
