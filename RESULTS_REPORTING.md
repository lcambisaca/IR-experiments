# Reporting compatible with the teammate's live experiments

Reviewed branch: `live-results-agent-v2-and-caption-mix`, commit
`51be8325b2aaa53281c91a8250b0fac547ccffff` (2026-11-09).

Their agent v2 changes **question selection**: Qwen2.5-VL-3B answers proposed
questions on the current top 20 candidate photos; the question with the most
varied answers wins, with a KL fallback. Our method instead uses three search
roles and peer critique, followed by common-reference KL. Matching reporting
does not require importing their probing algorithm or changing our ranking rule.

## Their testing protocol

- Same 50,000-image cached BLIP pool and matched VisDial targets across methods.
- State zero is the original caption; ten Q&A interactions give eleven states.
- Qwen3-4B-Instruct-2507 questioner/rewriter; BLIP-2 FLAN-T5-XL simulated user.
- Targets 1–30 used for development. Their final comparison uses targets 31–149
  (119 shared completed targets); PlugIR also completed target 150, which is excluded.
- Comparisons are paired by target identity. Final Recall@1/10/100, cumulative
  Hits@10, median rank, better/worse/same counts, and mean Recall@10 over rounds
  1–10. They use 5,000 paired target bootstrap resamples with seed 42.
- A 0.3 original-caption / 0.7 rewritten-query embedding blend is a separate
  display-only analysis. It does not feed back into that branch's dialogue loop.
- Their README reports one L4 and approximately 8 minutes/target for PlugIR and
  12 for agent v2. These are reported timings, not measurements of our method.

Their scripts are `final_results.py` and `scripts/make_figures.py`. The plotting
script assumes private `vm_results/` directories and several fixed method names,
cohorts and example images. The portable reporter here reads committed exports
or our existing run outputs, so a teammate does not need those private folders.

## Generate our reports after a run

Install the optional plotting extra into the project environment:

```bash
python -m pip install -e ".[analysis]"
```

Then point the reporter at a completed common-evaluator output folder:

```bash
python scripts/report_results.py runs/experiments/YOUR_RUN --output results/analysis/YOUR_RUN
```

The output directory must be new. Analysis never runs a model, starts a VM,
changes a saved result or computes a new caption mix.

For a focused critique comparison:

```bash
python scripts/report_results.py runs/experiments/YOUR_RUN --methods agents_no_critique agents_critique --compare agents_critique agents_no_critique --output results/analysis/YOUR_RUN_CRITIQUE
```

If the same run includes the live baseline, select it alongside the two agent
methods and add `--compare agents_critique live_plugir_qwen`. Saved GPT-3.5
queries are a reference with potentially different conversations/model responses,
not a same-model causal control.

To enrich examples with the generated question/answer text, also pass the saved
`prepared/three_agents/agent_dialogues.json` as a second input. Rank conflicts
are rejected rather than silently overwritten.

## Input compatibility

- Our `report.json` or its containing result folder.
- Our `per_example.csv` long table.
- Teammate's `per_target_ranks.csv` wide table.
- Compact conversation exports via `METHOD=path.json`.
- Our `agent_dialogues.json` prepared export, which already contains method names.
- Their `saved_plugir_gpt35_ranks.json` saved reference (defaults to `supplied_plugir`).

Use `--methods NAME ...` to choose methods, `--states N` for an explicitly chosen
common prefix, and `--allow-intersection` only after inspecting missing/incomplete
targets. Excluded IDs are exported. Different target sets are rejected by default.

Target order for the cumulative-effect chart defaults to input order, which is
not necessarily execution order. Use `--target-order path.json` with a list of
target IDs, a JSON `targets` list, or the matched-query list to supply a known
order. All plotted methods use the same matched cohort.

## Figures and statistics

- `fig3_recall_hits_by_round`: same main Recall@10/Hits@10 curve format.
- `fig4_effect_over_targets`: first comparison's final Recall@10 difference as
  targets accumulate, with descriptive pointwise bootstrap bands.
- `fig5_effect_sizes`: final paired Recall@10 differences and 95% intervals.
- `fig_final_metrics`: Recall@1/10/100, Hits@10, lower median rank and BRI.
- `example_*_rank_trajectory`: per-target log-rank plots and a conversation table.
- `fig1_setup`: our pipeline diagram when agent methods are present.

Figures are saved as PNG and vector PDF. CSV/JSON tables contain full-precision
values and paired intervals. BRI follows the original natural-log best-so-far
rank trapezoidal formula; lower is better. The lower-middle median matches the
teammate's `torch.median` convention even for an even target count.

Their extra-search fusion figure is not applicable until we actually run fusion;
the reporter does not create data for an experiment we have not performed.

Five-thousand bootstrap intervals apply to final comparisons. The accumulating
figure uses 2,000 resamples per plotted prefix (as their figure does), with at
most 40 prefixes for scalability. These are pointwise descriptive intervals,
not an early-stopping or sequential-significance test. One-target pilots get no
bootstrap CI and must be treated as debugging output.

Equal graph/statistic formats do not make methods equally expensive: our three
branches and peer critique use different calls/searches than their five-question
VLM-probing pipeline. Keep target pool, target split, rounds, models, metrics and
cost accounting explicit in the report. Do not tune our prompts on their final
119 targets and later describe those targets as untouched held-out data for us.
