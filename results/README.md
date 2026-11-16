# Results

| Folder | Contents |
|---|---|
| [baseline/](baseline/) | Saved-query baseline on all 2,064 matched targets, 10 rounds: `caption_only`, `raw_dialogue` (caption + saved VisDial dialogue) and `supplied_plugir` (the PlugIR authors' saved queries). Recall@10 at round 10: 71.1 / 59.4 / 74.3. `final_metrics.csv`, `per_round.csv` and per-target ranks in `rank_comparison.csv`. |
| [conversations/](conversations/) | Compact live conversations: PlugIR + Qwen on targets 31–150 and 151–250, and end-to-end fact fusion on 151–250. |
| [fact_fusion/](fact_fusion/) | Post-hoc scoring of rewrite, caption mix and fact fusion; see [CAPTION_MIX_FACT_FUSION.md](../CAPTION_MIX_FACT_FUSION.md). |
| [fact_fusion_confirm/](fact_fusion_confirm/) | Pre-registered protocols, frozen script hash, and the end-to-end comparison. |

New runs write to `runs/` (git-ignored); copy only curated outputs here.
