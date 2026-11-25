# Results

| Folder | Contents |
|---|---|
| [baseline/](baseline/) | Saved-query baseline on all 2,064 matched targets, 10 rounds: `caption_only`, `raw_dialogue` (caption + saved VisDial dialogue) and `supplied_plugir` (the PlugIR authors' saved queries). Recall@10 at round 10: 71.1 / 59.4 / 74.3. `final_metrics.csv`, `per_round.csv` and per-target ranks in `rank_comparison.csv`. |
| [conversations/](conversations/) | Compact live conversations: PlugIR + Qwen on targets 31–150 and 151–250, end-to-end fact fusion on 151–250, and the answer-entropy agent on 31–149. |
| [agent_v2_vlm_probe/](agent_v2_vlm_probe/) | Answer-entropy agent vs live PlugIR on 119 targets (with and without the caption mix); see its README. |
| [fact_fusion/](fact_fusion/) | Post-hoc scoring of rewrite, caption mix and fact fusion; see [CAPTION_MIX_FACT_FUSION.md](../CAPTION_MIX_FACT_FUSION.md). |
| [fact_fusion_confirm/](fact_fusion_confirm/) | Pre-registered protocols, frozen script hash, and the end-to-end comparison. |
| [paraphrase/](paraphrase/) | Qwen3-4B paraphrases of every caption (5 each) and their round-0 ranks on all 2,064 targets. Reused by the rewrite-fusion study. |
| [rewrite_fusion/](rewrite_fusion/) | Round-0 rewrite strategies (detail, refocus, decompose) and fusion rules on the 200-caption sample. |
| [vision_rerank/](vision_rerank/) | BLIP top-30 shortlists, all Qwen2.5-VL-7B answers, and per-target ranks for the 47-target pilot. |

New runs write to `runs/` (git-ignored); copy only curated outputs here.
