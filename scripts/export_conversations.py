"""Export the live conversations into compact JSON files that can be committed and analysed.

The raw checkpoints (one JSON per target, ~800 KB each, mostly model-call logs and RNG state) stay
local in vm_results/. This keeps, for every target and round: the description searched (rewrite),
the question asked, BLIP-2's answer, the candidate questions Qwen proposed, the captions of the
cluster representatives shown to Qwen, the target's rank, and (agent runs) the agent's selection log.

Usage: `python scripts/export_conversations.py` from the repo root.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vm_results/final_pull/runs/live_upstream"
OUTPUT = ROOT / "results/live_experiments/conversations"
RUNS = {  # file name: (folders, description)
    "plugir_qwen_targets_1_30": (["1237be4ebfc611a85a2e"], "Live PlugIR + Qwen3-4B, targets 1-30"),
    "agent_v1_probe_split_targets_1_30": (["f7cf7e3550889608e932"], "Agent v1 (probe_split), targets 1-30"),
    "plugir_qwen_targets_31_150": (["0364674fdf0478954425", "355f786724fdc410f0f7"], "Live PlugIR + Qwen3-4B, targets 31-150"),
    "agent_v2_probe_vlm_targets_31_149": (["10db503e990c5e70feb9", "5064c43d1a55fc1d6877"], "Agent v2 (probe_vlm), targets 31-149"),
}
KEEP = ("round", "query", "question", "answer", "candidate_questions", "candidate_caption_prompt", "target_rank", "selection")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (folders, description) in RUNS.items():
        records = []
        for folder in folders:
            for path in sorted((SOURCE / folder / "dialogues").glob("*.json")):
                raw = json.loads(path.read_text())
                if raw.get("status") != "complete":
                    continue
                identity = raw["identity"]
                records.append({"target": identity["target"], "caption": identity["initial_caption"],
                                "selection": identity.get("selection", "plugir_kl"),
                                "states": [{k: s[k] for k in KEEP if k in s} for s in raw["states"]]})
        records.sort(key=lambda r: r["target"])
        (OUTPUT / f"{name}.json").write_text(json.dumps({"description": description, "n_targets": len(records),
                                                         "conversations": records}, indent=1, ensure_ascii=False))
        print(f"{name}: {len(records)} conversations")


if __name__ == "__main__":
    main()
