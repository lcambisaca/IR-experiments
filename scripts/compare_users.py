"""Hits@1 / Hits@10 / Recall@10 per round: PlugIR with the BLIP-2 user vs the Qwen2.5-VL user (CPU, saved runs).

    python scripts/compare_users.py
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1] / "results" / "conversations"
BLIP2 = ROOT / "plugir_qwen_targets_151_250.json"
QWEN = ROOT / "plugir_qwen_qwenuser_targets_151_250.json"


def ranks(path):
    return {c["target"]: [s["target_rank"] for s in c["states"]] for c in json.load(open(path))["conversations"]}


def hits(rows, k, t):
    """Cumulative: the target reached the top k at any round up to t."""
    return 100 * sum(any(x <= k for x in r[:t + 1]) for r in rows) / len(rows)


def main():
    blip2, qwen = ranks(BLIP2), ranks(QWEN)
    targets = sorted(set(blip2) & set(qwen))
    rounds = range(11)
    print(f"{len(targets)} matched targets")
    for name, d in (("BLIP-2 user", blip2), ("Qwen user", qwen)):
        rows = [d[t] for t in targets]
        print(f"{name:>12} Hits@10   " + " ".join(f"{hits(rows, 10, t):5.1f}" for t in rounds))
        print(f"{'':>12} Hits@1    " + " ".join(f"{hits(rows, 1, t):5.1f}" for t in rounds))
        print(f"{'':>12} Recall@10 " + " ".join(f"{100 * sum(r[t] <= 10 for r in rows) / len(rows):5.1f}" for t in rounds))
    diff = [any(x <= 10 for x in qwen[t]) - any(x <= 10 for x in blip2[t]) for t in targets]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diff) for _ in diff) * 100 / len(diff) for _ in range(4000))
    print(f"Hits@10 round 10, Qwen - BLIP-2: {100 * sum(diff) / len(diff):+.1f} (95% CI {boots[100]:+.1f} to {boots[3899]:+.1f})")


if __name__ == "__main__":
    main()
