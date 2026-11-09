"""Final results: agent v2 (probe_vlm) vs live PlugIR + Qwen, each with and without the caption mix.

Inputs are the finished live conversations copied from the GPU VM (INPUT); only targets finished by
both methods are compared. Ranks without the mix are the ones saved during the live run; ranks with
the mix are recomputed (0.7 * rewrite + 0.3 * original caption, normalized), which is exactly what a
live run with that display would show, because the display does not change the conversation.

Writes per-round metrics, per-target ranks, paired differences and agent behaviour to OUTPUT.
Usage: `PYTHONPATH=src python final_results.py`.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import random

import torch
from torch.nn.functional import normalize

from agentir.cache import load_cache, read_json
from agentir.retrieval import BlipRetriever, ranks_for_targets

# SETTINGS
INPUT = "vm_results/final_pull/runs/live_upstream"  # Full copy of the VM runs (Nov 9).
METHODS = {"plugir": ("0364674fdf0478954425", "355f786724fdc410f0f7"),
           "agent_v2": ("10db503e990c5e70feb9", "5064c43d1a55fc1d6877")}
CAPTION_SHARE = 0.3
ROUNDS = 11
KS = (1, 10, 100)
BOOTSTRAP = 5000
SEED = 42
OUTPUT = "results/agent_v2_vlm_probe"


def load(root, folders):
    out = {}
    for folder in folders:
        for path in (root / INPUT / folder / "dialogues").glob("*.json"):
            state = read_json(path)
            if state.get("status") == "complete" and len(state["states"]) >= ROUNDS:
                out[state["identity"]["target"]] = state
    return out


def bootstrap(difference, rng):
    n = len(difference)
    means = sorted(sum(difference[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return [round(100 * sum(difference) / n, 1), round(100 * means[int(0.025 * BOOTSTRAP)], 1),
            round(100 * means[int(0.975 * BOOTSTRAP)], 1)]


def main():
    root = Path(__file__).resolve().parent
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    retriever = BlipRetriever(cache, device="cpu", threads=4, local_files_only=True)
    path_to_id = {p: i for i, p in enumerate(cache.paths)}
    order = {p: i for i, p in enumerate(cache.paths)}

    conversations = {name: load(root, folders) for name, folders in METHODS.items()}
    targets = sorted(set.intersection(*(set(c) for c in conversations.values())), key=order.get)
    ids = [path_to_id[t] for t in targets]
    print(f"{len(targets)} targets finished by both methods.")

    ranks = {}
    for name, convs in conversations.items():
        saved = torch.tensor([[s["target_rank"] for s in convs[t]["states"][:ROUNDS]] for t in targets])
        caption = retriever.embed_texts([convs[t]["states"][0]["query"] for t in targets])
        mixed = []
        for r in range(ROUNDS):
            rewrite = retriever.embed_texts([convs[t]["states"][r]["query"] for t in targets])
            check = ranks_for_targets(retriever.score_vectors(rewrite), ids)
            assert torch.equal(check, saved[:, r]), f"{name} round {r}: recomputed ranks differ from the live run"
            blend = normalize((1 - CAPTION_SHARE) * rewrite + CAPTION_SHARE * caption, dim=-1)
            mixed.append(ranks_for_targets(retriever.score_vectors(blend), ids))
        ranks[f"{name}"] = saved
        ranks[f"{name}+mix"] = torch.stack(mixed, 1)

    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    # Per-round metrics.
    rows = []
    for key, R in ranks.items():
        for r in range(ROUNDS):
            row = {"method": key, "round": r}
            for k in KS:
                row[f"recall@{k}"] = round(100 * (R[:, r] <= k).float().mean().item(), 1)
            row["hits@10"] = round(100 * (R[:, :r + 1] <= 10).any(1).float().mean().item(), 1)
            row["median_rank"] = int(R[:, r].median().item())
            rows.append(row)
    with (out / "per_round.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Per-target ranks.
    with (out / "per_target_ranks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "method"] + [f"round_{r}" for r in range(ROUNDS)])
        for i, t in enumerate(targets):
            for key, R in ranks.items():
                writer.writerow([t, key] + R[i].tolist())

    # Paired differences at round 10 (and Recall@10 averaged over rounds 1-10).
    rng = random.Random(SEED)
    last = ROUNDS - 1
    at = lambda R, k: (R[:, last] <= k).float()
    hits = lambda R: (R <= 10).any(1).float()
    avg = lambda R: (R[:, 1:] <= 10).float().mean(1)
    comparisons = {"agent_v2 - plugir": ("agent_v2", "plugir"),
                   "plugir+mix - plugir": ("plugir+mix", "plugir"),
                   "agent_v2+mix - agent_v2": ("agent_v2+mix", "agent_v2"),
                   "agent_v2+mix - plugir+mix": ("agent_v2+mix", "plugir+mix")}
    differences = {}
    for label, (a, b) in comparisons.items():
        A, B = ranks[a], ranks[b]
        differences[label] = {
            "recall@1": bootstrap((at(A, 1) - at(B, 1)).tolist(), rng),
            "recall@10": bootstrap((at(A, 10) - at(B, 10)).tolist(), rng),
            "hits@10": bootstrap((hits(A) - hits(B)).tolist(), rng),
            "recall@10_avg_rounds_1_10": bootstrap((avg(A) - avg(B)).tolist(), rng),
            "final_rank_better_worse_same": [int((A[:, last] < B[:, last]).sum()), int((A[:, last] > B[:, last]).sum()),
                                             int((A[:, last] == B[:, last]).sum())]}
    # Agent v2 R@10 difference by target block (first 30 fresh targets vs the rest), to show the early lead.
    first_run = set(load(root, METHODS["plugir"][:1]))  # The first folder holds targets 31-60.
    first = torch.tensor([t in first_run for t in targets])
    blocks = {}
    for label, mask in (("targets_31_60", first), ("targets_61_on", ~first)):
        d = (at(ranks["agent_v2"][mask], 10) - at(ranks["plugir"][mask], 10)).tolist()
        blocks[label] = {"n": len(d), "recall@10_difference": bootstrap(d, rng)}

    # Agent behaviour from its selection logs.
    logs = [s["selection"] for t in targets for s in conversations["agent_v2"][t]["states"][1:ROUNDS] if s.get("selection")]
    chosen = [max((c["score"] for c in log["candidates"] if c["question"] == log["chosen"]), default=None) for log in logs]
    behaviour = {"rounds": len(logs),
                 "fallback_to_plugir": sum(log["fallback_to_plugir"] for log in logs),
                 "rounds_with_opinion_question_skipped": sum(bool(log["skipped_opinion"]) for log in logs),
                 "mean_factual_candidates": round(sum(len(log["candidates"]) for log in logs) / len(logs), 2),
                 "vlm_answers": sum(sum(c["answers"].values()) for log in logs for c in log["candidates"]),
                 "mean_chosen_entropy_bits": round(sum(c for c in chosen if c is not None) / sum(c is not None for c in chosen), 2)}

    summary = {"n_targets": len(targets), "caption_share": CAPTION_SHARE,
               "round_10": {key: next(r for r in rows if r["method"] == key and r["round"] == last) for key in ranks},
               "differences_round_10": differences, "agent_lead_by_block": blocks, "agent_behaviour": behaviour}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("\nRecall@10 by round:")
    for key in ranks:
        print(f"  {key:14}" + " ".join(f"{r['recall@10']:5.1f}" for r in rows if r["method"] == key))


if __name__ == "__main__":
    main()
