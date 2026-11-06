"""Stage A: Matan's extra-search strategies on live PlugIR + Qwen conversations.

Each round of a live conversation has a current description (round 0 = the caption, later rounds =
Qwen's rewrite after BLIP-2's answer). For every round and strategy, Qwen (the same pinned model that
ran the conversation) writes 5 variant searches; together with the original description that makes
k = 6 searches over all 50,000 photos. Their top-10 lists are merged (max score, mean score) into the
top 10 the user would see.

Showing the merged list instead of the original does not change PlugIR's questions (PlugIR picks them
from its own search), so applying the strategies to finished live conversations gives exactly what a
live run with this display would give.

Reported per round, for each strategy:
  - Recall@1/10 and Hits@10 of the merged top 10 vs the original top 10 (same number of photos shown);
  - Matan's "winning version" (target in any search's top 10) next to the fair control (the original
    search showing as many photos as the pooled lists hold).
Nothing is tuned: k, the strategies and the merge rules are fixed in advance from Matan's email.

Usage: edit SETTINGS, then `python matan_strategies.py`.
"""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import random
import re

import torch

from agentir.cache import load_cache, read_json
from agentir.evaluation import matched_queries
from agentir.extra_searches import MERGES, N_VARIANTS, STRATEGIES, messages, parse_variants, round_outcome
from agentir.retrieval import BlipRetriever

# SETTINGS
CONVERSATIONS = "vm_results/1237be4ebfc611a85a2e/dialogues"  # Live PlugIR + Qwen checkpoints (one JSON per target).
N_TARGETS = None       # None = every complete conversation in the folder.
ROUNDS = 11            # States 0..10 (caption + ten rounds); fewer for a quick check.
LIVE_DEVICE = "cuda"   # Qwen device: "cuda" on the GPU machine, "mps" on a Mac.
OFFLINE = True         # Use already-downloaded model files.
MAX_ATTEMPTS = 3       # Retries when Qwen's reply is unreadable; the first attempt is greedy (temperature 0).
BOOTSTRAP = 5000
SEED = 42
OUTPUT = "runs/extra_searches"


def normalize_text(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def rephrase(qwen, strategy, description, attempt):
    """One Qwen call; the function name is recorded as the call's stage."""
    options = {"temperature": 0} if attempt == 0 else {"temperature": 0.7, "top_p": 0.8}
    response = qwen.create(model="local", messages=messages(strategy, description), max_tokens=400, n=1, **options)
    return response["choices"][0]["message"]["content"]


def variants_for(qwen, target, round_index, strategy, description):
    qwen.begin_round(round_index, f"{target}|{strategy}")
    for attempt in range(MAX_ATTEMPTS):
        items = [v for v in parse_variants(strategy, rephrase(qwen, strategy, description, attempt))
                 if normalize_text(v) != normalize_text(description)]
        if len(items) >= 2:
            return items, attempt
    return items, MAX_ATTEMPTS


def load_conversations(root, pairs):
    """Complete live conversations, in the matched-target order used by every other run."""
    by_target = {}
    for path in (root / CONVERSATIONS).glob("*.json"):
        state = read_json(path)
        if state.get("status") == "complete" and len(state["states"]) >= ROUNDS:
            by_target[state["identity"]["target"]] = [s["query"] for s in state["states"][:ROUNDS]]
    ordered = [(raw["img"], by_target[raw["img"]]) for raw, _ in pairs if raw["img"] in by_target]
    return ordered[:N_TARGETS] if N_TARGETS else ordered


def bootstrap(difference, rng):
    n = len(difference)
    means = [sum(difference[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP)]
    means.sort()
    return 100 * sum(difference) / n, 100 * means[int(0.025 * BOOTSTRAP)], 100 * means[int(0.975 * BOOTSTRAP)]


def main():
    root = Path(__file__).resolve().parent
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]), read_json(paths["supplied_queries"]), cache.paths)
    conversations = load_conversations(root, pairs)
    print(f"{len(conversations)} live conversations, {ROUNDS} states, strategies {STRATEGIES}, k = {1 + N_VARIANTS}.")
    retriever = BlipRetriever(cache, device="cpu", threads=4, local_files_only=OFFLINE)
    path_to_id = {p: i for i, p in enumerate(cache.paths)}

    from agentir.live_models import Qwen3
    qwen = Qwen3(root / OUTPUT / "qwen_calls", device=LIVE_DEVICE, offline=OFFLINE, seed=1021)
    rows, variants = [], {}
    try:
        for number, (target, descriptions) in enumerate(conversations, 1):
            target_id = path_to_id[target]
            for round_index, description in enumerate(descriptions):
                for strategy in STRATEGIES:
                    items, retries = variants_for(qwen, target, round_index, strategy, description)
                    variants[f"{target}|{round_index}|{strategy}"] = items
                    vectors = retriever.embed_texts([description] + items)
                    outcome = round_outcome(vectors @ retriever.images.T, target_id)
                    rows.append({"target": target, "round": round_index, "strategy": strategy,
                                 "n_variants": len(items), "retries": retries, **outcome})
            print(f"  {number}/{len(conversations)} conversations done", flush=True)
    finally:
        qwen.park()

    destination = root / OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True)
    with (destination / "per_round.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "variants.json").write_text(json.dumps(variants, indent=1, ensure_ascii=False))

    # Summary: rates per round, cumulative Hits@10, and paired differences against the original top 10.
    rng = random.Random(SEED)
    targets = [t for t, _ in conversations]
    found = lambda rank, k: rank is not None and rank <= k
    summary = []
    print(f"\n{'strategy':<8}{'round':>6}  R@10 orig/max/mean   H@10 orig/max/mean   any-top10 / same-count control")
    for strategy in STRATEGIES:
        table = {(r["target"], r["round"]): r for r in rows if r["strategy"] == strategy}
        hits = {method: {t: False for t in targets} for method in ("original",) + MERGES}
        for round_index in range(ROUNDS):
            line = {"strategy": strategy, "round": round_index}
            for method in ("original",) + MERGES:
                ranks = [table[(t, round_index)][method] for t in targets]
                for t, rank in zip(targets, ranks):
                    hits[method][t] |= found(rank, 10)
                line[f"recall@1_{method}"] = round(100 * sum(found(r, 1) for r in ranks) / len(targets), 1)
                line[f"recall@10_{method}"] = round(100 * sum(found(r, 10) for r in ranks) / len(targets), 1)
                line[f"hits@10_{method}"] = round(100 * sum(hits[method].values()) / len(targets), 1)
            line["any_top10"] = round(100 * sum(table[(t, round_index)]["any_top10"] for t in targets) / len(targets), 1)
            line["control_same_count"] = round(100 * sum(table[(t, round_index)]["control_same_count"] for t in targets) / len(targets), 1)
            line["mean_pool_size"] = round(sum(table[(t, round_index)]["pool_size"] for t in targets) / len(targets), 1)
            summary.append(line)
            print(f"{strategy:<8}{round_index:>6}  {line['recall@10_original']:5.1f}/{line['recall@10_max']:5.1f}/{line['recall@10_mean']:5.1f}"
                  f"    {line['hits@10_original']:5.1f}/{line['hits@10_max']:5.1f}/{line['hits@10_mean']:5.1f}"
                  f"    {line['any_top10']:5.1f} / {line['control_same_count']:5.1f}")
        last = ROUNDS - 1
        for method in MERGES:
            difference = [found(table[(t, last)][method], 10) - found(table[(t, last)]["original"], 10) for t in targets]
            print(f"  {strategy} {method}: Recall@10 change at round {last} = %+.1f [%+.1f, %+.1f]" % bootstrap(difference, rng))
        difference = [table[(t, last)]["any_top10"] - table[(t, last)]["control_same_count"] for t in targets]
        print(f"  {strategy} any-top10 minus same-count control at round {last} = %+.1f [%+.1f, %+.1f]" % bootstrap(difference, rng))
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved to {destination}")


if __name__ == "__main__":
    main()
