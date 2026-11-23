"""Every number in the paper's rewrite-fusion and vision re-ranking results, from committed files.

Reads only results/rewrite_fusion/ and results/vision_rerank/ (no BLIP cache, GPU or Ollama),
so it runs on any laptop in a second:

    python rewrite_vision_numbers.py

Confidence intervals are paired bootstrap 95% intervals (5,000 resamples, seed 42), the same
procedure as the live experiments. For the vision pilot, targets are resampled within their
BLIP-rank group and each group is weighted by its share of the 200-caption sample.
"""
import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
REWRITES = ROOT / "results" / "rewrite_fusion"
VISION = ROOT / "results" / "vision_rerank"
RESAMPLES = 5000
SEED = 42
YES_THRESHOLD = 5   # Rule 2 on the stored 0-10 scale (10 x P(yes)), i.e. P(yes) >= 0.5.


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def interval(values):
    """Point estimate and 95% range of the mean of per-target values (in points)."""
    rng = random.Random(SEED)
    n = len(values)
    boots = sorted(100 * sum(rng.choices(values, k=n)) / n for _ in range(RESAMPLES))
    return 100 * sum(values) / n, boots[int(0.025 * RESAMPLES) - 1], boots[int(0.975 * RESAMPLES) - 1]


def show(label, point, low, high):
    print(f"  {label}: {point:+.1f} [{low:+.1f}, {high:+.1f}]")


def rewrite_numbers():
    print("Fusing caption rewrites (round 0)")
    single = {r["method"]: float(r["recall@10"]) for r in read_csv(REWRITES / "strategies_summary.csv")}
    print(f"  caption alone R@10: {single['original caption']:.1f}")
    for name in ("paraphrase", "detail", "refocus", "decompose"):
        print(f"  one {name} rewrite (avg) R@10: {single[f'{name}: single variant (avg)']:.1f}")

    def change(rows, method, original, k=10):
        return [(int(r[method]) <= k) - (int(r[original]) <= k) for r in rows]

    sample = read_csv(REWRITES / "ranks_sample_200.csv")
    oracle = [int(r["ALL | best of all queries (oracle*)"]) <= 10 for r in sample]
    print(f"  oracle over all 21 queries R@10: {100 * sum(oracle) / len(oracle):.1f}")
    show("oracle over all 21 vs caption", *interval(change(sample, "ALL | best of all queries (oracle*)", "ALL | original")))
    show("all 21 fused (mean embedding, caption x3) vs caption",
         *interval(change(sample, "ALL | mean embedding (orig x3)", "ALL | original")))
    show("paraphrase z-max vs caption, 200 targets", *interval(change(sample, "paraphrase | z-max", "paraphrase | original")))
    full = read_csv(REWRITES / "ranks_paraphrase_all_2064.csv")
    show(f"paraphrase z-max vs caption, all {len(full)} targets",
         *interval(change(full, "paraphrase | z-max", "paraphrase | original")))


def vision_numbers():
    print("\nVision re-ranking (round 0, stratified pilot)")
    settings = json.loads((VISION / "shortlists_top30.json").read_text(encoding="utf-8"))["settings"]
    strata = {name: (low, high) for name, (low, high, _) in settings["strata"].items()}
    sample_ranks = settings["sample_ranks"]
    group = lambda rank: next((g for g, (low, high) in strata.items() if low <= rank <= high), None)
    size = {g: sum(group(r) == g for r in sample_ranks) for g in strata}
    rows = read_csv(VISION / "per_target.csv")
    for r in rows:
        r["group"] = group(int(r["blip_rank"]))
    by_group = {g: [r for r in rows if r["group"] == g] for g in strata}

    def weighted_change(groups, key, k):
        return 100 / len(sample_ranks) * sum(
            size[g] * sum((int(r[key]) <= k) - (int(r["blip_rank"]) <= k) for r in members) / len(members)
            for g, members in groups.items())

    for key, label in (("vlm_rank", "rule 1 (vision order)"), ("split_rank", "rule 2 (vision filter)")):
        for k in (1, 10):
            rng = random.Random(SEED)  # A fresh seed per estimate, so each number reproduces on its own.
            point = weighted_change(by_group, key, k)
            boots = sorted(weighted_change({g: rng.choices(m, k=len(m)) for g, m in by_group.items()}, key, k)
                           for _ in range(RESAMPLES))
            show(f"{label} R@{k} vs BLIP", point, boots[int(0.025 * RESAMPLES) - 1], boots[int(0.975 * RESAMPLES) - 1])
    for key, label in (("vlm_rank", "rule 1"), ("split_rank", "rule 2")):
        for g, members in by_group.items():
            before = sum(int(r["blip_rank"]) <= 10 for r in members)
            after = sum(int(r[key]) <= 10 for r in members)
            print(f"  {label}, BLIP {g}: top 10 {before} -> {after} of {len(members)}")
    crowded = sum(sum(s > 9.6 for s in json.loads(r["scores"])) >= 10 for r in rows)
    print(f"  shortlists where 10+ of the 30 photos scored above 0.96: {crowded} of {len(rows)}")

    # Prompt choice: the first 6 targets that `vision_rerank_experiment.py compare` uses
    # (round-robin over the rank groups), scored with the yes-probability prompt.
    answers = json.loads(next(VISION.glob("qwen2.5vl_7b_yes_prob_*.json")).read_text(encoding="utf-8"))
    targets = json.loads((VISION / "shortlists_top30.json").read_text(encoding="utf-8"))["targets"]
    mixed = [t for trio in zip(*[[t for t in targets if t["stratum"] == g] for g in strata]) for t in trio][:6]
    own = [answers[f"{t['caption']}\t{t['target']}"]["score"] for t in mixed]
    rest = [answers[f"{t['caption']}\t{i}"]["score"] for t in mixed for i in t["shortlist"] if i != t["target"]]
    print(f"  prompt check, 6 targets: target {sum(own) / len(own) / 10:.2f} vs others {sum(rest) / len(rest) / 10:.2f}")


if __name__ == "__main__":
    rewrite_numbers()
    vision_numbers()
