"""End-to-end facts search vs live PlugIR on the same targets (see results/fact_fusion_confirm/PROTOCOL_E2E.md).

    PYTHONPATH=src python scripts/compare_e2e.py <plugir_run_folder> <e2e_run_folder> [posthoc_ranks.csv]

Primary (pre-registered): e2e facts (its own target_rank) minus PlugIR caption mix, Recall@10 at round 10,
paired bootstrap (5,000, seed 42); passes if the 95% CI lower bound > 0. Everything else is descriptive.
"""
import csv
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn.functional import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentir.cache import load_cache, read_json  # noqa: E402
from agentir.retrieval import BlipRetriever, ranks_for_targets  # noqa: E402

ROUNDS, BOOT, SEED, SHARE = 11, 5000, 42, 0.3


def load(folder):
    out = {}
    for p in (Path(folder) / "dialogues").glob("*.json"):
        s = read_json(p)
        if s.get("status") == "complete" and len(s["states"]) >= ROUNDS:
            out[s["identity"]["target"]] = s
    return out


def boot(d, rng):
    n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return [round(100 * sum(d) / n, 1), round(100 * m[int(.025 * BOOT)], 1), round(100 * m[int(.975 * BOOT)], 1)]


def main():
    root = Path(__file__).resolve().parents[1]
    base, e2e = load(sys.argv[1]), load(sys.argv[2])
    posthoc = {}
    if len(sys.argv) > 3 and Path(sys.argv[3]).is_file():
        for r in csv.DictReader(open(sys.argv[3])):
            if r["method"] == "facts_only":
                posthoc[r["target"]] = [int(r[f"round_{i}"]) for i in range(ROUNDS)]
    T = sorted(set(base) & set(e2e))
    print(f"PlugIR complete {len(base)}, e2e complete {len(e2e)}, common {len(T)}; "
          f"e2e incomplete: {sorted(set(base) - set(e2e))}")
    cfg = read_json(root / "experiment.json")
    paths = {k: (root / v).resolve() for k, v in cfg.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    ret = BlipRetriever(cache, device="cpu", threads=4, local_files_only=True)
    pid = {p: i for i, p in enumerate(cache.paths)}
    ids = [pid[t] for t in T]
    R = {"plugir_rewrite": torch.tensor([[s["target_rank"] for s in base[t]["states"][:ROUNDS]] for t in T])}
    cap = ret.embed_texts([base[t]["states"][0]["query"] for t in T])
    mix = []
    for r in range(ROUNDS):
        rw = ret.embed_texts([base[t]["states"][r]["query"] for t in T])
        assert torch.equal(ranks_for_targets(ret.score_vectors(rw), ids), R["plugir_rewrite"][:, r])
        mix.append(ranks_for_targets(ret.score_vectors(normalize((1 - SHARE) * rw + SHARE * cap, dim=-1)), ids))
    R["plugir_caption_mix"] = torch.stack(mix, 1)
    R["e2e_facts"] = torch.tensor([[s["target_rank"] for s in e2e[t]["states"][:ROUNDS]] for t in T])
    R["e2e_rewrite_only"] = torch.tensor([[s.get("rewrite_only_rank", s["target_rank"])
                                           for s in e2e[t]["states"][:ROUNDS]] for t in T])
    if posthoc and all(t in posthoc for t in T):
        R["posthoc_facts_on_plugir"] = torch.tensor([posthoc[t] for t in T])

    rows = {}
    for m, X in R.items():
        rows[m] = {**{f"R@10 r{r}": round(100 * (X[:, r] <= 10).float().mean().item(), 1) for r in (0, 2, 5, 10)},
                   "R@10 avg r1-10": round(100 * (X[:, 1:] <= 10).float().mean().item(), 1),
                   "R@1 r10": round(100 * (X[:, -1] <= 1).float().mean().item(), 1),
                   "Hits@10 r10": round(100 * (X <= 10).any(1).float().mean().item(), 1),
                   "median rank r10": int(X[:, -1].median().item()),
                   "R@10 by round": [round(100 * (X[:, r] <= 10).float().mean().item(), 1) for r in range(ROUNDS)]}
    rng = random.Random(SEED)
    A = R["e2e_facts"]
    diffs = {}
    for b in [k for k in R if k != "e2e_facts"]:
        B = R[b]
        diffs[f"e2e_facts - {b}"] = {
            "R@10 r10": boot(((A[:, -1] <= 10).float() - (B[:, -1] <= 10).float()).tolist(), rng),
            "R@10 avg r1-10": boot(((A[:, 1:] <= 10).float().mean(1) - (B[:, 1:] <= 10).float().mean(1)).tolist(), rng),
            "Hits@10": boot(((A <= 10).any(1).float() - (B <= 10).any(1).float()).tolist(), rng),
            "better/worse/same r10": [int((A[:, -1] < B[:, -1]).sum()), int((A[:, -1] > B[:, -1]).sum()),
                                      int((A[:, -1] == B[:, -1]).sum())]}
    primary = diffs["e2e_facts - plugir_caption_mix"]["R@10 r10"]
    facts_per_target = sum(len(e2e[t].get("facts", [])) for t in T) / max(len(T), 1)
    report = {"n_common": len(T), "metrics": rows, "paired": diffs, "facts_per_target": round(facts_per_target, 2),
              "primary": {"test": "e2e_facts - plugir_caption_mix, R@10 r10", "diff_ci": primary,
                          "passes": primary[1] > 0}}
    out = root / "results/fact_fusion_confirm/e2e_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    with open(out.with_name("e2e_ranks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "method"] + [f"round_{r}" for r in range(ROUNDS)])
        for i, t in enumerate(T):
            for m, X in R.items():
                w.writerow([t, m] + X[i].tolist())
    for m, row in rows.items():
        print(f"{m:26s} " + " | ".join(f"{k} {v}" for k, v in row.items() if k != "R@10 by round"))
    for k, v in diffs.items():
        print(f"{k:45s} {v}")
    print(f"facts per target {facts_per_target:.1f}\nPRIMARY {primary} -> {'PASS' if primary[1] > 0 else 'NOT PASSED'}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
