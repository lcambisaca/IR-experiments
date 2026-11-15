"""Why does facts-driven search inside the loop not beat facts as the final ranking? Descriptive diagnostics.

    python scripts/diagnose_e2e.py <plugir_run_folder> <e2e_run_folder> <fact_cache.json>

Compares the two conversations on the same (complete) targets: question repeats, PlugIR's fallback question,
caption-echo answers, fact yield and novelty, and turnover of the candidate captions that seed questions.
"""
import json
from pathlib import Path
import re
import sys

STOP = set("a an the of in on at with and or is are to its it this that there image photo picture".split())
words = lambda t: {w for w in re.findall(r"[a-z]+", (t or "").lower()) if w not in STOP}


def load(folder):
    out = {}
    for p in (Path(folder) / "dialogues").glob("*.json"):
        s = json.loads(p.read_text())
        if s.get("status") == "complete" and len(s["states"]) >= 11:
            out[s["identity"]["target"]] = s
    return out


def stats(convs, fact_of):
    n = rep = other = echo = facts = novel = 0
    turnover, cand_q = [], []
    for s in convs:
        caption = s["states"][0]["query"]
        seen_q, known = set(), words(caption)
        prev = None
        for r in s["states"][1:11]:
            n += 1
            q = re.sub(r"\W+", " ", r["question"].lower()).strip()
            rep += q in seen_q
            seen_q.add(q)
            other += "other object" in q
            a = words(r["answer"])
            echo += bool(a) and len(a & words(caption)) / len(a) >= 0.8
            f = fact_of(s, r)
            if f:
                facts += 1
                new = words(f) - known
                novel += bool(new)
                known |= words(f)
            cands = {c.strip() for c in (r.get("candidate_caption_prompt") or "").split("\n") if c.strip()}
            if prev is not None and (cands or prev):
                turnover.append(1 - len(cands & prev) / max(len(cands | prev), 1))
            prev = cands
            cand_q.append(len({x.lower() for x in r.get("candidate_questions", [])}))
    pct = lambda x: round(100 * x / n, 1)
    return {"rounds": n, "repeated question %": pct(rep), "'other object' fallback %": pct(other),
            "caption-echo answer %": pct(echo), "rounds giving a fact %": pct(facts),
            "facts adding new words %": round(100 * novel / max(facts, 1), 1),
            "candidate-caption turnover per round %": round(100 * sum(turnover) / max(len(turnover), 1), 1),
            "distinct candidate questions per round": round(sum(cand_q) / max(len(cand_q), 1), 2)}


def main():
    base, e2e = load(sys.argv[1]), load(sys.argv[2])
    cache = json.loads(Path(sys.argv[3]).read_text())
    T = sorted(set(base) & set(e2e))
    print(f"{len(T)} common complete targets")
    plug_fact = lambda s, r: cache.get(json.dumps((s["states"][0]["query"], r["question"], r["answer"] or "")))
    rows = {"PlugIR conversations": stats([base[t] for t in T], plug_fact),
            "facts-driven conversations": stats([e2e[t] for t in T], lambda s, r: r.get("fact"))}
    keys = list(next(iter(rows.values())))
    print(f"{'':40s}" + "".join(f"{k:>30s}" for k in rows))
    for k in keys:
        print(f"{k:40s}" + "".join(f"{rows[m][k]:>30}" for m in rows))
    # Same questions? How often the two loops ask the identical question in the same round.
    same = sum(re.sub(r"\W+", " ", base[t]["states"][r]["question"].lower()) ==
               re.sub(r"\W+", " ", e2e[t]["states"][r]["question"].lower()) for t in T for r in range(1, 11))
    print(f"identical question in the same round: {100 * same / (10 * len(T)):.1f}%")
    first_diff = [next((r for r in range(1, 11) if base[t]["states"][r]["question"] != e2e[t]["states"][r]["question"]), 11)
                  for t in T]
    print(f"first round where the questions differ (median): {sorted(first_diff)[len(first_diff) // 2]}")


if __name__ == "__main__":
    main()
