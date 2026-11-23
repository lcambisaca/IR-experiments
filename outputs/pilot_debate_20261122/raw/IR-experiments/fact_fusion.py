"""Fact fusion: does feeding the dialogue's answers to BLIP as separate facts beat one rewritten sentence?

No new dialogue and no GPU: the conversations already exist, and only the final search query changes,
so (as with the caption mix) every method below sees exactly the same questions and answers.

PRE-REGISTERED (fixed before any result was seen, Nov 12):
  Data      primary: the 119 live PlugIR + Qwen3-4B conversations (targets 31+, the reported baseline).
            secondary: PlugIR targets 1-30, and ES-EIG v2 targets 1-30 (finished ones).
  Facts     Qwen3-4B-Instruct-2507 (the live runs' checkpoint, local transformers, greedy) turns each question + answer into one short
            positive visual phrase, or NONE (non-answer, repeats the caption, absent / not visible, unsure).
            Phrases with a negation word are dropped. Fact query i = "<caption>, <fact i>".
  Methods   rewrite        PlugIR's rewritten sentence (as run)
            caption_mix    normalize(0.7 rewrite + 0.3 caption)          <- current best, the bar to beat
            history_mean   normalize(mean of the rewrites of rounds 0..t)
            facts_only     normalize(mean of caption and fact queries)   (no rewriter at all)
            rewrite_facts  normalize(0.7 rewrite + 0.3 mean(caption and fact queries))
                           (= caption_mix until the first fact)
            facts_rrf      reciprocal-rank fusion (k=60) of caption_mix and every fact query
  Pass      a method passes if its paired bootstrap 95% CI vs caption_mix on the 119 primary targets is
            above 0 for Recall@10 at round 10 OR for Recall@10 averaged over rounds 1-10. All six are reported.

Usage: `PYTHONPATH=src .venv/bin/python fact_fusion.py [name=/abs/run/folder ...]`.
"""
import csv
import json
from pathlib import Path
import random
import re

import torch
from torch.nn.functional import normalize

from agentir.cache import load_cache, read_json
from agentir.retrieval import BlipRetriever

# SETTINGS (pre-registered; do not tune on the primary targets)
PLUGIR_FOLDERS = ["vm_results/final_pull/runs/live_upstream/0364674fdf0478954425",
                  "vm_results/final_pull/runs/live_upstream/355f786724fdc410f0f7"]
MATCH_FOLDERS = ["vm_results/final_pull/runs/live_upstream/10db503e990c5e70feb9",   # agent_v2 (probe): only
                 "vm_results/final_pull/runs/live_upstream/5064c43d1a55fc1d6877"]   # to select the same 119
PLUGIR_1_30 = "results/live_experiments/conversations/plugir_qwen_targets_1_30.json"
CAPTION_SHARE, FACT_SHARE, RRF_K = 0.3, 0.3, 60
ROUNDS, BOOTSTRAP, SEED = 11, 5000, 42
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
def model_path():
    import os
    hubs = [Path(os.environ["HF_HOME"]) / "hub"] if os.environ.get("HF_HOME") else []
    hubs += [Path.home() / ".cache/huggingface/hub"]
    for hub in hubs:
        found = sorted((hub / "models--Qwen--Qwen3-4B-Instruct-2507/snapshots").glob("*/config.json"))
        if found:
            return found[0].parent
    raise FileNotFoundError("Qwen3-4B-Instruct-2507 is not in the local Hugging Face cache.")
FACT_BATCH, FACT_TOKENS = 16, 24
FACT_CACHE = "results/fact_fusion/facts_qwen3_4b_instruct_2507.json"
OUTPUT = "results/fact_fusion"

FACT_PROMPT = (
    "You turn one question and answer about a photo into a short visual fact for an image search engine. "
    "Reply with one short noun phrase (3 to 8 words) that names the object together with the new detail. "
    "Example: Question: What color is the wetsuit? Answer: white -> a white wetsuit. "
    "Reply NONE if the answer does not answer the question, only repeats the description, says something "
    "is absent or not visible, or is unsure. Reply with the phrase or NONE only.")
NEGATION = re.compile(r"\b(no|not|none|nothing|without|neither|nor|absent|n't)\b", re.I)


def clean_fact(text):
    text = text.strip().strip('"').splitlines()[0].strip().strip('"').rstrip(".").strip() if text.strip() else ""
    return None if not text or text.upper().startswith("NONE") or NEGATION.search(text) else text


def facts_for(conversations, store, path):
    """Extract (and cache) one fact per question + answer, batched, greedy, with the live runs' Qwen."""
    todo = [(c["caption"], s["question"], s["answer"] or "") for c in conversations
            for s in c["states"][1:ROUNDS] if s.get("question")]
    missing = [x for x in dict.fromkeys(todo) if json.dumps(x) not in store]
    print(f"Facts: {len(todo) - len(missing)} cached, {len(missing)} to extract with {MODEL}.", flush=True)
    if not missing:
        return store
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    MODEL_PATH = model_path()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float16).to(device).eval()
    for start in range(0, len(missing), FACT_BATCH):
        batch = missing[start:start + FACT_BATCH]
        prompts = [tok.apply_chat_template([{"role": "system", "content": FACT_PROMPT},
                                            {"role": "user", "content": f"Description: {c}\nQuestion: {q}\nAnswer: {a}"}],
                                           tokenize=False, add_generation_prompt=True) for c, q, a in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=FACT_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        for x, out in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            store[json.dumps(x)] = clean_fact(out)
        path.write_text(json.dumps(store, indent=1, ensure_ascii=False))
        print(f"  {min(start + FACT_BATCH, len(missing))}/{len(missing)}", flush=True)
    del model
    return store


def load_folder_runs(root, folders):
    out = {}
    for folder in folders:
        base = root / folder
        if not base.is_dir():  # On the GPU VM the runs live directly under runs/live_upstream.
            base = root / "runs/live_upstream" / Path(folder).name
        for p in (base / "dialogues").glob("*.json"):
            s = read_json(p)
            if s.get("status") == "complete" and len(s["states"]) >= ROUNDS:
                out[s["identity"]["target"]] = {"target": s["identity"]["target"],
                                                "caption": s["states"][0]["query"], "states": s["states"]}
    return out


def bootstrap(d, rng):
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return [round(100 * sum(d) / n, 1), round(100 * means[int(0.025 * BOOTSTRAP)], 1),
            round(100 * means[int(0.975 * BOOTSTRAP)], 1)]


def evaluate(name, convs, retriever, pid, store, rng, out):
    T = sorted(convs, key=pid.get)
    ids = torch.tensor([pid[t] for t in T])
    emb = lambda texts: retriever.embed_texts(list(texts))
    images = retriever.images.cpu()
    methods = ["rewrite", "caption_mix", "history_mean", "facts_only", "rewrite_facts", "facts_rrf"]
    ranks = {m: torch.zeros(len(T), ROUNDS, dtype=torch.long) for m in methods}
    n_facts = 0
    for i, t in enumerate(T):
        c = convs[t]
        cap = emb([c["caption"]])[0]
        rewrites = emb([s["query"] for s in c["states"][:ROUNDS]])
        facts = []
        for r in range(ROUNDS):
            if r and c["states"][r].get("question"):
                f = store.get(json.dumps((c["caption"], c["states"][r]["question"], c["states"][r]["answer"] or "")))
                if f:
                    facts.append(f"{c['caption']}, {f}")
            fact_vecs = emb(facts) if facts else torch.empty(0, cap.shape[0])
            anchor = torch.cat([cap[None], fact_vecs]).mean(0)
            mix = normalize((1 - CAPTION_SHARE) * rewrites[r] + CAPTION_SHARE * cap, dim=-1)
            q = {"rewrite": rewrites[r], "caption_mix": mix,
                 "history_mean": normalize(rewrites[:r + 1].mean(0), dim=-1),
                 "facts_only": normalize(anchor, dim=-1),
                 "rewrite_facts": normalize((1 - FACT_SHARE) * rewrites[r] + FACT_SHARE * anchor, dim=-1)}
            scores = images @ torch.stack(list(q.values())).T  # [50k, 5]
            for j, m in enumerate(q):
                ranks[m][i, r] = int((scores[:, j] > scores[ids[i], j]).sum()) + 1
            # RRF over caption_mix and each fact query (full rankings needed).
            lists = torch.cat([mix[None], fact_vecs]) if facts else mix[None]
            s = images @ lists.T
            pos = torch.empty_like(s, dtype=torch.long)
            pos.scatter_(0, s.argsort(0, descending=True), torch.arange(1, len(s) + 1)[:, None].expand_as(s))
            fused = (1.0 / (RRF_K + pos.float())).sum(1)
            ranks["facts_rrf"][i, r] = int((fused > fused[ids[i]]).sum()) + 1
        n_facts += len(facts)
    # Sanity: saved PlugIR ranks are reproduced (ties aside).
    saved = torch.tensor([[s["target_rank"] for s in convs[t]["states"][:ROUNDS]] for t in T])
    agree = (saved == ranks["rewrite"]).float().mean().item()

    rows = []
    for m, R in ranks.items():
        rows.append({"data": name, "method": m, "n": len(T),
                     **{f"R@10 r{r}": round(100 * (R[:, r] <= 10).float().mean().item(), 1) for r in (0, 2, 5, 10)},
                     "R@10 avg r1-10": round(100 * (R[:, 1:] <= 10).float().mean().item(), 1),
                     "R@1 r10": round(100 * (R[:, -1] <= 1).float().mean().item(), 1),
                     "Hits@10 r10": round(100 * (R <= 10).any(1).float().mean().item(), 1),
                     "median rank r10": int(R[:, -1].median().item())})
    base = ranks["caption_mix"]
    diffs = {}
    for m in methods:
        if m == "caption_mix":
            continue
        A = ranks[m]
        diffs[m] = {"R@10 r10": bootstrap(((A[:, -1] <= 10).float() - (base[:, -1] <= 10).float()).tolist(), rng),
                    "R@10 avg r1-10": bootstrap(((A[:, 1:] <= 10).float().mean(1) - (base[:, 1:] <= 10).float().mean(1)).tolist(), rng),
                    "better/worse/same r10": [int((A[:, -1] < base[:, -1]).sum()), int((A[:, -1] > base[:, -1]).sum()),
                                              int((A[:, -1] == base[:, -1]).sum())]}
    with (out / f"ranks_{name}.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["target", "method"] + [f"round_{r}" for r in range(ROUNDS)])
        for i, t in enumerate(T):
            for m, R in ranks.items():
                w.writerow([t, m] + R[i].tolist())
    print(f"\n== {name}: {len(T)} targets, {n_facts / len(T):.1f} facts per conversation, "
          f"saved-rank agreement {100 * agree:.1f}%")
    keys = [k for k in rows[0] if k not in ("data", "method", "n")]
    print(f"{'method':14s} " + " | ".join(keys))
    for row in rows:
        print(f"{row['method']:14s} " + " | ".join(f"{row[k]:>{len(k)}}" for k in keys))
    print("paired vs caption_mix: mean difference [95% CI], points")
    for m, d in diffs.items():
        print(f"  {m:14s} R@10 r10 {d['R@10 r10']}  R@10 avg {d['R@10 avg r1-10']}  better/worse/same {d['better/worse/same r10']}")
    return rows, diffs


def main():
    import sys
    root = Path(__file__).resolve().parent
    # Extra datasets: name=/absolute/run/folder (each folder holds dialogues/*.json).
    extra = dict(arg.split("=", 1) for arg in sys.argv[1:])
    config = read_json(root / "experiment.json")
    paths = {k: (root / v).resolve() for k, v in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    retriever = BlipRetriever(cache, device="cpu", threads=4, local_files_only=True)
    pid = {p: i for i, p in enumerate(cache.paths)}
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    plugir = load_folder_runs(root, PLUGIR_FOLDERS)
    matched = load_folder_runs(root, MATCH_FOLDERS)
    datasets = {"plugir_119": {t: c for t, c in plugir.items() if t in matched}}
    early = read_json(root / PLUGIR_1_30)["conversations"] if (root / PLUGIR_1_30).is_file() else []
    datasets["plugir_1_30"] = {c["target"]: {"target": c["target"], "caption": c["caption"], "states": c["states"]}
                               for c in early if len(c["states"]) >= ROUNDS}
    if not (root / PLUGIR_1_30).is_file():
        del datasets["plugir_1_30"]
    for name, folder in extra.items():
        datasets[name] = load_folder_runs(Path("/"), [str(Path(folder).resolve())[1:]])

    store_path = root / FACT_CACHE
    store = json.loads(store_path.read_text()) if store_path.is_file() else {}
    for convs in datasets.values():
        store = facts_for(list(convs.values()), store, store_path)
    store_path.write_text(json.dumps(store, indent=1, ensure_ascii=False))

    rng = random.Random(SEED)
    report = {}
    for name, convs in datasets.items():
        rows, diffs = evaluate(name, convs, retriever, pid, store, rng, out)
        report[name] = {"metrics": rows, "paired_vs_caption_mix": diffs}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
