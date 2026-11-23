"""Matan's reranking agent (Nov 19 meeting): a vision model reads the dialogue so far and one photo,
and says accept / partial / reject.

    sanity  - per target, rounds 0/5/10: the target, 5 random photos, 5 near misses (top BLIP non-targets).
              Can the judge tell the target apart at all?
    choose  - pick the judge model with the best target-vs-near-miss separation (AUC of P(accept)).
    rerank  - judge PlugIR's top 50 at every round of saved live PlugIR conversations.
    score   - (CPU) Hits@10 / Recall@10 per round: PlugIR order vs reranked order.

Rerank rule fixed before any reranking ran (Matan's naive rule): accepted photos first, in PlugIR's
original order, then everything else in original order. Secondary rules: three tiers
(accept > partial > reject), and sorting by P(accept). Ranks below 50 are never changed.
Reranking only changes what the user is shown; PlugIR's next question comes from its top 100
clusters, not the shown order, so scoring reranks on saved conversations is exact.

    PYTHONPATH=src python scripts/vlm_judge.py sanity --model 3b --conversations results/conversations/plugir_qwen_targets_151_250.json
    PYTHONPATH=src python scripts/vlm_judge.py score --model 3b --dir results/vlm_rerank --output /tmp/summary.json  # CPU, saved run
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = {"3b": "Qwen/Qwen2.5-VL-3B-Instruct", "7b": "Qwen/Qwen2.5-VL-7B-Instruct"}
MAX_PIXELS = 256 * 28 * 28
DEPTH = 50
SANITY_ROUNDS = (0, 5, 10)
N_RANDOM = N_NEAR = 5
SEED = 42
LABELS = ("accept", "partial", "reject")
PROMPT = ("A user is searching a photo collection for one specific photo. Everything the user has told us so far:\n"
          "{dialogue}\n\n"
          "Is this photo the one the user is looking for? Reply with exactly one word:\n"
          "accept - it fits everything the user said,\n"
          "partial - it fits some of it, or you cannot tell,\n"
          "reject - it clearly contradicts what the user said.")
OUT = ROOT / "runs" / "vlm_judge"


def dialogue_text(conversation, round_index):
    """Caption plus the user's answers so far; never PlugIR's rewritten query."""
    lines = [f"Caption: {conversation['caption']}"]
    for state in conversation["states"][1:round_index + 1]:
        lines.append(f"Q: {state['question']}\nA: {state['answer']}")
    return "\n".join(lines)


class Judge:
    def __init__(self, key):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch, self.key = torch, key
        self.processor = AutoProcessor.from_pretrained(MODELS[key], max_pixels=MAX_PIXELS)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODELS[key], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to("cuda")
        ids = [self.processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in LABELS]
        assert len(set(ids)) == 3, ids
        self.ids = ids

    def __call__(self, dialogue, paths, batch=10):
        from PIL import Image
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": PROMPT.format(dialogue=dialogue)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = []
        for i in range(0, len(paths), batch):
            images = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.processor(text=[text] * len(images), images=images, padding=True,
                                    return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :].float()
            full = logits.softmax(-1)
            three = logits[:, self.ids].softmax(-1)
            for row, mass in zip(three.tolist(), full[:, self.ids].sum(-1).tolist()):
                out.append({"p_accept": row[0], "p_partial": row[1], "p_reject": row[2],
                            "label": LABELS[max(range(3), key=row.__getitem__)], "label_mass": mass})
        return out


def setup():
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from agentir.cache import load_cache, read_json
    from agentir.live import resolve_target_image
    from agentir.retrieval import BlipRetriever
    config = read_json(ROOT / "experiment.json")
    cache = load_cache((ROOT / config["cache"]).resolve(), (ROOT / config["manifest"]).resolve())
    retriever = BlipRetriever(cache, device="cuda", local_files_only=True)
    index = {p: i for i, p in enumerate(cache.paths)}
    image = lambda name: resolve_target_image(ROOT / "data" / "images", name, download=True)
    return retriever, index, image


def top_list(retriever, query, depth):
    scores = retriever.scores([query])[0]
    return scores, scores.topk(depth).indices.tolist()


def done_keys(path):
    if not path.is_file():
        return set()
    with open(path) as f:
        return {(r["target"], r["round"], r["image"]) for r in map(json.loads, f)}


def run(args):
    retriever, index, image = setup()
    judge = Judge(args.model)
    paths = retriever.cache.paths
    conversations = [c for f in args.conversations for c in json.load(open(ROOT / f))["conversations"]]
    out = OUT / f"{args.mode}_{args.model}{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done, started, judged, mismatches = done_keys(out), time.time(), 0, 0
    with open(out, "a") as sink:
        for n, conv in enumerate(conversations, 1):
            target = index[conv["target"]]
            rounds = SANITY_ROUNDS if args.mode == "sanity" else range(len(conv["states"]))
            for r in rounds:
                if r >= len(conv["states"]):
                    continue
                state = conv["states"][r]
                scores, top = top_list(retriever, state["query"], DEPTH)
                rank = int((scores > scores[target]).sum()) + 1
                mismatches += rank != state["target_rank"]
                if args.mode == "sanity":
                    near = [i for i in top if i != target][:N_NEAR]
                    rng = random.Random(f"{SEED}-{conv['target']}-{r}")
                    pool = set(top) | {target}
                    rand = []
                    while len(rand) < N_RANDOM:
                        i = rng.randrange(len(paths))
                        if i not in pool and i not in rand:
                            rand.append(i)
                    items = [(target, "target", rank)] + [(i, "near_miss", top.index(i) + 1) for i in near] \
                        + [(i, "random", None) for i in rand]
                else:
                    items = [(i, "target" if i == target else "other", k + 1) for k, i in enumerate(top)]
                items = [it for it in items if (conv["target"], r, paths[it[0]]) not in done]
                if not items:
                    continue
                with ThreadPoolExecutor(8) as pool:
                    files = list(pool.map(lambda it: image(paths[it[0]]), items))
                verdicts = judge(dialogue_text(conv, r), files)
                for (i, kind, blip_rank), v in zip(items, verdicts):
                    sink.write(json.dumps({"target": conv["target"], "round": r, "image": paths[i], "kind": kind,
                        "blip_rank": blip_rank, "target_blip_rank": rank, "saved_target_rank": state["target_rank"],
                        "model": args.model, **v}) + "\n")
                judged += len(items)
            sink.flush()
            rate = judged / max(time.time() - started, 1)
            print(f"{time.strftime('%H:%M:%S')} {args.mode} {args.model} target {n}/{len(conversations)} "
                  f"judged {judged} ({rate:.1f}/s) rank mismatches {mismatches}", flush=True)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def sanity_table(model):
    rows = [json.loads(l) for l in open(OUT / f"sanity_{model}.jsonl")]
    table = {}
    for kind in ("target", "near_miss", "random"):
        sub = [r for r in rows if r["kind"] == kind]
        table[kind] = {lab: round(100 * sum(r["label"] == lab for r in sub) / max(len(sub), 1), 1) for lab in LABELS}
        table[kind]["n"] = len(sub)
    p = lambda kind: [r["p_accept"] for r in rows if r["kind"] == kind]
    table["auc_target_vs_near"] = round(auc(p("target"), p("near_miss")), 3)
    table["auc_target_vs_random"] = round(auc(p("target"), p("random")), 3)
    return table


def choose(_args):
    tables = {m: sanity_table(m) for m in MODELS if (OUT / f"sanity_{m}.jsonl").is_file()}
    best = max(tables, key=lambda m: (tables[m]["auc_target_vs_near"], m == "3b"))
    # Prefer the cheaper 3B unless 7B separates clearly better (fixed before seeing results).
    if best == "7b" and "3b" in tables and tables["7b"]["auc_target_vs_near"] < tables["3b"]["auc_target_vs_near"] + 0.03:
        best = "3b"
    json.dump({"tables": tables, "chosen": best}, open(OUT / "choice.json", "w"), indent=1)
    print(json.dumps({"tables": tables, "chosen": best}, indent=1))


def rerank_rank(items, rule):
    """New 1-based position of the target among PlugIR's top 50 (items sorted by blip_rank)."""
    if rule == "accept_first":
        key = lambda r: (r["label"] != "accept", r["blip_rank"])
    elif rule == "three_tier":
        key = lambda r: (LABELS.index(r["label"]), r["blip_rank"])
    else:
        key = lambda r: (-r["p_accept"], r["blip_rank"])
    order = sorted(items, key=key)
    return next(k for k, r in enumerate(order, 1) if r["kind"] == "target")


def score(args):
    path = OUT / f"rerank_{args.model}{args.tag}.jsonl"
    if not path.is_file() and path.with_suffix(".jsonl.gz").is_file():
        import gzip
        rows = [json.loads(l) for l in gzip.open(path.with_suffix(".jsonl.gz"), "rt")]
    else:
        rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault((r["target"], r["round"]), []).append(r)
    targets = sorted({t for t, _ in by})
    rounds = sorted({r for _, r in by})
    complete = [t for t in targets if all(len(by.get((t, r), [])) == DEPTH for r in rounds)]
    rules = ["plugir", "accept_first", "three_tier", "p_accept"]
    ranks = {rule: {} for rule in rules}
    for t in complete:
        for r in rounds:
            items = by[(t, r)]
            base = items[0]["target_blip_rank"]
            ranks["plugir"][(t, r)] = base
            for rule in rules[1:]:
                ranks[rule][(t, r)] = rerank_rank(items, rule) if base <= DEPTH else base
    lines = []
    for rule in rules:
        hits = [100 * sum(any(ranks[rule][(t, q)] <= 10 for q in rounds if q <= r) for t in complete) / len(complete)
                for r in rounds]
        rec = [100 * sum(ranks[rule][(t, r)] <= 10 for t in complete) / len(complete) for r in rounds]
        r1 = [100 * sum(ranks[rule][(t, r)] == 1 for t in complete) / len(complete) for r in rounds]
        lines.append({"rule": rule, "hits10": hits, "recall10": rec, "recall1": r1})
    # Paired bootstrap on Hits@10 at the last round, accept_first minus PlugIR.
    rng = random.Random(SEED)
    last = rounds[-1]
    hit = lambda rule, t: any(ranks[rule][(t, q)] <= 10 for q in rounds)
    diff = [hit("accept_first", t) - hit("plugir", t) for t in complete]
    boots = sorted(sum(rng.choice(diff) for _ in diff) * 100 / len(diff) for _ in range(5000))
    by_round = []
    for r in rounds:
        d = [any(ranks["accept_first"][(t, q)] <= 10 for q in rounds if q <= r)
             - any(ranks["plugir"][(t, q)] <= 10 for q in rounds if q <= r) for t in complete]
        b = sorted(sum(rng.choice(d) for _ in d) * 100 / len(d) for _ in range(2000))
        by_round.append([r, round(100 * sum(d) / len(d), 1), round(b[50], 1), round(b[1949], 1)])
    summary = {"model": args.model, "hits10_diff_by_round [round, mean, 2.5%, 97.5%]": by_round, "targets": len(complete), "rounds": rounds, "table": lines,
               "accept_first_minus_plugir_hits10_final": [round(100 * sum(diff) / len(diff), 1),
                                                         round(boots[125], 1), round(boots[4874], 1)],
               "only_rerank_hits": sum(d == 1 for d in diff), "only_plugir_hits": sum(d == -1 for d in diff),
               "label_rates_all_photos": {lab: round(100 * sum(r["label"] == lab for r in rows) / len(rows), 1) for lab in LABELS},
               "target_label_rates": {lab: round(100 * sum(r["label"] == lab for r in rows if r["kind"] == "target")
                                                  / max(1, sum(r["kind"] == "target" for r in rows)), 1) for lab in LABELS}}
    dest = Path(args.output) if args.output else OUT / f"rerank_{args.model}{args.tag}_summary.json"
    json.dump(summary, open(dest, "w"), indent=1)
    print(f"{len(complete)} complete targets, last round {last}")
    for line in lines:
        print(f"{line['rule']:>13}  H@10 " + " ".join(f"{x:5.1f}" for x in line["hits10"]))
        print(f"{'':>13}  R@10 " + " ".join(f"{x:5.1f}" for x in line["recall10"]))
    print("accept_first - plugir H@10 by round [round, mean, 2.5%, 97.5%]:", by_round)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sanity", "choose", "rerank", "score"])
    parser.add_argument("--model", choices=list(MODELS), default="3b")
    parser.add_argument("--conversations", nargs="+", default=["results/conversations/plugir_qwen_targets_151_250.json"])
    parser.add_argument("--output", default="")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. _qwenuser")
    parser.add_argument("--dir", default="", help="judgement folder (default runs/vlm_judge); results/vlm_rerank has the saved run")
    args = parser.parse_args()
    if args.dir:
        OUT = ROOT / args.dir
    {"sanity": run, "rerank": run, "choose": choose, "score": score}[args.mode](args)
