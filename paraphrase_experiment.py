"""Paraphrase experiment: does rewriting the initial caption help BLIP find the target?

Open this file in VS Code, check the settings below, then click Run.
Follows the plan agreed with our mentor (Oct 23): a fixed random sample of captions,
5 paraphrases each from ONE local open-source model (Qwen via Ollama), searched
against all 50,000 saved image vectors. No image encoding, no paid API calls.

Requires the Ollama app running locally with the model pulled:
    ollama serve            # in a separate terminal, if not already running
    ollama pull qwen3:4b
Paraphrases are saved under data/paraphrases/ and reused on later runs.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time
import unicodedata
import urllib.request

import torch

from agentir.cache import load_cache, read_json
from agentir.evaluation import matched_queries
from agentir.retrieval import BlipRetriever, ranks_for_targets


# SETTINGS: keep these fixed when comparing runs across teammates.
N_EXAMPLES = 100        # Random captions from the 2,064 matched targets; None for all.
SEED = 42               # Fixes which examples are sampled and the model's sampling.
N_PARAPHRASES = 5
MODEL = "qwen3:4b"      # One agent model for every experiment (mentor's request).
TEMPERATURE = 0.7       # Some variety between the 5 rewrites; seed keeps it repeatable.
K_VALUES = [1, 10, 100]
RRF_CONSTANT = 60       # Reciprocal Rank Fusion smoothing; not a top-K cutoff.
DEVICE = "cpu"
CPU_THREADS = 4
OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = (
    "You rewrite image descriptions for an image search engine. Write exactly {n} different "
    "paraphrases of the user's description. Each paraphrase must keep every fact: objects, counts, "
    "colors, attributes, actions, prepositions, spatial relationships, negations and uncertainty. "
    "Add nothing new, do not guess details, and do not drop anything. Vary wording and sentence "
    "structure. Each paraphrase is one plain description, not a question or a list."
)


def normalize_text(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def warning_flags(original, candidate):
    """Cheap checks borrowed from the Colab pilot; flags are recorded, not filtered."""
    flags = []
    words, other = normalize_text(original).split(), normalize_text(candidate).split()
    names = dict(zip("zero one two three four five six seven eight nine ten".split(), map(str, range(11))))
    numbers = lambda ws: Counter(names.get(w, w) for w in ws if w in names or w.isdigit())
    if numbers(words) != numbers(other):
        flags.append("numbers_changed")
    palette = set("black white red blue green yellow orange purple pink brown gray grey beige silver gold".split())
    colors = lambda ws: Counter("gray" if w == "grey" else w for w in ws if w in palette)
    if colors(words) != colors(other):
        flags.append("colors_changed")
    negations = set("no not without never none nor".split())
    if bool(set(words) & negations) != bool(set(other) & negations):
        flags.append("negation_changed")
    if normalize_text(original) == normalize_text(candidate):
        flags.append("identical_to_original")
    return flags


def ask_model(caption, attempt=0):
    """One local Ollama request returning exactly N_PARAPHRASES strings, or None."""
    schema = {"type": "object", "required": ["paraphrases"], "properties": {"paraphrases": {
        "type": "array", "items": {"type": "string"},
        "minItems": N_PARAPHRASES, "maxItems": N_PARAPHRASES}}}
    body = {"model": MODEL, "think": False, "stream": False, "format": schema,
            "options": {"temperature": TEMPERATURE, "seed": SEED + attempt},
            "messages": [{"role": "system", "content": PROMPT.format(n=N_PARAPHRASES)},
                         {"role": "user", "content": caption}]}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content = json.load(response)["message"]["content"]
        items = json.loads(content)["paraphrases"]
    except Exception as error:  # Local model: a failed attempt costs nothing, so we retry once.
        print(f"  attempt {attempt + 1} failed: {error}", flush=True)
        return None
    items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    return items if len(items) == N_PARAPHRASES else None


def generate_paraphrases(captions, store_path):
    """Reuse saved paraphrases; ask the model only for captions not yet stored."""
    store = read_json(store_path) if store_path.is_file() else {}
    missing = [c for c in dict.fromkeys(captions) if c not in store]
    print(f"Paraphrases: {len(captions) - len(missing)} reused, {len(missing)} to generate with {MODEL}.")
    started = time.perf_counter()
    for i, caption in enumerate(missing, 1):
        items = ask_model(caption) or ask_model(caption, attempt=1)
        store[caption] = items  # None records a failure; that example stays original-only.
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
        if i % 10 == 0 or i == len(missing):
            print(f"  {i}/{len(missing)} generated ({time.perf_counter() - started:.0f}s)", flush=True)
    return store


def positions(scores):
    """Zero-based rank of every image for each query row."""
    ordering = torch.argsort(scores, dim=1, descending=True, stable=True)
    result = torch.empty_like(ordering)
    result.scatter_(1, ordering, torch.arange(scores.shape[1]).expand_as(ordering))
    return result


def recall(ranks, k):
    return 100 * sum(r <= k for r in ranks) / len(ranks)


def main():
    root = Path(__file__).resolve().parent
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]),
                            read_json(paths["supplied_queries"]), cache.paths)
    indices = sorted(random.Random(SEED).sample(range(len(pairs)), N_EXAMPLES)) if N_EXAMPLES else list(range(len(pairs)))
    examples = [(i, pairs[i][0]["dialog"][0], pairs[i][0]["img"]) for i in indices]
    print(f"Evaluating {len(examples)}/{len(pairs)} captions against {len(cache.paths):,} images.")

    prompt_id = hashlib.sha256(json.dumps([PROMPT, N_PARAPHRASES, TEMPERATURE, SEED]).encode()).hexdigest()[:12]
    store_path = root / "data" / "paraphrases" / f"{MODEL.replace(':', '_')}_{prompt_id}.json"
    store = generate_paraphrases([caption for _, caption, _ in examples], store_path)

    retriever = BlipRetriever(cache, device=DEVICE, threads=CPU_THREADS)
    path_to_id = {p: i for i, p in enumerate(cache.paths)}
    rows = []
    for index, caption, target in examples:
        paraphrases = store.get(caption) or []
        texts = [caption, *paraphrases]
        vectors = retriever.embed_texts(texts)
        scores = vectors @ retriever.images.T
        target_id = path_to_id[target]
        ranks = ranks_for_targets(scores, [target_id] * len(texts)).tolist()
        # Fusion 1: average the original and paraphrase embeddings into one query vector.
        mean = torch.nn.functional.normalize(vectors.mean(0, keepdim=True), dim=-1)
        mean_rank = int(ranks_for_targets(mean @ retriever.images.T, [target_id])[0])
        # Fusion 2: Reciprocal Rank Fusion of the full rankings, sum 1 / (60 + rank).
        fused = (1.0 / (RRF_CONSTANT + positions(scores).double() + 1)).sum(0, keepdim=True)
        rrf_rank = int(ranks_for_targets(fused, [target_id])[0])
        # Label-free selection: choose ONE of the queries without looking at the target.
        top_scores, top_ids = scores.topk(10, dim=1)
        confident = int(top_scores[:, 0].argmax())  # Strongest best-match similarity.
        sets = [set(row.tolist()) for row in top_ids]
        agreement = [sum(len(a & b) / len(a | b) for b in sets if b is not a) for a in sets]
        consensus = max(range(len(sets)), key=lambda q: (agreement[q], -q))  # Ties keep the original.
        rows.append({
            "example_index": index, "target": target, "original": caption,
            "original_rank": ranks[0], "paraphrase_ranks": ranks[1:],
            "best_paraphrase_rank": min(ranks[1:]) if paraphrases else None,
            "mean_embedding_rank": mean_rank, "rrf_rank": rrf_rank,
            "confident_pick_rank": ranks[confident], "consensus_pick_rank": ranks[consensus],
            "paraphrases": paraphrases,
            "flags": [warning_flags(caption, p) for p in paraphrases],
        })

    with_paraphrases = [r for r in rows if r["paraphrases"]]
    methods = {
        "original caption": [r["original_rank"] for r in rows],
        "single paraphrase (avg of 5)": None,
        "mean embedding (orig + 5)": [r["mean_embedding_rank"] for r in rows],
        "RRF fusion (orig + 5)": [r["rrf_rank"] for r in rows],
        "pick most confident (orig + 5)": [r["confident_pick_rank"] for r in rows],
        "pick consensus (orig + 5)": [r["consensus_pick_rank"] for r in rows],
        "best of 5 paraphrases (oracle*)": [r["best_paraphrase_rank"] or r["original_rank"] for r in rows],
        "best of orig + 5 (oracle*)": [min(r["original_rank"], r["best_paraphrase_rank"] or r["original_rank"]) for r in rows],
    }
    print(f"\n{'Method':<34}" + "".join(f"{f'R@{k}':>9}" for k in K_VALUES) + f"{'median rank':>13}")
    table = []
    for name, ranks in methods.items():
        if ranks is None:  # Each paraphrase searched alone, averaged over the 5 slots.
            values = [statistics.mean(recall([r["paraphrase_ranks"][j] for r in with_paraphrases], k)
                                      for j in range(N_PARAPHRASES)) for k in K_VALUES]
            median = statistics.median(x for r in with_paraphrases for x in r["paraphrase_ranks"])
        else:
            values, median = [recall(ranks, k) for k in K_VALUES], statistics.median(ranks)
        table.append({"method": name, **{f"recall@{k}": v for k, v in zip(K_VALUES, values)}, "median_rank": median})
        print(f"{name:<34}" + "".join(f"{v:>8.1f}%" for v in values) + f"{median:>13.0f}")
    print("* Oracle rows use the known target to pick the best query: an upper bound, not a usable method.")

    changes = [o - p for r in with_paraphrases for o, p in zip([r["original_rank"]] * N_PARAPHRASES, r["paraphrase_ranks"])]
    better, worse = sum(c > 0 for c in changes), sum(c < 0 for c in changes)
    print(f"\nEach paraphrase vs its original ({len(changes)} pairs): {better} moved the target up, "
          f"{worse} down, {len(changes) - better - worse} unchanged.")
    for k in (10, 100):
        rescued = sum(r["original_rank"] > k and r["best_paraphrase_rank"] <= k for r in with_paraphrases)
        lost = sum(r["original_rank"] <= k and r["best_paraphrase_rank"] > k for r in with_paraphrases)
        print(f"Top {k}: original missed but some paraphrase hit in {rescued} examples; "
              f"original hit but every paraphrase missed in {lost}.")
    flagged = sum(bool(f) for r in rows for f in r["flags"])
    print(f"Paraphrases with warning flags: {flagged}/{len(changes)}; failed generations: {len(rows) - len(with_paraphrases)}.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "runs" / "paraphrase" / stamp
    destination.mkdir(parents=True)
    settings = {"n_examples": len(rows), "seed": SEED, "model": MODEL, "temperature": TEMPERATURE,
                "n_paraphrases": N_PARAPHRASES, "rrf_constant": RRF_CONSTANT, "prompt": PROMPT,
                "paraphrase_store": str(store_path.relative_to(root)), "cache": cache.report}
    (destination / "report.json").write_text(json.dumps(
        {"settings": settings, "summary": table, "examples": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, data in [("summary.csv", table), ("per_example.csv", rows)]:
        with (destination / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                              for k, v in row.items()} for row in data)
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"\nSaved results to: {destination}")


if __name__ == "__main__":
    main()
