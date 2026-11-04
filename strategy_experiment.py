"""Rewrite-strategy experiment: which kind of extra search finds the target most often?

Follows the mentor's suggestion: besides plain paraphrasing, try rewrites that change the
details, change the focus, or break the caption into several single-focus prompts, and
compare which strategy gives the best "winning version" (best of the extra searches).

Each strategy produces N_VARIANTS rewrites of the round-0 caption with ONE local model
(Qwen via Ollama). Every rewrite is searched against all 50,000 saved image vectors.
"paraphrase" reuses the saved rewrites in results/paraphrase/, so it needs no generation.

Requires the Ollama app/server running locally with the model pulled:
    ollama serve            # in a separate terminal, if not already running
    ollama pull qwen3:4b
Rewrites are saved under data/rewrites/ and reused on later runs.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import urllib.request

import torch

from agentir.cache import load_cache, read_json
from agentir.evaluation import matched_queries
from agentir.retrieval import BlipRetriever, ranks_for_targets
from paraphrase_experiment import normalize_text, positions, recall, warning_flags


# SETTINGS: keep these fixed when comparing runs across teammates.
N_EXAMPLES = 200        # Random captions from the 2,064 matched targets; None for all.
SEED = 42               # Same sampling as paraphrase_experiment.py for the same N.
N_VARIANTS = 5          # Rewrites per strategy; equal N keeps best-of-N comparisons fair.
MODEL = "qwen3:4b"      # One agent model for every experiment (mentor's request).
TEMPERATURE = 0.7
K_VALUES = [1, 10, 100]
RRF_CONSTANT = 60
PRF_TOP = 5             # Pseudo-relevance feedback: how many top images to fold into the query.
PRF_WEIGHT = 0.5        # Weight of their mean vector relative to the query vector.
DEVICE = "cpu"
CPU_THREADS = 4
OLLAMA_URL = "http://localhost:11434/api/chat"
SAVED_PARAPHRASES = "results/paraphrase/qwen3_4b_rewrites.json"

MAX_ATTEMPTS = 3        # Retry (new seed) when the model repeats itself or copies the caption.

# One entry per strategy: a system prompt plus worked examples sent as earlier chat turns
# (small models follow examples far better than instructions). Example captions are made up,
# not taken from VisDial. Add a new entry here to test another strategy.
STRATEGIES = {
    "paraphrase": None,  # Reused from SAVED_PARAPHRASES (same wording-only prompt as before).
    "detail": {
        "system": (
            "You rewrite image descriptions for an image search engine. Write exactly {n} versions "
            "of the user's description that each use a DIFFERENT level of detail: some shorter, "
            "keeping only the main content; some longer, spelling out what the scene would plainly "
            "look like when it is strongly implied. Never contradict the description and never add "
            "unusual objects. Every version must differ from the original and from each other."
        ),
        "examples": [
            ("a plate of food on a wooden table", [
                "a plate of food",
                "food on a table",
                "a plate with a cooked meal sitting on a brown wooden table",
                "a dinner plate of food placed on a wooden dining table indoors",
                "a full plate of food resting on the wooden surface of a table"]),
            ("two dogs playing with a red ball in a park", [
                "dogs playing in a park",
                "two dogs and a red ball",
                "two dogs running on green grass in a park, chasing a red ball",
                "a pair of dogs playing together with a bright red ball on a park lawn",
                "two playful dogs with a red ball outdoors in a grassy park"]),
        ],
    },
    "refocus": {
        "system": (
            "You rewrite image descriptions for an image search engine. Write exactly {n} versions "
            "of the user's description, each making a DIFFERENT element the main subject: a "
            "different object, person, animal, or the setting/background, with the other elements "
            "as secondary context. Use only elements stated in the description; if it names only "
            "one or two things, shift the focus between them and the setting. Never add new "
            "objects. Every version must differ from the original and from each other."
        ),
        "examples": [
            ("a ginger cat sleeping in a busy street, people in the background", [
                "man and woman standing in a busy street with a ginger cat in front of them",
                "a busy street full of people with a ginger cat asleep on the ground",
                "people walking along a crowded street past a sleeping ginger cat",
                "a city street scene with pedestrians and a ginger cat lying down",
                "a sleeping ginger cat in the street, people behind it"]),
            ("a plate of food on a wooden table", [
                "a wooden table with a plate of food on it",
                "food served on a plate sitting on a wooden table",
                "a wooden tabletop holding a plate of food",
                "a meal on a plate, placed on a wooden table",
                "a table made of wood with a plate of food"]),
        ],
    },
    "decompose": {
        "system": (
            "You split image descriptions into focused search queries for an image search engine. "
            "Write exactly {n} short descriptions, each focusing on ONE element or relationship of "
            "the user's scene (one object with its attributes, one person or group, one action, or "
            "the setting), phrased like a photo caption. Together they cover the whole description. "
            "Never add new objects. Every description must differ from the original and from each other."
        ),
        "examples": [
            ("a ginger cat sleeping in a busy street, people in the background", [
                "a photo of a ginger cat in the street",
                "man and woman in a busy street",
                "a busy street with people and a cat",
                "a ginger cat sleeping",
                "people in the background of a city street"]),
            ("two dogs playing with a red ball in a park", [
                "two dogs playing together",
                "a red ball on the grass",
                "a dog chasing a ball",
                "a grassy park",
                "dogs in a park"]),
        ],
    },
}


def ask_model(strategy, caption, attempt=0):
    """One local Ollama request returning exactly N_VARIANTS strings, or None."""
    schema = {"type": "object", "required": ["variants"], "properties": {"variants": {
        "type": "array", "items": {"type": "string"}, "minItems": N_VARIANTS, "maxItems": N_VARIANTS}}}
    messages = [{"role": "system", "content": strategy["system"].format(n=N_VARIANTS)}]
    for example, answer in strategy["examples"]:
        messages += [{"role": "user", "content": example},
                     {"role": "assistant", "content": json.dumps({"variants": answer})}]
    messages.append({"role": "user", "content": caption})
    body = {"model": MODEL, "think": False, "stream": False, "format": schema,
            "options": {"temperature": TEMPERATURE, "seed": SEED + attempt}, "messages": messages}
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            items = json.loads(json.load(response)["message"]["content"])["variants"]
    except Exception as error:
        print(f"  attempt {attempt + 1} failed: {error}", flush=True)
        return None
    items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    return items if len(items) == N_VARIANTS else None


def distinct(caption, items):
    """How many variants are new: not a copy of the caption and not a repeat of each other."""
    seen = {normalize_text(caption)}
    return sum(1 for item in items if not (normalize_text(item) in seen or seen.add(normalize_text(item))))


def ask_until_distinct(strategy, caption):
    """Retry with new seeds until all variants are new; keep the most varied attempt."""
    best = None
    for attempt in range(MAX_ATTEMPTS):
        items = ask_model(strategy, caption, attempt)
        if items and (best is None or distinct(caption, items) > distinct(caption, best)):
            best = items
        if best and distinct(caption, best) == N_VARIANTS:
            break
    return best


def generate(name, system, captions, root):
    """Reuse stored rewrites for this strategy; ask the model only for missing captions."""
    if system is None:
        saved = read_json(root / SAVED_PARAPHRASES)
        return {c: saved.get(c) for c in captions}
    prompt_id = hashlib.sha256(json.dumps([system, N_VARIANTS, TEMPERATURE, SEED, MODEL, MAX_ATTEMPTS]).encode()).hexdigest()[:12]
    path = root / "data" / "rewrites" / f"{MODEL.replace(':', '_')}_{name}_{prompt_id}.json"
    store = read_json(path) if path.is_file() else {}
    missing = [c for c in dict.fromkeys(captions) if c not in store]
    print(f"[{name}] {len(captions) - len(missing)} reused, {len(missing)} to generate.", flush=True)
    started = time.perf_counter()
    for i, caption in enumerate(missing, 1):
        store[caption] = ask_until_distinct(system, caption)  # None records a failure.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
        if i % 10 == 0 or i == len(missing):
            elapsed = time.perf_counter() - started
            print(f"  [{name}] {i}/{len(missing)} ({elapsed:.0f}s, ~{elapsed / i * (len(missing) - i) / 60:.0f} min left)", flush=True)
    return store


def ollama_identity():
    """Record the Ollama version and exact model weights, so teammates can compare runs."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=10) as response:
            version = json.load(response)["version"]
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
            digest = next((m["digest"] for m in json.load(response)["models"] if m["name"] == MODEL), None)
    except Exception as error:  # Not fatal: a run that only reuses saved rewrites needs no server.
        return {"ollama_version": None, "model_digest": None, "error": str(error)}
    return {"ollama_version": version, "model_digest": digest}


def rank_of(scores, target_id):
    return int(ranks_for_targets(scores, [target_id])[0])


def fusions(vectors, scores, images, target_id):
    """Label-free ways to turn several queries into one ranking (row 0 is the original)."""
    mean = torch.nn.functional.normalize(vectors.mean(0, keepdim=True), dim=-1)
    rrf = (1.0 / (RRF_CONSTANT + positions(scores).double() + 1)).sum(0, keepdim=True)
    return {
        "mean_embedding": rank_of(mean @ images.T, target_id),
        "rrf": rank_of(rrf, target_id),
        "max_score": rank_of(scores.max(0, keepdim=True).values, target_id),
        "mean_embedding+prf": rank_of(prf(mean, images), target_id),
    }


def prf(query, images):
    """Pseudo-relevance feedback: move the query toward its own top results, then search again."""
    top = (query @ images.T).topk(PRF_TOP, dim=1).indices[0]
    moved = torch.nn.functional.normalize(query + PRF_WEIGHT * images[top].mean(0, keepdim=True), dim=-1)
    return moved @ images.T


def main():
    root = Path(__file__).resolve().parent
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]),
                            read_json(paths["supplied_queries"]), cache.paths)
    indices = sorted(random.Random(SEED).sample(range(len(pairs)), N_EXAMPLES)) if N_EXAMPLES else list(range(len(pairs)))
    examples = [(i, pairs[i][0]["dialog"][0], pairs[i][0]["img"]) for i in indices]
    captions = [c for _, c, _ in examples]
    print(f"Evaluating {len(examples)}/{len(pairs)} captions against {len(cache.paths):,} images.")

    identity = ollama_identity()
    print(f"Ollama {identity['ollama_version']}, {MODEL} digest {identity['model_digest']}")
    stores = {name: generate(name, system, captions, root) for name, system in STRATEGIES.items()}

    retriever = BlipRetriever(cache, device=DEVICE, threads=CPU_THREADS)
    images = retriever.images
    path_to_id = {p: i for i, p in enumerate(cache.paths)}
    rows = []
    for index, caption, target in examples:
        target_id = path_to_id[target]
        original_vector = retriever.embed_texts([caption])
        original_scores = original_vector @ images.T
        row = {"example_index": index, "target": target, "original": caption,
               "original_rank": rank_of(original_scores, target_id),
               "original+prf_rank": rank_of(prf(original_vector, images), target_id)}
        all_vectors = [original_vector]
        for name in STRATEGIES:
            variants = stores[name].get(caption) or []
            row[f"{name}_variants"] = variants
            if not variants:
                row[f"{name}_ranks"] = []
                continue
            vectors = retriever.embed_texts(variants)
            all_vectors.append(vectors)
            both = torch.cat([original_vector, vectors])
            scores = both @ images.T
            ranks = ranks_for_targets(scores, [target_id] * len(both)).tolist()
            row[f"{name}_ranks"] = ranks[1:]
            for method, rank in fusions(both, scores, images, target_id).items():
                row[f"{name}_{method}_rank"] = rank
            row[f"{name}_flags"] = [warning_flags(caption, v) for v in variants]
        combined = torch.cat(all_vectors)
        scores = combined @ images.T
        row["all_ranks_best"] = int(ranks_for_targets(scores, [target_id] * len(combined)).min())
        for method, rank in fusions(combined, scores, images, target_id).items():
            row[f"all_{method}_rank"] = rank
        rows.append(row)

    def best(row, name):  # Oracle: best of the original and this strategy's variants.
        return min([row["original_rank"], *row[f"{name}_ranks"]])

    def fused(row, key):  # Fall back to the original when a generation failed.
        return row.get(key, row["original_rank"])

    methods = {"original caption": [r["original_rank"] for r in rows],
               "original + PRF (no LLM)": [r["original+prf_rank"] for r in rows]}
    for name in STRATEGIES:
        n = len(rows[0][f"{name}_ranks"]) or N_VARIANTS
        methods[f"{name}: single variant (avg)"] = ("single", name)
        for method in ("mean_embedding", "rrf", "max_score", "mean_embedding+prf"):
            methods[f"{name}: {method} (orig + {n})"] = [fused(r, f"{name}_{method}_rank") for r in rows]
        methods[f"{name}: best of orig + {n} (oracle*)"] = [best(r, name) for r in rows]
    for method in ("mean_embedding", "rrf", "max_score", "mean_embedding+prf"):
        methods[f"ALL strategies: {method}"] = [r[f"all_{method}_rank"] for r in rows]
    methods["ALL strategies: best of all (oracle*)"] = [r["all_ranks_best"] for r in rows]

    width = max(map(len, methods)) + 2
    print(f"\n{'Method':<{width}}" + "".join(f"{f'R@{k}':>9}" for k in K_VALUES) + f"{'median':>9}")
    table = []
    for name, ranks in methods.items():
        if isinstance(ranks, tuple):
            strategy = ranks[1]
            have = [r for r in rows if r[f"{strategy}_ranks"]]
            slots = range(min(len(r[f"{strategy}_ranks"]) for r in have))
            values = [statistics.mean(recall([r[f"{strategy}_ranks"][j] for r in have], k) for j in slots) for k in K_VALUES]
            median = statistics.median(x for r in have for x in r[f"{strategy}_ranks"])
        else:
            values, median = [recall(ranks, k) for k in K_VALUES], statistics.median(ranks)
        table.append({"method": name, **{f"recall@{k}": v for k, v in zip(K_VALUES, values)}, "median_rank": median})
        print(f"{name:<{width}}" + "".join(f"{v:>8.1f}%" for v in values) + f"{median:>9.0f}")
        if name.startswith("original + PRF") or "oracle" in name:
            print()
    print("* Oracle rows use the known target to pick the best query: an upper bound, not a usable method.")
    print("  The ALL oracle picks from many more queries than the per-strategy oracles, so it is not a fair comparison.")

    for name in STRATEGIES:
        have = [r for r in rows if r[f"{name}_ranks"]]
        rescued = sum(r["original_rank"] > 10 and best(r, name) <= 10 for r in have)
        flagged = sum(bool(f) for r in have for f in r[f"{name}_flags"])
        print(f"[{name}] rescued into top 10: {rescued}/{sum(r['original_rank'] > 10 for r in have)} misses; "
              f"flagged variants: {flagged}/{len(have) * N_VARIANTS}; failed generations: {len(rows) - len(have)}.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "runs" / "strategies" / stamp
    destination.mkdir(parents=True)
    settings = {"n_examples": len(rows), "seed": SEED, "model": MODEL, "temperature": TEMPERATURE,
                "n_variants": N_VARIANTS, "rrf_constant": RRF_CONSTANT, "prf_top": PRF_TOP,
                "prf_weight": PRF_WEIGHT, "strategies": STRATEGIES, **identity, "cache": cache.report}
    (destination / "report.json").write_text(json.dumps(
        {"settings": settings, "summary": table, "examples": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, data in [("summary.csv", table), ("per_example.csv", rows)]:
        with (destination / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                              for k, v in row.items()} for row in data)
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"\nSaved results to: {destination}")


if __name__ == "__main__":
    main()
