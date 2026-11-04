"""Paraphrase experiment: does rewriting the initial caption help BLIP find the target?

Open this file in VS Code, check the settings below, then click Run.
Follows the plan agreed with our mentor (Oct 23): a fixed random sample of captions,
5 paraphrases each from ONE local open-source model (Qwen via Ollama), searched
against all 50,000 saved image vectors. No image encoding, no paid API calls.

By default, replay the team's saved rewrites; Ollama is NOT needed.
Only PARAPHRASE_SOURCE = "generate" requires Ollama with the model pulled:
    ollama serve            # in a separate terminal, if not already running
    ollama pull qwen3:4b
Paraphrases are saved under data/paraphrases/ and reused on later runs.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import importlib.metadata
from pathlib import Path
import platform
import random
import re
import statistics
import time
import unicodedata
import urllib.request

import torch

from agentir.cache import digest, load_cache, read_json
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
DEVICE = "cpu"         # BLIP device: set "cuda" only in a working NVIDIA GPU environment.
TEXT_BATCH_SIZE = 8    # Texts encoded together; does not change the candidate pool.
CPU_THREADS = 4         # Limits CPU work; it does not choose a GPU or change the image pool.
OLLAMA_URL = "http://localhost:11434/api/chat"  # Used only by explicit Ollama generation mode.
PARAPHRASE_SOURCE = "saved"  # "saved" replays without Ollama; "generate" opts into inference.
SAVED_PARAPHRASES = "results/paraphrase/qwen3_4b_rewrites.json"
FILTER_FLAGGED = True  # Conservative heuristic checks; not a semantic guarantee.
DEDUPLICATE = True     # Original and each normalized wording get one vote.
# Set BOTH switches False to replay the original unfiltered six-query experiment.

PROMPT = (
    "You rewrite image descriptions for an image search engine. Write exactly {n} different "
    "paraphrases of the user's description. Each paraphrase must keep every fact: objects, counts, "
    "colors, attributes, actions, prepositions, spatial relationships, negations and uncertainty. "
    "Add nothing new, do not guess details, and do not drop anything. Vary wording and sentence "
    "structure. Each paraphrase is one plain description, not a question or a list."
)


def normalize_text(text):
    """Ignore casing, punctuation and Unicode formatting when spotting repeated wording."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def warning_flags(original, candidate):
    """Flag count/color/negation changes or original copies; these checks can be wrong.
    They miss some meaning changes and may reject valid synonyms. The run's
    FILTER_FLAGGED setting decides whether flagged candidates are excluded.
    """
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


def select_paraphrases(original, candidates, *, filter_flagged=True, deduplicate=True):
    """Keep usable rewrites without target labels; log every acceptance/rejection.
    Preserve the original separately, remove repeated normalized wording, and
    optionally exclude heuristic warnings. Return accepted texts and an audit.
    """
    accepted, audit = [], []
    seen = {normalize_text(original)}
    for candidate in candidates:
        flags = warning_flags(original, candidate)
        key = normalize_text(candidate)
        reasons = []
        if not key:
            reasons.append("empty_wording")
        if filter_flagged:
            reasons.extend(flag for flag in flags if flag != "identical_to_original")
        if deduplicate and key in seen:
            reasons.append("duplicate_wording")
        if not reasons:
            accepted.append(candidate)
            seen.add(key)
        audit.append({"text": candidate, "flags": flags, "accepted": not reasons,
                      "rejection_reasons": reasons})
    return accepted, audit


def valid_rewrites(items):
    """Require the expected number of nonempty strings before reusing a saved entry."""
    return (isinstance(items, list) and len(items) == N_PARAPHRASES
            and all(isinstance(text, str) and text.strip() for text in items))


def ollama_request(endpoint, body=None):
    """Read local Ollama JSON, or POST a JSON body; never invoked during saved replay."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    request = urllib.request.Request(
        base + endpoint, data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def ollama_provenance():
    """Record the server version and installed model digest/quantization before generation.
    Unknown fields stay unknown; a model tag alone does not identify exact weights.
    """
    result = {"model_tag": MODEL, "captured_at": datetime.now(timezone.utc).isoformat()}
    try:
        result["server_version"] = ollama_request("/api/version")["version"]
        models = ollama_request("/api/tags")["models"]
        match = next((m for m in models if MODEL in (m.get("name"), m.get("model"))), None)
        result.update({"digest": None if match is None else match.get("digest"),
                       "details": {} if match is None else match.get("details", {})})
    except (OSError, ValueError, KeyError) as error:
        result["unavailable_reason"] = str(error)
    return result


def ask_model(caption, attempt=0):
    """One local Ollama request returning exactly N_PARAPHRASES strings, or None."""
    schema = {"type": "object", "required": ["paraphrases"], "properties": {"paraphrases": {
        "type": "array", "items": {"type": "string"},
        "minItems": N_PARAPHRASES, "maxItems": N_PARAPHRASES}}}
    body = {"model": MODEL, "think": False, "stream": False, "format": schema,
            "options": {"temperature": TEMPERATURE, "seed": SEED + attempt},
            "messages": [{"role": "system", "content": PROMPT.format(n=N_PARAPHRASES)},
                         {"role": "user", "content": caption}]}
    try:
        content = ollama_request("/api/chat", body)["message"]["content"]
        items = json.loads(content)["paraphrases"]
        if not isinstance(items, list):
            return None
        items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    except Exception as error:  # Local model: a failed attempt costs nothing, so we retry once.
        print(f"  attempt {attempt + 1} failed: {error}", flush=True)
        return None
    return items if valid_rewrites(items) else None


def generate_paraphrases(captions, store_path, *, allow_generation=False):
    """Replay a bank without network requests by default; fail if entries are missing.
    Explicit generation mode retries missing, failed or malformed entries, saves
    progress after each caption, and records provenance only for new generations.
    """
    store = read_json(store_path) if store_path.is_file() else {}
    if not isinstance(store, dict):
        raise ValueError("Paraphrase bank must map original captions to lists of rewrites.")
    captions = list(dict.fromkeys(captions))
    missing = [c for c in captions if not valid_rewrites(store.get(c))]
    if missing and not allow_generation:
        raise ValueError(
            f"Saved bank {store_path} has {len(missing)} missing/invalid entries. "
            'No generation started. Check SAVED_PARAPHRASES and N_PARAPHRASES, '
            'or explicitly set PARAPHRASE_SOURCE = "generate".'
        )
    print(f"Paraphrases: {len(captions) - len(missing)} reused, {len(missing)} to generate with {MODEL}.")
    if not missing:
        return store
    metadata_path = store_path.with_suffix(".metadata.json")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {"by_caption": {}}
    provenance = ollama_provenance()
    started = time.perf_counter()
    try:
        for i, caption in enumerate(missing, 1):
            items = ask_model(caption) or ask_model(caption, attempt=1)
            store[caption] = items  # Failed entries fall back to original and can be retried next run.
            metadata["by_caption"][caption] = {"provenance": provenance, "success": items is not None,
                                              "prompt": PROMPT, "temperature": TEMPERATURE,
                                              "seed": SEED, "n_paraphrases": N_PARAPHRASES}
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            if i % 10 == 0 or i == len(missing):
                print(f"  {i}/{len(missing)} generated ({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        # Release Qwen's memory before loading BLIP on the same machine.
        try:
            ollama_request("/api/generate", {"model": MODEL, "keep_alive": 0, "stream": False})
        except (OSError, ValueError) as error:
            raise RuntimeError("Could not unload Qwen. Stop its Ollama model before loading BLIP.") from error
    return store


def positions(scores):
    """Zero-based rank of every image for each query row."""
    ordering = torch.argsort(scores, dim=1, descending=True, stable=True)
    result = torch.empty_like(ordering)
    result.scatter_(1, ordering, torch.arange(scores.shape[1], device=scores.device).expand_as(ordering))
    return result


def recall(ranks, k):
    """Percentage of examples with their known target in the first K results."""
    return 100 * sum(r <= k for r in ranks) / len(ranks)


def main():
    """Search with original/accepted rewrites, compare both fusions and save a new run.
    Known targets are used only to measure ranks and explicitly labeled oracles.
    This is initial-caption evaluation, not a live question/answer agent loop.
    """
    root = Path(__file__).resolve().parent
    if PARAPHRASE_SOURCE not in ("saved", "generate"):
        raise ValueError('PARAPHRASE_SOURCE must be "saved" or "generate".')
    if N_PARAPHRASES < 1 or RRF_CONSTANT <= 0:
        raise ValueError("N_PARAPHRASES and RRF_CONSTANT must be positive.")
    if torch.device(DEVICE).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Set DEVICE = "cpu" or check NVIDIA drivers and the PyTorch CUDA build.')
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]),
                            read_json(paths["supplied_queries"]), cache.paths)
    if N_EXAMPLES is not None and not 1 <= N_EXAMPLES <= len(pairs):
        raise ValueError(f"N_EXAMPLES must be None or between 1 and {len(pairs)}.")
    if not K_VALUES or any(not isinstance(k, int) or not 1 <= k <= len(cache.paths) for k in K_VALUES):
        raise ValueError("K_VALUES must contain positive integers inside the image pool.")
    indices = sorted(random.Random(SEED).sample(range(len(pairs)), N_EXAMPLES)) if N_EXAMPLES else list(range(len(pairs)))
    examples = [(i, pairs[i][0]["dialog"][0], pairs[i][0]["img"]) for i in indices]
    print(f"Evaluating {len(examples)}/{len(pairs)} captions against {len(cache.paths):,} images.")

    prompt_id = hashlib.sha256(json.dumps([MODEL, PROMPT, N_PARAPHRASES, TEMPERATURE, SEED]).encode()).hexdigest()[:12]
    if PARAPHRASE_SOURCE == "saved":
        store_path = (root / SAVED_PARAPHRASES).resolve()
    else:
        store_path = root / "data" / "paraphrases" / f"{MODEL.replace(':', '_')}_{prompt_id}.json"
    store = generate_paraphrases([caption for _, caption, _ in examples], store_path,
                                 allow_generation=PARAPHRASE_SOURCE == "generate")

    retriever = BlipRetriever(cache, device=DEVICE, batch_size=TEXT_BATCH_SIZE, threads=CPU_THREADS)
    print(f"Retrieval device: {retriever.device}; filter flagged: {FILTER_FLAGGED}; deduplicate: {DEDUPLICATE}")
    path_to_id = {p: i for i, p in enumerate(cache.paths)}
    rows = []
    for index, caption, target in examples:
        generated = store.get(caption)
        generated = generated if valid_rewrites(generated) else []
        paraphrases, audit = select_paraphrases(caption, generated, filter_flagged=FILTER_FLAGGED,
                                               deduplicate=DEDUPLICATE)
        texts = [caption, *paraphrases]
        vectors = retriever.embed_texts(texts)
        # Queries are cached on CPU; score_vectors moves them to the image device.
        # It returns CPU scores so ranking, RRF and selections share one device.
        scores = retriever.score_vectors(vectors)
        target_id = path_to_id[target]
        ranks = ranks_for_targets(scores, [target_id] * len(texts)).tolist()
        # Fusion 1: average the original and paraphrase embeddings into one query vector.
        mean = torch.nn.functional.normalize(vectors.mean(0, keepdim=True), dim=-1)
        mean_rank = int(ranks_for_targets(retriever.score_vectors(mean), [target_id])[0])
        # Fusion 2: Reciprocal Rank Fusion of the full rankings, sum 1 / (60 + rank).
        fused = (1.0 / (RRF_CONSTANT + positions(scores).double() + 1)).sum(0, keepdim=True)
        rrf_rank = int(ranks_for_targets(fused, [target_id])[0])
        # Label-free selection: choose ONE of the queries without looking at the target.
        # Stable ties match target ranking; use up to 10 images even in tiny tests.
        top_ids = torch.argsort(scores, descending=True, dim=1, stable=True)[:, :10]
        top_scores = scores.gather(1, top_ids)
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
            "generated_paraphrases": generated, "rewrite_audit": audit,
            "effective_query_count": len(texts), "generation_failed": not generated,
            "flags": [item["flags"] for item in audit],
        })

    with_paraphrases = [r for r in rows if r["paraphrases"]]
    methods = {
        "original caption": [r["original_rank"] for r in rows],
        "single rewrite (per-example avg)": None,
        "mean embedding (orig + accepted)": [r["mean_embedding_rank"] for r in rows],
        "RRF fusion (orig + accepted)": [r["rrf_rank"] for r in rows],
        "pick most confident": [r["confident_pick_rank"] for r in rows],
        "pick consensus": [r["consensus_pick_rank"] for r in rows],
        "best accepted (oracle*)": [r["best_paraphrase_rank"] or r["original_rank"] for r in rows],
        "best orig + accepted (oracle*)": [min(r["original_rank"], r["best_paraphrase_rank"] or r["original_rank"]) for r in rows],
    }
    print(f"\n{'Method':<34}" + "".join(f"{f'R@{k}':>9}" for k in K_VALUES) + f"{'median rank':>13}")
    table = []
    for name, ranks in methods.items():
        if ranks is None:
            # Each example has equal weight even if it has fewer accepted rewrites.
            # With none accepted, evaluate its original; never silently drop a target.
            choices = [r["paraphrase_ranks"] or [r["original_rank"]] for r in rows]
            values = [statistics.mean(recall(choices_for_example, k) for choices_for_example in choices)
                      for k in K_VALUES]
            median = statistics.median(rank for choices_for_example in choices for rank in choices_for_example)
        else:
            values, median = [recall(ranks, k) for k in K_VALUES], statistics.median(ranks)
        table.append({"method": name, **{f"recall@{k}": v for k, v in zip(K_VALUES, values)}, "median_rank": median})
        print(f"{name:<34}" + "".join(f"{v:>8.1f}%" for v in values) + f"{median:>13.0f}")
    print("* Oracle rows use the known target to pick the best query: an upper bound, not a usable method.")

    changes = [r["original_rank"] - p for r in with_paraphrases for p in r["paraphrase_ranks"]]
    better, worse = sum(c > 0 for c in changes), sum(c < 0 for c in changes)
    print(f"\nEach paraphrase vs its original ({len(changes)} pairs): {better} moved the target up, "
          f"{worse} down, {len(changes) - better - worse} unchanged.")
    for k in (10, 100):
        rescued = sum(r["original_rank"] > k and r["best_paraphrase_rank"] <= k for r in with_paraphrases)
        lost = sum(r["original_rank"] <= k and r["best_paraphrase_rank"] > k for r in with_paraphrases)
        print(f"Top {k}: original missed but some paraphrase hit in {rescued} examples; "
              f"original hit but every paraphrase missed in {lost}.")
    flagged = sum(bool(f) for r in rows for f in r["flags"])
    generated_count = sum(len(r["generated_paraphrases"]) for r in rows)
    print(f"Generated rewrites with warning flags: {flagged}/{generated_count}; "
          f"accepted: {len(changes)}; rejected: {generated_count - len(changes)}; "
          f"original-only examples: {len(rows) - len(with_paraphrases)}; "
          f"failed generations: {sum(r['generation_failed'] for r in rows)}.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "runs" / "paraphrase" / stamp
    destination.mkdir(parents=True)
    metadata_path = store_path.with_suffix(".metadata.json")
    generation_metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    provenance = {caption: generation_metadata.get("by_caption", {}).get(caption)
                  for _, caption, _ in examples}
    settings = {"n_examples": len(rows), "seed": SEED, "generation_model_setting": MODEL, "temperature": TEMPERATURE,
                "n_paraphrases": N_PARAPHRASES, "rrf_constant": RRF_CONSTANT, "prompt": PROMPT,
                "filter_flagged": FILTER_FLAGGED, "deduplicate": DEDUPLICATE,
                "paraphrase_source": PARAPHRASE_SOURCE, "paraphrase_store": str(store_path),
                "paraphrase_store_sha256": digest(store_path), "cache": cache.report,
                "generation_provenance_by_caption": provenance,
                "provenance_note": "Null means the original generation model digest/quantization is unknown; "
                                   "current settings are not proof of a saved bank's origin.",
                "single_rewrite_metric": "Mean of per-example recall across accepted rewrites; "
                                         "fallback to original when none accepted. Median pools those ranks.",
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                            "cuda_build": torch.version.cuda, "device": str(retriever.device),
                            "gpu": torch.cuda.get_device_name(retriever.device) if retriever.device.type == "cuda" else None,
                            "text_batch_size": TEXT_BATCH_SIZE, "cpu_threads": CPU_THREADS}}
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
