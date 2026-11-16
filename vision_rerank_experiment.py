"""Vision-model re-ranking pilot: does looking at the photos fix BLIP's ordering?

BLIP ranks all 50,000 photos by one vector similarity. The target is usually near the top
but often just outside the top 10 (round 0: 95.5% in the top 100, 74.5% in the top 10).
This pilot keeps BLIP's shortlist (top SHORTLIST photos for the original caption), asks a
vision-language model to score how well each photo matches the caption, and reorders the
shortlist by that score (ties keep BLIP's order). Photos outside the shortlist keep their
BLIP rank, so the re-ranker can only move the target within the shortlist.

Round 0 only: the description is the original VisDial caption, no dialogue yet.
Targets come from the same 200-caption sample as strategy_experiment.py, chosen by BLIP rank
(STRATA): every target just outside the top 10 (the only ones re-ranking can rescue), plus a
few already in the top 10 (to measure harm). Each stratum is weighted by its share of the 200,
so the reported recall estimates the effect on the whole sample, not on the hand-picked mix.

Two phases, so the vision model can run on a GPU machine (e.g. Kaggle) without the BLIP cache:

    python vision_rerank_experiment.py shortlist [--all]     # local: BLIP, CPU, seconds
    python vision_rerank_experiment.py compare SHORTLISTS    # GPU: try each prompt on a few targets
    python vision_rerank_experiment.py rerank SHORTLISTS     # GPU: the pilot
    python vision_rerank_experiment.py                       # both phases on one machine

The rerank and compare phases need only the Python standard library, Pillow and an Ollama
server with the vision model pulled (`ollama serve`, `ollama pull qwen2.5vl:7b`). Shortlisted
photos are downloaded once from images.cocodataset.org into IMAGE_DIR. Model answers are saved
under data/vlm_scores/ and reused on later runs. Notebook for GPU runs: notebooks/vision_rerank_kaggle.ipynb. Guide and results: REWRITE_FUSION_VISION_RERANK.md.
"""
from datetime import datetime, timezone
import argparse
import base64
import csv
import gc
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import statistics
import sys
import time
import urllib.error
import urllib.request


# SETTINGS: keep these fixed when comparing runs across teammates.
SAMPLE_SIZE = 200       # Same sample as strategy_experiment.py (N_EXAMPLES = 200, SEED = 42).
SEED = 42
SHORTLIST = 30          # BLIP top-K that the vision model re-ranks.
# Pilot targets by BLIP rank: (lowest rank, highest rank, how many; None = all of them).
STRATA = {"rank 1": (1, 1, 10), "rank 2-10": (2, 10, 10), f"rank 11-{SHORTLIST}": (11, SHORTLIST, None)}
K_VALUES = [1, 5, 10]
MODEL = "qwen2.5vl:7b"      # Chosen by `compare` on Kaggle T4s (3b separated the target less well).
PROMPT_NAME = "yes_prob"    # Which entry of PROMPTS the pilot uses; pick it with `compare` first.
# Rule 2 (fixed before the pilot): photos scoring >= YES_THRESHOLD form a "yes" group placed
# ahead of the rest; inside each group BLIP's order is kept. 5 = the model leans "yes" over "no".
YES_THRESHOLD = 5
IMAGE_SIDE = 448        # Longest side sent to the model; smaller is faster, larger sees more detail.
MAX_ATTEMPTS = 3        # Retries with a new seed when the answer is malformed.
MAX_TOKENS = 256        # Cap per answer: a small model can loop on the mismatch list forever.
REQUEST_TIMEOUT = 120   # Seconds; a healthy answer takes 1-3 s on a T4.
DEVICE = "cpu"
CPU_THREADS = 4
OLLAMA_URL = "http://localhost:11434/api/chat"
COCO_URL = "http://images.cocodataset.org/"
IMAGE_DIR = Path.home() / "agentir-data" / "images"

# Each prompt yields a "score" from 0 to 10 (JSON prompts: an integer; yes_prob: a probability x 10).
PROMPTS = {
    "checklist": {
        "system": (
            "You check whether a photo matches a description written by someone looking for that "
            "exact photo. First list the details of the description that this photo clearly does "
            "NOT show or contradicts (objects, counts, colours, actions, setting). Leave the list "
            "empty when everything fits; vague words like 'busy' or 'lots of' fit if roughly true. "
            "Then score the match from 0 to 10: 10 = everything fits, 8 = one small detail is off, "
            "5 = the main subject fits but important details are wrong, 2 = only the general kind "
            "of scene fits, 0 = a different scene. An empty list must score 9 or 10."
        ),
        "schema": {"type": "object", "required": ["mismatches", "score"], "properties": {
            "mismatches": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "score": {"type": "integer", "minimum": 0, "maximum": 10}}},
    },
    "score_only": {
        "system": (
            "Someone is searching for one exact photo using a description. Rate how likely it is "
            "that this photo is the one they described, from 0 to 10: 10 = it fits every detail, "
            "5 = same kind of scene but some details differ, 0 = a different scene. "
            "Reply with JSON only."
        ),
        "schema": {"type": "object", "required": ["score"], "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 10}}},
    },
    # Not a JSON prompt: the score is 10 x the model's probability of answering "yes" (vs "no"),
    # read from token log-probabilities. Fine-grained, almost never tied. Needs Ollama >= 0.12.11.
    "yes_prob": {
        "kind": "yes_prob",
        "system": (
            "Someone is searching for one exact photo using a description. Look carefully at the "
            "objects, counts, colours, actions and setting. Answer 'yes' only if this photo matches "
            "the description, otherwise 'no'. Answer with one word: yes or no."
        ),
        "question": "Is this the photo described? Answer yes or no.",
    },
}


def prompt_id(name):
    """Fingerprint of everything that changes the model's answers, for the answer store."""
    return hashlib.sha256(json.dumps([PROMPTS[name], MODEL, IMAGE_SIDE, SEED]).encode()).hexdigest()[:12]


def fetch_image(name, folder=IMAGE_DIR):
    """Return the local copy of one manifest photo, downloading it the first time."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path: {name}")
    path = Path(folder, *relative.parts)
    if not path.is_file():
        with urllib.request.urlopen(COCO_URL + name, timeout=60) as response:
            data = response.read(20 * 1024**2 + 1)
        if len(data) > 20 * 1024**2:
            raise ValueError(f"Unexpectedly large image: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def encode_image(path, side=IMAGE_SIDE):
    """Shrink the photo so the longest side is at most `side`, as base64 JPEG."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def parse_answer(content):
    """Model output to {'score': int 0-10, 'mismatches': [str]}, or None if malformed."""
    try:
        answer = json.loads(content)
        score = answer["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        mismatches = [str(m).strip() for m in answer.get("mismatches", []) if str(m).strip()]
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return {"score": max(0, min(10, int(round(score)))), "mismatches": mismatches}


class SetupError(RuntimeError):
    """A problem with the server or its version: stop the run instead of recording failures."""


class ServerDown(SetupError):
    """Ollama is not reachable."""


def parse_yes_prob(reply):
    """Ollama reply with logprobs to {'score': 10 * P(yes) / (P(yes) + P(no)), 'p_yes', 'p_no'}.
    None when neither word is among the likely first tokens (retry)."""
    import math
    entries = reply.get("logprobs")
    if not entries:
        raise SetupError("Ollama returned no logprobs: yes_prob needs Ollama 0.12.11 or newer.")
    first = entries[0]
    candidates = [first, *(first.get("top_logprobs") or [])]
    seen, p = set(), {"yes": 0.0, "no": 0.0}
    for item in candidates:
        token = item.get("token", "")
        word = token.strip().lower()
        if word in p and token not in seen:
            seen.add(token)
            p[word] += math.exp(item["logprob"])
    if p["yes"] + p["no"] == 0:
        return None
    return {"score": 10 * p["yes"] / (p["yes"] + p["no"]), "p_yes": p["yes"], "p_no": p["no"],
            "mismatches": []}


def ask_vlm(caption, image_b64, prompt_name=PROMPT_NAME):
    """One photo, one description: retry malformed or timed-out answers with a new seed;
    None on failure. Raises ServerDown when nothing is listening at OLLAMA_URL."""
    prompt = PROMPTS[prompt_name]
    for attempt in range(MAX_ATTEMPTS):
        yes_prob = prompt.get("kind") == "yes_prob"
        user = f"Description: {caption}" + (f"\n{prompt['question']}" if yes_prob else "")
        body = {"model": MODEL, "stream": False,
                "options": {"temperature": 0, "seed": SEED + attempt, "num_predict": 1 if yes_prob else MAX_TOKENS},
                "messages": [{"role": "system", "content": prompt["system"]},
                             {"role": "user", "content": user, "images": [image_b64]}]}
        if yes_prob:
            body.update(logprobs=True, top_logprobs=10)
        else:
            body["format"] = prompt["schema"]
        request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                reply = json.load(response)
            answer = parse_yes_prob(reply) if yes_prob else parse_answer(reply["message"]["content"])
        except SetupError:
            raise
        except urllib.error.URLError as error:
            if isinstance(error.reason, ConnectionRefusedError):
                raise ServerDown(f"Ollama is not running at {OLLAMA_URL}: start it (ollama serve) and rerun.") from error
            print(f"  attempt {attempt + 1} failed: {error}", flush=True)
            answer = None
        except Exception as error:
            print(f"  attempt {attempt + 1} failed: {error}", flush=True)
            answer = None
        if answer is not None:
            return answer
    return None


class AnswerStore:
    """Saved model answers for one prompt, keyed by caption and photo; reused across runs."""

    def __init__(self, root, prompt_name):
        self.prompt_name = prompt_name
        self.path = Path(root) / "data" / "vlm_scores" / f"{MODEL.replace(':', '_')}_{prompt_name}_{prompt_id(prompt_name)}.json"
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        # Only real answers are kept: failures (older versions saved them as null) are asked again.
        self.answers = {key: answer for key, answer in saved.items() if answer is not None}

    def missing(self, rows):
        return sum(f"{r['caption']}\t{i}" not in self.answers for r in rows for i in r["shortlist"])

    def get(self, caption, image):
        """Return the saved answer, asking the model (and saving) the first time.
        A failed answer returns None and is not saved, so the next run asks again."""
        key = f"{caption}\t{image}"
        if key in self.answers:
            return self.answers[key]
        answer = ask_vlm(caption, encode_image(fetch_image(image)), self.prompt_name)
        if answer is not None:
            self.answers[key] = answer
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.answers, indent=2, ensure_ascii=False), encoding="utf-8")
        return answer


def rerank(shortlist, scores):
    """Order shortlist photos by vision score, highest first; ties keep BLIP's order.
    A photo without a score (failed answer) gets the shortlist's mean score, so a failure
    neither promotes nor buries it."""
    known = [s for s in scores.values() if s is not None]
    neutral = statistics.mean(known) if known else 0
    position = {image: i for i, image in enumerate(shortlist)}
    value = {image: neutral if scores.get(image) is None else scores[image] for image in shortlist}
    return sorted(shortlist, key=lambda image: (-value[image], position[image]))


def split_rerank(shortlist, scores, threshold=None):
    """Rule 2: "yes" photos (score >= threshold) first, then the rest, each group in BLIP's order.
    The vision model only demotes photos it rejects; it never reorders photos it accepts.
    A photo without a score (failed answer) stays in the "yes" group, i.e. keeps its BLIP place."""
    threshold = YES_THRESHOLD if threshold is None else threshold
    rejected = {image for image in shortlist if scores.get(image) is not None and scores[image] < threshold}
    return [i for i in shortlist if i not in rejected] + [i for i in shortlist if i in rejected]


def reranked_rank(target, blip_rank, order):
    """Target's rank after re-ranking; targets outside the shortlist keep their BLIP rank."""
    return order.index(target) + 1 if target in order else blip_rank


def recall(ranks, k):
    return 100 * sum(r <= k for r in ranks) / len(ranks)


def score_row(row, store):
    """Ask the model about every shortlisted photo of one target and re-rank them."""
    answers = {image: store.get(row["caption"], image) for image in row["shortlist"]}
    scores = {image: (a or {}).get("score") for image, a in answers.items()}
    order = rerank(row["shortlist"], scores)
    split = split_rerank(row["shortlist"], scores)
    return {**row, "answers": answers, "scores": scores, "reranked": order,
            "target_score": scores.get(row["target"]),
            "vlm_rank": reranked_rank(row["target"], row["blip_rank"], order),
            "split_rank": reranked_rank(row["target"], row["blip_rank"], split)}


def separation(rows):
    """How well the scores single out the target among its lookalikes (targets in the shortlist)."""
    inside = [r for r in rows if r["target"] in r["shortlist"] and r["target_score"] is not None]
    others = [s for r in inside for i, s in r["scores"].items() if i != r["target"] and s is not None]
    return {
        "targets_in_shortlist": len(inside),
        "target_mean_score": statistics.mean(r["target_score"] for r in inside) if inside else None,
        "others_mean_score": statistics.mean(others) if others else None,
        "target_has_top_score": sum(r["target_score"] >= max(s for s in r["scores"].values() if s is not None)
                                    for r in inside),
        "target_uniquely_top": sum(sum(s is not None and s >= r["target_score"] for s in r["scores"].values()) == 1
                                   for r in inside),
        "mean_distinct_scores_per_shortlist": statistics.mean(
            len({s for s in r["scores"].values() if s is not None}) for r in rows),
        "failed_answers": sum(s is None for r in rows for s in r["scores"].values()),
    }


def stratum_of(rank, strata=None):
    strata = STRATA if strata is None else strata
    return next((name for name, (low, high, _) in strata.items() if low <= rank <= high), None)


def weighted_recall(rows, sample_ranks, k, key, strata=None):
    """Recall@k over the whole sample: each pilot target stands for its stratum's share of the
    sample; sample targets outside every stratum keep their BLIP rank (re-ranking can't reach them)."""
    strata = STRATA if strata is None else strata
    sizes = {name: sum(stratum_of(r, strata) == name for r in sample_ranks) for name in strata}
    picked = {name: [r for r in rows if r["stratum"] == name] for name in strata}
    hits = sum(r <= k for r in sample_ranks if stratum_of(r, strata) is None)
    for name, members in picked.items():
        if members:
            hits += sizes[name] * sum(r[key] <= k for r in members) / len(members)
    return 100 * hits / len(sample_ranks)


def summarize(rows, sample_ranks):
    """Recall before and after, plus diagnostics of the model's ability to tell photos apart."""
    inside = [r for r in rows if r["blip_rank"] <= SHORTLIST]
    strata = {name: {"in_sample": sum(stratum_of(r) == name for r in sample_ranks),
                     "in_pilot": sum(r["stratum"] == name for r in rows),
                     **{f"top{k}_before": sum(r["stratum"] == name and r["blip_rank"] <= k for r in rows) for k in K_VALUES},
                     **{f"top{k}_after": sum(r["stratum"] == name and r["vlm_rank"] <= k for r in rows) for k in K_VALUES},
                     **{f"top{k}_after_split": sum(r["stratum"] == name and r["split_rank"] <= k for r in rows)
                        for k in K_VALUES}}
              for name in STRATA}
    # Every shortlisted target scored: the recall is exact, not a weighted estimate.
    scope = "all targets" if all(s["in_pilot"] == s["in_sample"] for s in strata.values()) else "weighted estimate"
    table = [{"method": f"BLIP (original caption), all {len(sample_ranks)}",
              **{f"recall@{k}": weighted_recall(rows, sample_ranks, k, "blip_rank") for k in K_VALUES}},
             {"method": f"rule 1: BLIP top {SHORTLIST} re-ranked by {MODEL} score ({scope})",
              **{f"recall@{k}": weighted_recall(rows, sample_ranks, k, "vlm_rank") for k in K_VALUES}},
             {"method": f"rule 2: {MODEL} yes/no at {YES_THRESHOLD:g}, BLIP order inside ({scope})",
              **{f"recall@{k}": weighted_recall(rows, sample_ranks, k, "split_rank") for k in K_VALUES}},
             {"method": f"ceiling: target in BLIP top {SHORTLIST}",
              **{f"recall@{k}": recall(sample_ranks, SHORTLIST) for k in K_VALUES}}]
    diagnostics = {
        "targets": len(rows),
        "strata": strata,
        "rescued_into_top10": sum(r["blip_rank"] > 10 and r["vlm_rank"] <= 10 for r in rows),
        "lost_from_top10": sum(r["blip_rank"] <= 10 and r["vlm_rank"] > 10 for r in rows),
        "rule2_rescued_into_top10": sum(r["blip_rank"] > 10 and r["split_rank"] <= 10 for r in rows),
        "rule2_lost_from_top10": sum(r["blip_rank"] <= 10 and r["split_rank"] > 10 for r in rows),
        "rule2_moved_down": sum(r["split_rank"] > r["blip_rank"] for r in inside),
        "target_rejected": sum(r["target_score"] is not None and r["target_score"] < YES_THRESHOLD for r in inside),
        "moved_up": sum(r["vlm_rank"] < r["blip_rank"] for r in inside),
        "moved_down": sum(r["vlm_rank"] > r["blip_rank"] for r in inside),
        "mean_rank_in_shortlist_blip": statistics.mean(r["blip_rank"] for r in inside) if inside else None,
        "mean_rank_in_shortlist_vlm": statistics.mean(r["vlm_rank"] for r in inside) if inside else None,
        **separation(rows),
    }
    return table, diagnostics


def ollama_identity():
    """Record the Ollama version and exact model weights, so teammates can compare runs."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=10) as response:
            version = json.load(response)["version"]
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
            digest = next((m["digest"] for m in json.load(response)["models"] if m["name"] == MODEL), None)
    except Exception as error:
        return {"ollama_version": None, "model_digest": None, "error": str(error)}
    return {"ollama_version": version, "model_digest": digest}


def pick_targets(ranked, strata):
    """Pilot targets: per stratum, all members (count None) or a seeded sample of `count`.
    Targets ranked outside every stratum are never picked (re-ranking can't reach them)."""
    rows = []
    for name, (_, _, count) in strata.items():
        members = [r for r in ranked if r["stratum"] == name]
        rows += members if count is None or count >= len(members) else random.Random(SEED).sample(members, count)
    return sorted(rows, key=lambda r: r["example_index"])


def build_shortlists(root, every_target=False):
    """Phase 1 (BLIP, local): each target's rank and its caption's top-SHORTLIST photos.
    every_target=True keeps every sample target inside the shortlist (exact recall, ~4x the work)."""
    from agentir.cache import load_cache, read_json
    from agentir.evaluation import matched_queries
    from agentir.retrieval import BlipRetriever, ranks_for_targets
    config = read_json(root / "experiment.json")
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]), read_json(paths["supplied_queries"]), cache.paths)
    sample = sorted(random.Random(SEED).sample(range(len(pairs)), SAMPLE_SIZE))
    retriever = BlipRetriever(cache, device=DEVICE, threads=CPU_THREADS)
    path_to_id = {p: i for i, p in enumerate(cache.paths)}
    ranked = []
    for index in sample:
        caption, target = pairs[index][0]["dialog"][0], pairs[index][0]["img"]
        rank = int(ranks_for_targets(retriever.scores([caption]), [path_to_id[target]])[0])
        ranked.append({"example_index": index, "target": target, "caption": caption,
                       "blip_rank": rank, "stratum": stratum_of(rank)})
    strata = {name: (low, high, None) for name, (low, high, _) in STRATA.items()} if every_target else STRATA
    rows = pick_targets(ranked, strata)
    for row in rows:
        row["shortlist"] = [hit["image"] for hit in retriever.search(row["caption"], top_k=SHORTLIST)]
    settings = {"sample_size": SAMPLE_SIZE, "seed": SEED, "shortlist": SHORTLIST, "strata": strata,
                "sample_ranks": [r["blip_rank"] for r in ranked], "cache": cache.report}
    del retriever, cache
    gc.collect()  # Free BLIP's memory before a local vision model loads.
    path = root / "runs" / "vision_rerank" / f"shortlists_top{SHORTLIST}{'_all' if every_target else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"settings": settings, "targets": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {name: sum(r["stratum"] == name for r in rows) for name in STRATA}
    print(f"Saved {len(rows)} shortlists {counts} to: {path}")
    return path


def load_shortlists(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["settings"]["shortlist"] != SHORTLIST:
        raise ValueError(f"Shortlists were built with top {data['settings']['shortlist']}, SHORTLIST is {SHORTLIST}.")
    return data


def compare_prompts(root, shortlist_path, n_targets=5, prompt_names=None):
    """Run every prompt on a few targets whose target is in the shortlist; print separation."""
    by_stratum = [[r for r in load_shortlists(shortlist_path)["targets"] if r["stratum"] == name] for name in STRATA]
    mixed = [r for group in zip(*by_stratum) for r in group]  # Take targets from every stratum in turn.
    rows = mixed[:n_targets]
    names = prompt_names or list(PROMPTS)
    print(f"Comparing {len(names)} prompts on {len(rows)} targets x {SHORTLIST} photos with {MODEL}.", flush=True)
    for name in names:
        store = AnswerStore(root, name)
        todo, started = store.missing(rows), time.perf_counter()
        scored = [score_row(r, store) for r in rows]
        elapsed = time.perf_counter() - started
        result = separation(scored)
        print(f"\n[{name}] {elapsed / todo if todo else 0:.1f} s per new answer")
        for key, value in result.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        for r in scored:
            show = lambda x: "-" if x is None else f"{x:g}" if isinstance(x, int) else f"{x:.1f}"
            others = " ".join(show(r["scores"][i]) for i in r["shortlist"] if i != r["target"])
            print(f"  BLIP rank {r['blip_rank']:>2} -> {r['vlm_rank']:>2}  target {show(r['target_score'])}  "
                  f"others [{others}]  | {r['caption']}")


def run_pilot(root, shortlist_path, prompt_name=PROMPT_NAME):
    """Phase 2 (vision model): re-rank every shortlist and report recall before and after."""
    data = load_shortlists(shortlist_path)
    identity = ollama_identity()
    print(f"Ollama {identity['ollama_version']}, {MODEL} digest {identity['model_digest']}, prompt {prompt_name}", flush=True)
    store = AnswerStore(root, prompt_name)
    todo = store.missing(data["targets"])
    print(f"{len(data['targets']) * SHORTLIST - todo} answers reused, {todo} to ask.", flush=True)
    started, rows = time.perf_counter(), []
    for n, row in enumerate(data["targets"], 1):
        rows.append(score_row(row, store))
        done = todo - store.missing(data["targets"])
        if done:
            elapsed = time.perf_counter() - started
            print(f"  {n}/{len(data['targets'])} targets; BLIP rank {row['blip_rank']} -> {rows[-1]['vlm_rank']} "
                  f"({elapsed:.0f}s, ~{elapsed / done * (todo - done) / 60:.0f} min left)", flush=True)

    table, diagnostics = summarize(rows, data["settings"]["sample_ranks"])
    width = max(len(t["method"]) for t in table) + 2
    print(f"\n{'Method':<{width}}" + "".join(f"{f'R@{k}':>9}" for k in K_VALUES))
    for t in table:
        print(f"{t['method']:<{width}}" + "".join(f"{t[f'recall@{k}']:>8.1f}%" for k in K_VALUES))
    print("\nPer stratum (targets in the top k: BLIP -> rule 1 / rule 2):")
    for name, s in diagnostics["strata"].items():
        print(f"  {name}: {s['in_pilot']} of {s['in_sample']} in sample; " + ", ".join(
            f"top {k}: {s[f'top{k}_before']} -> {s[f'top{k}_after']} / {s[f'top{k}_after_split']}" for k in K_VALUES))
    print("\nDiagnostics:")
    for name, value in diagnostics.items():
        if name != "strata":
            print(f"  {name}: {value:.2f}" if isinstance(value, float) else f"  {name}: {value}")
    print("\nThe estimate rests on a few targets per stratum: read the per-stratum counts, not just the percentages.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination, n = root / "runs" / "vision_rerank" / stamp, 1
    while destination.exists():  # Two runs in the same second get separate folders.
        destination, n = root / "runs" / "vision_rerank" / f"{stamp}_{n}", n + 1
    destination.mkdir(parents=True)
    settings = {**data["settings"], "model": MODEL, "image_side": IMAGE_SIDE, "prompt_name": prompt_name,
                "prompt": PROMPTS[prompt_name], "prompt_id": prompt_id(prompt_name), **identity}
    (destination / "report.json").write_text(json.dumps(
        {"settings": settings, "summary": table, "diagnostics": diagnostics, "targets": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (destination / "per_target.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["example_index", "target", "caption", "blip_rank", "vlm_rank", "split_rank", "target_score", "scores"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "scores": json.dumps([r["scores"][i] for i in r["shortlist"]])} for r in rows)
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"\nSaved results to: {destination}")
    return destination


def main():
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("phase", nargs="?", choices=["shortlist", "compare", "rerank"])
    parser.add_argument("shortlists", nargs="?", help="Shortlist JSON from the shortlist phase.")
    parser.add_argument("--prompt", default=PROMPT_NAME, choices=list(PROMPTS))
    parser.add_argument("--targets", type=int, default=5, help="Targets per prompt for compare.")
    parser.add_argument("--model", default=MODEL, help="Ollama vision model, e.g. qwen2.5vl:7b.")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), help="Prompts for compare (default: all).")
    parser.add_argument("--all", action="store_true",
                        help=f"shortlist: keep every sample target in BLIP's top {SHORTLIST}, not a stratified few.")
    args = parser.parse_args()
    MODEL = args.model
    root = Path(__file__).resolve().parent
    if args.phase in ("compare", "rerank") and not args.shortlists:
        parser.error(f"{args.phase} needs the shortlist JSON path.")
    try:
        if args.phase == "shortlist":
            build_shortlists(root, every_target=args.all)
        elif args.phase == "compare":
            compare_prompts(root, args.shortlists, args.targets, args.prompts)
        elif args.phase == "rerank":
            run_pilot(root, args.shortlists, args.prompt)
        else:
            run_pilot(root, build_shortlists(root, every_target=args.all), args.prompt)
    except SetupError as error:
        sys.exit(f"\nStopped: {error} Answers so far are saved and will be reused.")


if __name__ == "__main__":
    main()
