"""Matched raw/provided-query evaluation using PlugIR's metric definitions."""
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch

from .retrieval import ranks_for_targets

METHOD_LABELS = {"raw": "VisDial raw dialogue", "supplied": "Supplied PlugIR queries"}


@dataclass(frozen=True)
class ExperimentContext:
    """Information an experiment may use for one target's current search state.
    Contains the initial caption and raw dialogue so far, never the target ID,
    target vector, future dialogue, supplied PlugIR queries, or evaluation ranks.
    Previous query/state belongs only to this method and example in this run.
    """

    example_index: int
    round_index: int
    initial_caption: str
    dialogue: tuple[str, ...]
    current_query: str
    previous_query: str | None = None
    previous_state: object = None


@dataclass
class SearchOutput:
    """One method's final score for every image in original manifest order.
    Use score_kind='cosine' for normalized embedding dot products; use 'fusion'
    or another descriptive label for other scores. Query and notes are logged.
    Optional state is passed to this method's next round, not saved in reports.
    """

    scores: torch.Tensor
    score_kind: str
    query: str | None = None
    notes: dict = field(default_factory=dict)
    state: object = None


def text_output(retriever, text, *, notes=None, state=None):
    """Encode one description and score it against all saved image vectors.
    Return cosine scores and the searched text for the shared evaluator.
    Experiments may also return a plain string for this common case.
    """
    return SearchOutput(retriever.scores([text])[0], "cosine", text,
                        {} if notes is None else notes, state)


def vector_output(retriever, vector, *, query=None, notes=None, state=None):
    """Normalize one custom query vector and compare it with every image vector.
    Accept [embedding_size] or [1, embedding_size], reject invalid/zero vectors,
    and return cosine scores. This supports experiments that combine embeddings
    directly instead of producing a final description.
    """
    with torch.inference_mode():
        images = retriever.images
        vector = torch.as_tensor(vector, dtype=images.dtype, device=images.device)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.shape != (images.shape[1],) or not torch.isfinite(vector).all():
            raise ValueError("Expected one finite query vector matching the image embedding size.")
        if vector.norm().item() == 0:
            raise ValueError("A zero query vector has no cosine direction.")
        vector = torch.nn.functional.normalize(vector, dim=-1)
        scores = (vector @ images.T).cpu()
    return SearchOutput(scores, "cosine", query, {} if notes is None else notes, state)


def matched_queries(baseline, supplied, image_paths, required_rounds=11):
    """Pair raw VisDial and supplied PlugIR examples for the same target images.
    Check matching initial captions, enough rounds, and targets in the image pool;
    return the paired examples so both methods use the same evaluation targets."""
    by_image = {}
    for row in supplied:
        if row["img"] in by_image:
            raise ValueError("Duplicate supplied-query target.")
        by_image[row["img"]] = row
    pool = set(image_paths)
    pairs, seen = [], set()
    for row in baseline:
        name = row["img"]
        if name in seen or name not in by_image:
            continue
        other = by_image[name]
        if name not in pool:
            raise ValueError(f"Target missing from candidate pool: {name}")
        if len(row["dialog"]) < required_rounds or len(other["dialog"]) < required_rounds:
            continue
        if row["dialog"][0].strip().lower() != other["dialog"][0].strip().lower():
            raise ValueError(f"Mismatched initial caption for {name}")
        pairs.append((row, other))
        seen.add(name)
    if not pairs:
        raise ValueError("No matched query examples.")
    return pairs


def query_text(row, method, round_index):
    """Build the search description for one method and round.
    Raw queries join the caption and dialogue so far; supplied queries select
    the authors' saved description for that round. No LLM is called here."""
    if method == "raw":
        return ", ".join(row["dialog"][:round_index + 1])
    if method == "supplied":
        return row["dialog"][round_index]
    raise ValueError(f"Unknown method: {method}")


def metrics_from_ranks(ranks, ks):
    """Turn target ranks into Recall@K, cumulative Hits@K, and BRI.
    Input axes are [round, example], with rank 1 meaning first place.
    Recall measures this round; Hits remembers earlier successes; lower BRI
    rewards reaching good best-so-far ranks earlier. Return metric rows and BRI."""
    ranks = torch.as_tensor(ranks, dtype=torch.long)
    if ranks.ndim != 2 or min(ranks.shape) < 1 or (ranks < 1).any():
        raise ValueError("Ranks must be a nonempty [round, example] matrix of positive ranks.")
    best = torch.cummin(ranks, dim=0).values
    rows = []
    for r in range(len(ranks)):
        for k in ks:
            if k < 1:
                raise ValueError("K must be positive.")
            rows.append({"round": r, "k": k,
                         "recall_percent": float((ranks[r] <= k).float().mean() * 100),
                         "hits_percent": float((best[r] <= k).float().mean() * 100)})
    bri = None
    if len(ranks) > 1:
        # Natural log of best-so-far ONE-based rank, trapezoids over rounds;
        # equals upstream log(zero_based_rank + 1), average over examples/intervals.
        log_best = best.float().log()
        bri = float(((log_best[:-1] + log_best[1:]) / 2).mean(dim=1).mean())
    return rows, bri


def benchmark(retriever, pairs, methods, rounds, ks):
    """Search the saved image pool for each method, round, and target query.
    Measure where each known target ranks, then calculate retrieval metrics.
    Return per-query records, per-round metrics, and summaries; this function
    uses saved dialogue and does not generate new questions or answers."""
    if not 1 <= rounds <= 11:
        raise ValueError("Use 1..11 query states.")
    path_to_id = {p: i for i, p in enumerate(retriever.cache.paths)}
    target_ids = [path_to_id[left["img"]] for left, _ in pairs]
    metrics, examples, summary = [], [], {}
    for method in methods:
        history = []
        rows = [pair[0 if method == "raw" else 1] for pair in pairs]
        for r in range(rounds):
            texts = [query_text(row, method, r) for row in rows]
            scores = retriever.scores(texts)
            ranks = ranks_for_targets(scores, target_ids)
            history.append(ranks)
            for index, (row, text, target_id, rank) in enumerate(zip(rows, texts, target_ids, ranks)):
                examples.append({"method": method, "round": r, "example_index": index,
                                 "target": row["img"], "target_rank": int(rank),
                                 "target_cosine": float(scores[index, target_id]), "query": text})
            print(f"{method}: round {r}/{rounds - 1} complete ({len(rows)} examples)", flush=True)
        rows_of_metrics, bri = metrics_from_ranks(torch.stack(history), ks)
        metrics += [dict(method=method, **row) for row in rows_of_metrics]
        summary[method] = {"examples": len(pairs), "rounds": rounds, "BRI": bri,
                           "final": [row for row in rows_of_metrics if row["round"] == rounds - 1]}
    return {"per_round": metrics, "per_example": examples, "summary": summary}


def compare_colab(result, pairs, run_config_path, per_round_path, summary_path):
    """Check local results against our saved 32-target, 11-state Colab run.
    Require the same targets and query contents, then compare Recall@10,
    Hits@10, and BRI within small tolerances. Return a match flag and differences
    so a new environment can be checked before testing new ideas."""
    run = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
    expected = run["targets"]
    actual = [left["img"] for left, _ in pairs]
    if actual != expected or any(row["rounds"] != run["rounds"] for row in result["summary"].values()):
        raise ValueError("Colab comparison requires the same first 32 targets and 11 query states.")
    for field, index in (("baseline_sha256", 0), ("plugir_sha256", 1)):
        encoded = json.dumps([pair[index] for pair in pairs]).encode()
        if hashlib.sha256(encoded).hexdigest() != run[field]:
            raise ValueError(f"Query bytes do not match Colab: {field}")
    reference = {}
    with Path(per_round_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            reference[(row["Method"], int(row["Round"]))] = row
    differences = []
    for row in result["per_round"]:
        if row["k"] != run["K"]:
            continue
        ref = reference[(METHOD_LABELS[row["method"]], row["round"])]
        for actual_key, expected_key in (("recall_percent", "Recall@10 (%)"), ("hits_percent", "Hits@10 (%)")):
            if abs(row[actual_key] - float(ref[expected_key])) > 0.011:
                differences.append({"method": row["method"], "round": row["round"],
                                    "metric": actual_key, "local": row[actual_key], "colab": float(ref[expected_key])})
    with Path(summary_path).open(encoding="utf-8", newline="") as stream:
        bri_reference = {row["Method"]: float(row["BRI (lower is better)"]) for row in csv.DictReader(stream)}
    for method, summary in result["summary"].items():
        target = bri_reference[METHOD_LABELS[method]]
        if abs(summary["BRI"] - target) > 1e-4:
            differences.append({"method": method, "metric": "BRI", "local": summary["BRI"], "colab": target})
    return {"matches_colab": not differences, "differences": differences,
            "percentage_tolerance": 0.011, "bri_tolerance": 1e-4,
            "note": "CPU versus CUDA float arithmetic and deterministic tie handling can affect borderline ranks."}


def evaluate_experiments(retriever, pairs, experiments=None, *, rounds=3, ks=(1, 10, 50)):
    """Compare registered experiments with caption, raw-dialogue, and PlugIR references.
    Each callback receives (context, retriever) and returns text or SearchOutput.
    Only this evaluator uses target labels to measure ranks after scoring; those
    ranks never feed back into callbacks. Return per-example rank changes,
    per-round Recall/Hits, and overall BRI on exactly the same targets and pool.
    This is saved-dialogue evaluation, not a live question/answer generation loop.
    """
    experiments = dict(experiments or {})
    references = ("caption_only", "raw_dialogue", "supplied_plugir")
    if not pairs or not 1 <= rounds <= 11:
        raise ValueError("Provide matched examples and 1..11 query states.")
    ks = sorted(set(ks))
    if not ks or any(not isinstance(k, int) or not 1 <= k <= len(retriever.cache.paths) for k in ks):
        raise ValueError("K values must be positive integers inside the image pool.")
    reserved_names = {*references, "round", "example_index", "target"}
    if any(not isinstance(name, str) or not name.strip() or name in reserved_names
           or not callable(callback) for name, callback in experiments.items()):
        raise ValueError("Use unique nonempty experiment names, distinct from references, and callable methods.")
    path_to_id = {path: i for i, path in enumerate(retriever.cache.paths)}
    target_ids = []
    for raw, supplied in pairs:
        if raw["img"] != supplied["img"] or raw["img"] not in path_to_id:
            raise ValueError("Every pair must share a target in the candidate pool.")
        if min(len(raw["dialog"]), len(supplied["dialog"])) < rounds:
            raise ValueError("Not enough saved dialogue states for the requested rounds.")
        if raw["dialog"][0].strip().lower() != supplied["dialog"][0].strip().lower():
            raise ValueError("Paired methods must start from the same caption.")
        target_ids.append(path_to_id[raw["img"]])
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Duplicate evaluation targets would bias aggregate metrics.")

    methods = [*references, *experiments]
    per_example, per_round, summary = [], [], {}
    for method in methods:
        history = []
        previous_outputs = [None] * len(pairs)
        for r in range(rounds):
            round_ranks = []
            for index, ((raw, supplied), target_id) in enumerate(zip(pairs, target_ids)):
                previous = previous_outputs[index]
                if method == "caption_only":
                    output = text_output(retriever, raw["dialog"][0])
                elif method == "raw_dialogue":
                    output = text_output(retriever, query_text(raw, "raw", r))
                elif method == "supplied_plugir":
                    output = text_output(retriever, query_text(supplied, "supplied", r))
                else:
                    context = ExperimentContext(
                        example_index=index, round_index=r,
                        initial_caption=raw["dialog"][0],
                        dialogue=tuple(raw["dialog"][1:r + 1]),
                        current_query=query_text(raw, "raw", r),
                        previous_query=None if previous is None else previous.query,
                        previous_state=None if previous is None else previous.state,
                    )
                    output = experiments[method](context, retriever)
                    if isinstance(output, str):
                        output = text_output(retriever, output)
                if not isinstance(output, SearchOutput):
                    raise TypeError(f"{method}: return a description string or SearchOutput.")
                scores = torch.as_tensor(output.scores).detach().cpu()
                if scores.shape != (len(retriever.cache.paths),) or not torch.isfinite(scores).all():
                    raise ValueError(f"{method}: return one finite score per image in manifest order.")
                if not isinstance(output.score_kind, str) or not output.score_kind.strip():
                    raise ValueError(f"{method}: provide a descriptive score_kind.")
                if output.query is not None and not isinstance(output.query, str):
                    raise TypeError(f"{method}: query must be a string or None.")
                # Check notes here so unserializable metadata fails near its source.
                if not isinstance(output.notes, dict):
                    raise TypeError(f"{method}: notes must be a JSON-serializable dictionary.")
                json.dumps(output.notes, allow_nan=False)
                rank = int(ranks_for_targets(scores.unsqueeze(0), [target_id])[0])
                previous_rank = None if r == 0 else int(history[-1][index])
                best_rank = min(rank, *(int(row[index]) for row in history)) if history else rank
                top_ids = torch.argsort(scores, descending=True, stable=True)[:10].tolist()
                per_example.append({
                    "method": method, "round": r, "example_index": index,
                    "target": raw["img"], "target_rank": rank,
                    "previous_target_rank": previous_rank, "best_target_rank": best_rank,
                    "rank_improvement_vs_previous": None if previous_rank is None else previous_rank - rank,
                    "query": output.query,
                    "previous_query": None if previous is None else previous.query,
                    "score_kind": output.score_kind, "target_score": float(scores[target_id]),
                    "target_cosine": float(scores[target_id]) if output.score_kind == "cosine" else None,
                    "top_image_ids": top_ids, "notes": output.notes,
                })
                round_ranks.append(rank)
                # Keep only the method's state/text, not 50k scores for every past round.
                previous_outputs[index] = SearchOutput(torch.empty(0), output.score_kind,
                                                       output.query, state=output.state)
            history.append(torch.tensor(round_ranks))
            print(f"{method}: state {r}/{rounds - 1}, {len(pairs)} targets complete", flush=True)
        metric_rows, bri = metrics_from_ranks(torch.stack(history), ks)
        per_round.extend(dict(method=method, **row) for row in metric_rows)
        summary[method] = {"examples": len(pairs), "rounds": rounds, "BRI": bri,
                           "initial_mean_rank": float(history[0].float().mean()),
                           "final_mean_rank": float(history[-1].float().mean()),
                           "final": [row for row in metric_rows if row["round"] == rounds - 1]}

    # Compare the same example at the same state; positive rank improvement is better.
    reference_rows = {(row["method"], row["round"], row["example_index"]): row
                      for row in per_example if row["method"] in references}
    for row in per_example:
        for reference in references:
            other = reference_rows[reference, row["round"], row["example_index"]]
            row[f"rank_improvement_vs_{reference}"] = other["target_rank"] - row["target_rank"]
    metric_lookup = {(row["method"], row["round"], row["k"]): row for row in per_round}
    for row in per_round:
        for reference in references:
            other = metric_lookup[reference, row["round"], row["k"]]
            row[f"recall_change_vs_{reference}_pp"] = row["recall_percent"] - other["recall_percent"]
            row[f"hits_change_vs_{reference}_pp"] = row["hits_percent"] - other["hits_percent"]
    for values in summary.values():
        values["BRI_reduction_vs_caption_only"] = (
            None if values["BRI"] is None else summary["caption_only"]["BRI"] - values["BRI"])
    return {
        "per_example": per_example, "per_round": per_round, "summary": summary,
        "protocol": {
            "pool": len(retriever.cache.paths), "targets": [raw["img"] for raw, _ in pairs],
            "rounds": rounds, "k": ks, "methods": methods,
            "tie_break": "Descending score, then original manifest index",
            "scope": "Saved raw dialogue is shared with custom methods. Supplied PlugIR queries are a saved reference with potentially different dialogue, not live generation or a controlled rewrite-only comparison.",
            "labels": "Targets are used only after a method returns scores. Callbacks receive no target labels or measured ranks. The retriever exposes the whole candidate pool, so experiment authors must also avoid external label lookups.",
            "round_zero": "Initial-caption state; later states add one saved raw dialogue turn each. Callbacks must not ask for future turns.",
            "costs": "Agent calls are internal computation, not extra user rounds. This runner does not call APIs or answer new questions.",
        },
    }


def save_experiment_results(result, destination):
    """Save one completed experiment comparison as JSON and spreadsheet-friendly CSVs.
    Record per-query rank changes, per-round metrics, final metrics with BRI,
    and a wide rank table for comparing all methods on the same target/state.
    Use a new folder so previous experiment results cannot be overwritten.
    """
    destination = Path(destination)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "report.json").write_text(payload, encoding="utf-8")
    final_rows = [dict(method=method, BRI=values["BRI"], **row)
                  for method, values in result["summary"].items() for row in values["final"]]
    wide = {}
    for row in result["per_example"]:
        key = (row["round"], row["example_index"])
        wide.setdefault(key, {"round": row["round"], "example_index": row["example_index"],
                              "target": row["target"]})[row["method"]] = row["target_rank"]
    tables = {"per_example.csv": result["per_example"], "per_round.csv": result["per_round"],
              "final_metrics.csv": final_rows, "rank_comparison.csv": list(wide.values())}
    for filename, rows in tables.items():
        if not rows:
            continue
        with (destination / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
    return destination


def print_experiment_comparison(result, k=10):
    """Print a compact side-by-side metric table for one chosen K at every state.
    Also print overall BRI for each method; one-state runs have no BRI.
    Saved CSVs contain the detailed per-target rank and query comparisons.
    """
    print(f"\n{'Method':<30} {'State':>5} {f'Recall@{k}':>12} {f'Hits@{k}':>12}")
    for row in result["per_round"]:
        if row["k"] == k:
            print(f"{row['method']:<30} {row['round']:>5} {row['recall_percent']:>11.2f}% {row['hits_percent']:>11.2f}%")
    print("\nOverall BRI (lower is better):")
    for method, values in result["summary"].items():
        bri = "not defined for one state" if values["BRI"] is None else f"{values['BRI']:.6f}"
        print(f"  {method}: {bri}")
