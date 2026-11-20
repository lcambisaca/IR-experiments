"""Evaluate saved baselines and optional experiments using cached image vectors.

Edit the settings below, then run this file. Live models load only when enabled.
"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random

import torch

from agentir.cache import digest, load_cache, read_json
from agentir.evaluation import (
    PreparedExperiment, evaluate_experiments, matched_queries, print_experiment_comparison,
    save_experiment_results,
)
from agentir.retrieval import BlipRetriever
from agentir.live import LiveSettings
from agentir.multagent_experiment.setup import (
    RUN_LABEL as PILOT_LABEL, prepare_three_agents, select_pilot_pairs,
)


# SETTINGS: start small; every search still includes all 50,000 candidate images.
START_TARGET = 0      # Choose a different first matched example without changing the image pool.
K_VALUES = [1, 10, 100]
TEXT_BATCH_SIZE = 8   # Maximum texts per encoding call; methods below run one case at a time.
CPU_THREADS = 4
OFFLINE = False       # True requires BLIP model files already on this machine.
SEED = 42
RUN_LABEL = "baseline_full_50k"

N_TARGETS = 32         # Set to 32 for our reference sample, or None for all 2,064.
N_STATES = 11           # 1 = caption only; 11 = caption plus ten dialogue turns.
DEVICE = "cuda"        # Use "cuda" only with a working NVIDIA PyTorch setup.

# Saved comparisons are optional; live methods use their own generated dialogue.
INCLUDE_SAVED_REFERENCES = False
SAVED_REFERENCES = ("raw_dialogue",)  # Also available: "caption_only", "supplied_plugir".

RUN_LIVE_PLUGIR = True
RUN_MULTI_AGENT = True
USE_PILOT_TARGETS = True  # True: cake and giraffe; False: START_TARGET and N_TARGETS.

LIVE_DEVICE = "cuda"       # Qwen/BLIP-2 generation GPU; DEVICE above controls BLIP retrieval.
SEARCH = "rewrite"         # "rewrite" = original PlugIR search; "facts" = fact fusion inside the live loop (facts_e2e.py).
SELECTION = "plugir_kl"    # "plugir_kl" = original PlugIR question choice; "probe_vlm" = answer-entropy agent (probe_agent.py).
MAX_INPUT_TOKENS = 4096    # Stop rather than silently truncate older Qwen dialogue.

CANDIDATE_CAPTIONS = "assets/protocol/visdial_captions.json"  # Public candidate catalogue.
IMAGE_ROOT = "data/images" # Target JPEGs 
IMAGE_ARCHIVE = None       # Optional local corpus tar; restore only selected target JPEGs.
DOWNLOAD_MISSING_TARGETS = True  # Fetch only chosen missing COCO JPEGs; OFFLINE disables this.
LIVE = LiveSettings(
    seed=1021,              # Local reproducibility; also original reconstruction seed.
    q_n=5,                  # Generate five alternatives sequentially, as in supplied filename.
    recall_hitting=10,      # Exclude current top 10 from original KL comparison.
    threshold_low=500,      # Original candidate window: 500 down to 50 across ten interactions.
    reconstruct=True,       # Rewrite caption using generated answers after each round.
    referring=True,         # Original ten-cluster/entropy candidate-caption prompting.
    filtering=True,         # Original sequential question filtering and retry/fallback logic.
    select=True,            # Original minimum-KL question choice.
)


def prepare_qwen_live(root, retriever, pairs, *, n_states, offline):
    """Generate or resume live conversations for the selected targets."""
    from agentir.live import prepare_live_experiment
    return prepare_live_experiment(root, retriever, pairs, n_states=n_states, settings=LIVE,
        device=LIVE_DEVICE, offline=offline, candidate_captions=CANDIDATE_CAPTIONS,
        image_root=IMAGE_ROOT, image_archive=IMAGE_ARCHIVE, download_missing=DOWNLOAD_MISSING_TARGETS,
        max_input_tokens=MAX_INPUT_TOKENS, search=SEARCH, selection=SELECTION)


def collect_experiments(root, retriever, pairs, *, n_states=None):
    """Prepare enabled experiments and collect their search functions and outputs."""
    methods = {}
    prepared_experiments = {}
    preparers = {}
    if RUN_MULTI_AGENT:
        preparers["three_agents"] = prepare_three_agents

    if RUN_LIVE_PLUGIR:
        preparers["qwen_live"] = prepare_qwen_live

    for name, prepare in preparers.items():
        prepared_experiment = prepare(
            root, retriever, pairs,
            n_states=N_STATES if n_states is None else n_states,
            offline=OFFLINE,
        )
        validate_prepared_experiment(prepared_experiment, methods)
        methods.update(prepared_experiment.methods)
        prepared_experiments[name] = prepared_experiment

    return methods, prepared_experiments


def select_target_pairs(pairs):
    """Select one shared target batch for every enabled method."""
    if USE_PILOT_TARGETS:
        return select_pilot_pairs(pairs)
    available = len(pairs)
    if type(START_TARGET) is not int or not 0 <= START_TARGET < available:
        raise ValueError("START_TARGET must select an available matched example.")
    pairs = pairs[START_TARGET:]
    if N_TARGETS is not None:
        if type(N_TARGETS) is not int or not 1 <= N_TARGETS <= len(pairs):
            raise ValueError(f"N_TARGETS must be 1..{len(pairs)} or None for all remaining targets.")
        pairs = pairs[:N_TARGETS]
    return pairs


def main():
    """Load the cache, evaluate enabled methods, and save metrics and run details."""
    references = SAVED_REFERENCES if INCLUDE_SAVED_REFERENCES else ()
    if not references and not (RUN_LIVE_PLUGIR or RUN_MULTI_AGENT):
        raise ValueError("Enable RUN_LIVE_PLUGIR, RUN_MULTI_AGENT, or INCLUDE_SAVED_REFERENCES.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    root = Path(__file__).resolve().parent
    config_path = root / "experiment.json"
    config = read_json(config_path)
    paths = {name: (root / value).resolve() for name, value in config.items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]),
                            read_json(paths["supplied_queries"]), cache.paths)
    available = len(pairs)
    n_states = N_STATES
    device = LIVE_DEVICE if RUN_MULTI_AGENT else DEVICE

    pairs = select_target_pairs(pairs)
    if type(n_states) is not int or not 1 <= n_states <= 11:
        raise ValueError("N_STATES must be 1..11.")
    if not K_VALUES or any(type(k) is not int or not 1 <= k <= len(cache.paths) for k in K_VALUES):
        raise ValueError("K_VALUES must contain positive integers inside the image pool.")
    print(f"Evaluating {len(pairs)}/{available} targets, {n_states} states, {len(cache.paths):,} candidate images.")
    print(f"Saved references: {', '.join(references) or 'disabled'}")
    retriever = BlipRetriever(cache, device=device, batch_size=TEXT_BATCH_SIZE,
                              threads=CPU_THREADS, local_files_only=OFFLINE)
    methods, prepared_experiments = collect_experiments(root, retriever, pairs, n_states=n_states)
    print(f"Enabled comparison methods: {', '.join(methods) or 'saved references only'}")
    result = evaluate_experiments(
        retriever, pairs, methods, rounds=n_states, ks=K_VALUES, references=references,
    )
    result["run"] = {
        "label": PILOT_LABEL if USE_PILOT_TARGETS else RUN_LABEL, "seed": SEED, "device": device,
        "text_batch_size": TEXT_BATCH_SIZE, "threads": CPU_THREADS,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "tokenizers", "pillow", "numpy")},
        "cache": cache.report,
        "input_sha256": {name: digest(paths[name]) for name in
                         ("baseline_queries", "supplied_queries", "manifest")},
        "runner_sha256": digest(Path(__file__)),
        "target_selection": "pilot" if USE_PILOT_TARGETS else "range",
        "selected_target_start": None if USE_PILOT_TARGETS else START_TARGET,
        "selected_target_images": [raw["img"] for raw, _ in pairs],
        "prepared_experiments": {name: {"methods": list(prepared_experiment.methods), "metadata": prepared_experiment.metadata,
            "artifacts": [f"prepared/{name}/{filename}" for filename in prepared_experiment.artifacts]}
            for name, prepared_experiment in prepared_experiments.items()},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = save_experiment_results(result, root / "runs" / "experiments" / stamp)
    # Save the editable runner used for this run, so later edits remain traceable.
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name, prepared_experiment in prepared_experiments.items():
        folder = destination / "prepared" / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, content in prepared_experiment.artifacts.items():
            (folder / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print_experiment_comparison(result, k=10 if 10 in K_VALUES else min(K_VALUES))
    print(f"\nSaved comparison files to: {destination}")
    print("Open rank_comparison.csv for side-by-side target ranks; report.json includes query history.")


def validate_prepared_experiment(prepared_experiment, existing_methods):
    """Check search functions and saved outputs before adding an experiment."""
    reserved = {"caption_only", "raw_dialogue", "supplied_plugir", "round", "example_index", "target"}
    if not isinstance(prepared_experiment, PreparedExperiment) or not isinstance(prepared_experiment.methods, dict) or not prepared_experiment.methods:
        raise TypeError("A preparation must return PreparedExperiment with at least one method.")
    for method, callback in prepared_experiment.methods.items():
        if not isinstance(method, str) or not method.strip() or method in reserved or method in existing_methods or not callable(callback):
            raise ValueError(f"Prepared method names must be unique and callable: {method}")
    if not isinstance(prepared_experiment.metadata, dict) or not isinstance(prepared_experiment.artifacts, dict):
        raise TypeError("Prepared metadata and artifacts must be dictionaries.")
    for filename in prepared_experiment.artifacts:
        if not isinstance(filename, str) or not filename.endswith(".json") or "/" in filename or "\\" in filename or ":" in filename:
            raise ValueError("Prepared artifacts must use plain .json filenames.")
    json.dumps({"metadata": prepared_experiment.metadata, "artifacts": prepared_experiment.artifacts}, allow_nan=False)


if __name__ == "__main__":
    main()