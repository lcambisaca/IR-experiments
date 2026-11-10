"""Open this file in VS Code, edit the settings/methods below, then click Run.

One entry point for saved baselines, custom methods and optional live experiments.
Uses the saved 50k image vectors. No image re-encoding or paid API calls.
Live Qwen/BLIP-2 inference/downloads happen only when explicitly enabled below.
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


# SETTINGS: start small; every search still includes all 50,000 candidate images.
N_TARGETS = 1         # Set to 32 for our reference sample, or None for all 2,064.
N_STATES = 2       # 1 = caption only; 11 = caption plus ten dialogue turns.
START_TARGET = 0      # Choose a different first matched example without changing the image pool.
K_VALUES = [1, 10, 100]
DEVICE = "cuda"        # Use "cuda" only with a working NVIDIA PyTorch setup.
TEXT_BATCH_SIZE = 8   # Maximum texts per encoding call; methods below run one case at a time.
CPU_THREADS = 4
OFFLINE = False       # True requires BLIP model files already on this machine.
SEED = 42
RUN_LABEL = "three_agents_pilot"

# OPTIONAL LIVE METHOD: False keeps your existing CPU/saved-baseline workflow.
# On GCP, start with N_TARGETS=1, N_STATES=3, DEVICE="cuda", then enable this.
RUN_LIVE_PLUGIR = False
LIVE_DEVICE = "cuda"       # Qwen/BLIP-2 generation GPU; DEVICE above controls BLIP retrieval.
MAX_INPUT_TOKENS = 4096    # Stop rather than silently truncate older Qwen dialogue.
# Output lengths/temperatures come directly from the original PlugIR functions:
# questions 32 tokens, filters 10, reformulation/paraphrase 512. BLIP-2 uses checkpoint defaults.
CANDIDATE_CAPTIONS = "assets/protocol/visdial_captions.json"  # Public candidate catalogue.
IMAGE_ROOT = "data/images" # Target JPEGs are needed by BLIP-2, not by saved-baseline retrieval.
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

# THREE-AGENT EXPERIMENT: new prompts/peer revision; original clustering and KL helpers.
# For the first GCP pilot set N_TARGETS=1, N_STATES=2, DEVICE="cuda", then enable.
# Both modes produce their own shared-answer histories and neutral official rankings.
RUN_THREE_AGENTS = True
AGENT_INPUT_TOKENS = 8192
AGENT_MODES = (False, True)  # no critique, then one peer-critique/revision round
from agentir.multi_agent import AgentSettings
AGENTS = AgentSettings(
    seed=1021, threshold_low=500, exclude_top_k=10,
    replacement_attempts=1, max_calls_per_round=40,
    diagnostic_top_k=10, max_targets=2,
)


def my_first_experiment(context, retriever):
    """Replace this function body with your own query-building idea.
    For now it returns the unchanged raw query, so enabling it should reproduce
    raw_dialogue, not claim a new improvement. Context has no target label.
    Return a description, vector_output(...), or SearchOutput(...) for fusion.
    """
    return context.current_query


# ADD METHODS HERE. Each name appears as its own comparison column.
# No template method runs until you uncomment its registration below.
EXPERIMENTS = {
    # "my_first_experiment": my_first_experiment,
}


def prepare_qwen_live(root, retriever, pairs, *, n_states, offline):
    """Build the live Qwen/BLIP-2 method using the SAME selected targets and BLIP cache.
    This preparation runs only when enabled. It generates/resumes conversations
    and returns ordinary search callbacks plus provenance and saved dialogue.
    """
    from agentir.live import prepare_live_experiment
    return prepare_live_experiment(root, retriever, pairs, n_states=n_states, settings=LIVE,
        device=LIVE_DEVICE, offline=offline, candidate_captions=CANDIDATE_CAPTIONS,
        image_root=IMAGE_ROOT, image_archive=IMAGE_ARCHIVE, download_missing=DOWNLOAD_MISSING_TARGETS,
        max_input_tokens=MAX_INPUT_TOKENS)


def prepare_three_agents(root, retriever, pairs, *, n_states, offline):
    """Prepare both agent conditions for the same target selection and common evaluator."""
    from agentir.multi_agent import prepare_agent_experiment
    return prepare_agent_experiment(root, retriever, pairs, n_states=n_states,
        settings=AGENTS, device=LIVE_DEVICE, offline=offline,
        candidate_captions=CANDIDATE_CAPTIONS, image_root=IMAGE_ROOT,
        image_archive=IMAGE_ARCHIVE, download_missing=DOWNLOAD_MISSING_TARGETS,
        max_input_tokens=AGENT_INPUT_TOKENS, modes=AGENT_MODES)


# FUTURE EXPERIMENTS THAT NEED SETUP/GENERATION FIRST:
# Add a function with prepare_qwen_live's signature returning PreparedExperiment.
# Ordinary per-state text/vector/fusion callbacks still belong in EXPERIMENTS above.
PREPARED_EXPERIMENTS = {
    # "my_agent_tree": prepare_my_agent_tree,
}


def collect_experiments(root, retriever, pairs):
    """Combine direct callbacks and enabled preparation-stage methods into one comparison.
    Check unique names and serializable provenance/artifacts. A preparation may
    generate live histories; metric callbacks receive no target labels/ranks.
    Return all callbacks and bundles to be exported with the common run.
    """
    methods, bundles = dict(EXPERIMENTS), {}
    preparers = dict(PREPARED_EXPERIMENTS)
    if RUN_LIVE_PLUGIR:
        if "qwen_live" in preparers:
            raise ValueError("qwen_live is reserved for the built-in live preparation.")
        preparers["qwen_live"] = prepare_qwen_live
    if RUN_THREE_AGENTS:
        if "three_agents" in preparers:
            raise ValueError("three_agents is reserved for the built-in agent preparation.")
        preparers["three_agents"] = prepare_three_agents
    reserved = {"caption_only", "raw_dialogue", "supplied_plugir", "round", "example_index", "target"}
    if any(not isinstance(name, str) or not name.strip() or name in reserved or not callable(callback)
           for name, callback in methods.items()):
        raise ValueError("Use unique, non-reserved names and callable EXPERIMENTS.")
    for name, prepare in preparers.items():
        if not isinstance(name, str) or not name.isidentifier() or not callable(prepare):
            raise ValueError("Use identifier names and callable PREPARED_EXPERIMENTS.")
        bundle = prepare(root, retriever, pairs, n_states=N_STATES, offline=OFFLINE)
        if not isinstance(bundle, PreparedExperiment) or not isinstance(bundle.methods, dict) or not bundle.methods:
            raise TypeError("A preparation must return PreparedExperiment with at least one method.")
        for method, callback in bundle.methods.items():
            if not isinstance(method, str) or not method.strip() or method in reserved or method in methods or not callable(callback):
                raise ValueError(f"Prepared method names must be unique and callable: {method}")
        if not isinstance(bundle.metadata, dict) or not isinstance(bundle.artifacts, dict):
            raise TypeError("Prepared metadata and artifacts must be dictionaries.")
        for filename in bundle.artifacts:
            if not isinstance(filename, str) or not filename.endswith(".json") or "/" in filename or "\\" in filename or ":" in filename:
                raise ValueError("Prepared artifacts must use plain .json filenames.")
        json.dumps({"metadata": bundle.metadata, "artifacts": bundle.artifacts}, allow_nan=False)
        methods.update(bundle.methods)
        bundles[name] = bundle
    return methods, bundles


def main():
    """Load the shared cache/model and evaluate the registered methods together.
    Resolve paths relative to this file, print metrics, and save a new result
    folder. Click Run again after adding an idea to EXPERIMENTS to compare it
    against the same caption/raw/supplied references, targets, and image pool.
    Optional preparers run first; all enabled methods then share the metric/export
    pipeline. Saved-only defaults do not load Qwen or BLIP-2.
    """
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
    if type(START_TARGET) is not int or not 0 <= START_TARGET < available:
        raise ValueError("START_TARGET must select an available matched example.")
    pairs = pairs[START_TARGET:]
    if N_TARGETS is not None:
        if isinstance(N_TARGETS, bool) or not isinstance(N_TARGETS, int) or not 1 <= N_TARGETS <= len(pairs):
            raise ValueError(f"N_TARGETS must be 1..{len(pairs)} or None for all remaining targets.")
        pairs = pairs[:N_TARGETS]
    if type(N_STATES) is not int or not 1 <= N_STATES <= 11:
        raise ValueError("N_STATES must be 1..11.")
    if not K_VALUES or any(type(k) is not int or not 1 <= k <= len(cache.paths) for k in K_VALUES):
        raise ValueError("K_VALUES must contain positive integers inside the image pool.")
    print(f"Evaluating {len(pairs)}/{available} targets, {N_STATES} states, {len(cache.paths):,} candidate images.")
    print("References: caption_only, raw_dialogue, supplied_plugir (saved queries, not a live LLM).")
    print(f"Custom experiments: {', '.join(EXPERIMENTS) or 'none registered yet'}")
    if RUN_THREE_AGENTS:
        AGENTS.validate(len(cache.paths), N_STATES)
        if len(pairs) > AGENTS.max_targets:
            raise ValueError("Set N_TARGETS=1 for the first agent pilot, or explicitly increase AGENTS.max_targets.")
    retriever = BlipRetriever(cache, device=DEVICE, batch_size=TEXT_BATCH_SIZE,
                              threads=CPU_THREADS, local_files_only=OFFLINE)
    methods, prepared = collect_experiments(root, retriever, pairs)
    print(f"Enabled comparison methods: {', '.join(methods) or 'saved references only'}")
    result = evaluate_experiments(retriever, pairs, methods, rounds=N_STATES, ks=K_VALUES)
    result["run"] = {
        "label": RUN_LABEL, "seed": SEED, "device": DEVICE,
        "text_batch_size": TEXT_BATCH_SIZE, "threads": CPU_THREADS,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "tokenizers", "pillow", "numpy")},
        "cache": cache.report,
        "input_sha256": {name: digest(paths[name]) for name in
                         ("baseline_queries", "supplied_queries", "manifest")},
        "runner_sha256": digest(Path(__file__)),
        "selected_target_start": START_TARGET,
        "prepared_experiments": {name: {"methods": list(bundle.methods), "metadata": bundle.metadata,
            "artifacts": [f"prepared/{name}/{filename}" for filename in bundle.artifacts]}
            for name, bundle in prepared.items()},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = save_experiment_results(result, root / "runs" / "experiments" / stamp)
    # Save the editable runner used for this run, so later edits remain traceable.
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name, bundle in prepared.items():
        folder = destination / "prepared" / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, content in bundle.artifacts.items():
            (folder / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print_experiment_comparison(result, k=10 if 10 in K_VALUES else min(K_VALUES))
    print(f"\nSaved comparison files to: {destination}")
    print("Open rank_comparison.csv for side-by-side target ranks; report.json includes query history.")


if __name__ == "__main__":
    main()
