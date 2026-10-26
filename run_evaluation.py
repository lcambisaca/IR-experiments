"""Open this file in VS Code, edit the settings/methods below, then click Run.

Uses our saved 50k image vectors. No image encoding, paid API calls.
First retrieval may download BLIP if not cached yet.
"""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import random

import torch

from agentir.cache import digest, load_cache, read_json
from agentir.evaluation import (
    evaluate_experiments, matched_queries, print_experiment_comparison,
    save_experiment_results,
)
from agentir.retrieval import BlipRetriever


# SETTINGS: start small; every search still includes all 50,000 candidate images.
N_TARGETS = 4            # Set to 32 for our reference sample, or None for all 2,064.
N_STATES = 3           # 1 = caption only; 11 = caption plus ten dialogue turns.
K_VALUES = [1, 10, 50]
DEVICE = "cpu"        # Use "cuda" only with a working NVIDIA PyTorch setup.
TEXT_BATCH_SIZE = 8   # Maximum texts per encoding call; methods below run one case at a time.
CPU_THREADS = 4
OFFLINE = False       # True requires BLIP model files already on this machine.
SEED = 42
RUN_LABEL = "first_comparison"


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


def main():
    """Load the shared cache/model and evaluate the registered methods together.
    Resolve paths relative to this file, print metrics, and save a new result
    folder. Click Run again after adding an idea to EXPERIMENTS to compare it
    against the same caption/raw/supplied references, targets, and image pool.
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
    if N_TARGETS is not None:
        if isinstance(N_TARGETS, bool) or not isinstance(N_TARGETS, int) or not 1 <= N_TARGETS <= available:
            raise ValueError(f"N_TARGETS must be 1..{available} or None for all targets.")
        pairs = pairs[:N_TARGETS]
    if not isinstance(N_STATES, int) or not 1 <= N_STATES <= 11:
        raise ValueError("N_STATES must be 1..11.")
    print(f"Evaluating {len(pairs)}/{available} targets, {N_STATES} states, {len(cache.paths):,} candidate images.")
    print("References: caption_only, raw_dialogue, supplied_plugir (saved queries, not a live LLM).")
    print(f"Custom experiments: {', '.join(EXPERIMENTS) or 'none registered yet'}")
    retriever = BlipRetriever(cache, device=DEVICE, batch_size=TEXT_BATCH_SIZE,
                              threads=CPU_THREADS, local_files_only=OFFLINE)
    result = evaluate_experiments(retriever, pairs, EXPERIMENTS,
                                  rounds=N_STATES, ks=K_VALUES) # Here we the experiments are evaluated with the specified number of rounds and K values.
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
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = save_experiment_results(result, root / "runs" / "experiments" / stamp)
    # Save the editable runner used for this run, so later edits remain traceable.
    (destination / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print_experiment_comparison(result, k=10 if 10 in K_VALUES else min(K_VALUES))
    print(f"\nSaved comparison files to: {destination}")
    print("Open rank_comparison.csv for side-by-side target ranks; report.json includes query history.")


if __name__ == "__main__":
    main()
