"""Multi-agent configuration and optional pilot targets for the shared evaluator."""
from dataclasses import replace
from .multi_agent import AgentSettings, prepare_agent_experiment

TARGET_IMAGES = (
    "unlabeled2017/000000364869.jpg",  # Group posing in front of a cake; CPU initial rank 9.
    "unlabeled2017/000000562109.jpg",  # Giraffe standing in an open field; CPU initial rank 89.
)
N_STATES = 11  # Default for direct calls; run_evaluation.py supplies its own N_STATES.
RUN_LABEL = "specialists_shared_cake_giraffe"
AGENT_MODES = (True,)  # Critique baseline; add False to include minimum-KL separately.
AGENT_INPUT_TOKENS = 8192
AGENTS = AgentSettings(
    seed=1021, threshold_low=500, exclude_top_k=10,
    max_calls_per_round=6, questions_per_agent=5,
    max_targets=2,
)


def select_pilot_pairs(pairs):
    """Select cake and giraffe by ID in pilot order, before preparing any method."""
    matches = {name: [] for name in TARGET_IMAGES}
    for pair in pairs:
        name = pair[0]["img"]
        if name in matches:
            matches[name].append(pair)
    for name, found in matches.items():
        if len(found) != 1:
            raise ValueError(f"Expected one matched pair for {name}; found {len(found)}.")
    return [matches[name][0] for name in TARGET_IMAGES]


def prepare_three_agents(root, retriever, pairs, *, n_states=N_STATES, offline=False,
                         settings=None, modes=AGENT_MODES, device="cuda",
                         candidate_captions="assets/protocol/visdial_captions.json",
                         image_root="data/images", image_archive=None,
                         download_missing=True, max_input_tokens=AGENT_INPUT_TOKENS,
                         run_name="shared_agents_simple_v1", answerer="blip2"):
    
    """Prepare the supplied matched targets and return a PreparedExperiment.

    The caller owns target selection, scoring, and export, so the same cases
    can be evaluated against teammates' methods. Importing starts no models.
    """
    if settings is None:
        settings = replace(AGENTS, max_targets=max(AGENTS.max_targets, len(pairs)))
    return prepare_agent_experiment(
        root, retriever, pairs, n_states=n_states, offline=offline,
        settings=settings, modes=modes, device=device,
        candidate_captions=candidate_captions, image_root=image_root,
        image_archive=image_archive, download_missing=download_missing,
        max_input_tokens=max_input_tokens, run_name=run_name, answerer=answerer,
    )
