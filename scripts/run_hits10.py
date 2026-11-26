"""Run Debate and live PlugIR until the first hit or the round limit."""

import argparse
from dataclasses import asdict, replace
import importlib.metadata
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_evaluation as runner
from agentir.cache import digest, load_cache, read_json
from agentir.evaluation import matched_queries
from agentir.live import generate_live_dialogue, load_candidate_captions, resolve_target_image, save_json, signature
from agentir.live_models import QWEN_ID, QWEN_REVISION, Qwen3, QwenVLAnswerer, require_cuda
from agentir.multagent_experiment.debate import DEBATE_POLICY, MAX_CALLS_PER_ROUND, generate_debate_dialogue
from agentir.multagent_experiment.setup import AGENTS, TARGET_IMAGES
from agentir.retrieval import BlipRetriever, ranks_for_targets


TOP_K = 10
TARGET_COUNT = 99
ANSWERER_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
METHODS = ("plugir", "debate")


def checkpoint_path(out, method, target):
    return out / "checkpoints" / method / (signature(target)[:20] + ".json")


def history_summary(out, method, target):
    path = checkpoint_path(out, method, target["image"])
    states = read_json(path)["states"] if path.is_file() else []
    ranks = [state["target_rank"] for state in states] or [target["initial_rank"]]
    return {"first_hit": next((r for r, rank in enumerate(ranks) if rank <= TOP_K), None),
            "ranks": ranks, "checkpoint": str(path.relative_to(out)) if states else None}


def covered(history, round_index):
    return history["first_hit"] is not None or len(history["ranks"]) > round_index


def save_report(out, targets, rounds, *, debate_policy=DEBATE_POLICY):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    histories = {target["image"]: {method: history_summary(out, method, target) for method in METHODS}
                 for target in targets}
    coverage = [sum(all(covered(history, r) for history in row.values())
                    for row in histories.values()) for r in range(rounds + 1)]
    end = max(r for r, count in enumerate(coverage) if count == len(targets))
    partial = end != rounds
    curves = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {"plugir": "Live PlugIR", "debate": f"Debate {debate_policy.rsplit('_', 1)[-1]}"}
    for method, label in labels.items():
        hits = [sum((hit := histories[target][method]["first_hit"]) is not None and hit <= r
                    for target in histories) for r in range(end + 1)]
        curves[method] = {"hits": hits, "percent": [100 * count / len(targets) for count in hits]}
        ax.plot(range(end + 1), curves[method]["percent"], marker="o", label=label)
    ax.set(xlabel="Interaction round (0 = starting caption)", ylabel="Cumulative Hits@10 (%)",
           xticks=range(end + 1), ylim=(0, 105),
           title=f"{len(targets)} paired targets through round {end}")
    if partial:
        ax.set_title(f"Partial: {coverage[rounds]}/{len(targets)} paired through round {rounds}\n"
                     f"Figure includes all {len(targets)} targets through round {end}")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hits_at_10_by_round.png", dpi=180)
    plt.close(fig)
    save_json(out / "first_hits.json", {"requested_round": rounds, "plotted_round": end,
              "partial": partial, "paired_coverage": coverage, "plotted_targets": list(histories),
              "curves": curves, "histories": histories})
    print(f"Saved Hits@10 through round {end}; {coverage[rounds]}/{len(targets)} paired "
          f"through requested round {rounds}: {out}", flush=True)


def run(out, rounds, offline):
    device = require_cuda(runner.LIVE_DEVICE)
    paths = {key: ROOT / value for key, value in read_json(ROOT / "experiment.json").items()}
    cache = load_cache(paths["cache"], paths["manifest"])
    pairs = matched_queries(read_json(paths["baseline_queries"]), read_json(paths["supplied_queries"]), cache.paths)
    selected = [(raw["img"], raw["dialog"][0]) for raw, _ in pairs if raw["img"] not in TARGET_IMAGES][:TARGET_COUNT]
    if len(selected) != TARGET_COUNT or len({name for name, _ in selected}) != TARGET_COUNT:
        raise ValueError(f"Expected {TARGET_COUNT} distinct non-pilot targets.")
    debate_settings = replace(AGENTS, max_calls_per_round=MAX_CALLS_PER_ROUND, max_targets=TARGET_COUNT)
    caption_path = ROOT / runner.CANDIDATE_CAPTIONS
    identity = {"targets": [list(item) for item in selected], "top_k": TOP_K, "retrieval": cache.report,
                "catalogue": digest(caption_path), "plugir_settings": asdict(runner.LIVE),
                "debate_settings": asdict(debate_settings), "policy": DEBATE_POLICY,
                "text_model": [QWEN_ID, QWEN_REVISION], "answerer": [QwenVLAnswerer.IDENTITY[0], ANSWERER_REVISION],
                "token_limit": runner.MAX_INPUT_TOKENS, "gpu": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda,
                "runtime": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "fast-pytorch-kmeans")},
                "sources": {name: digest(ROOT / name) for name in (
                    "scripts/run_hits10.py", "src/agentir/live.py", "src/agentir/live_models.py",
                    "src/agentir/retrieval.py", "src/agentir/plugir_upstream.py",
                    "src/agentir/multagent_experiment/debate.py", "src/agentir/multagent_experiment/multi_agent.py",
                    "src/agentir/multagent_experiment/agent_prompts.py",
                )}}
    manifest = out / "manifest.json"
    if manifest.is_file():
        saved = read_json(manifest)
        if saved["identity"] != identity:
            raise ValueError("Run inputs changed. Resume with the original code and inputs or use a fresh output folder.")
    retriever = BlipRetriever(cache, device=runner.LIVE_DEVICE, batch_size=runner.TEXT_BATCH_SIZE,
                              threads=runner.CPU_THREADS, local_files_only=offline)
    ids = {name: index for index, name in enumerate(cache.paths)}
    if manifest.is_file():
        targets = saved["targets"]
    else:
        targets = [{"image": name, "caption": caption,
                    "initial_rank": int(ranks_for_targets(retriever.scores([caption]), [ids[name]])[0])}
                   for name, caption in selected]
        save_json(manifest, {"identity": identity, "targets": targets})

    try:
        catalogue = load_candidate_captions(caption_path, cache.paths)
        questioner = Qwen3(out / "qwen_calls", device=runner.LIVE_DEVICE, offline=offline,
                           max_input_tokens=runner.MAX_INPUT_TOKENS, max_output_tokens=1024,
                           seed=runner.LIVE.seed)
        answerer = QwenVLAnswerer(out / "answerer_calls", device=runner.LIVE_DEVICE, offline=offline,
                                  revision=ANSWERER_REVISION)
        experiment_id = signature(identity)
        image_manifest = out / "image_hashes.json"
        image_hashes = read_json(image_manifest) if image_manifest.is_file() else {}
        for target in targets:
            name = target["image"]
            image = None
            for method in METHODS:
                if covered(history_summary(out, method, target), rounds):
                    continue
                if image is None:
                    image = resolve_target_image(ROOT / runner.IMAGE_ROOT, name,
                        archive=None if runner.IMAGE_ARCHIVE is None else ROOT / runner.IMAGE_ARCHIVE,
                        download=runner.DOWNLOAD_MISSING_TARGETS and not offline)
                    image_hash = digest(image)
                    if name in image_hashes and image_hashes[name] != image_hash:
                        raise ValueError(f"Target image changed since checkpoint: {name}")
                    if name not in image_hashes:
                        image_hashes[name] = image_hash
                        save_json(image_manifest, image_hashes)
                common = dict(initial_caption=target["caption"], target_id=ids[name], image_path=image,
                              captions=catalogue, questioner=questioner, answerer=answerer,
                              n_states=rounds + 1, checkpoint_path=checkpoint_path(out, method, name),
                              stop_at_rank=TOP_K)
                print(f"{method} · {name} · through round {rounds}", flush=True)
                try:
                    if method == "plugir":
                        generate_live_dialogue(retriever, settings=runner.LIVE, run_identity=experiment_id, **common)
                    else:
                        generate_debate_dialogue(retriever, settings=debate_settings, **common)
                finally:
                    questioner.records.clear()
                    answerer.records.clear()
    finally:
        save_report(out, targets, rounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Run folder. Reuse it to resume.")
    parser.add_argument("--rounds", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--offline", action="store_true", help="Use cached models and target images.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate the figure from saved checkpoints.")
    args = parser.parse_args()
    if args.report_only:
        manifest = read_json(args.out / "manifest.json")
        save_report(args.out, manifest["targets"], args.rounds, debate_policy=manifest["identity"]["policy"])
    else:
        # Check the plotting dependency before reserving GPU model memory.
        import matplotlib
        run(args.out, args.rounds, args.offline)


if __name__ == "__main__":
    main()
