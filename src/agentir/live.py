"""Storage and runtime wrapper around the authors' actual PlugIR generation code.

The scientific loop/functions live unchanged in third_party/PlugIR. This file
supplies local models, device placement, selected image paths and checkpoints.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from types import SimpleNamespace
from typing import Optional
import urllib.request

from PIL import Image
import torch
from torch.nn.functional import normalize

from .cache import digest, read_json
from .retrieval import ranks_for_targets
from .plugir_upstream import bind_original_text_encoder, load_generator, provenance


@dataclass(frozen=True)
class LiveSettings:
    """Original CLI arguments; defaults match the authors' supplied q_n_5 file.
    These are not training settings. Ten clusters and three filter attempts are
    constants inside upstream code, so we do not introduce replacement options.
    """
    seed: int = 1021
    q_n: int = 5
    recall_hitting: int = 10
    threshold_low: int = 500
    reconstruct: bool = True
    referring: bool = True
    filtering: bool = True
    select: bool = True

    def validate(self, pool_size, n_states=11):
        """Reject settings that would make the unchanged source index an empty pool.
        Upstream's non-referring branch indexes nonexistent completions when q_n
        exceeds one; reject that combination rather than silently patch its logic.
        """
        if type(n_states) is not int or not 1 <= n_states <= 11:
            raise ValueError('Use 1..11 states: caption plus up to ten interactions.')
        if any(type(n) is not int or n < 1 for n in (self.q_n, self.recall_hitting, self.threshold_low)):
            raise ValueError('Original count settings must be positive integers.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Use an integer seed in 0..2**32-1.')
        if any(type(flag) is not bool for flag in (self.reconstruct, self.referring, self.filtering, self.select)):
            raise ValueError('Original switches must be boolean.')
        if self.recall_hitting > pool_size or self.threshold_low > pool_size:
            raise ValueError('Original search windows cannot exceed the image pool.')
        smallest = int(self.threshold_low - max(0, n_states - 2) * self.threshold_low / 10.)
        if n_states > 1 and ((self.referring and smallest < 10) or (self.select and smallest <= self.recall_hitting)):
            raise ValueError('Final window must support ten clusters and leave candidates outside top K.')
        if not self.referring and self.q_n != 1:
            raise ValueError('Unchanged upstream non-referring branch requires q_n=1.')


def signature(value):
    """Fingerprint JSON settings/text so compatible checkpoints and model calls can be reused."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_json(path, value):
    """Save complete JSON through a temporary file, preserving the previous checkpoint on interruption.
    Use one writer per run folder; this is crash protection, not a concurrency lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_candidate_captions(path, image_paths):
    """Map the public catalogue captions to exact manifest filenames, never by assumed row order.
    These describe all retrieval candidates; they are not hidden answers from the target dialogue.
    """
    rows = read_json(path)
    result = {}
    for row in rows:
        name, captions = row["id"], row["caption"]
        if name in result or not isinstance(captions, list) or not captions or not isinstance(captions[0], str) or not captions[0].strip():
            raise ValueError("Duplicate image or missing catalogue caption.")
        result[name] = captions
    if set(result) != set(image_paths):
        raise ValueError("Candidate captions must cover exactly the saved image manifest.")
    return result


def resolve_target_image(root, name, *, archive=None, download=False):
    """Find one target JPEG, optionally restore only it from a tar or fetch it from COCO.
    Never extract/download all images. Validate the filename and image before
    saving bytes under IMAGE_ROOT; downloads contain no model or API credentials.
    """
    from io import BytesIO
    from PIL import Image
    relative = PurePosixPath(name)
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not re.fullmatch(r"unlabeled2017/\d{12}\.jpg", name) or not path.is_relative_to(root):
        raise ValueError("Expected a safe COCO unlabeled2017 filename from the manifest.")
    if path.is_file():
        with Image.open(path) as image:
            image.verify()
        return path
    data = None
    limit = 20 * 1024**2
    if archive is not None:
        with tarfile.open(archive, "r:*") as stream:
            for member in stream:
                # Read a matching regular file; do not trust/extract archive paths or links.
                if member.isfile() and (member.name == name or member.name.endswith("/" + name)):
                    if member.size > limit:
                        raise ValueError("Archived target image is unexpectedly large.")
                    with stream.extractfile(member) as image_stream:
                        data = image_stream.read(limit + 1)
                    break
    if data is None and download:
        print(f"Downloading one target image: {name}", flush=True)
        with urllib.request.urlopen("http://images.cocodataset.org/" + name, timeout=30) as response:
            data = response.read(limit + 1)
    if data is None:
        raise FileNotFoundError(f"Missing {path}. Set IMAGE_ARCHIVE or enable DOWNLOAD_MISSING_TARGETS.")
    if len(data) > limit:
        raise ValueError("Target download exceeds the image size limit.")
    with Image.open(BytesIO(data)) as image:
        image.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jpg.writing")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def generated_query_method(conversations, *, raw=False):
    """Wrap completed live queries for the existing side-by-side evaluation interface.
    The callback ignores saved human dialogue and returns only this method's
    generated text at that state. Raw=True is a same-conversation diagnostic.
    """
    def method(context, retriever):
        """Select the pre-generated live state, without passing target labels to a model."""
        return conversations[context.example_index]["states"][context.round_index]["raw_query" if raw else "query"]
    return method


def capture_rng():
    """Store Torch CPU/CUDA random states as JSON so KMeans can resume exactly.
    Qwen isolates its own sampling RNG. The original clustering library consumes
    Torch randomness, so completed-round checkpoints must include this state.
    """
    return {"cpu": torch.get_rng_state().tolist(),
            "cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    """Restore a checkpoint's Torch random stream before the next original round."""
    torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint GPU count changed; use a fresh run identity.")
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]])


def unchanged_progress(items):
    """Supply the original target-loop iterator without adding another progress bar."""
    return items


def generate_live_dialogue(retriever, *, initial_caption, target_id, image_path, captions,
                           questioner, answerer, settings, n_states, checkpoint_path, run_identity,
                           _kmeans=None):
    """Execute/resume the authors' loop using our selected cache and local transports.
    Only the initial caption enters the dialogue. BLIP-2 alone sees target pixels;
    target ranks are measured after a round and never influence question choice.
    _kmeans is a test-double seam; normal runs always use fast-pytorch-kmeans.
    """
    import time
    settings.validate(len(retriever.cache.paths), n_states)
    if _kmeans is None:
        from fast_pytorch_kmeans import KMeans
        _kmeans = KMeans(n_clusters=10, mode="cosine", verbose=0)
    identity = {"run": run_identity, "initial_caption": initial_caption,
                "target": retriever.cache.paths[target_id], "image_sha256": digest(image_path),
                "settings": asdict(settings), "upstream": provenance()}
    case_id = signature(identity)
    path = Path(checkpoint_path)
    namespace = {"torch": torch, "normalize": normalize, "Optional": Optional, "Image": Image,
                 "time": time, "BlipForImageTextRetrieval": type(retriever.model),
                 "SEED": settings.seed, "q_n": settings.q_n, "recall_hitting": settings.recall_hitting,
                 "threshold_low": settings.threshold_low, "reconstruct": settings.reconstruct,
                 "referring": settings.referring, "filtering": settings.filtering, "select": settings.select,
                 "openai": SimpleNamespace(ChatCompletion=questioner), "device": answerer.device,
                 "img_embs": retriever.images, "search_space": retriever.cache.paths,
                 "captions": [{"caption": captions[name]} for name in retriever.cache.paths],
                 "visdial": [{"img": identity["target"], "dialog": [initial_caption]}],
                 "s_idx": 0, "e_idx": 1, "tqdm": unchanged_progress, "kmeans": _kmeans,
                 "target_image_path": str(image_path), "blip2": None, "processor2": None,
                 "dial_new": [], "dial_new_recon": []}
    original_answer, loop_code, audit = load_generator(namespace)
    bind_original_text_encoder(namespace, retriever)

    def observed_rank(text):
        """Log rank with the original generation encoder; this value never enters its loop."""
        scores = normalize(namespace["get_text_features"](text), dim=-1) @ retriever.images.T
        return int(ranks_for_targets(scores.cpu(), [target_id])[0])

    def cached_answer(img_path="", query="", model_caps=None, processor_caps=None):
        """Park Qwen, run/cache the original BLIP-2 helper, then release answerer VRAM."""
        questioner.park()
        try:
            return answerer.answer_original(original_answer, img_path, query)
        finally:
            answerer.park()

    namespace["generate_answer"] = cached_answer
    # Keep RNG changes local to this case. Explicit KMeans seeding is an operational
    # reproducibility choice: upstream exposes SEED but does not seed Torch itself.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(settings.seed)
        if path.is_file():
            state = read_json(path)
            if state["identity"] != identity:
                raise ValueError("Checkpoint identity mismatch; keep source/models/cache/settings together.")
        else:
            state = {"identity": identity, "source_audit": audit, "dialogue": [],
                     "dialog": [initial_caption], "reformulated": [initial_caption],
                     "states": [{"round": 0, "query": initial_caption, "raw_query": initial_caption,
                                 "target_rank": observed_rank(initial_caption)}], "rng": capture_rng()}
            save_json(path, state)
        call_start = 0

        def restore_round_state():
            """Replace the source's initial state with the last completed prefix on resume."""
            namespace["dial_gen"] = list(state["dialog"])
            namespace["captions_recon"] = list(state["reformulated"])
            namespace["caption_recon"] = state["reformulated"][-1]
            restore_rng(state["rng"])

        def begin_round(rnd):
            """Reset per-round completion slots; replaying an interrupted round reuses calls."""
            nonlocal call_start
            call_start = len(questioner.records)
            questioner.begin_round(rnd, case_id)

        def checkpoint_round():
            """Save an original completed round plus observation-only ranks and call traces."""
            rnd = namespace["rnd"] + 1
            record = {"round": rnd, "query": namespace["caption_recon"],
                      "raw_query": ", ".join(namespace["dial_gen"]),
                      "question": namespace["question_final"], "answer": namespace["answer_generated"],
                      "candidate_questions": list(namespace["questions"]),
                      "candidate_caption_prompt": namespace["prompt_fake"],
                      "target_rank": observed_rank(namespace["caption_recon"]),
                      "llm_calls": questioner.records[call_start:]}
            state["dialog"] = list(namespace["dial_gen"])
            state["reformulated"] = list(namespace["captions_recon"])
            state["dialogue"].append({"question": record["question"], "answer": record["answer"]})
            state["states"].append(record)
            state["rng"], state["status"] = capture_rng(), "partial"
            state.pop("last_error", None)
            save_json(path, state)
            print(f"State {rnd}: {record['question']} -> {record['answer']}\n"
                  f"Query: {record['query']}\nTarget rank: {record['target_rank']}/{len(retriever.images)} (saved)", flush=True)

        namespace.update(restore_round_state=restore_round_state, begin_round=begin_round,
                         checkpoint_round=checkpoint_round, start_round=len(state["states"]) - 1,
                         max_rounds=n_states - 1)
        try:
            if len(state["states"]) < n_states:
                exec(loop_code, namespace)
            state["status"] = "complete"
            save_json(path, state)
            return {**state, "states": state["states"][:n_states], "dialogue": state["dialogue"][:n_states - 1],
                    "dialog": state["dialog"][:n_states], "reformulated": state["reformulated"][:n_states]}
        except BaseException as error:
            state["status"], state["last_error"] = "interrupted", str(error)
            save_json(path, state)
            raise
        finally:
            questioner.park()
            answerer.park()


def prepare_live_experiment(root, retriever, pairs, *, n_states, settings, device="cuda", offline=False,
                            candidate_captions="assets/protocol/visdial_captions.json",
                            image_root="data/images", image_archive=None, download_missing=True,
                            max_input_tokens=4096):
    """Generate original-loop Qwen conversations, then enter the common evaluation.
    Reuse the existing image cache and target selection. Save source/model identity,
    original prompts/responses and each completed round. No saved human answers or
    supplied reformulations are given to live generation.
    """
    import importlib.metadata
    import importlib.util
    import platform
    from .evaluation import PreparedExperiment
    from .live_models import BLIP2_ID, BLIP2_REVISION, QWEN_ID, QWEN_REVISION, Blip2Answerer, Qwen3, require_cuda

    require_cuda(device)
    if any(importlib.util.find_spec(name) is None for name in ("accelerate", "sentencepiece", "fast_pytorch_kmeans", "psutil")):
        raise RuntimeError('Install live extras in the GPU environment: python -m pip install -e ".[live,test]"')
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("MAX_INPUT_TOKENS must be a positive integer.")
    root = Path(root)
    settings.validate(len(retriever.cache.paths), n_states)
    caption_path = (root / candidate_captions).resolve()
    catalogue = load_candidate_captions(caption_path, retriever.cache.paths)
    image_paths = [resolve_target_image(root / image_root, raw["img"],
        archive=None if image_archive is None else root / image_archive,
        download=download_missing and not offline) for raw, _ in pairs]
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(torch.device(device)), "cuda_build": torch.version.cuda,
               "generation_device": device, "retrieval_device": str(retriever.device),
               "versions": {name: importlib.metadata.version(name) for name in
                   ("torch", "torchvision", "transformers", "accelerate", "sentencepiece", "pillow", "fast-pytorch-kmeans", "psutil")}}
    identity = {"method": "upstream_plugir_qwen_v1", "settings": asdict(settings), "source": provenance(),
                "qwen": [QWEN_ID, QWEN_REVISION], "answerer": [BLIP2_ID, BLIP2_REVISION],
                "cache": retriever.cache.report, "candidate_captions_sha256": digest(caption_path),
                "max_input_tokens": max_input_tokens, "runtime": runtime,
                "source_sha256": {name: digest(root / name) for name in
                    ("src/agentir/live.py", "src/agentir/live_models.py", "src/agentir/plugir_upstream.py", "src/agentir/retrieval.py")}}
    folder = root / "runs" / "live_upstream" / signature(identity)[:20]
    save_json(folder / "identity.json", identity)
    questioner = Qwen3(folder / "qwen_calls", device=device, offline=offline,
                       max_input_tokens=max_input_tokens, seed=settings.seed)
    answerer = Blip2Answerer(folder / "blip2_calls", device=device, offline=offline)
    conversations = []
    path_to_id = {name: i for i, name in enumerate(retriever.cache.paths)}
    print(f"Original PlugIR + Qwen: {len(pairs)} target(s), {n_states} states. Checkpoints: {folder}", flush=True)
    try:
        for (raw, _), image_path in zip(pairs, image_paths):
            conversations.append(generate_live_dialogue(retriever, initial_caption=raw["dialog"][0],
                target_id=path_to_id[raw["img"]], image_path=image_path, captions=catalogue,
                questioner=questioner, answerer=answerer, settings=settings, n_states=n_states,
                checkpoint_path=folder / "dialogues" / (signature(raw["img"]) + ".json"),
                run_identity=signature(identity)))
    finally:
        questioner.park()
        answerer.park()
    return PreparedExperiment(
        methods={"live_plugir_qwen": generated_query_method(conversations),
                 "live_raw_same_dialogue": generated_query_method(conversations, raw=True)},
        metadata={"identity": identity, "checkpoint_folder": str(folder),
                  "n_states": n_states, "n_targets": len(pairs),
                  "limitations": "Original algorithm with Qwen replacing GPT; not identical paper numbers. "
                                  "BLIP-2 answers may err. Same-dialogue raw is a reformulation diagnostic."},
        artifacts={"generated_dialogues.json": conversations, "upstream_source.json": provenance()},
    )


