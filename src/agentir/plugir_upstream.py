"""Execute checksum-verified PlugIR definitions and its interaction loop.

The vendored files stay unchanged. We skip their top-level CLI/model downloads,
inject local runtime objects, and edit only loop paths/bounds/checkpoint hooks.
No prompt, clustering, filtering, KL or reformulation function is rewritten.
"""
import ast
import copy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

COMMIT = "ce79c9ff834aabc9de698efb706c6cb9427ecf35"
GENERATOR_SHA256 = "fa61dcd1bf56fe02d249b849d37cdf186bfd76e11b99eb9024965b9b16ec4d7f"
EVALUATOR_SHA256 = "2f5ed55dd5fcfb7959d502a8d3b10aa59b77adf6ab94126b77574bfba72263ea"
VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "PlugIR"


class UpstreamExecutionStop(BaseException):
    """Stop on a local runtime error instead of entering upstream's network retry loops.
    Original helpers catch Exception, so this control signal deliberately derives
    from BaseException. Our outer orchestrator still saves progress and cleans up.
    """


def verified_source(name, *, folder=None):
    """Read exact pinned bytes and refuse modified upstream source before compiling it."""
    expected = {"generate_dialog.py": GENERATOR_SHA256, "eval.py": EVALUATOR_SHA256}[name]
    path = (VENDOR if folder is None else Path(folder)) / name
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Upstream checksum mismatch: {path}. Restore the pinned file; do not edit it.")
    return data.decode("utf-8"), path


def ast_fingerprint(node):
    """Fingerprint executable AST structure while ignoring source-location metadata."""
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def generator_parts(*, folder=None):
    """Extract unchanged definitions and the original target loop without executing CLI/setup.
    Return AST nodes for compilation and independent parity tests; the source hash
    is checked first, so extraction cannot silently target a different revision.
    """
    source, path = verified_source("generate_dialog.py", folder=folder)
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    loops = [node for node in tree.body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "idx"]
    if len(loops) != 1 or len(definitions) != 12:
        raise ValueError("Unexpected upstream generator structure.")
    return definitions, loops[0], path


def adapted_loop(original):
    """Add only four declared orchestration edits to a copy of the original target loop.
    Map the JPEG path, restore a completed prefix, bound/resume the round range,
    and add begin/end hooks. The body between hooks remains the authors' code.
    """
    loop = copy.deepcopy(original)
    paths = [node for node in loop.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "img_path" for t in node.targets)]
    rounds = [node for node in loop.body if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == "rnd"]
    if len(paths) != 1 or len(rounds) != 1:
        raise ValueError("Unexpected upstream loop paths/rounds.")
    paths[0].value = ast.Name(id="target_image_path", ctx=ast.Load())
    round_loop = rounds[0]
    round_loop.iter = ast.parse("range(start_round, max_rounds)", mode="eval").body
    round_loop.body.insert(0, ast.parse("begin_round(rnd)").body[0])
    round_loop.body.append(ast.parse("checkpoint_round()").body[0])
    loop.body.insert(loop.body.index(round_loop), ast.parse("restore_round_state()").body[0])
    # Prove these edits are reversible and no scientific branch was changed.
    restored = copy.deepcopy(loop)
    restored.body = [n for n in restored.body if not (isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "restore_round_state")]
    restored_round = next(n for n in restored.body if isinstance(n, ast.For) and n.target.id == "rnd")
    restored_round.iter = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.For) and n.target.id == "rnd").iter)
    restored_round.body = restored_round.body[1:-1]
    for node in restored.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "img_path" for t in node.targets):
            node.value = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "img_path" for t in n.targets)).value)
    if ast_fingerprint(restored) != ast_fingerprint(original):
        raise AssertionError("An undeclared upstream loop change was detected.")
    return loop


def load_generator(namespace):
    """Install original functions/classes in an injected runtime and compile the adapted loop.
    Every definition AST is compiled unchanged, including examples, temperatures,
    question parsing, filtering quirks, min-KL selection and paraphrase fallback.
    Return the original answer function, compiled loop and a provenance audit.
    """
    definitions, original_loop, path = generator_parts()
    module = ast.Module(body=copy.deepcopy(definitions), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    loop = adapted_loop(original_loop)
    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), "exec")
    audit = {"commit": COMMIT, "generator_sha256": GENERATOR_SHA256,
             "definitions": {node.name: {"ast_sha256": ast_fingerprint(node), "unchanged": True}
                             for node in definitions},
             "loop_edits": ["target image path", "round bounds/resume", "prefix restoration",
                            "begin/end checkpoint hooks"], "loop_otherwise_unchanged": True}
    return namespace["generate_answer"], code, audit


class DeviceProcessor:
    """Keep the original processing calls while moving batches to the correct device."""

    def __init__(self, processor, device, *, dtype=None):
        """Remember a real HF processor, device and optional floating-image dtype."""
        self.processor, self.device, self.dtype = processor, device, dtype

    def __call__(self, *args, **kwargs):
        """Apply exactly the supplied preprocessing options, then transfer the resulting tensors.
        Do not add truncation to upstream generation. BatchFeature.to casts floating
        pixels only, keeping tokenizer IDs integer when dtype is specified.
        """
        batch = self.processor(*args, **kwargs)
        return batch.to(self.device) if self.dtype is None else batch.to(self.device, self.dtype)

    def decode(self, *args, **kwargs):
        """Delegate answer decoding to the same processor used by upstream generation."""
        return self.processor.decode(*args, **kwargs)


def bind_original_text_encoder(namespace, retriever):
    """Use PlugIR's original BLIP feature method on the already-loaded identical HF model.
    This reuses weights without implementing a second encoder or re-encoding images.
    DeviceProcessor fixes input placement; all original text options remain intact.
    """
    method = namespace["BlipForRetrieval"].get_text_features

    def get_text_features(*args, **kwargs):
        """Call the unmodified upstream projection/normalization on the shared BLIP model."""
        return method(retriever.model, *args, **kwargs)

    namespace["processor"] = DeviceProcessor(retriever.processor, retriever.device)
    namespace["blip"] = SimpleNamespace(get_text_features=get_text_features)


@lru_cache(maxsize=1)
def metric_code():
    """Compile the authors' actual Hits/Recall functions and BRI block, without CLI/model setup."""
    source, path = verified_source("eval.py")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("get_first_hitting_time", "cumulative_hits_per_round")]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlugIREval")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "bri" for t in n.targets))
    end = next(i for i in range(start, len(run.body)) if isinstance(run.body[i], ast.AugAssign)
               and isinstance(run.body[i].target, ast.Name) and run.body[i].target.id == "bri")
    if len(functions) != 2:
        raise ValueError("Unexpected upstream metric structure.")
    return (compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            compile(ast.Module(body=run.body[start:end + 1], type_ignores=[]), str(path), "exec"))


def upstream_metrics(ranks, ks):
    """Measure one-based [state, example] ranks with the original zero-based metric code.
    Only convert indexing/layout. BRI is undefined for one state (upstream's
    division by zero); retain the evaluator's explicit None for that extension.
    """
    definitions, bri_block = metric_code()
    zero_based = ranks.long().cpu() - 1
    namespace = {"torch": torch, "args": SimpleNamespace(num_rounds=len(ranks)),
                 "min_ranks": list(torch.cummin(zero_based, dim=0).values)}
    exec(definitions, namespace)
    rows = []
    for k in ks:
        hits, recall = namespace["cumulative_hits_per_round"](zero_based.flatten(), None, None, hitting_recall=k)
        rows.extend({"round": r, "k": k, "recall_percent": float(recall[r]), "hits_percent": float(hits[r])}
                    for r in range(len(ranks)))
    if len(ranks) > 1:
        exec(bri_block, namespace)
        bri = float(namespace["bri"])
    else:
        bri = None
    return sorted(rows, key=lambda row: (row["round"], row["k"])), bri


def provenance():
    """Expose exact upstream source identifiers for reports and code review."""
    return {"commit": COMMIT, "generator_sha256": GENERATOR_SHA256,
            "evaluator_sha256": EVALUATOR_SHA256,
            "metrics": "Original get_first_hitting_time, cumulative_hits_per_round and PlugIREval.run BRI block"}
