"""Read the actual Colab cache format; no image indexing or JPEG download code."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch

MODEL_ID = "Salesforce/blip-itm-large-coco"


def digest(path): # Calculates a file's fingerprint using SHA-256. Reads the file in 8MB blocks to avoid memory issues with large files.
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()
# Just a tool to verify there was no issues retieriving our images

def read_json(path): # Loads its json content into a form our code can use
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ImageCache:
    ids: torch.Tensor
    vectors: torch.Tensor
    paths: list[str]
    metadata: dict
    path: Path

    @property
    def revision(self):
        return self.metadata["identity"]["model_revision"]

    @property
    def report(self):
        lengths = self.vectors.norm(dim=-1)
        return {
            "images": len(self.paths), "shape": list(self.vectors.shape),
            "dtype": str(self.vectors.dtype), "device": str(self.vectors.device),
            "norm_min": lengths.min().item(), "norm_max": lengths.max().item(),
            "model": MODEL_ID, "model_revision": self.revision,
            "cache_sha256": self.metadata["sha256"],
            "manifest_sha256": self.metadata["identity"]["manifest_sha256"],
            "encoding_versions": self.metadata["identity"],
        }


def load_cache(cache_path, manifest_path, *, expected_count=50_000):
    """Require matching bytes, model, image order, IDs and unit-length vectors."""
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing saved cache/metadata. Put {cache_path.name} and {metadata_path.name} "
            f"in {cache_path.parent}. See README for private Drive download links."
        )
    metadata = read_json(metadata_path)
    identity = metadata["identity"]
    if identity.get("model") != MODEL_ID or identity.get("mode") != "eval":
        raise ValueError("Expected the team's eval-mode BLIP retrieval cache.")
    revision = identity.get("model_revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Cache is missing a pinned model revision.")
    if digest(cache_path) != metadata["sha256"]:
        raise ValueError("Cache checksum failed; download the original file again, do not re-encode.")
    if digest(manifest_path) != identity["manifest_sha256"]:
        raise ValueError("Image manifest does not match the saved vectors. Refusing to mix image order/pools.")
    paths = read_json(manifest_path)
    if not isinstance(paths, list) or len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count:,} candidate images.")
    for name in paths:
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            raise ValueError("Invalid manifest image path.")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest image path.")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate manifest paths.")
    value = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Expected the notebook's (ids, vectors) cache tuple.")
    ids, vectors = value
    if not isinstance(ids, torch.Tensor) or not isinstance(vectors, torch.Tensor):
        raise ValueError("Expected tensor IDs and vectors.")
    if ids.dtype != torch.int64 or ids.shape != (expected_count,) or not torch.equal(ids, torch.arange(expected_count)):
        raise ValueError("Image IDs do not match manifest order.")
    if vectors.dtype != torch.float32 or vectors.shape != (expected_count, 256):
        raise ValueError("Expected 50k x 256 float32 BLIP vectors.")
    if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=-1), torch.ones(expected_count), atol=1e-4, rtol=1e-4):
        raise ValueError("Nonfinite vectors or invalid vector normalization.")
    return ImageCache(ids, vectors.contiguous(), paths, metadata, cache_path)
