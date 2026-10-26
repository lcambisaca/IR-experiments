"""The existing BLIP text encoder searching already encoded image vectors."""
import importlib.metadata
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Our BlipRetriever class wraps the existing model with functions for searching your saved image vectors.
# it holds BLIP, the processor, the saved image vectors, and our search functions together.

import torch
from torch.nn.functional import normalize

from .cache import MODEL_ID

"""Find each known target image's position in its query's ranked results.
Use those positions to evaluate retrieval performance.
"""
def ranks_for_targets(scores, target_ids):
    """One-based target ranks; exact ties break by original manifest index."""
    if scores.ndim != 2 or len(target_ids) != len(scores):
        raise ValueError("Expected one target ID per score row.")
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite retrieval scores.")
    ordering = torch.argsort(scores, descending=True, dim=-1, stable=True)
    targets = torch.as_tensor(target_ids, device=ordering.device).unsqueeze(1)
    if (targets < 0).any() or (targets >= scores.shape[1]).any():
        raise ValueError("Target outside the candidate pool.")
    return (ordering == targets).to(torch.int64).argmax(dim=1).cpu() + 1


class BlipRetriever:
    def __init__(self, cache, *, device="cpu", batch_size=8, threads=4, local_files_only=False):
        if batch_size < 1 or threads < 1:
            raise ValueError("Batch size and CPU threads must be positive.")
        installed = importlib.metadata.version("transformers")
        saved = cache.metadata["identity"]["transformers"]
        if installed != saved:
            raise ValueError(f"Use transformers=={saved}, matching the saved cache. Installed: {installed}.")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use --device cpu for local retrieval testing.")
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.cache, self.batch_size = cache, batch_size
        self.images = cache.vectors.to(self.device)
        from transformers import AutoProcessor, BlipForImageTextRetrieval
        print(f"Loading pinned BLIP text retriever on {self.device}; first use downloads weights.", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=cache.revision, local_files_only=local_files_only
        )
        self.model = BlipForImageTextRetrieval.from_pretrained(
            MODEL_ID, revision=cache.revision, torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self._text_cache = {}

    def embed_texts(self, texts):
        """Encode new descriptions into normalized 256-dimensional vectors
        in the same embedding space as our saved image vectors.
        Reuse previously computed embeddings for identical text.
        """
        if not texts or any(not isinstance(s, str) or not s.strip() for s in texts):
            raise ValueError("Queries must be nonempty text strings.")
        missing = list(dict.fromkeys(s for s in texts if s not in self._text_cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                output = self.model.text_encoder(**inputs, return_dict=True)
                features = normalize(self.model.text_proj(output.last_hidden_state[:, 0, :]), dim=-1)
                # Upstream's evaluator normalizes the dialog encoder output again.
                features = normalize(features, dim=-1).cpu()
                for text, feature in zip(batch, features):
                    self._text_cache[text] = feature
        return torch.stack([self._text_cache[text] for text in texts])

    def scores(self, texts):
        """Compare each description's embedding with every saved image embedding.
        Return cosine similarity scores; higher means a closer model match.
        """
        with torch.inference_mode():
            return (self.embed_texts(texts).to(self.device) @ self.images.T).cpu()

    def search(self, text, top_k=10):
        """Rank images by similarity to the description.
        Return the top matches with their filenames, ranks, and scores.
        """
        if not 1 <= top_k <= len(self.cache.paths):
            raise ValueError("top_k must fit inside the candidate pool.")
        scores = self.scores([text])[0]
        ids = torch.argsort(scores, descending=True, stable=True)[:top_k]
        return [{"rank": rank, "image_id": int(i), "image": self.cache.paths[i],
                 "cosine_similarity": float(scores[i])} for rank, i in enumerate(ids, 1)]
