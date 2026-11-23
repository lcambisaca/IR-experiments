"""Shared retriever device and input-validation regression checks."""
import pytest
import torch

from agentir.retrieval import BlipRetriever, ranks_for_targets

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a CUDA-enabled PyTorch build and NVIDIA GPU"
))]


@pytest.mark.parametrize("device", DEVICES)
def test_cpu_queries_score_against_images_on_either_device(device):
    """Exercise the actual scorer without BLIP, including CPU queries vs CUDA images."""
    retriever = BlipRetriever.__new__(BlipRetriever)
    retriever.images = torch.eye(3, device=device)
    vectors = torch.nn.functional.normalize(torch.tensor([[3., 4., 0.], [0., 0., 2.]]), dim=-1)
    retriever.embed_texts = lambda texts: vectors
    scores = retriever.scores(["one", "two"])
    assert scores.device.type == "cpu"
    torch.testing.assert_close(scores, torch.tensor([[.6, .8, 0.], [0., 0., 1.]]))
    assert ranks_for_targets(scores, [1, 2]).tolist() == [1, 1]
    # Already-GPU query vectors must work too; return contract remains CPU.
    torch.testing.assert_close(retriever.score_vectors(vectors.to(device)), scores)
    for bad in (torch.zeros(1, 3), torch.full((1, 3), float("nan")),
                torch.ones(1, 3), torch.ones(3), torch.empty(0, 3), torch.ones(1, 4)):
        with pytest.raises(ValueError):
            retriever.score_vectors(bad)
