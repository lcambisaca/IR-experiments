"""Device, replay and fusion regression checks; no models, downloads or API calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agentir.retrieval import BlipRetriever, ranks_for_targets

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires a CUDA-enabled PyTorch build and NVIDIA GPU"
))]


@pytest.fixture
def pilot():
    """Import the actual click-Run script without launching its main function."""
    path = Path(__file__).resolve().parents[1] / "paraphrase_experiment.py"
    spec = importlib.util.spec_from_file_location("paraphrase_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any accidental model download or HTTP request fail immediately."""
    def forbidden(*args, **kwargs):
        """Reject network access so saved replay stays offline."""
        raise AssertionError("Unexpected network access")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)


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


@pytest.mark.parametrize("device", DEVICES)
def test_full_rrf_matches_hand_calculation_with_stable_ties(pilot, device):
    """Invert rankings on their input device and check RRF by image, including ties."""
    scores = torch.tensor([[.2, .9, .9, .1], [.5, .4, .3, .2]], device=device)
    positions = pilot.positions(scores)
    assert positions.device == scores.device
    assert positions.cpu().tolist() == [[2, 0, 1, 3], [0, 1, 2, 3]]
    fused = (1 / (60 + positions.double() + 1)).sum(0).cpu().tolist()
    expected = [1 / 63 + 1 / 61, 1 / 61 + 1 / 62, 1 / 62 + 1 / 63, 2 / 64]
    assert fused == pytest.approx(expected)


def test_selection_audits_changes_and_prevents_extra_duplicate_votes(pilot):
    """Preserve original, reject changed counts/colors/negation and log repeated wording."""
    original = "Two red dogs without hats."
    candidates = ["TWO RED DOGS WITHOUT HATS!", "Without hats are two red dogs.",
                  "without hats are TWO red dogs!", "Three red dogs without hats.",
                  "Two blue dogs without hats.", "Two red dogs with hats."]
    accepted, audit = pilot.select_paraphrases(original, candidates)
    assert accepted == [candidates[1]]
    assert audit[0]["rejection_reasons"] == ["duplicate_wording"]
    assert audit[2]["rejection_reasons"] == ["duplicate_wording"]
    assert "numbers_changed" in audit[3]["rejection_reasons"]
    assert "colors_changed" in audit[4]["rejection_reasons"]
    assert "negation_changed" in audit[5]["rejection_reasons"]
    legacy, audit = pilot.select_paraphrases(original, candidates, filter_flagged=False, deduplicate=False)
    assert legacy == candidates and all(item["accepted"] for item in audit)


def test_saved_bank_reuses_unicode_without_requests_and_rejects_missing(pilot, tmp_path):
    """Saved mode must neither generate nor overwrite files, even when entries are bad."""
    path = tmp_path / "bank.json"
    bank = {"a room — 室": ["rewrite " + str(i) for i in range(5)]}
    path.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    before = path.read_bytes()
    assert pilot.generate_paraphrases(list(bank), path) == bank
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="No generation started"):
        pilot.generate_paraphrases(["missing caption"], path)
    assert path.read_bytes() == before


def test_generation_retries_failed_entries_records_provenance_and_unloads(pilot, tmp_path, monkeypatch):
    """Explicit generation retries None/malformed entries and releases Qwen before BLIP."""
    path = tmp_path / "bank.json"
    valid = [f"a valid rewrite {i}" for i in range(5)]
    path.write_text(json.dumps({"failed": None, "bad": [17], "good": valid}), encoding="utf-8")
    calls, unloaded = [], []

    def answer(caption, attempt=0):
        """Simulate one failed attempt and a successful retry without any model."""
        calls.append((caption, attempt))
        return None if caption == "failed" and attempt == 0 else valid

    monkeypatch.setattr(pilot, "ask_model", answer)
    monkeypatch.setattr(pilot, "ollama_provenance", lambda: {"digest": "test-digest", "details": {"quantization_level": "test"}})
    monkeypatch.setattr(pilot, "ollama_request", lambda endpoint, body: unloaded.append((endpoint, body)))
    store = pilot.generate_paraphrases(["failed", "bad", "good"], path, allow_generation=True)
    assert calls == [("failed", 0), ("failed", 1), ("bad", 0)]
    assert all(value == valid for value in store.values())
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["by_caption"]["failed"]["provenance"]["digest"] == "test-digest"
    assert "good" not in metadata["by_caption"]  # Never retroactively claim old weights.
    assert unloaded == [("/api/generate", {"model": pilot.MODEL, "keep_alive": 0, "stream": False})]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("policy", ["legacy", "filtered", "all_rejected"])
def test_main_pipeline_fusion_exports_and_original_only_fallback(pilot, tmp_path, monkeypatch, device, policy):
    """Run the real pipeline against a small pool and check ranks/metrics without models."""
    config = {name: name + ".json" for name in ("cache", "manifest", "baseline_queries", "supplied_queries")}
    (tmp_path / "experiment.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("baseline_queries", "supplied_queries"):
        (tmp_path / config[name]).write_text("[]", encoding="utf-8")
    (tmp_path / "pilot.py").write_text("# Synthetic runner snapshot", encoding="utf-8")
    bank = {"a": ["a", "a", "a", "a", "a"], "b": ["b", "b", "b", "b", "b"]}
    if policy != "all_rejected":
        bank = {caption: [caption + chr(99 + i) for i in range(5)] for caption in bank}
    (tmp_path / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
    monkeypatch.setattr(pilot, "__file__", str(tmp_path / "pilot.py"))
    for name, value in {"N_EXAMPLES": None, "K_VALUES": [1, 2, 3], "DEVICE": device,
                        "SAVED_PARAPHRASES": "bank.json", "PARAPHRASE_SOURCE": "saved",
                        "FILTER_FLAGGED": policy != "legacy", "DEDUPLICATE": policy != "legacy"}.items():
        monkeypatch.setattr(pilot, name, value)
    pairs = [({"img": f"{caption}.jpg", "dialog": [caption]}, {}) for caption in bank]
    cache = SimpleNamespace(paths=["a.jpg", "b.jpg", "c.jpg"], report={"images": 3})
    vectors = torch.nn.functional.normalize(torch.tensor([
        [3., 1., 2.], [1., 3., 2.], [1., 2., 3.], [1., 4., 2.], [2., 3., 1.], [1., 2., 4.],
        [3., 2., 1.], [1., 3., 2.], [1., 2., 4.], [2., 3., 1.], [4., 1., 2.], [1., 3., 1.],
    ]), dim=-1)
    features = {}
    for index, (caption, rewrites) in enumerate(bank.items()):
        for j, text in enumerate([caption, *rewrites]):
            features.setdefault(text, vectors[index * 6 + j])

    class FakeRetriever(BlipRetriever):
        """Use real device-aware scoring with fixed CPU text embeddings instead of BLIP."""

        def __init__(self, saved_cache, *, device, batch_size, threads):
            """Place only the synthetic images on the requested CPU/CUDA device."""
            self.cache, self.device = saved_cache, torch.device(device)
            self.images = torch.eye(3, device=self.device)

        def embed_texts(self, texts):
            """Return CPU embeddings, reproducing the BLIP wrapper's cache contract."""
            return torch.stack([features[text] for text in texts])

    monkeypatch.setattr(pilot, "BlipRetriever", FakeRetriever)
    monkeypatch.setattr(pilot, "load_cache", lambda *args: cache)
    monkeypatch.setattr(pilot, "matched_queries", lambda *args: pairs)
    pilot.main()
    report_path = next((tmp_path / "runs/paraphrase").glob("*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["examples"]) == 2
    assert len(report["summary"]) == 8
    assert all(value is None for value in report["settings"]["generation_provenance_by_caption"].values())
    assert report["settings"]["runtime"]["device"] == device
    for index, row in enumerate(report["examples"]):
        texts = [row["original"], *row["paraphrases"]]
        batch = torch.stack([features[text] for text in texts])
        orders = batch.argsort(dim=1, descending=True, stable=True).tolist()
        ranks = [order.index(index) + 1 for order in orders]
        assert row["original_rank"] == ranks[0] and row["paraphrase_ranks"] == ranks[1:]
        assert row["effective_query_count"] == len(texts)
        mean_order = batch.mean(0).argsort(descending=True, stable=True).tolist()
        assert row["mean_embedding_rank"] == mean_order.index(index) + 1
        fused = [sum(1 / (60 + order.index(image) + 1) for order in orders) for image in range(3)]
        fused_order = sorted(range(3), key=lambda image: (-fused[image], image))
        assert row["rrf_rank"] == fused_order.index(index) + 1
        if policy == "all_rejected":
            assert row["effective_query_count"] == 1 and not row["generation_failed"]
    if policy == "all_rejected":
        original = report["summary"][0]
        assert all(row["recall@1"] == original["recall@1"] for row in report["summary"])
