"""Small synthetic retrieval tests; no BLIP downloads, image encoding, or APIs."""
from types import SimpleNamespace

import pytest
import torch

from agentir.evaluation import (
    SearchOutput, benchmark, evaluate_experiments, save_experiment_results,
    vector_output,
)


class FakeRetriever:
    """Provide deterministic scores for a four-image pool without loading a model."""

    def __init__(self):
        """Define known query scores whose target ranks can be checked by hand."""
        self.cache = SimpleNamespace(paths=["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        self.images = torch.eye(4)
        self.table = {
            "a": [1., .9, .8, .7], "b": [.9, .8, 1., .7],
            "a, first": [.9, 1., .8, .7], "b, first": [.8, 1., .9, .7],
            "a, first, second": [.8, .9, .7, 1.],
            "b, first, second": [1., .8, .9, .7],
            "a refined": [1., .9, .8, .7], "b refined": [.8, 1., .9, .7],
            "a final": [1., .8, .9, .7], "b final": [.9, .8, 1., .7],
        }

    def scores(self, texts):
        """Return one predetermined score row per description, in manifest order."""
        return torch.tensor([self.table[text] for text in texts])


@pytest.fixture
def pairs():
    """Supply two matched targets with an initial caption and two saved turns."""
    return [
        ({"img": "a.jpg", "dialog": ["a", "first", "second"]},
         {"img": "a.jpg", "dialog": ["a", "a refined", "a final"]}),
        ({"img": "b.jpg", "dialog": ["b", "first", "second"]},
         {"img": "b.jpg", "dialog": ["b", "b refined", "b final"]}),
    ]


def test_references_match_existing_benchmark_and_identity_control(pairs):
    """Preserve old raw/supplied metrics and verify a no-op experiment matches raw."""
    retriever = FakeRetriever()
    old = benchmark(retriever, pairs, ["raw", "supplied"], 3, [1, 2, 4])
    result = evaluate_experiments(retriever, pairs,
                                  {"identity": lambda context, model: context.current_query},
                                  rounds=3, ks=[1, 2, 4])
    for new_name, old_name in [("raw_dialogue", "raw"), ("supplied_plugir", "supplied"),
                               ("identity", "raw")]:
        assert result["summary"][new_name]["BRI"] == old["summary"][old_name]["BRI"]
        new_rows = [row for row in result["per_round"] if row["method"] == new_name]
        old_rows = [row for row in old["per_round"] if row["method"] == old_name]
        for left, right in zip(new_rows, old_rows):
            assert left["recall_percent"] == right["recall_percent"]
            assert left["hits_percent"] == right["hits_percent"]
    a = [row for row in result["per_example"] if row["method"] == "identity" and row["example_index"] == 0]
    assert [row["target_rank"] for row in a] == [1, 2, 3]
    assert [row["rank_improvement_vs_previous"] for row in a] == [None, -1, -1]
    assert [row["best_target_rank"] for row in a] == [1, 1, 1]
    assert all(row["rank_improvement_vs_raw_dialogue"] == 0 for row in a)
    caption_metrics = [row for row in result["per_round"] if row["method"] == "caption_only" and row["k"] == 1]
    assert [row["hits_percent"] for row in caption_metrics] == [50., 50., 50.]


def test_context_has_only_past_raw_dialogue_and_method_local_state(pairs):
    """Keep labels/future turns out of callbacks and state separate per example/run."""
    seen = []

    def method(context, retriever):
        """Inspect the method's input boundary and carry a round counter forward."""
        assert not hasattr(context, "target_id") and not hasattr(context, "target")
        assert not hasattr(context, "target_rank") and not hasattr(context, "supplied_query")
        assert len(context.dialogue) == context.round_index
        assert context.previous_state == (None if context.round_index == 0 else context.round_index)
        assert context.previous_query == (None if context.round_index == 0 else
                                          ", ".join([context.initial_caption, *context.dialogue[:-1]]))
        seen.append((context.round_index, context.example_index))
        return SearchOutput(retriever.scores([context.current_query])[0], "cosine",
                            context.current_query, state=context.round_index + 1)

    for _ in range(2):
        evaluate_experiments(FakeRetriever(), pairs, {"stateful": method}, rounds=3, ks=[1])
    assert len(seen) == 12


def test_vector_output_normalizes_and_matches_cosine_scores():
    """Custom vectors must rank against the same image order and reject zero/NaN."""
    retriever = FakeRetriever()
    output = vector_output(retriever, torch.tensor([[3., 4., 0., 0.]]))
    assert output.score_kind == "cosine"
    assert output.scores.tolist() == pytest.approx([.6, .8, 0., 0.])
    for bad in [torch.zeros(4), torch.ones(3), torch.tensor([1., float("nan"), 0., 0.])]:
        with pytest.raises(ValueError):
            vector_output(retriever, bad)


def test_fusion_scores_are_not_reported_as_cosine_and_exports_align(tmp_path, pairs):
    """Export all methods on the same rows and preserve non-cosine score labels."""
    import csv
    import json

    def fused(context, retriever):
        """Return arbitrary final fusion scores to exercise the generic score interface."""
        return SearchOutput(torch.tensor([.01, .04, .02, .03]), "fusion",
                            notes={"agent_queries": ["one", "two"]})

    result = evaluate_experiments(FakeRetriever(), pairs, {"fusion_test": fused}, rounds=3, ks=[1, 2])
    rows = [row for row in result["per_example"] if row["method"] == "fusion_test"]
    assert all(row["target_cosine"] is None for row in rows)
    b = next(row for row in rows if row["example_index"] == 1 and row["round"] == 0)
    assert b["rank_improvement_vs_caption_only"] == 2  # Target b moves from 3 to 1.
    destination = save_experiment_results(result, tmp_path / "results")
    saved = json.loads((destination / "report.json").read_text())
    assert saved["protocol"]["targets"] == ["a.jpg", "b.jpg"]
    with (destination / "rank_comparison.csv").open(newline="") as stream:
        wide = list(csv.DictReader(stream))
    assert len(wide) == 6
    assert wide[1]["caption_only"] == "3" and wide[1]["fusion_test"] == "1"
    assert {"per_example.csv", "per_round.csv", "final_metrics.csv"} <= {p.name for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        save_experiment_results(result, destination)


@pytest.mark.parametrize("scores", [torch.ones(3), torch.full((4,), float("nan")), torch.ones(1, 4)])
def test_invalid_experiment_scores_fail_before_metrics(pairs, scores):
    """Reject incomplete pools, nonfinite scores, or an ambiguous score matrix."""
    with pytest.raises(ValueError, match="one finite score per image"):
        evaluate_experiments(FakeRetriever(), pairs,
                             {"bad": lambda context, model: SearchOutput(scores, "fusion")},
                             rounds=1, ks=[1])


def test_single_state_works_and_bad_registration_is_rejected(pairs):
    """Keep one-state metrics meaningful and stop accidental reference replacement."""
    result = evaluate_experiments(FakeRetriever(), pairs, rounds=1, ks=[1])
    assert all(values["BRI"] is None for values in result["summary"].values())
    with pytest.raises(ValueError, match="unique nonempty"):
        evaluate_experiments(FakeRetriever(), pairs, {"raw_dialogue": lambda context, model: "a"}, ks=[1])
    with pytest.raises(ValueError, match="Duplicate evaluation targets"):
        evaluate_experiments(FakeRetriever(), [pairs[0], pairs[0]], rounds=1, ks=[1])
