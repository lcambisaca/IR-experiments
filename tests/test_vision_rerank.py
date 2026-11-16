"""Vision re-ranking pilot: ordering, weighting and a full rerank phase with a fake model."""
import json

import pytest

import vision_rerank_experiment as v


def test_rerank_orders_by_score_and_keeps_blip_order_for_ties():
    order = v.rerank(["a", "b", "c", "d"], {"a": 3, "b": 9, "c": 3, "d": 9})
    assert order == ["b", "d", "a", "c"]


def test_failed_answer_gets_the_mean_score():
    # Mean of known scores is 5: the unscored photo ranks between 8 and 2, not first or last.
    assert v.rerank(["a", "b", "c"], {"a": 2, "b": None, "c": 8}) == ["c", "b", "a"]


def test_target_outside_shortlist_keeps_its_blip_rank():
    assert v.reranked_rank("z", 57, ["a", "b"]) == 57
    assert v.reranked_rank("b", 2, ["b", "a"]) == 1


@pytest.mark.parametrize("content, expected", [
    ('{"mismatches": [], "score": 9}', {"score": 9, "mismatches": []}),
    ('{"score": 12.4}', {"score": 10, "mismatches": []}),
    ('{"mismatches": [" no dog ", ""], "score": -1}', {"score": 0, "mismatches": ["no dog"]}),
    ('{"score": "high"}', None), ('{"score": true}', None), ("not json", None), ('{"mismatches": []}', None),
])
def test_parse_answer(content, expected):
    assert v.parse_answer(content) == expected


def test_weighted_recall_scales_each_stratum_to_its_sample_share():
    strata = {"top": (1, 10, 2), "near": (11, 30, None)}
    sample_ranks = [1] * 6 + [15] * 2 + [500] * 2          # 10 targets: 6 top, 2 near, 2 unreachable.
    rows = [{"stratum": "top", "blip_rank": 1, "vlm_rank": 1}, {"stratum": "top", "blip_rank": 1, "vlm_rank": 12},
            {"stratum": "near", "blip_rank": 15, "vlm_rank": 3}, {"stratum": "near", "blip_rank": 15, "vlm_rank": 15}]
    assert v.weighted_recall(rows, sample_ranks, 10, "blip_rank", strata) == pytest.approx(60.0)
    # Half the top stratum is lost (6 -> 3), half the near stratum is rescued (0 -> 1).
    assert v.weighted_recall(rows, sample_ranks, 10, "vlm_rank", strata) == pytest.approx(40.0)


def test_rerank_phase_end_to_end_with_a_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "SHORTLIST", 3)
    monkeypatch.setattr(v, "STRATA", {"rank 1": (1, 1, None), "rank 2-3": (2, 3, None)})
    monkeypatch.setattr(v, "fetch_image", lambda name: name)
    monkeypatch.setattr(v, "encode_image", lambda path: path)
    monkeypatch.setattr(v, "ask_vlm", lambda caption, image, prompt: {"score": 9 if "target" in image else 2, "mismatches": []})
    monkeypatch.setattr(v, "ollama_identity", lambda: {"ollama_version": "test", "model_digest": None})
    targets = [
        {"example_index": 0, "caption": "c0", "target": "t0_target", "blip_rank": 1, "stratum": "rank 1",
         "shortlist": ["t0_target", "x", "y"]},
        {"example_index": 1, "caption": "c1", "target": "t1_target", "blip_rank": 3, "stratum": "rank 2-3",
         "shortlist": ["p", "q", "t1_target"]},
    ]
    shortlists = tmp_path / "shortlists.json"
    shortlists.write_text(json.dumps({"settings": {"shortlist": 3, "sample_ranks": [1, 3, 99]}, "targets": targets}))
    destination = v.run_pilot(tmp_path, shortlists, "checklist")
    report = json.loads((destination / "report.json").read_text())
    assert [t["vlm_rank"] for t in report["targets"]] == [1, 1]
    recall_at_1 = {row["method"].split(" (")[0].split(" re-ranked")[0]: row["recall@1"] for row in report["summary"]}
    assert recall_at_1["BLIP"] == pytest.approx(100 / 3)
    assert report["summary"][1]["recall@1"] == pytest.approx(200 / 3)
    assert report["diagnostics"]["target_uniquely_top"] == 2
    # Answers are stored and reused: a second run asks nothing new.
    monkeypatch.setattr(v, "ask_vlm", lambda *a: pytest.fail("should reuse saved answers"))
    v.run_pilot(tmp_path, shortlists, "checklist")


def test_failed_answers_are_not_saved_and_old_failures_are_asked_again(tmp_path, monkeypatch):
    store = v.AnswerStore(tmp_path, "checklist")
    store.path.parent.mkdir(parents=True)
    store.path.write_text(json.dumps({"cap\told.jpg": None, "cap\tgood.jpg": {"score": 7, "mismatches": []}}))
    monkeypatch.setattr(v, "fetch_image", lambda name: name)
    monkeypatch.setattr(v, "encode_image", lambda path: path)
    monkeypatch.setattr(v, "ask_vlm", lambda *a: None)
    store = v.AnswerStore(tmp_path, "checklist")
    assert store.missing([{"caption": "cap", "shortlist": ["old.jpg", "good.jpg"]}]) == 1
    assert store.get("cap", "old.jpg") is None
    # Still missing after reloading: the next run asks again.
    assert v.AnswerStore(tmp_path, "checklist").missing([{"caption": "cap", "shortlist": ["old.jpg"]}]) == 1


def test_server_down_stops_instead_of_recording_failures(monkeypatch):
    import urllib.error

    def refuse(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    monkeypatch.setattr(v.urllib.request, "urlopen", refuse)
    with pytest.raises(v.ServerDown):
        v.ask_vlm("cap", "image")


def test_yes_prob_reads_the_probability_of_yes_against_no():
    import math
    reply = {"logprobs": [{"token": "Yes", "logprob": math.log(0.6), "top_logprobs": [
        {"token": "Yes", "logprob": math.log(0.6)}, {"token": "No", "logprob": math.log(0.2)},
        {"token": " yes", "logprob": math.log(0.1)}, {"token": "The", "logprob": math.log(0.05)}]}]}
    answer = v.parse_yes_prob(reply)
    assert answer["score"] == pytest.approx(10 * 0.7 / 0.9)   # "Yes" counted once, " yes" added.
    assert v.parse_yes_prob({"logprobs": [{"token": "The", "logprob": 0.0, "top_logprobs": []}]}) is None


def test_yes_prob_without_logprobs_stops_the_run():
    with pytest.raises(v.SetupError, match="0.12.11"):
        v.parse_yes_prob({"message": {"content": "yes"}})


def test_yes_prob_request_asks_for_one_token_with_logprobs(monkeypatch):
    import io, math
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return io.BytesIO(json.dumps({"logprobs": [{"token": "no", "logprob": math.log(0.8),
                                                    "top_logprobs": [{"token": "yes", "logprob": math.log(0.2)}]}]}).encode())
    monkeypatch.setattr(v.urllib.request, "urlopen", fake_urlopen)
    answer = v.ask_vlm("a cat", "IMG", "yes_prob")
    assert answer["score"] == pytest.approx(2.0)
    assert sent["logprobs"] is True and sent["options"]["num_predict"] == 1 and "format" not in sent
    assert sent["messages"][1]["content"].endswith("Answer yes or no.")


def test_rule2_demotes_rejected_photos_and_keeps_blip_order_inside_each_group():
    scores = {"a": 9.9, "b": 0.4, "c": 9.6, "d": 9.98, "e": 1.2, "f": None}
    # Accepted (and unscored) photos keep BLIP order; rejected ones follow, also in BLIP order.
    assert v.split_rerank(["a", "b", "c", "d", "e", "f"], scores, 5) == ["a", "c", "d", "f", "b", "e"]
    assert v.split_rerank(["a", "b"], {"a": 4.99, "b": 5}, 5) == ["b", "a"]


def test_pick_targets_samples_partial_strata_and_takes_all_otherwise():
    ranked = [{"example_index": i, "stratum": "top" if i < 6 else "near" if i < 9 else None} for i in range(12)]
    picked = v.pick_targets(ranked, {"top": (1, 1, 2), "near": (2, 30, None)})
    assert [r["example_index"] for r in picked if r["stratum"] == "near"] == [6, 7, 8]
    assert sum(r["stratum"] == "top" for r in picked) == 2
    everything = v.pick_targets(ranked, {"top": (1, 1, None), "near": (2, 30, None)})
    assert [r["example_index"] for r in everything] == list(range(9))   # Never targets outside the strata.


def test_report_says_exact_when_every_target_was_scored(monkeypatch):
    monkeypatch.setattr(v, "SHORTLIST", 3)
    monkeypatch.setattr(v, "STRATA", {"rank 1-3": (1, 3, None)})
    rows = [{"stratum": "rank 1-3", "target": "t", "shortlist": ["x", "t", "y"], "blip_rank": 2, "vlm_rank": 1,
             "split_rank": 1, "target_score": 9, "scores": {"x": 1, "t": 9, "y": 2}}]
    table, diagnostics = v.summarize(rows, [2, 50])
    assert "all targets" in table[1]["method"] and table[2]["method"].startswith("rule 2")
    assert table[2]["recall@1"] == pytest.approx(50.0)
    assert diagnostics["rule2_lost_from_top10"] == 0 and diagnostics["target_rejected"] == 0
