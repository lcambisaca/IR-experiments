import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agentir.cache import read_json
from agentir.live import generate_live_dialogue, save_json
from agentir.multagent_experiment.debate import generate_debate_dialogue
from test_debate import DebateTransport
from test_live import RoundRobinKMeans, ScriptedAnswerer, ScriptedQuestioner, case
from test_multi_agent import agent_case


@pytest.mark.parametrize("method", ["plugir", "debate"])
@pytest.mark.parametrize("ranks", [[7], [30, 20, 15, 8], [30, 20, 15, 14, 13]])
def test_first_hit_resume_matches_continuous_run(method, ranks, case, agent_case, tmp_path, monkeypatch):
    play = case if method == "plugir" else agent_case
    if method == "plugir":
        options = {**play, "initial_caption": "a room", "target_id": 0,
                   "_kmeans": RoundRobinKMeans(), "run_identity": "offline"}
        generator, transport = generate_live_dialogue, ScriptedQuestioner
    else:
        options = {**play, "settings": replace(play["settings"], max_calls_per_round=14)}
        generator, transport = generate_debate_dialogue, DebateTransport

    def generate(path, n_states, remaining):
        expected = iter(remaining)

        def scores(_queries):
            scores = torch.arange(100, 0, -1).float()
            scores[0] = 100 - next(expected) + .5
            return scores[None, :]

        monkeypatch.setattr(play["retriever"], "scores", scores)
        questioner = transport()
        return generator(**options, questioner=questioner, answerer=ScriptedAnswerer(questioner),
                         n_states=n_states, checkpoint_path=path, stop_at_rank=10)

    fresh = generate(tmp_path / "full.json", 5, ranks)
    path = tmp_path / "staged.json"
    prefix = generate(path, 3, ranks[:3])
    resumed = generate(path, 5, ranks[len(prefix["states"]):])
    assert resumed["states"] == fresh["states"]
    assert [state["target_rank"] for state in resumed["states"]] == ranks


def test_report_counts_first_hits_and_uses_paired_coverage(tmp_path):
    pytest.importorskip("matplotlib")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_hits10.py"
    spec = importlib.util.spec_from_file_location("run_hits10", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    targets = [{"image": "early", "initial_rank": 7}, {"image": "later", "initial_rank": 30}]
    save_json(runner.checkpoint_path(tmp_path, "plugir", "later"),
              {"states": [{"target_rank": rank} for rank in (30, 9)]})
    save_json(runner.checkpoint_path(tmp_path, "debate", "later"),
              {"states": [{"target_rank": rank} for rank in (30, 20, 15)]})
    runner.save_report(tmp_path, targets, 4)
    report = read_json(tmp_path / "first_hits.json")
    assert report["partial"] and report["plotted_round"] == 2
    assert report["curves"]["plugir"]["percent"] == [50, 100, 100]
    assert report["curves"]["debate"]["percent"] == [50, 50, 50]
    assert (tmp_path / "hits_at_10_by_round.png").stat().st_size > 0
    runner.save_report(tmp_path, [*targets, {"image": "pending", "initial_rank": 40}], 4)
    partial = read_json(tmp_path / "first_hits.json")
    assert partial["plotted_round"] == 0 and partial["paired_coverage"] == [3, 2, 2, 1, 1]
