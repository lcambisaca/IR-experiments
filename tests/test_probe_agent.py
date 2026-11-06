"""Agent question selection: scoring, parsing and wrapper integration with fake models."""
import json
import math

import torch
from torch.nn.functional import normalize

from agentir.live import generate_live_dialogue
from agentir.plugir_upstream import UpstreamExecutionStop
from agentir.probe_agent import parse_answers, split_score
from test_live import RoundRobinKMeans, ScriptedAnswerer, ScriptedQuestioner, case  # noqa: F401


class AgentQuestioner(ScriptedQuestioner):
    """Scripted PlugIR responses plus guessed answers for the agent's probe step."""
    def create(self, **kwargs):
        import inspect
        stage = inspect.currentframe().f_back.f_code.co_name
        assert ".jpg" not in json.dumps(kwargs) and "target_rank" not in json.dumps(kwargs)
        self.ordinal += 1
        if stage == "guess_answers":
            content = "1. White\n2. dark wood.\n- gray"
        elif stage.startswith("generate_questions"):
            code = abs(hash(json.dumps(kwargs, sort_keys=True))) % 997
            content = f"What is detail {code}? trailing text"
        elif stage == "filter_questions":
            content = "Uncertain"
        elif stage == "reconstruct_dialog":
            content = f"a room with detail {self.round}"
            self.round += 1
            self.ordinal = 0
        else:
            raise UpstreamExecutionStop(f"unexpected stage {stage}")  # escapes upstream retry loops
        result = {"choices": [{"message": {"content": content}}]}
        self.records.append({"stage": stage, "request": kwargs, "response": result})
        return result


def run(case, tmp_path, selection, target_id=0, n_states=3):
    qwen = AgentQuestioner()
    result = generate_live_dialogue(**case, initial_caption="a room", target_id=target_id, questioner=qwen,
        answerer=ScriptedAnswerer(qwen), n_states=n_states, checkpoint_path=tmp_path / f"{selection}{target_id}.json",
        run_identity="test", _kmeans=RoundRobinKMeans(), selection=selection)
    return result, qwen


def test_split_score_rewards_even_splits_and_ignores_identical_probes():
    images = normalize(torch.randn(300, 8, generator=torch.Generator().manual_seed(0)), dim=-1)
    same = images[:1].repeat(3, 1)
    assert abs(split_score(same, images)[0]) < 1e-6
    distinct = torch.eye(8)[:3]
    clean = normalize(torch.cat([distinct[i].repeat(100, 1) for i in range(3)]), dim=-1)
    assert abs(split_score(distinct, clean)[0] - math.log2(3)) < 1e-3
    lopsided = normalize(torch.cat([distinct[0].repeat(290, 1), distinct[1].repeat(5, 1), distinct[2].repeat(5, 1)]), dim=-1)
    assert split_score(distinct, lopsided)[0] < split_score(distinct, clean)[0] / 2


def test_parse_answers():
    assert parse_answers("1. White\n2) dark wood.\n- gray\n- white\nmore") == ["white", "dark wood", "gray"]
    assert parse_answers("") == []
    assert parse_answers("laptop, printer, keyboard, mouse") == ["laptop", "printer", "keyboard"]
    assert parse_answers("blue bed  \nred bed  \ngreen bed") == ["blue bed", "red bed", "green bed"]
    assert parse_answers("desk with laptops, printer, and other electronics") == ["desk with laptops", "printer", "other electronics"]


def test_probe_split_runs_in_the_original_loop_and_logs_choices(case, tmp_path):
    result, qwen = run(case, tmp_path, "probe_split")
    assert len(result["states"]) == 3
    for state in result["states"][1:]:
        log = state["selection"]
        assert log["chosen"] == state["question"] in state["candidate_questions"]
        assert all(row["answers"] == ["white", "dark wood", "gray"] for row in log["candidates"])
    assert any(r["stage"] == "guess_answers" for r in qwen.records)


def test_agent_never_depends_on_the_target(case, tmp_path):
    first, q1 = run(case, tmp_path / "a", "probe_split", target_id=0)
    second, q2 = run(case, tmp_path / "b", "probe_split", target_id=90)
    assert q1.records == q2.records and first["dialog"] == second["dialog"]


def test_default_selection_is_untouched_plugir(case, tmp_path):
    result, qwen = run(case, tmp_path, "plugir_kl")
    assert all(state.get("selection") is None for state in result["states"][1:])
    assert not any(r["stage"] == "guess_answers" for r in qwen.records)


def test_random_check_is_reproducible(case, tmp_path):
    first, _ = run(case, tmp_path / "a", "random")
    second, _ = run(case, tmp_path / "b", "random")
    assert first["dialog"] == second["dialog"]
