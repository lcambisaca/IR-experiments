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
        return self.respond(inspect.currentframe().f_back.f_code.co_name, kwargs)

    def respond(self, stage, kwargs):
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


from pathlib import Path

from PIL import Image
from agentir.probe_agent import answer_entropy, bucket


class FakeVision:
    """Agent vision model double: 'color' questions get one answer for every photo, others vary by photo."""
    def __init__(self):
        self.calls, self.parked = [], 0
    def answer(self, paths, question):
        self.calls.append((list(paths), question))
        if "color" in question.lower():
            return ["White."] * len(paths)
        return [f"thing {int(str(p)[-6:-4]) % 4}" for p in paths]
    def park(self):
        self.parked += 1


class ColorQuestioner(AgentQuestioner):
    """Alternate between a colour question and a varied question as PlugIR's candidates."""
    def respond(self, stage, kwargs):
        if stage.startswith("generate_questions"):
            self.ordinal += 1
            content = "What color is the wall?" if self.ordinal % 4 == 1 else "What object is on the table?"
            result = {"choices": [{"message": {"content": content}}]}
            self.records.append({"stage": stage, "request": kwargs, "response": result})
            return result
        return super().respond(stage, kwargs)


def test_bucket_and_entropy():
    assert bucket("A red bus.") == bucket("red bus") == "red bus"
    assert bucket("I can't tell") == bucket("unknown") == "(unsure)"
    assert answer_entropy(["white"] * 20) == 0
    assert abs(answer_entropy(["red", "blue"] * 10) - 1) < 1e-6


def run_vlm(case, tmp_path, target_id=0):
    root = tmp_path / "images"
    for name in case["retriever"].cache.paths:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(root / name)
    qwen, vision = ColorQuestioner(), FakeVision()
    result = generate_live_dialogue(**case, initial_caption="a room", target_id=target_id, questioner=qwen,
        answerer=ScriptedAnswerer(qwen), n_states=3, checkpoint_path=tmp_path / "vlm.json", run_identity="test",
        _kmeans=RoundRobinKMeans(), selection="probe_vlm", vlm=vision, image_root=root)
    return result, qwen, vision


def test_probe_vlm_skips_questions_everyone_answers_the_same(case, tmp_path):
    result, _, vision = run_vlm(case, tmp_path)
    for state in result["states"][1:]:
        log = state["selection"]
        assert not log["fallback_to_plugir"] and "color" not in state["question"].lower()
        assert any(row["answers"] == {"white": 20} and row["score"] == 0 for row in log["candidates"])
    assert all(len(paths) == 20 for paths, _ in vision.calls) and vision.parked >= 2


def test_probe_vlm_never_depends_on_the_target(case, tmp_path):
    first, q1, v1 = run_vlm(case, tmp_path / "a", target_id=0)
    second, q2, v2 = run_vlm(case, tmp_path / "b", target_id=90)
    names = lambda vision: [([Path(x).name for x in paths], q) for paths, q in vision.calls]
    assert first["dialog"] == second["dialog"] and names(v1) == names(v2)


def test_opinion_questions_are_skipped():
    from agentir.probe_agent import OPINION
    assert OPINION.search("What might the green paint symbolize in this scene?")
    assert OPINION.search("What mood does the room suggest?")
    assert not OPINION.search("What color is the bag on the towel rack?")
    assert not OPINION.search("How many people are sitting at the table?")


def test_plugir_identity_ignores_agent_code(tmp_path, monkeypatch):
    """Changing the agent's file must not invalidate saved PlugIR conversations, only agent ones."""
    import agentir.live as live
    captured = {}
    monkeypatch.setattr(live, "save_json", lambda path, value: captured.setdefault("identity", value))
    def stop(*args, **kwargs):
        raise RuntimeError("stop after identity")
    monkeypatch.setattr("agentir.live_models.Qwen3", stop)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(live, "require_cuda", lambda device: device, raising=False)
    import agentir.live_models as models
    monkeypatch.setattr(models, "require_cuda", lambda device: device)
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    from types import SimpleNamespace
    names = [f"unlabeled2017/{i:012d}.jpg" for i in range(100)]
    retriever = SimpleNamespace(cache=SimpleNamespace(paths=names, report={}), device="cpu")
    for selection in ("plugir_kl", "probe_vlm"):
        captured.clear()
        monkeypatch.setattr(live, "load_candidate_captions", lambda path, paths: {})
        try:
            live.prepare_live_experiment(root, retriever, [], n_states=3, settings=live.LiveSettings(threshold_low=100, recall_hitting=2),
                                         device="cuda", selection=selection)
        except RuntimeError:
            pass
        has_agent = "src/agentir/probe_agent.py" in captured["identity"]["source_sha256"]
        assert has_agent == (selection != "plugir_kl")
