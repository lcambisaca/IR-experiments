"""Answer-entropy question selection (agent v2) inside the original PlugIR loop, with fake models (CPU)."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from agentir.live import generate_live_dialogue
from agentir.plugir_upstream import UpstreamExecutionStop
from agentir.probe_agent import OPINION, answer_entropy, bucket
from test_live import RoundRobinKMeans, ScriptedAnswerer, ScriptedQuestioner, case  # noqa: F401


class ColorQuestioner(ScriptedQuestioner):
    """Scripted PlugIR responses; candidates alternate between a colour question and a varied one."""
    def create(self, **kwargs):
        import inspect
        stage = inspect.currentframe().f_back.f_code.co_name
        assert ".jpg" not in json.dumps(kwargs) and "target_rank" not in json.dumps(kwargs)
        self.ordinal += 1
        if stage.startswith("generate_questions"):
            content = "What color is the wall?" if self.ordinal % 4 == 1 else "What object is on the table?"
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


def run(case, tmp_path, selection="probe_vlm", target_id=0):
    root = tmp_path / "images"
    for name in case["retriever"].cache.paths:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(root / name)
    qwen, vision = ColorQuestioner(), FakeVision()
    result = generate_live_dialogue(**case, initial_caption="a room", target_id=target_id, questioner=qwen,
        answerer=ScriptedAnswerer(qwen), n_states=3, checkpoint_path=tmp_path / f"{selection}.json",
        run_identity="test", _kmeans=RoundRobinKMeans(), selection=selection, vlm=vision, image_root=root)
    return result, vision


def test_bucket_and_entropy():
    assert bucket("A red bus.") == bucket("red bus") == "red bus"
    assert bucket("I can't tell") == bucket("unknown") == "(unsure)"
    assert answer_entropy(["white"] * 20) == 0
    assert abs(answer_entropy(["red", "blue"] * 10) - 1) < 1e-6
    assert abs(answer_entropy(["white"] * 18 + ["red"] * 2) - 0.469) < 1e-3  # The paper's example.
    assert abs(answer_entropy(["black", "white", "silver", "brown"] * 5) - 2) < 1e-6


def test_skips_questions_every_photo_answers_the_same(case, tmp_path):
    result, vision = run(case, tmp_path)
    for state in result["states"][1:]:
        log = state["selection"]
        assert not log["fallback_to_plugir"] and "color" not in state["question"].lower()
        assert log["chosen"] == state["question"]
        assert any(row["answers"] == {"white": 20} and row["score"] == 0 for row in log["candidates"])
    assert all(len(paths) == 20 for paths, _ in vision.calls) and vision.parked >= 2


def test_never_depends_on_the_target(case, tmp_path):
    first, v1 = run(case, tmp_path / "a", target_id=0)
    second, v2 = run(case, tmp_path / "b", target_id=90)
    names = lambda vision: [([Path(x).name for x in paths], q) for paths, q in vision.calls]
    assert first["dialog"] == second["dialog"] and names(v1) == names(v2)


def test_default_selection_is_untouched_plugir(case, tmp_path):
    result, vision = run(case, tmp_path, selection="plugir_kl")
    assert all("selection" not in state for state in result["states"][1:]) and not vision.calls


def test_opinion_questions_are_skipped():
    assert OPINION.search("What might the green paint symbolize in this scene?")
    assert OPINION.search("What mood does the room suggest?")
    assert not OPINION.search("What color is the bag on the towel rack?")
    assert not OPINION.search("How many people are sitting at the table?")


def test_plugir_identity_ignores_agent_code(tmp_path, monkeypatch):
    """Changing the agent's file must not invalidate saved PlugIR conversations, only agent ones."""
    import torch
    import agentir.live as live
    import agentir.live_models as models
    captured = {}
    monkeypatch.setattr(live, "save_json", lambda path, value: captured.setdefault("identity", value))

    def stop(*args, **kwargs):
        raise RuntimeError("stop after identity")
    monkeypatch.setattr(models, "Qwen3", stop)
    monkeypatch.setattr(models, "require_cuda", lambda device: device)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    monkeypatch.setattr(live, "load_candidate_captions", lambda path, paths: {})
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    root = Path(__file__).resolve().parents[1]
    names = [f"unlabeled2017/{i:012d}.jpg" for i in range(100)]
    retriever = SimpleNamespace(cache=SimpleNamespace(paths=names, report={}), device="cpu")
    for selection in ("plugir_kl", "probe_vlm"):
        captured.clear()
        with pytest.raises(RuntimeError, match="stop after identity"):
            live.prepare_live_experiment(root, retriever, [], n_states=3,
                settings=live.LiveSettings(threshold_low=100, recall_hitting=2), device="cuda", selection=selection)
        has_agent = "src/agentir/probe_agent.py" in captured["identity"]["source_sha256"]
        assert has_agent == (selection != "plugir_kl")
