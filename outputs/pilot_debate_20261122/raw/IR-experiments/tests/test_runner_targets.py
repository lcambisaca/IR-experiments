"""Shared target selection and runner exports without model loading or GPU calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agentir.evaluation import PreparedExperiment, SearchOutput
from agentir.multagent_experiment import setup


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("target_selection_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pilot,start,count,states", [
    (False, 0, 32, 3), (False, 4, 32, 11), (False, 4, None, 2),
    (True, 999, 1, 3), (True, 0, 32, 11),
])
def test_three_methods_share_selected_targets_and_user_states(
        runner, monkeypatch, tmp_path, pilot, start, count, states):
    paths = [f"image_{i}.jpg" for i in range(40)]
    paths[35], paths[38] = setup.TARGET_IMAGES
    pairs = [({"img": name, "dialog": [f"caption {i}"] + ["answer"] * 10},
              {"img": name, "dialog": [f"caption {i}"] + ["rewrite"] * 10})
             for i, name in enumerate(paths)]
    config = {key: key + ".json" for key in
              ("cache", "manifest", "baseline_queries", "supplied_queries")}
    (tmp_path / "experiment.json").write_text(json.dumps(config))
    for filename in config.values():
        (tmp_path / filename).write_text("[]")
    (tmp_path / "run_evaluation.py").write_text("# fixture runner")
    cache = SimpleNamespace(paths=paths, report={"pool": len(paths)})
    retriever = SimpleNamespace(cache=cache)
    calls = []

    def preparer(name):
        def prepare(root, model, selected, *, n_states, offline, **options):
            calls.append((name, model, selected, n_states))

            def search(context, model):
                return SearchOutput(torch.arange(len(paths), dtype=torch.float32),
                                    "fixture", f"{name} state {context.round_index}")

            return PreparedExperiment({name: search}, metadata={"n_states": n_states},
                                      artifacts={"dialogues.json": [p[0]["img"] for p in selected]})
        return prepare

    for key, value in {
        "__file__": str(tmp_path / "run_evaluation.py"), "RUN_MULTI_AGENT": True,
        "RUN_LIVE_PLUGIR": True, "RUN_MULTI_AGENT_DEBATE": True, "USE_PILOT_TARGETS": pilot,
        "INCLUDE_SAVED_REFERENCES": False, "START_TARGET": start,
        "N_TARGETS": count, "N_STATES": states, "K_VALUES": [1, 10],
        "load_cache": lambda *args: cache, "matched_queries": lambda *args: pairs,
        "BlipRetriever": lambda *args, **kwargs: retriever,
        "prepare_three_agents": preparer("agents_fixture"),
        "prepare_debate_experiment": preparer("debate_fixture"),
        "prepare_qwen_live": preparer("live_fixture"),
    }.items():
        monkeypatch.setattr(runner, key, value)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "fixture")
    runner.main()

    expected = setup.select_pilot_pairs(pairs) if pilot else pairs[start:][:count]
    assert [call[0] for call in calls] == ["agents_fixture", "debate_fixture", "live_fixture"]
    assert calls[0][1] is calls[1][1] is calls[2][1] is retriever
    assert calls[0][2] is calls[1][2] is calls[2][2]
    assert calls[0][2] == expected
    assert calls[0][3] == calls[1][3] == calls[2][3] == states
    destination = next((tmp_path / "runs/experiments").iterdir())
    report = json.loads((destination / "report.json").read_text())
    assert report["protocol"]["targets"] == [raw["img"] for raw, _ in expected]
    assert report["protocol"]["rounds"] == states
    assert report["protocol"]["saved_references"] == []
    assert set(report["summary"]) == {"agents_fixture", "debate_fixture", "live_fixture"}
    assert report["run"]["target_selection"] == ("pilot" if pilot else "range")
    assert report["run"]["selected_target_start"] == (None if pilot else start)
    for name in ("three_agents", "agent_debate", "qwen_live"):
        saved = json.loads((destination / "prepared" / name / "dialogues.json").read_text())
        assert saved == report["protocol"]["targets"]


def test_multi_agent_allowance_follows_batch_without_changing_defaults(monkeypatch):
    observed = []
    sentinel = object()

    def prepare(*args, **kwargs):
        observed.append(kwargs)
        return sentinel

    monkeypatch.setattr(setup, "prepare_agent_experiment", prepare)
    default_limit = setup.AGENTS.max_targets
    pairs = [({}, {}) for _ in range(32)]
    assert setup.prepare_three_agents("root", "retriever", pairs, n_states=4) is sentinel
    assert observed[-1]["settings"].max_targets >= 32
    assert observed[-1]["n_states"] == 4
    assert setup.AGENTS.max_targets == default_limit
    setup.prepare_three_agents("root", "retriever", pairs, n_states=3, settings=setup.AGENTS)
    assert observed[-1]["settings"] is setup.AGENTS


@pytest.mark.parametrize("start,count", [(True, 1), (-1, 1), (2, 1), (0, 0),
                                        (0, True), (0, 3)])
def test_normal_target_selection_rejects_invalid_ranges(runner, monkeypatch, start, count):
    monkeypatch.setattr(runner, "USE_PILOT_TARGETS", False)
    monkeypatch.setattr(runner, "START_TARGET", start)
    monkeypatch.setattr(runner, "N_TARGETS", count)
    with pytest.raises(ValueError):
        runner.select_target_pairs([({}, {}), ({}, {})])


def test_debate_switch_runs_without_other_methods(runner,monkeypatch):
    calls=[]
    def prepare(root,model,pairs,*,n_states,**options):
        calls.append((pairs,n_states,options))
        return PreparedExperiment({'agents_debate':lambda *args: 'query'})
    def forbidden(*args,**kwargs):
        raise AssertionError('Disabled generation must not start.')
    monkeypatch.setattr(runner,'RUN_MULTI_AGENT',False)
    monkeypatch.setattr(runner,'RUN_LIVE_PLUGIR',False)
    monkeypatch.setattr(runner,'RUN_MULTI_AGENT_DEBATE',True)
    monkeypatch.setattr(runner,'prepare_three_agents',forbidden)
    monkeypatch.setattr(runner,'prepare_qwen_live',forbidden)
    monkeypatch.setattr(runner,'prepare_debate_experiment',prepare)
    pairs=[({}, {})]
    methods,prepared=runner.collect_experiments('root','retriever',pairs,n_states=4)
    assert list(methods)==['agents_debate'] and list(prepared)==['agent_debate']
    assert calls[0][0] is pairs and calls[0][1]==4
    assert calls[0][2]['candidate_captions']==runner.CANDIDATE_CAPTIONS
    assert calls[0][2]['device']==runner.LIVE_DEVICE
    assert calls[0][2]['max_input_tokens']==runner.MAX_INPUT_TOKENS
