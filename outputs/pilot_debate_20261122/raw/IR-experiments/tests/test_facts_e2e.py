"""End-to-end facts-only search inside the original PlugIR loop (scripted models, CPU)."""
import inspect
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn.functional import normalize

from agentir.facts_e2e import FACT_PROMPT, NEGATION, clean_fact
from agentir.live import generate_live_dialogue
from agentir.plugir_upstream import UpstreamExecutionStop
from agentir.retrieval import ranks_for_targets
from test_live import RoundRobinKMeans, ScriptedAnswerer, ScriptedQuestioner, case, wrapped  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FactsQuestioner(ScriptedQuestioner):
    """ScriptedQuestioner plus the new extract_fact stage (a fact, NONE, or a negative phrase)."""

    def create(self, **kwargs):
        stage = inspect.currentframe().f_back.f_code.co_name
        if stage != "extract_fact":
            namespace = {"ScriptedQuestioner": ScriptedQuestioner}
            exec(f"def {stage}(q, kw):\n    return ScriptedQuestioner.create(q, **kw)", namespace)
            return namespace[stage](self, kwargs)  # parent sees the real stage name
        assert ".jpg" not in json.dumps(kwargs) and kwargs["temperature"] == 0
        # Like greedy Qwen, the reply depends only on the request (here: on the question's detail code).
        question = kwargs["messages"][1]["content"].split("Question: ")[1].split("\n")[0]
        n = int(question.split("detail ")[1][:6], 16) if "detail " in question else 0
        content = ["a blue lamp", "NONE", "no chair visible", f"a red detail {n % 97}"][n % 4]
        result = {"choices": [{"message": {"content": content}}]}
        self.records.append({"stage": stage, "request": kwargs, "response": result})
        return result


def run(case, folder, *, questioner=None, n_states=11):
    qwen = questioner or FactsQuestioner()
    return generate_live_dialogue(**case, initial_caption="a room", target_id=0, questioner=qwen,
                                  answerer=ScriptedAnswerer(qwen), n_states=n_states,
                                  checkpoint_path=folder / "case.json", run_identity="test",
                                  _kmeans=RoundRobinKMeans(), search="facts"), qwen


def vec(retriever, text):
    inputs = retriever.processor(text=text, padding=True, return_tensors="pt")
    return normalize(retriever.model.text_encoder(**inputs).last_hidden_state[:, 0, :], dim=-1)[0]


def test_prompt_and_cleaning_identical_to_frozen_fact_fusion():
    import fact_fusion
    assert FACT_PROMPT == fact_fusion.FACT_PROMPT and NEGATION.pattern == fact_fusion.NEGATION.pattern
    for text in ["a white wetsuit.", "NONE", "no trees", '"red skis"\nextra', "  ", "None of these"]:
        assert clean_fact(text) == fact_fusion.clean_fact(text)


def test_facts_search_runs_end_to_end_and_ranks_with_facts(case, tmp_path):
    state, qwen = run(case, tmp_path)
    assert state["status"] == "complete" and len(state["states"]) == 11
    rounds = state["states"][1:]
    # Only real, positive facts are kept: NONE and negative replies give no fact.
    replies = [x["response"]["choices"][0]["message"]["content"] for x in qwen.records if x["stage"] == "extract_fact"]
    assert [r["fact"] for r in rounds] == [None if x == "NONE" or x.startswith("no ") else x for x in replies]
    assert state["facts"] == [r["fact"] for r in rounds if r["fact"]] and state["facts"]
    assert any(r["fact"] is None for r in rounds)
    retriever = case["retriever"]
    for r in rounds:  # the rank the loop shows is the facts-only search, recomputed independently
        q = normalize(torch.stack([vec(retriever, "a room")] +
                                  [vec(retriever, f"a room, {f}") for f in r["facts"]]).mean(0), dim=-1)
        assert r["target_rank"] == int(ranks_for_targets((q @ retriever.images.T)[None], [0])[0])
        rw = vec(retriever, r["query"]) @ retriever.images.T
        assert r["rewrite_only_rank"] == int(ranks_for_targets(rw[None], [0])[0])
    # The facts search changes what PlugIR sees, so the conversation differs from the original loop.
    plain, _, _ = wrapped(case, tmp_path / "plain")
    assert [r["question"] for r in plain["states"][1:]] != [r["question"] for r in rounds]
    assert "search" not in plain["identity"] and state["identity"]["search"] == "facts_only"


def test_facts_search_resumes_identically(case, tmp_path):
    full, _ = run(case, tmp_path / "full")
    with pytest.raises(UpstreamExecutionStop, match="deliberate"):
        run(case, tmp_path / "cut", questioner=FactsQuestioner(fail_at=(3, 1)))
    resumed, _ = run(case, tmp_path / "cut")
    keep = lambda s: [(r.get("question"), r.get("fact"), r["target_rank"]) for r in s["states"]]
    assert keep(resumed) == keep(full) and resumed["facts"] == full["facts"]
