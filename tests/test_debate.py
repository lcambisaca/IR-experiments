"""Offline debate contracts: no downloaded weights, target access, or GPU calls."""
from copy import deepcopy
from dataclasses import replace
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from agentir.multagent_experiment import debate
from agentir.multagent_experiment.agent_prompts import ROLES
from agentir.multagent_experiment.multi_agent import generate_agent_dialogue
from agentir.evaluation import evaluate_experiments, save_experiment_results
from test_multi_agent import agent_case, AgentTransport
from test_live import RoundRobinKMeans, ScriptedAnswerer


class DebateTransport(AgentTransport):
    """Script structured discussion, with controllable final filtering and failures."""
    def __init__(self, *, filter_statuses=(), bad_evidence_first=False, fail_stage=None,
                 malformed=False, fallback=False, flat_opening=False, candidate_question=None):
        super().__init__()
        self.filter_statuses = iter(filter_statuses)
        self.bad_evidence_first = bad_evidence_first
        self.coordinator_calls = 0
        self.fail_stage = fail_stage
        self.malformed = malformed
        self.fallback = fallback
        self.flat_opening = flat_opening
        self.candidate_question = candidate_question

    def create(self, **kwargs):
        system = kwargs['messages'][0]['content']
        if system.startswith('Your role is to '):
            response = super().create(**kwargs)
            if self.fallback and self.records[-1]['stage'] == 'reconstruct_dialog':
                response['choices'][0]['message']['content'] = 'a room'
                self.records[-1]['result'] = 'a room'
            return response
        packet = json.loads(kwargs['messages'][1]['content'])
        stage, data = packet['stage'], packet['reference_data']
        if stage == 'filter_batch':
            # The production debate uses the unchanged baseline filter.
            malformed = self.malformed
            self.malformed = False
            response = super().create(**kwargs)
            self.malformed = malformed
            result = json.loads(response['choices'][0]['message']['content'])
            status = next(self.filter_statuses, 'unknown')
            for decision in result['decisions']:
                decision.update(status=status, reason='Scripted final grounding check.')
            response['choices'][0]['message']['content'] = json.dumps(result)
            self.records[-1]['result'] = deepcopy(result)
            return response
        assert stage in ('opening','discussion','coordinate')
        serialized = json.dumps(data)
        assert '.jpg' not in serialized and 'target_rank' not in serialized
        assert 'HUMAN_SECRET' not in serialized and 'FUTURE_SECRET' not in serialized
        if self.fail_stage == stage:
            self.fail_stage = None
            raise RuntimeError('deliberate debate interruption')
        self.active = True
        role = ('attributes' if 'ROLE: attributes.' in system else
                'relations' if 'ROLE: relationships.' in system else 'full')
        ids = [data['candidates'][0]['id']]
        key = json.dumps(kwargs, sort_keys=True)
        if key in self.cache:
            result = self.cache[key]
        else:
            self.new_calls += 1
            if stage == 'opening':
                result = {'observations':[{'text':'Candidates differ in visible details.', 'evidence_ids':ids}],
                          'question':f'What is the {role} opening detail?', 'evidence_ids':ids,
                          'limitation':'Caption evidence is incomplete.'}
            elif stage == 'discussion':
                result = {'comment':'The peer suggestion could conflate two attributes; ask one.',
                          'question':f'What is the {role} discussion detail {data["discussion_round"]}?',
                          'evidence_ids':ids, 'limitation':'Answerability is uncertain.'}
            else:
                self.coordinator_calls += 1
                revision = ' revised' if 'revision_feedback' in data else ''
                result = {'question':f'What is the shared distinguishing detail round {self.interaction}{revision}?',
                          'evidence_ids':[999999] if self.bad_evidence_first and self.coordinator_calls==1 else ids,
                          'reason':'Combined the evidence while removing a compound assumption.',
                          'unresolved_disagreement':'The relative value of colour and position is uncertain.'}
            if stage == "opening" and self.flat_opening and not self.records:
                result["observations"] = ["Candidates differ in visible details."]
            if stage == "coordinate" and self.candidate_question and self.coordinator_calls == 1:
                result["question"] = self.candidate_question
            self.cache[key] = result
        self.records.append({'stage':stage,'payload':deepcopy(data),'result':deepcopy(result)})
        text = 'not JSON' if self.malformed else json.dumps(result)
        return {'choices':[{'message':{'content':text},'finish_reason':'stop'}]}


def run_debate(case, path, *, n_states=3, transport=None):
    qwen = transport if transport is not None else DebateTransport()
    answerer = ScriptedAnswerer(qwen)
    options = {**case, 'settings':replace(case['settings'], max_calls_per_round=14)}
    state = debate.generate_debate_dialogue(**options, questioner=qwen, answerer=answerer,
                                          checkpoint_path=path, n_states=n_states)
    return state, qwen, answerer


@pytest.mark.parametrize("stage", ["opening", "discussion", "coordinate"])
def test_debate_prompt_examples_match_live_output_schema(stage):
    role = "coordinator" if stage == "coordinate" else "full"
    messages = debate.debate_messages(stage, role, {"candidates": [{"id": 97}]})
    examples = [line for line in messages[0]["content"].splitlines() if line.startswith("{")]
    assert len(examples) == (2 if stage == "opening" else 1)
    assert '"id": 97' in messages[1]["content"]
    for example in examples:
        result = debate.DebateChat.validate_debate_output(
            stage, {"candidates": [{"id": 12}, {"id": 28}]}, example, "stop")
        if result["question"] != "abstain":
            assert debate.valid_question(result["question"])
            assert not debate.CANDIDATE_QUESTION.search(result["question"])


def test_debate_independent_openings_and_two_synchronous_rounds(agent_case, tmp_path, monkeypatch):
    retrievals = []
    from agentir.multagent_experiment.multi_agent import RetrievalContext
    original = RetrievalContext.candidates
    def tracked(self,*args):
        retrievals.append(args)
        return original(self,*args)
    monkeypatch.setattr(RetrievalContext,'candidates',tracked)
    state,qwen,answerer = run_debate(agent_case,tmp_path/'debate.json',n_states=2)
    current = state['states'][1]
    audit = current['audit']
    assert len(retrievals)==1 and answerer.model.calls==1
    assert current['llm_calls']==12
    assert len(audit['transcript'])==9
    assert [len(r['messages']) for r in audit['discussion_rounds']]==[3,3]
    openings = [r for r in qwen.records if r['stage']=='opening']
    assert len(openings)==3 and all('transcript' not in r['payload'] for r in openings)
    for number,length in ((1,3),(2,6)):
        responses = [r for r in qwen.records if r['stage']=='discussion'
                     and r['payload']['discussion_round']==number]
        assert len(responses)==3
        assert responses[0]['payload']==responses[1]['payload']==responses[2]['payload']
        assert len(responses[0]['payload']['transcript'])==length
        assert all(item['discussion_round']<number for item in responses[0]['payload']['transcript'])
    for record in openings+[r for r in qwen.records if r['stage']=='discussion']:
        assert record['payload']['candidates']==audit['shared_retrieval']['captions']
    assert current['question'] not in {item['content']['question'] for item in audit['transcript']}
    assert current['question']==audit['coordinator_attempts'][0]['proposal']['question']
    assert audit["question_source"] == "coordinator" and audit["fallback_attempts"] == []
    assert current['skip_reason'] is None
    assert answerer.processor.prompts==['Question: a room. '+current['question']+' Answer:']
    rewrite = next(r for r in qwen.records if r['stage']=='reconstruct_dialog')
    assert rewrite['payload']['messages'][0]['content'].startswith('Your role is to reconstruct')
    assert state['method_policy']==debate.DEBATE_POLICY


@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize("status, guidance", [
    ("unsupported", "Revise once to address this rejection"),
    ("already_known", "Choose a different unresolved detail"),
])
def test_rejected_final_question_gets_only_one_revision(agent_case,tmp_path,fallback,status,guidance):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'revision.json',n_states=2,
        transport=DebateTransport(filter_statuses=[status,'unknown'],fallback=fallback))
    current=state['states'][1]
    audit=current['audit']
    assert len(audit['coordinator_attempts'])==2 and audit['coordinator_revision_used']
    assert [a['accepted'] for a in audit['coordinator_attempts']]==[False,True]
    assert answerer.model.calls==1 and current['llm_calls']==14+int(fallback)
    coordinator=[r for r in qwen.records if r['stage']=='coordinate']
    assert coordinator[0]['payload']['transcript']==coordinator[1]['payload']['transcript']
    assert coordinator[1]['payload']['revision_feedback']['rejection']['status']==status
    original, revision = [debate.debate_messages("coordinate", "coordinator", c["payload"])[0]["content"]
                          for c in coordinator]
    assert guidance not in original and guidance in revision
    assert sum(r['stage']=='opening' for r in qwen.records)==3
    assert sum(r['stage']=='discussion' for r in qwen.records)==6


@pytest.mark.parametrize("malformed_coordinator", [False, True])
def test_debate_fallback_checks_target_answerability(agent_case, tmp_path, malformed_coordinator):
    class NeedsFallback(DebateTransport):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            entry = self.records[-1]
            if entry["stage"] == "coordinate" and malformed_coordinator:
                response["choices"][0]["message"]["content"] = "not JSON"
            if (entry["stage"] == "discussion" and entry["payload"]["discussion_round"] == 2
                    and "full discussion" in entry["result"]["question"]):
                result = {**entry["result"], "question": "Does candidate: 21232 show a lamp?"}
                response["choices"][0]["message"]["content"] = json.dumps(result)
            return response

    statuses = [] if malformed_coordinator else ["unsupported", "ambiguous"]
    state, transport, answerer = run_debate(agent_case, tmp_path / "fallback.json", n_states=2,
        transport=NeedsFallback(filter_statuses=statuses))
    current = state["states"][1]
    audit = current["audit"]
    assert all(not attempt["accepted"] for attempt in audit["coordinator_attempts"])
    assert len(audit["coordinator_attempts"]) == (1 if malformed_coordinator else 2)
    first, second, third = audit["fallback_attempts"]
    assert first["filter"]["status"] == "unsupported"
    assert "target photo" in first["filter"]["reason"]
    assert current["question"] == second["proposal"]["question"]
    assert audit["question_source"] == "attributes"
    assert third["filter"]["status"] == "unknown"
    filters = [call["payload"] for call in transport.records if call["stage"] == "filter_batch"]
    assert len(filters) == (1 if malformed_coordinator else 3)
    assert [proposal["slot"] for proposal in filters[-1]["proposals"]] == [1, 2]
    assert answerer.processor.prompts == ["Question: a room. " + current["question"] + " Answer:"]
    assert current["llm_calls"] == (13 if malformed_coordinator else 15)


def test_rejected_fallback_skips_without_answer_or_rewrite(agent_case,tmp_path):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'skip.json',n_states=2,
        transport=DebateTransport(filter_statuses=['unsupported','ambiguous','unsupported']))
    current=state['states'][1]
    assert state['status']=='complete' and current['no_question']
    assert current['skip_reason']=='no_valid_debate_question'
    assert current['query']==state['states'][0]['query']
    assert current['target_rank']==state['states'][0]['target_rank']
    assert answerer.model.calls==0 and state['dialogue']==[]
    assert current['llm_calls']==14
    assert all(attempt["filter"]["status"] == "unsupported"
               for attempt in current["audit"]["fallback_attempts"])
    assert not any(r['stage'] in ('reconstruct_dialog','paraphrase') for r in qwen.records)


def test_unjudged_debate_question_is_revised_without_reaching_answerer(agent_case, tmp_path):
    class OmittingFirstFilter(DebateTransport):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            entry = self.records[-1]
            if entry["stage"] == "filter_batch" and self.coordinator_calls == 1:
                response["choices"][0]["message"]["content"] = '{"decisions":[]}'
            return response

    state, transport, answerer = run_debate(agent_case, tmp_path / "omitted.json", n_states=2,
                                            transport=OmittingFirstFilter())
    current = state["states"][1]
    first, second = current["audit"]["coordinator_attempts"]
    assert current["audit"]["filter_omitted_slots"] == [0]
    assert not first["accepted"] and second["accepted"]
    assert "Missing model decision" in first["validation"]["reason"]
    assert current["llm_calls"] == 15 and answerer.model.calls == 1
    filters = [call["payload"] for call in transport.records if call["stage"] == "filter_batch"]
    assert [payload.get("filter_retry", False) for payload in filters] == [False, True, False]
    assert current["question"] == second["proposal"]["question"]


def test_bad_coordinator_evidence_is_revised_before_filtering(agent_case,tmp_path):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'evidence.json',n_states=2,
                                  transport=DebateTransport(bad_evidence_first=True))
    attempts=state['states'][1]['audit']['coordinator_attempts']
    assert attempts[0]['validation']['status']=='unsupported'
    assert attempts[1]['accepted'] and answerer.model.calls==1
    assert sum(r['stage']=='filter_batch' for r in qwen.records)==1
    revision = next(r["payload"]["revision_feedback"] for r in qwen.records
                    if "revision_feedback" in r["payload"])
    assert "Check the supplied candidate IDs" in revision["instruction"]


def test_debate_resume_extension_and_failure_recovery(agent_case,tmp_path):
    transport=DebateTransport(fail_stage='discussion')
    path=tmp_path/'resume.json'
    with pytest.raises(RuntimeError,match='deliberate debate'):
        run_debate(agent_case,path,n_states=2,transport=transport)
    saved=json.loads(path.read_text())
    assert saved['status']=='interrupted' and len(saved['states'])==1
    assert saved['partial_round_audit'] and saved['dialogue']==[]
    short,_,_=run_debate(agent_case,path,n_states=2,transport=transport)
    reused,qwen,answerer=run_debate(agent_case,path,n_states=2)
    assert qwen.records==[] and answerer.model.calls==0
    assert short['states']==reused['states']
    extended,_,_=run_debate(agent_case,path,n_states=3)
    full,_,_=run_debate(agent_case,tmp_path/'full.json',n_states=3)
    assert extended['states']==full['states']


@pytest.mark.parametrize("question", [
    "Does candidate 21232 show someone cutting a cake?",
    "Does candidate: 21232 show a lamp?",
    "Does Candidate #21232 show a lamp?",
    "What is shown in photo ID: 21232?",
    "What is shown in photo_id=21232?",
    "Which of the retrieved photos best matches the description?",
])
def test_candidate_directed_question_is_revised_before_answering(agent_case, tmp_path, question):
    state, transport, answerer = run_debate(agent_case, tmp_path / "target.json", n_states=2,
        transport=DebateTransport(candidate_question=question))
    current = state["states"][1]
    first, second = current["audit"]["coordinator_attempts"]
    assert first["proposal"]["question"] == question
    assert not first["accepted"] and second["accepted"]
    assert "target photo" in first["validation"]["reason"]
    filtered, = [call for call in transport.records if call["stage"] == "filter_batch"]
    assert filtered["payload"]["require_target_question"]
    assert answerer.processor.prompts == ["Question: a room. " + current["question"] + " Answer:"]
    assert question not in answerer.processor.prompts[0]


def test_target_count_question_keeps_its_original_text(agent_case, tmp_path):
    question = "Are there 3 people in the photo?"
    state, _, answerer = run_debate(agent_case, tmp_path / "count.json", n_states=2,
        transport=DebateTransport(candidate_question=question))
    current = state["states"][1]
    assert current["question"] == question
    assert len(current["audit"]["coordinator_attempts"]) == 1
    assert answerer.processor.prompts == ["Question: a room. " + question + " Answer:"]


def test_flat_observation_list_gets_one_logged_correction(agent_case, tmp_path):
    state, _, answerer = run_debate(agent_case, tmp_path / "format.json", n_states=2,
        transport=DebateTransport(flat_opening=True))
    current = state["states"][1]
    original, retry = current["llm_audit"][:2]
    assert original["stage"] == retry["stage"] == "opening"
    assert "its own supplied evidence_ids" in original["validation_error"]
    assert retry["payload"]["format_retry"]["previous_response"] == original["response"]
    assert "validation_error" not in retry
    assert len(current["audit"]["transcript"]) == 9
    assert current["llm_calls"] == 13 and answerer.model.calls == 1


def test_specialist_keeps_valid_question_after_failed_correction(agent_case, tmp_path):
    class MalformedOpenings(DebateTransport):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            entry = self.records[-1]
            if entry["stage"] == "opening" and (
                    entry["payload"].get("format_retry") or "attributes opening" in entry["result"]["question"]):
                response["choices"][0]["message"]["content"] = "not JSON"
            return response

    state, transport, answerer = run_debate(agent_case, tmp_path / "recovered.json", n_states=2,
        transport=MalformedOpenings(flat_opening=True))
    current = state["states"][1]
    first_discussion = next(call for call in transport.records if call["stage"] == "discussion")
    generalist, attributes, _ = first_discussion["payload"]["transcript"]
    assert generalist["content"]["observations"] == []
    assert generalist["content"]["question"] == "What is the full opening detail?"
    assert generalist["content"]["evidence_ids"]
    assert generalist["content"]["format_recovery"] == "partial"
    assert attributes["content"]["question"] == "abstain"
    assert attributes["content"]["format_recovery"] == "abstained"
    assert sum("format_retry" in item["payload"] for item in current["llm_audit"]) == 1
    assert current["audit"]["question_source"] == "coordinator"
    assert answerer.model.calls == 1


def test_completely_malformed_debate_skips_and_continues(agent_case, tmp_path):
    state, _, answerer = run_debate(agent_case, tmp_path / "malformed.json",
                                    transport=DebateTransport(malformed=True))
    assert state["status"] == "complete"
    assert state["dialogue"] == [] and answerer.model.calls == 0
    assert len(state["states"]) == 3
    for current in state["states"][1:]:
        assert current["skip_reason"] == "no_valid_debate_question"
        assert current["query"] == state["states"][0]["query"]
        assert current["target_rank"] == state["states"][0]["target_rank"]
        assert len(current["llm_audit"]) == 11
        assert all("validation_error" in item for item in current["llm_audit"])
        assert sum("format_retry" in item["payload"] for item in current["llm_audit"]) == 1
        assert len(current["audit"]["fallback_attempts"]) == 3


def test_debate_bounds_rejection_context_but_blocks_old_questions(agent_case, tmp_path):
    class RejectedDebate(DebateTransport):
        first_question = None

        def create(self, **kwargs):
            response = super().create(**kwargs)
            entry = self.records[-1]
            if entry["stage"] == "discussion":
                result = entry["result"]
                result["question"] = result["question"].removesuffix("?") + f" in turn {self.interaction}?"
                response["choices"][0]["message"]["content"] = json.dumps(result)
            if entry["stage"] == "coordinate" and "revision_feedback" not in entry["payload"]:
                result = entry["result"]
                if self.first_question is None:
                    self.first_question = result["question"]
                if self.interaction == 3:
                    result["question"] = self.first_question
                    response["choices"][0]["message"]["content"] = json.dumps(result)
            return response

    qwen = RejectedDebate(filter_statuses=["unsupported"] * 20)
    state, _, answerer = run_debate(agent_case, tmp_path / "history.json", n_states=5, transport=qwen)
    assert state["status"] == "complete" and answerer.model.calls == 0
    assert len(state["question_memory"]) == 20
    current = state["states"][-1]
    modeled = [call for call in current["llm_audit"]
               if call["stage"] in ("opening", "discussion", "coordinate")]
    recent = modeled[0]["payload"]["attempted_proposals"]
    assert len(recent) == 10
    assert all(call["payload"]["attempted_proposals"] == recent for call in modeled)
    assert qwen.first_question not in {item["question"] for item in recent}
    first = current["audit"]["coordinator_attempts"][0]
    assert first["proposal"]["question"] == qwen.first_question
    assert first["validation"]["status"] == "previously_rejected"


def test_target_identity_changes_metrics_not_debate(agent_case,tmp_path):
    a,qa,_=run_debate(agent_case,tmp_path/'a.json')
    b,qb,_=run_debate({**agent_case,'target_id':90},tmp_path/'b.json')
    assert qa.records==qb.records and a['dialogue']==b['dialogue']
    assert [s['target_rank'] for s in a['states']] != [s['target_rank'] for s in b['states']]


def test_baseline_checkpoint_cannot_be_resumed_as_debate(agent_case,tmp_path):
    qwen=AgentTransport()
    path=tmp_path/'baseline.json'
    generate_agent_dialogue(**agent_case,questioner=qwen,answerer=ScriptedAnswerer(qwen),
                           n_states=2,checkpoint_path=path,critique=True)
    with pytest.raises(ValueError,match='different question-generation method'):
        run_debate(agent_case,path,n_states=3)


def test_debate_rejects_insufficient_budget_before_calls(agent_case,tmp_path):
    with pytest.raises(ValueError,match='at least 14'):
        debate.generate_debate_dialogue(**{**agent_case, 'settings':replace(agent_case['settings'],max_calls_per_round=13)},questioner=DebateTransport(),
            answerer=ScriptedAnswerer(DebateTransport()),n_states=2,
            checkpoint_path=tmp_path/'budget.json')


@pytest.mark.parametrize("answerer_name", ["blip2", "qwen_vl"])
def test_prepared_debate_exports_transcript_and_metrics(agent_case,tmp_path,monkeypatch, answerer_name):
    import agentir.live_models as models
    root=tmp_path/'project'
    root.mkdir()
    (root/'catalogue.json').write_text(json.dumps([
        {'id':name,'caption':captions} for name,captions in agent_case['captions'].items()]))
    qwen=DebateTransport()
    answerer=ScriptedAnswerer(qwen)
    monkeypatch.setattr(models,'require_cuda',lambda device:device)
    monkeypatch.setattr(models,'Qwen3',lambda *args,**kwargs:qwen)
    def make_answerer(*args, **kwargs):
        return answerer
    make_answerer.IDENTITY = models.QwenVLAnswerer.IDENTITY
    model_name = "QwenVLAnswerer" if answerer_name == "qwen_vl" else "Blip2Answerer"
    monkeypatch.setattr(models, model_name, make_answerer)
    monkeypatch.setattr(debate,'resolve_target_image',lambda *args,**kwargs:agent_case['image_path'])
    original=importlib.util.find_spec
    extras=('accelerate','sentencepiece','fast_pytorch_kmeans','psutil')
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:SimpleNamespace() if name in extras else original(name))
    monkeypatch.setitem(sys.modules,'fast_pytorch_kmeans',SimpleNamespace(KMeans=lambda **kwargs:RoundRobinKMeans()))
    target=agent_case['retriever'].cache.paths[0]
    pairs=[({'img':target,'dialog':['a room','HUMAN_SECRET','FUTURE_SECRET']},
            {'img':target,'dialog':['a room','saved one','saved two']})]
    prepared=debate.prepare_debate_experiment(root,agent_case['retriever'],pairs,n_states=3,
        settings=replace(agent_case['settings'],max_calls_per_round=14),candidate_captions='catalogue.json', answerer=answerer_name)
    assert set(prepared.methods)=={'agents_debate'}
    conversations=prepared.artifacts['debate_dialogues.json']
    assert len(conversations)==1 and len(conversations[0]['states'])==3
    assert len(conversations[0]['states'][1]['audit']['transcript'])==9
    result=evaluate_experiments(agent_case['retriever'],pairs,prepared.methods,
                               rounds=3,ks=[1,10],references=())
    folder=save_experiment_results(result,root/'results')
    assert (folder/'report.json').is_file() and result['protocol']['methods']==['agents_debate']
    assert prepared.metadata['identity']['rewrite_policy']==debate.REWRITE_POLICY
    assert prepared.metadata['identity']['filter_prompt_version']==debate.FILTER_PROMPT_VERSION
    json.dumps(prepared.artifacts,allow_nan=False)


def test_initial_debate_and_baseline_retrieval_evidence_match(agent_case,tmp_path):
    qwen=AgentTransport()
    baseline=generate_agent_dialogue(**agent_case,questioner=qwen,answerer=ScriptedAnswerer(qwen),
                                    n_states=2,checkpoint_path=tmp_path/'baseline_evidence.json',critique=True)
    discussed,_,_=run_debate(agent_case,tmp_path/'debate_evidence.json',n_states=2)
    assert baseline['states'][1]['audit']['shared_retrieval']==discussed['states'][1]['audit']['shared_retrieval']
