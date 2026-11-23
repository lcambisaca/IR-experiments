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
                 malformed=False, fallback=False):
        super().__init__()
        self.filter_statuses = iter(filter_statuses)
        self.bad_evidence_first = bad_evidence_first
        self.coordinator_calls = 0
        self.fail_stage = fail_stage
        self.malformed = malformed
        self.fallback = fallback

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
    assert current['skip_reason'] is None
    assert answerer.processor.prompts==['Question: a room. '+current['question']+' Answer:']
    rewrite = next(r for r in qwen.records if r['stage']=='reconstruct_dialog')
    assert rewrite['payload']['messages'][0]['content'].startswith('Your role is to reconstruct')
    assert state['method_policy']==debate.DEBATE_POLICY


@pytest.mark.parametrize('fallback',[False,True])
def test_rejected_final_question_gets_only_one_revision(agent_case,tmp_path,fallback):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'revision.json',n_states=2,
        transport=DebateTransport(filter_statuses=['unsupported','unknown'],fallback=fallback))
    current=state['states'][1]
    audit=current['audit']
    assert len(audit['coordinator_attempts'])==2 and audit['coordinator_revision_used']
    assert [a['accepted'] for a in audit['coordinator_attempts']]==[False,True]
    assert answerer.model.calls==1 and current['llm_calls']==14+int(fallback)
    coordinator=[r for r in qwen.records if r['stage']=='coordinate']
    assert coordinator[0]['payload']['transcript']==coordinator[1]['payload']['transcript']
    assert coordinator[1]['payload']['revision_feedback']['rejection']['status']=='unsupported'
    assert sum(r['stage']=='opening' for r in qwen.records)==3
    assert sum(r['stage']=='discussion' for r in qwen.records)==6


def test_second_rejection_skips_without_answer_or_rewrite(agent_case,tmp_path):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'skip.json',n_states=2,
        transport=DebateTransport(filter_statuses=['unsupported','ambiguous']))
    current=state['states'][1]
    assert state['status']=='complete' and current['no_question']
    assert current['skip_reason']=='no_valid_debate_question'
    assert current['query']==state['states'][0]['query']
    assert current['target_rank']==state['states'][0]['target_rank']
    assert answerer.model.calls==0 and state['dialogue']==[]
    assert current['llm_calls']==13
    assert not any(r['stage'] in ('reconstruct_dialog','paraphrase') for r in qwen.records)


def test_bad_coordinator_evidence_is_revised_before_filtering(agent_case,tmp_path):
    state,qwen,answerer=run_debate(agent_case,tmp_path/'evidence.json',n_states=2,
                                  transport=DebateTransport(bad_evidence_first=True))
    attempts=state['states'][1]['audit']['coordinator_attempts']
    assert attempts[0]['validation']['status']=='unsupported'
    assert attempts[1]['accepted'] and answerer.model.calls==1
    assert sum(r['stage']=='filter_batch' for r in qwen.records)==1


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


def test_debate_model_errors_remain_visible(agent_case,tmp_path):
    path=tmp_path/'malformed.json'
    with pytest.raises(ValueError,match='Invalid debate JSON'):
        run_debate(agent_case,path,transport=DebateTransport(malformed=True))
    saved=json.loads(path.read_text())
    assert saved['status']=='interrupted' and saved['partial_round_audit'][0]['stage']=='opening'
    assert saved['dialogue']==[]


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
