"""Offline scientific-contract tests; no downloaded models, APIs or GPU required."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from agentir.multagent_experiment.agent_prompts import messages_for
from agentir.evaluation import evaluate_experiments
from agentir.live import generated_query_method, signature
from agentir.multagent_experiment.multi_agent import (AgentChat, AgentSettings, RetrievalContext,
    rewrite_query, generate_agent_dialogue, question_key, unique_proposals)
from test_live import TinyRetriever, RoundRobinKMeans, ScriptedAnswerer


class AgentTransport:
    """Deterministic JSON transport exposing stage messages and interruption behavior."""

    def __init__(self, *, known=False, malformed=False, fail_stage=None):
        """Choose semantic branches without introducing target-dependent responses."""
        self.records, self.cache, self.active = [], {}, False
        self.known, self.malformed = known, malformed
        self.fail_stage = fail_stage
        self.new_calls = 0

    def begin_round(self, interaction, case):
        """Track reproducible slots as the real transport does."""
        self.interaction, self.case = interaction, case

    def create(self, **kwargs):
        """Answer each actual stage schema, rejecting hidden evaluation information."""
        system = kwargs['messages'][0]['content']
        if system.startswith('Your role is to reconstruct') or system.startswith('Your role is to paraphrase'):
            stage = 'reconstruct_dialog' if system.startswith('Your role is to reconstruct') else 'paraphrase'
            content = f"a room with confirmed detail {len(kwargs['messages'][-1]['content'])}"
            self.active = True
            self.records.append({'stage': stage, 'payload': deepcopy(kwargs), 'result': content})
            return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        packet = json.loads(kwargs['messages'][1]['content'])
        stage, data = packet['stage'], packet['reference_data']
        serialized = json.dumps(data)
        assert '.jpg' not in serialized and 'target_rank' not in serialized
        assert 'HUMAN_SECRET' not in serialized and 'FUTURE_SECRET' not in serialized
        if self.fail_stage == stage:
            self.fail_stage = None
            raise RuntimeError('deliberate agent interruption')
        self.active = True
        role_text = kwargs['messages'][0]['content']
        # Role strings are in the prompt; generation differs without peeking at labels.
        role = ('attributes' if 'ROLE: attributes.' in role_text else
                'relations' if 'ROLE: relationships.' in role_text else 'full')
        cache_key = signature(kwargs)
        if cache_key in self.cache:
            result = self.cache[cache_key]
        else:
            self.new_calls += 1
            if stage == 'propose_batch':
                result = {'proposals': [
                    {'question': f'What is the {role} detail {slot} round {self.interaction}?',
                     'evidence_ids': [data['candidates'][0]['id']]}
                    for slot in range(data['question_count'])]}
            elif stage == 'filter_batch':
                result = {'decisions': [
                    {'slot': p['slot'], 'status': 'already_known' if self.known else 'unknown',
                     'reason': 'scripted context check'} for p in data['proposals']]}
            elif stage == 'critique':
                result = {'slot': data['proposals'][-1]['slot'], 'reason': 'Best supported distinction.'}
            else:
                raise AssertionError(stage)
            self.cache[cache_key] = result
        self.records.append({'stage': stage, 'payload': deepcopy(data), 'result': deepcopy(result)})
        text = 'not JSON' if self.malformed else json.dumps(result)
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}

    def park(self):
        """Expose memory-parking state to the fake BLIP-2 answerer."""
        self.active = False


@pytest.fixture
def agent_case(tmp_path, monkeypatch):
    """Build a real original-helper runtime on deterministic tiny vectors and a JPEG."""
    def forbidden(*args, **kwargs):
        """Refuse accidental image/model network access in this module."""
        raise AssertionError('network forbidden')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    retriever = TinyRetriever()
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (4, 4), 'blue').save(image)
    return dict(retriever=retriever, initial_caption='a room', target_id=0, image_path=image,
        captions={n: [f'public caption {i}'] for i, n in enumerate(retriever.cache.paths)},
        settings=AgentSettings(threshold_low=100, exclude_top_k=2),
        _kmeans=RoundRobinKMeans())


def run_case(case, path, *, critique=True, transport=None, n_states=3):
    """Run the public generator using controlled language and answer transports."""
    qwen = transport if transport is not None else AgentTransport()
    answerer = ScriptedAnswerer(qwen)
    state = generate_agent_dialogue(**case, questioner=qwen, answerer=answerer,
        n_states=n_states, checkpoint_path=path, critique=critique)
    return state, qwen, answerer


def test_two_conditions_share_initial_proposals_and_answer_once(agent_case, tmp_path, monkeypatch):
    """Critique changes proposals, while each interaction gets only one target-aware answer."""
    baseline, qb, ab = run_case(agent_case, tmp_path / 'b.json', critique=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Critique must skip KL selection")
    monkeypatch.setattr(RetrievalContext, 'select', forbidden)
    debated, qd, ad = run_case(agent_case, tmp_path / 'd.json', critique=True)
    b, d = baseline['states'][1]['audit'], debated['states'][1]['audit']
    assert b['attempts'] == d['attempts']
    assert b['before_critique'] == d['before_critique']
    assert b['selector'] == 'min_kl' and b['kl']
    assert d['selector'] == 'critique' and d['kl'] == []
    assert d['selected_question'] == d['final_proposals'][d['critique_decision']['slot']]['question']
    assert [r['stage'] for r in qd.records][:6] == ['propose_batch'] * 3 + ['filter_batch', 'critique', 'reconstruct_dialog']
    assert ab.model.calls == ad.model.calls == 2
    assert all(len(s['audit']['final_proposals']) <= 3 * agent_case['settings'].questions_per_agent for s in debated['states'][1:])
    assert debated['states'][0]['query'] == 'a room'


def test_official_rank_each_state_enters_original_evaluator(agent_case, tmp_path):
    """Ensure the official trajectory is neutral queries, never an oracle-best branch."""
    state, qwen, _ = run_case(agent_case, tmp_path / 'case.json', n_states=11)
    target = agent_case['retriever'].cache.paths[0]
    pairs = [({'img': target, 'dialog': ['a room'] + ['HUMAN_SECRET'] * 10},
              {'img': target, 'dialog': ['a room'] + ['FUTURE_SECRET'] * 10})]
    result = evaluate_experiments(agent_case['retriever'], pairs,
        {'agents': generated_query_method([state])}, rounds=11, ks=[1,10])
    rows = [r for r in result['per_example'] if r['method'] == 'agents']
    assert len(rows) == 11
    assert [r['target_rank'] for r in rows] == [s['target_rank'] for s in state['states']]
    assert result['summary']['agents']['BRI'] is not None
    assert [s['audit']['shared_retrieval']['window_size'] for s in state['states'][1:]] == list(range(100,0,-10))


def test_no_target_rank_or_identity_enters_decisions(agent_case, tmp_path):
    """Changing the evaluation target alters measured ranks, not questions or critiques."""
    a, qa, _ = run_case(agent_case, tmp_path / 'a.json')
    b, qb, _ = run_case({**agent_case, 'target_id': 90}, tmp_path / 'b.json')
    assert qa.records == qb.records
    assert a['dialogue'] == b['dialogue']
    assert [s['target_rank'] for s in a['states']] != [s['target_rank'] for s in b['states']]


def test_completed_resume_and_extension(agent_case, tmp_path):
    """Completed states are reused without calls; extending preserves the completed prefix."""
    short, _, _ = run_case(agent_case, tmp_path / 'resume.json', n_states=2)
    reused, q, a = run_case(agent_case, tmp_path / 'resume.json', n_states=2)
    assert q.records == [] and a.model.calls == 0
    extended, _, _ = run_case(agent_case, tmp_path / 'resume.json', n_states=3)
    full, _, _ = run_case(agent_case, tmp_path / 'full.json', n_states=3)
    assert extended['states'] == full['states']
    assert extended['states'][:2] == short['states'] == reused['states']


def test_interrupted_round_replays_cached_calls(agent_case, tmp_path):
    """Partial critique failures keep the previous completed state and can resume exactly."""
    transport = AgentTransport(fail_stage='critique')
    with pytest.raises(RuntimeError, match='deliberate'):
        run_case(agent_case, tmp_path / 'partial.json', transport=transport)
    saved = json.loads((tmp_path / 'partial.json').read_text())
    assert len(saved['states']) == 1 and saved['status'] == 'interrupted'
    assert saved['dialogue'] == [] and saved['partial_round_audit']
    resumed, _, _ = run_case(agent_case, tmp_path / 'partial.json', transport=transport)
    full, _, _ = run_case(agent_case, tmp_path / 'full.json')
    assert resumed['states'] == full['states']


def test_all_redundant_proposals_do_not_force_answer(agent_case, tmp_path):
    """Bounded failed attempts stay internal and preserve an unchanged official ranking."""
    state, q, answer = run_case(agent_case, tmp_path / 'none.json', transport=AgentTransport(known=True))
    assert answer.model.calls == 0 and state['dialogue'] == []
    assert all(s['query'] == 'a room' for s in state['states'])
    for s in state['states'][1:]:
        assert s['no_question'] and len(s['audit']['attempts']) == 3 * agent_case['settings'].questions_per_agent
        assert s['audit']['critique_decision'] is None


def test_invalid_model_outputs_fail_closed(agent_case, tmp_path):
    """Do not silently repair malformed model responses."""
    with pytest.raises(ValueError, match='Invalid JSON'):
        run_case(agent_case, tmp_path / 'bad.json', transport=AgentTransport(malformed=True))
    saved = json.loads((tmp_path / 'bad.json').read_text())
    assert saved['status'] == 'interrupted' and not saved['dialogue']


@pytest.mark.parametrize("slot", [-1, 1, True, "0", None])
def test_critique_cannot_select_an_unlisted_slot(slot):
    from types import SimpleNamespace
    response = {'slot': slot, 'reason': 'Reviewed evidence.'}
    chat = AgentChat(SimpleNamespace(create=lambda **kwargs: {'choices': [
        {'message': {'content': json.dumps(response)}, 'finish_reason': 'stop'}]}), AgentSettings())
    with pytest.raises(ValueError, match='eligible proposal slot'):
        chat.complete('critique', 'full', {'proposals': [{'slot': 0}]})


def test_common_kl_is_exact_original_helper(agent_case):
    """Compare every candidate KL against independent math on the same neutral subset."""
    c = RetrievalContext(agent_case['retriever'], agent_case['captions'], agent_case['settings'], kmeans=RoundRobinKMeans())
    proposals = [{'owner':r,'question':f'What is {r}?'} for r in ('colour','location','action')]
    selected, logs = c.select('a room', proposals, 0)
    scores = c.scores('a room')
    indices = scores.argsort()[-100:-2]
    p = scores[indices].softmax(0)
    upstream = c.namespace["select_question"](
        caption_recon='a room', questions=[p['question'] for p in proposals],
        cossim_prev=scores, k=2, img_embs=c.retriever.images, threshold=100, round=1)
    assert selected == upstream
    expected = []
    for proposal in proposals:
        q = c.scores('a room, ' + proposal['question'])[indices].softmax(0)
        expected.append(float((p * (p.log() - q.log())).sum()))
    assert [r['kl'] for r in logs] == pytest.approx(expected)
    assert selected == proposals[min(range(3),key=lambda i:expected[i])]['question']


def test_duplicate_questions_get_one_selection_slot():
    """Exact rephrasings normalized by punctuation/case do not add voting weight."""
    proposals = [{'owner':'full','question':'What colour?'},
                 {'owner':'attributes','question':'WHAT COLOUR ?'}]
    result = unique_proposals(proposals)
    assert len(result) == 1 and result[0]['duplicate_owners'] == ['attributes']


def test_call_budget_fails_before_additional_model_execution():
    """Prevent a bad configuration from creating an unbounded proposal/critique loop."""
    q = AgentTransport()
    chat = AgentChat(q, AgentSettings(max_calls_per_round=1))
    chat.begin({'initial_caption':'room','dialogue':[]},0)
    chat.complete('critique','full', {'proposals':[{'slot':0}]})
    with pytest.raises(RuntimeError,match='budget'):
        chat.complete('critique','full', {'proposals':[{'slot':0}]})
    assert q.new_calls == 1


def test_three_batches_generate_five_questions_each_and_ask_one_answer(agent_case, tmp_path):
    """Three batch proposals and three batch audits produce only one target answer."""
    from dataclasses import replace
    agent_case['settings'] = replace(agent_case['settings'], questions_per_agent=5, max_calls_per_round=80)
    state, qwen, answerer = run_case(agent_case, tmp_path/'five.json', critique=False, n_states=2)
    audit = state['states'][1]['audit']
    assert len(audit['attempts']) == len(audit['final_proposals']) == 15
    for role in ('full','attributes','relations'):
        proposals = [a for a in audit['attempts'] if a['owner']==role]
        assert {a['slot'] for a in proposals} == set(range(5))
        calls = [r for r in qwen.records if r['stage']=='propose_batch' and r['payload']['agent_role']==role]
        assert len(calls) == 1 and calls[0]['payload']['candidates']==audit['shared_retrieval']['captions']
    stages = [r['stage'] for r in qwen.records]
    assert stages.count('propose_batch') == 3 and stages.count('filter_batch') == 1
    assert max(i for i,s in enumerate(stages) if s=='propose_batch') < stages.index('filter_batch')
    assert not any(s in ('propose','filter') for s in stages)
    assert answerer.model.calls == 1
    assert audit['critique_decision'] is None
    assert not any(r['stage'] in ('critique',) for r in qwen.records)
    assert state['states'][1]['question'] in {p['question'] for p in audit['final_proposals']}


def test_batch_duplicates_are_omitted_without_replacements(agent_case, tmp_path):
    """Duplicates within/across batches are omitted without a replacement call."""
    from dataclasses import replace
    class RepeatingTransport(AgentTransport):
        def create(self, **kwargs):
            if kwargs['messages'][0]['content'].startswith('Your role is to '):
                return super().create(**kwargs)
            packet = json.loads(kwargs['messages'][1]['content'])
            response = super().create(**kwargs)
            if packet['stage']=='propose_batch':
                result = json.loads(response['choices'][0]['message']['content'])
                for p in result['proposals']:
                    p['question'] = 'What colour is the wall?'
                response['choices'][0]['message']['content'] = json.dumps(result)
            return response
    agent_case['settings'] = replace(agent_case['settings'], questions_per_agent=5, max_calls_per_round=80)
    state, qwen, answerer = run_case(agent_case,tmp_path/'duplicates.json',critique=False,
                          transport=RepeatingTransport(),n_states=2)
    audit=state['states'][1]['audit']
    assert len(audit['attempts']) == 15
    assert len(audit['final_proposals']) == 1
    assert sum(a['filter']['status']=='duplicate' for a in audit['attempts']) == 14
    assert all(a['attempt']==0 for a in audit['attempts'])
    assert sum(r['stage']=='propose_batch' for r in qwen.records)==3
    assert sum(r['stage']=='filter_batch' for r in qwen.records)==1
    assert answerer.model.calls==1


def test_five_round_shared_evidence_uses_five_calls_and_one_answer_per_round(agent_case, tmp_path, monkeypatch):
    """Count real orchestration across five rounds; all role-local slots survive one global audit."""
    from dataclasses import replace
    calls = []
    original = RetrievalContext.candidates
    def tracked(self, query, interaction, role, record):
        calls.append((query, interaction, role))
        return original(self, query, interaction, role, record)
    monkeypatch.setattr(RetrievalContext, 'candidates', tracked)
    agent_case['settings'] = replace(agent_case['settings'], questions_per_agent=5, max_calls_per_round=5)
    state, qwen, answerer = run_case(agent_case,tmp_path/'five_rounds.json',critique=False,n_states=6)
    assert len(state['states'])==6 and len(state['dialogue'])==answerer.model.calls==5
    assert len(calls)==5 and all(role=='shared' for _,_,role in calls)
    assert len(qwen.records)==25
    for previous, current in zip(state['states'],state['states'][1:]):
        audit=current['audit']
        shared=audit['shared_retrieval']
        assert shared['query']==previous['query']
        assert len(audit['final_proposals'])==15 and current['llm_calls']==5
        stages=[entry['stage'] for entry in current['llm_audit']]
        assert stages==['propose_batch']*3+['filter_batch','reconstruct_dialog']
        decisions=next(entry for entry in current['llm_audit'] if entry['stage']=='filter_batch')
        assert [p['slot'] for p in decisions['payload']['proposals']]==list(range(15))
        assert all('owner' not in p for p in decisions['payload']['proposals'])
        assert len(decisions['payload']['record']['dialogue'])==current['round']-1


def test_memory_blocks_asked_and_rejected_but_keeps_unasked_options(agent_case, tmp_path):
    """Memory survives a checkpoint resume without treating proposals as confirmed answers."""
    from dataclasses import replace
    class FixedPool(AgentTransport):
        def create(self, **kwargs):
            if kwargs['messages'][0]['content'].startswith('Your role is to '):
                return super().create(**kwargs)
            packet = json.loads(kwargs['messages'][1]['content'])
            response = super().create(**kwargs)
            result = json.loads(response['choices'][0]['message']['content'])
            if packet['stage']=='propose_batch':
                for p in result['proposals']:
                    p['question'] = p['question'].split(' round ')[0] + '?'
                result['proposals'][0]['question'] = 'Is it a room?'
            elif packet['stage']=='filter_batch':
                for p,d in zip(packet['reference_data']['proposals'], result['decisions']):
                    if p['question']=='Is it a room?':
                        d.update(status='already_known', reason='Room is in the initial caption.')
            response['choices'][0]['message']['content'] = json.dumps(result)
            return response
    agent_case['settings'] = replace(agent_case['settings'], questions_per_agent=5)
    path = tmp_path/'memory.json'
    first, _, a = run_case(agent_case,path,critique=False,transport=FixedPool(),n_states=2)
    extended, qwen, b = run_case(agent_case,path,critique=False,transport=FixedPool(),n_states=3)
    assert a.model.calls == b.model.calls == 1
    assert len(extended['dialogue'])==2
    asked = first['dialogue'][0]['question']
    audit = extended['states'][2]['audit']
    assert any(x['proposal']['question']==asked and x['filter']['status']=='already_asked'
               for x in audit['attempts'])
    assert any(x['proposal']['question']=='Is it a room?' and x['filter']['status']=='previously_rejected'
               for x in audit['attempts'])
    assert audit['final_proposals']  # Useful questions from round one were not blacklisted.
    for call in qwen.records:
        if call['stage']=='filter_batch':
            assert all(p['question'] not in (asked,'Is it a room?') for p in call['payload']['proposals'])
        if call['stage']=='propose_batch':
            avoided = {p['question'] for p in call['payload']['attempted_proposals']}
            assert {asked,'Is it a room?'} <= avoided
            assert call['payload']['record']['dialogue']==first['dialogue']
    assert json.loads(path.read_text())['question_memory']==extended['question_memory']


@pytest.mark.parametrize('result', [
    {'proposals': 'not a list'},
    {'proposals': [{}]},
    {'proposals': [{'question':'What colour?','evidence_ids':[], 'unknown_detail':'colour',
                    'motivation':'test','assumption_check':'none'}]*6},
])
def test_invalid_proposal_batch_fails_closed(result):
    from types import SimpleNamespace
    transport = SimpleNamespace(create=lambda **kwargs: {
        'choices':[{'message':{'content':json.dumps(result)},'finish_reason':'stop'}]})
    with pytest.raises(ValueError):
        AgentChat(transport, AgentSettings()).complete('propose_batch','full',{'question_count':5})


@pytest.mark.parametrize('decisions', [[],
    [{'slot':0,'status':'unknown','reason':'test'}]*2,
    [{'slot':0,'status':'unknown','reason':'test'}, {'slot':True,'status':'unknown','reason':'test'}],
])
def test_batch_filter_requires_exact_slot_coverage(decisions):
    from types import SimpleNamespace
    transport = SimpleNamespace(create=lambda **kwargs: {
        'choices':[{'message':{'content':json.dumps({'decisions':decisions})},'finish_reason':'stop'}]})
    with pytest.raises(ValueError,match='one valid decision per slot'):
        AgentChat(transport, AgentSettings()).complete('filter_batch','full',{'proposals':[{'slot':0},{'slot':1}]})


def test_disabled_runner_does_not_prepare_models(monkeypatch):
    """Explicitly disabling live methods skips preparation regardless of edited defaults."""
    path = Path(__file__).resolve().parents[1] / 'run_evaluation.py'
    spec = importlib.util.spec_from_file_location('agent_runner_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def forbidden(*args, **kwargs):
        """Fail if disabled generation attempts any model preparation."""
        raise AssertionError('Disabled method attempted model preparation')
    for name, value in {'RUN_MULTI_AGENT': False, 'RUN_LIVE_PLUGIR': False,
                        'prepare_three_agents': forbidden, 'prepare_qwen_live': forbidden}.items():
        monkeypatch.setattr(runner, name, value)
    assert runner.collect_experiments(None, None, []) == ({}, {})


def test_pilot_selects_only_cake_and_giraffe_by_identity():
    from agentir.multagent_experiment import setup
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / 'assets/protocol/baseline_matched.json').read_text())
    selected = setup.select_pilot_pairs([(r, r) for r in reversed(rows)])
    assert [left['dialog'][0] for left, _ in selected] == [
        'a group of people posing in front of a cake', 'a giraffe is standing in an open field']
    assert len(selected) == 2
    assert setup.AGENTS.questions_per_agent == 5 and setup.AGENTS.max_calls_per_round == 6


@pytest.mark.parametrize('indices', [(0,), (0, 1, 1)])
def test_pilot_selection_rejects_missing_or_duplicate_targets(indices):
    from agentir.multagent_experiment import setup
    rows = [{'img': name} for name in setup.TARGET_IMAGES]
    with pytest.raises(ValueError, match='Expected one matched pair'):
        setup.select_pilot_pairs([(rows[i], rows[i]) for i in indices])


def test_specialist_prompts_share_grounding_and_demonstrate_different_questions():
    """Specialities change questions, while neutral rewrites never receive a specialist role."""
    from agentir.multagent_experiment.agent_prompts import EXAMPLES, ROLES, COMMON_EXAMPLE
    assert len(set(EXAMPLES.values())) == 3
    assert all(COMMON_EXAMPLE in example for example in EXAMPLES.values())
    assert 'colour' in EXAMPLES['attributes'] and 'table near' in EXAMPLES['relations']
    for role in ROLES:
        assert 'authoritative' in messages_for('filter_batch',role,{})[0]['content']
        assert ROLES[role] not in messages_for('filter_batch',role,{})[0]['content']
        assert ROLES[role] in messages_for('propose_batch',role,{})[0]['content']


def test_prepared_experiment_runs_both_modes_and_exports_dialogues(agent_case, tmp_path, monkeypatch):
    """Exercise model wiring, actual prepared callbacks, audit CSVs and common metrics offline."""
    import importlib.metadata
    import agentir.live_models as models
    import sys
    from types import SimpleNamespace
    from agentir.multagent_experiment.multi_agent import prepare_agent_experiment
    root = tmp_path.parent / 'agent_project'
    root.mkdir()
    paths = ('src/agentir/multagent_experiment/multi_agent.py', 'src/agentir/multagent_experiment/agent_prompts.py', 'src/agentir/live_models.py',
             'src/agentir/plugir_upstream.py', 'src/agentir/retrieval.py')
    for name in paths:
        file = root / name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('# source fixture',encoding='utf-8')
    (root / 'catalogue.json').write_text(json.dumps([
        {'id':n,'caption':c} for n,c in agent_case['captions'].items()]))
    qwen = AgentTransport()
    answerer = ScriptedAnswerer(qwen)
    monkeypatch.setattr(models,'require_cuda',lambda device: device)
    monkeypatch.setattr(models,'Qwen3',lambda *args,**kwargs: qwen)
    monkeypatch.setattr(models,'Blip2Answerer',lambda *args,**kwargs: answerer)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'test GPU')
    original_find = importlib.util.find_spec
    extras = ('accelerate','sentencepiece','fast_pytorch_kmeans','psutil')
    monkeypatch.setattr(importlib.util,'find_spec',lambda name: SimpleNamespace() if name in extras else original_find(name))
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name: 'test' if name=='fast-pytorch-kmeans' else original_version(name))
    monkeypatch.setitem(sys.modules,'fast_pytorch_kmeans',SimpleNamespace(KMeans=lambda **kwargs:RoundRobinKMeans()))
    monkeypatch.setattr('agentir.multagent_experiment.multi_agent.resolve_target_image',lambda *args,**kwargs:agent_case['image_path'])
    target = agent_case['retriever'].cache.paths[0]
    pairs = [({'img':target,'dialog':['a room','HUMAN_SECRET','FUTURE_SECRET']},
              {'img':target,'dialog':['a room','saved one','saved two']})]
    bundle = prepare_agent_experiment(root,agent_case['retriever'],pairs,n_states=3,
        settings=agent_case['settings'],candidate_captions='catalogue.json')
    assert set(bundle.methods) == {'agents_critique','agents_no_critique'}
    result = evaluate_experiments(agent_case['retriever'],pairs,bundle.methods,rounds=3,ks=[1,10])
    assert 'agents_critique' in result['summary'] and 'agents_no_critique' in result['summary']
    # Same first-round proposals are literally shared before histories diverge.
    cases = bundle.artifacts['agent_dialogues.json']
    assert cases['agents_critique'][0]['states'][1]['audit']['attempts'] == cases['agents_no_critique'][0]['states'][1]['audit']['attempts']
    # Also exercise the actual main template's enabled switch and generic exports.
    runner_file = Path(__file__).resolve().parents[1] / 'run_evaluation.py'
    spec = importlib.util.spec_from_file_location('enabled_agent_runner',runner_file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = {k:k+'.json' for k in ('cache','manifest','baseline_queries','supplied_queries')}
    (root/'experiment.json').write_text(json.dumps(config))
    for value in config.values():
        (root/value).write_text('[]')
    (root/'run_evaluation.py').write_text('# runner fixture')
    monkeypatch.setattr(runner,'load_cache',lambda *args:agent_case['retriever'].cache)
    monkeypatch.setattr(runner,'matched_queries',lambda *args:pairs)
    monkeypatch.setattr(runner,'BlipRetriever',lambda *args,**kwargs:agent_case['retriever'])
    monkeypatch.setattr(runner,'prepare_three_agents',lambda *args,**kwargs:bundle)
    monkeypatch.setattr(runner,'select_pilot_pairs',lambda selected: selected)
    for key,value in {'__file__':str(root/'run_evaluation.py'),'N_TARGETS':1,'N_STATES':3,
        'K_VALUES':[1,10],'RUN_MULTI_AGENT':True,'USE_PILOT_TARGETS':True,
        'RUN_LIVE_PLUGIR':False,'INCLUDE_SAVED_REFERENCES':False}.items():
        monkeypatch.setattr(runner,key,value)
    runner.main()
    exported = next((root/'runs/experiments').iterdir())
    report = json.loads((exported/'report.json').read_text())
    assert {'agents_critique','agents_no_critique'} <= set(report['summary'])
    saved = json.loads((exported/'prepared/three_agents/agent_dialogues.json').read_text())
    assert saved == cases
    assert all(s['audit']['critique_decision'] for s in saved['agents_critique'][0]['states'][1:])
    assert (exported/'rank_comparison.csv').is_file()


def test_agent_answer_prompt_matches_plugir_description_plus_question(agent_case, tmp_path):
    """The shared answerer sees the same prompt construction in both pipelines."""
    state, _, answerer = run_case(agent_case, tmp_path/'answer_prompt.json',
                                  critique=False, n_states=3)
    assert answerer.processor.prompts == [
        'Question: ' + previous['query'] + '. ' + current['question'] + ' Answer:'
        for previous, current in zip(state['states'], state['states'][1:])
    ]


def test_short_answer_grounding_examples_reach_neutral_stages():
    """Verify live prompts include the observed failure and its negative controls.

    This is a prompt contract check, not proof of pretrained model judgement.
    """
    from agentir.multagent_experiment.agent_prompts import QA_GROUNDING_EXAMPLE
    for stage in ('critique', 'filter_batch'):
        prompt = messages_for(stage, 'full', {})[0]['content']
        assert QA_GROUNDING_EXAMPLE in prompt
        assert 'Interpret each answer together with its question' in prompt
    assert 'If the answer were no' in QA_GROUNDING_EXAMPLE
    assert 'only repeated the original caption' in QA_GROUNDING_EXAMPLE
    assert 'does not confirm their count or species' in QA_GROUNDING_EXAMPLE


@pytest.mark.parametrize("fallback", [False, True])
def test_agent_rewrite_matches_original_plugir_requests(agent_case, fallback):
    """Compare exact original messages, sampling options, formatting, and fallback."""
    class Capture:
        def __init__(self):
            self.requests = []
        def create(self, **kwargs):
            self.requests.append(deepcopy(kwargs))
            is_rewrite = kwargs['messages'][0]['content'].startswith('Your role is to reconstruct')
            content = ('previous query' if fallback else 'new description') if is_rewrite else 'paraphrased description'
            return {'choices': [{'message': {'content': content}}]}

    context = RetrievalContext(agent_case['retriever'], agent_case['captions'],
                               agent_case['settings'], kmeans=RoundRobinKMeans())
    record = {'initial_caption': 'a room', 'dialogue': [
        {'question': 'What colour is the wall?', 'answer': 'blue'},
        {'question': 'Is there a chair?', 'answer': 'yes'},
    ]}
    transport = Capture()
    chat = AgentChat(transport, AgentSettings(max_calls_per_round=6))
    chat.calls = 5
    actual = rewrite_query(chat, record, 'previous query', context.namespace)
    requests = deepcopy(transport.requests)
    assert chat.calls == 6 + int(fallback)
    assert [a['stage'] for a in chat.audit] == ['reconstruct_dialog'] + (['paraphrase'] if fallback else [])
    transport.requests.clear()
    # These are the unchanged functions called by the original live PlugIR loop.
    expected = context.namespace['reconstruct_dialog']([
        'a room', 'What colour is the wall? blue', 'Is there a chair? yes',
    ])
    if expected == 'previous query':
        expected = context.namespace['paraphrase'](expected)
    assert actual == expected
    assert requests == transport.requests
    assert requests[0]['temperature'] == 0 and requests[0]['max_tokens'] == 512
    assert requests[0]['seed'] == chat.settings.seed
    assert requests[0]['messages'][-1]['content'] == (
        '[Caption]: a room [Dialogue]: What colour is the wall? blue, Is there a chair? yes  [New Caption]: ')
    if fallback:
        assert requests[1]['temperature'] == .7 and requests[1]['top_p'] == .8


def test_legacy_rewrite_checkpoint_is_not_reused(agent_case, tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'states': [{'query': 'old rewrite'}], 'dialogue': []}))
    with pytest.raises(ValueError, match='different rewrite policy'):
        run_case(agent_case, path)
