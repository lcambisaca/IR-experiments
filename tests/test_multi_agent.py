"""Offline scientific-contract tests; no downloaded models, APIs or GPU required."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from agentir.agent_prompts import messages_for
from agentir.evaluation import evaluate_experiments
from agentir.live import generated_query_method, signature
from agentir.multi_agent import (AgentChat, AgentSettings, RetrievalContext,
    generate_agent_dialogue, question_key, unique_proposals)
from test_live import TinyRetriever, RoundRobinKMeans, ScriptedAnswerer


class AgentTransport:
    """Deterministic JSON transport exposing stage messages and interruption behavior."""

    def __init__(self, *, known=False, bad_query=False, malformed=False, fail_stage=None):
        """Choose semantic branches without introducing target-dependent responses."""
        self.records, self.cache, self.active = [], {}, False
        self.known, self.bad_query, self.malformed = known, bad_query, malformed
        self.fail_stage = fail_stage
        self.new_calls = 0

    def begin_round(self, interaction, case):
        """Track reproducible slots as the real transport does."""
        self.interaction, self.case = interaction, case

    def create(self, **kwargs):
        """Answer each actual stage schema, rejecting hidden evaluation information."""
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
        role = ('attributes' if 'Emphasize confirmed colours' in role_text else
                'relations' if 'Emphasize confirmed actions' in role_text else 'full')
        cache_key = signature(kwargs)
        if cache_key in self.cache:
            result = self.cache[cache_key]
        else:
            self.new_calls += 1
            if stage == 'reformulate':
                r = data['record']
                result = {'query': r['initial_caption'] + ' ' + role + ' ' +
                          ' '.join(qa['answer'] for qa in r['dialogue']),
                          'emphasis': role, 'coverage_issue': 'none'}
            elif stage == 'query_check':
                result = {'faithful': not self.bad_query, 'reason': 'scripted audit'}
            elif stage == 'propose':
                result = {'question': f'What is the {role} detail?',
                    'evidence_ids': [data['candidates'][0]['id']],
                    'unknown_detail': role, 'motivation': 'caption evidence', 'assumption_check': 'none'}
            elif stage == 'filter':
                result = {'status': 'already_known' if self.known else 'unknown', 'reason': 'scripted context check'}
            elif stage == 'critique':
                result = {'verdict': 'revise', 'reason': 'make the detail more precise',
                    'evidence_ids': data['proposal']['evidence_ids'],
                    'suggested_question': data['proposal']['question'].replace('detail', 'specific detail')}
            elif stage == 'revise':
                result = {'action': 'revise', 'question': data['critique']['suggested_question'],
                    'evidence_ids': data['proposal']['evidence_ids'], 'response_to_critique': 'accepted'}
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
        settings=AgentSettings(threshold_low=100, exclude_top_k=2), run_identity='test',
        _kmeans=RoundRobinKMeans())


def run_case(case, path, *, critique=True, transport=None, n_states=3):
    """Run the public generator using controlled language and answer transports."""
    qwen = transport if transport is not None else AgentTransport()
    answerer = ScriptedAnswerer(qwen)
    state = generate_agent_dialogue(**case, questioner=qwen, answerer=answerer,
        n_states=n_states, checkpoint_path=path, critique=critique)
    return state, qwen, answerer


def test_two_conditions_share_initial_proposals_and_answer_once(agent_case, tmp_path):
    """Critique changes proposals, while each interaction gets only one target-aware answer."""
    baseline, qb, ab = run_case(agent_case, tmp_path / 'b.json', critique=False)
    debated, qd, ad = run_case(agent_case, tmp_path / 'd.json', critique=True)
    b, d = baseline['states'][1]['audit'], debated['states'][1]['audit']
    assert b['attempts'] == d['attempts']
    assert b['before_critique'] == d['before_critique']
    assert len(d['reviews']) == 3 and all(r['question_changed'] for r in d['reviews'])
    assert d['selected_question_changed']
    assert ab.model.calls == ad.model.calls == 2
    assert all(len(s['audit']['final_proposals']) <= 3 for s in debated['states'][1:])
    assert len(d['diversity']) == 3
    assert debated['states'][0]['query'] == 'a room'
    assert all('target_rank_diagnostic' in v for v in d['branches'].values())


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
    assert [s['audit']['branches']['full']['window_size'] for s in state['states'][1:]] == list(range(100,0,-10))


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
    transport = AgentTransport(fail_stage='revise')
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
        assert s['no_question'] and len(s['audit']['attempts']) == 6
        assert s['audit']['reviews'] == []


@pytest.mark.parametrize('kwargs,message', [({'bad_query':True}, 'fidelity'), ({'malformed':True}, 'Invalid JSON')])
def test_invalid_model_outputs_fail_closed(agent_case, tmp_path, kwargs, message):
    """Do not quietly add unsupported query facts or repair arbitrary malformed responses."""
    with pytest.raises(ValueError, match=message):
        run_case(agent_case, tmp_path / 'bad.json', transport=AgentTransport(**kwargs))
    saved = json.loads((tmp_path / 'bad.json').read_text())
    assert saved['status'] == 'interrupted' and not saved['dialogue']


def test_common_kl_is_exact_original_helper(agent_case):
    """Compare every candidate KL against independent math on the same neutral subset."""
    c = RetrievalContext(agent_case['retriever'], agent_case['captions'], agent_case['settings'], kmeans=RoundRobinKMeans())
    proposals = [{'owner':r,'question':f'What is {r}?'} for r in ('colour','location','action')]
    selected, logs = c.select('a room', proposals, 0)
    scores = c.scores('a room')
    indices = scores.argsort()[-100:-2]
    p = scores[indices].softmax(0)
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
    chat.complete('query_check','full', {'record':{},'query':'room'})
    with pytest.raises(RuntimeError,match='budget'):
        chat.complete('query_check','full', {'record':{},'query':'room'})
    assert q.new_calls == 1


def test_disabled_runner_keeps_saved_defaults():
    """Importing the normal template cannot activate the new models or change user settings."""
    path = Path(__file__).resolve().parents[1] / 'run_evaluation.py'
    spec = importlib.util.spec_from_file_location('agent_runner_test', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.RUN_THREE_AGENTS is False
    assert runner.N_TARGETS == 32 and runner.N_STATES == 2 and runner.DEVICE == 'cpu'
    assert runner.AGENT_MODES == (False, True)


def test_prompt_roles_use_different_scenes_and_abstention():
    """Check the reviewed pack includes concrete varied examples and shared evidence rules."""
    from agentir.agent_prompts import EXAMPLES, ROLES
    assert len(set(EXAMPLES.values())) == 3 and 'abstain' in EXAMPLES['relations']
    assert 'kitchen' in EXAMPLES['full'] and 'suitcase' in EXAMPLES['attributes']
    for role in ROLES:
        assert 'authoritative' in messages_for('reformulate',role,{})[0]['content']


def test_prepared_experiment_runs_both_modes_and_exports_diagnostics(agent_case, tmp_path, monkeypatch):
    """Exercise model wiring, actual prepared callbacks, audit CSVs and common metrics offline."""
    import importlib.metadata
    import agentir.live_models as models
    import sys
    from types import SimpleNamespace
    from agentir.multi_agent import prepare_agent_experiment
    root = tmp_path.parent / 'agent_project'
    root.mkdir()
    paths = ('src/agentir/multi_agent.py', 'src/agentir/agent_prompts.py', 'src/agentir/live_models.py',
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
    monkeypatch.setattr('agentir.multi_agent.resolve_target_image',lambda *args,**kwargs:agent_case['image_path'])
    target = agent_case['retriever'].cache.paths[0]
    pairs = [({'img':target,'dialog':['a room','HUMAN_SECRET','FUTURE_SECRET']},
              {'img':target,'dialog':['a room','saved one','saved two']})]
    bundle = prepare_agent_experiment(root,agent_case['retriever'],pairs,n_states=3,
        settings=agent_case['settings'],candidate_captions='catalogue.json')
    assert set(bundle.methods) == {'agents_critique','agents_no_critique'}
    tables = bundle.artifacts['diagnostics.json']
    assert len(tables['round_diagnostics']) == 4 and len(tables['critique_changes']) == 6
    assert all((Path(bundle.metadata['checkpoint_folder']) / (name+'.csv')).is_file() for name in tables)
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
    for key,value in {'__file__':str(root/'run_evaluation.py'),'N_TARGETS':1,'N_STATES':3,
        'K_VALUES':[1,10],'RUN_THREE_AGENTS':True,'AGENTS':agent_case['settings']}.items():
        monkeypatch.setattr(runner,key,value)
    runner.main()
    exported = next((root/'runs/experiments').iterdir())
    report = json.loads((exported/'report.json').read_text())
    assert {'agents_critique','agents_no_critique'} <= set(report['summary'])
    assert (exported/'prepared/three_agents/diagnostics.json').is_file()
    assert (exported/'rank_comparison.csv').is_file()
