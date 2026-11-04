"""Original-source parity and wrapper tests; no pretrained downloads or API calls."""
import ast
from copy import deepcopy
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Optional

from PIL import Image
import pytest
import torch
from torch.nn.functional import normalize
from transformers.feature_extraction_utils import BatchFeature

from agentir.evaluation import evaluate_experiments, metrics_from_ranks
from agentir.live import LiveSettings, generate_live_dialogue, generated_query_method, load_candidate_captions
from agentir.live_models import Blip2Answerer, Qwen3
from agentir.plugir_upstream import DeviceProcessor, UpstreamExecutionStop, verified_source


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail immediately if a unit test attempts a model/image download."""
    def forbidden(*args, **kwargs):
        """Expose accidental network use as a test failure."""
        raise AssertionError('Network forbidden during tests')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)


class TextProcessor:
    """Create deterministic token IDs for short fake text without pretrained files."""
    def __call__(self, text=None, **kwargs):
        """Hash every supplied string to one token while checking upstream options."""
        assert kwargs == {'padding': True, 'return_tensors': 'pt'}
        texts = [text] if isinstance(text, str) else text
        ids = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 1000 for t in texts]
        return BatchFeature({'input_ids': torch.tensor(ids)[:, None], 'attention_mask': torch.ones(len(ids), 1, dtype=torch.long)})


class TinyBlip:
    """Supply HF-shaped encoder/projection fields for the unchanged BLIP method."""
    def __init__(self):
        """Use deterministic distinct vectors and the same return-dict contract as BLIP."""
        self.config = SimpleNamespace(use_return_dict=True)
        self.text_proj = torch.nn.Identity()
    def text_encoder(self, input_ids, attention_mask=None, return_dict=True):
        """Return an eight-number CLS vector derived from the supplied token ID."""
        ids = input_ids.float()
        hidden = torch.sin(ids[..., None] * torch.arange(1, 9) / 13.)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyRetriever:
    """Exercise original retrieval calculations against a small reproducible corpus."""
    def __init__(self):
        """Create 100 normalized images and public captions; no target-aware text encoding."""
        self.images = normalize(torch.randn(100, 8, generator=torch.Generator().manual_seed(7)), dim=-1)
        self.device, self.batch_size = torch.device('cpu'), 8
        self.processor, self.model = TextProcessor(), TinyBlip()
        self.cache = SimpleNamespace(paths=[f'unlabeled2017/{i:012d}.jpg' for i in range(100)], vectors=self.images, report={})
    def scores(self, texts):
        """Compute the same deterministic normalized query/image dot products."""
        inputs = self.processor(text=texts, padding=True, return_tensors='pt')
        hidden = self.model.text_encoder(**inputs).last_hidden_state[:, 0, :]
        return normalize(hidden, dim=-1) @ self.images.T


class ScriptedQuestioner:
    """Return controlled completions while recording every original request verbatim."""
    def __init__(self, known=False, fail_at=None):
        """Choose the upstream filtering fallback branch or one deliberate interruption."""
        self.known, self.fail_at = known, fail_at
        self.records, self.active = [], False
        self.round, self.ordinal, self.cache = 0, 0, {}
    def begin_round(self, rnd, case):
        """Reset call slots so resuming a partial round can replay completed calls."""
        self.round, self.ordinal = rnd, 0
    def create(self, **kwargs):
        """Return text in the original response format, never inspect target information."""
        stage = inspect.currentframe().f_back.f_code.co_name
        assert '.jpg' not in json.dumps(kwargs) and 'target_rank' not in json.dumps(kwargs)
        slot = (self.round, self.ordinal)
        self.ordinal += 1
        if slot == self.fail_at:
            self.fail_at = None
            raise UpstreamExecutionStop('deliberate interruption')
        self.active = True
        if stage.startswith('generate_questions'):
            code = hashlib.sha256(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:6]
            content = f'What is detail {code}? trailing text ignored by original'
        elif stage == 'filter_questions':
            content = 'yes' if self.known else 'Uncertain'
        elif stage == 'reconstruct_dialog':
            # Trigger the original paraphrase fallback in first round as well.
            content = 'a room' if self.round == 0 else f'a room with detail {self.round}'
            self.round += 1
            self.ordinal = 0
        else:
            assert stage == 'paraphrase'
            content = 'a room clarified'
        result = {'choices': [{'message': {'content': content}}]}
        self.records.append({'stage': stage, 'request': deepcopy(kwargs), 'response': result})
        return result
    def park(self):
        """Simulate releasing Qwen GPU memory before answering."""
        self.active = False


class AnswerProcessor:
    """Return realistic mixed dtypes and retain the exact original answer prompt."""
    def __init__(self):
        """Start a log of target-aware VLM prompts."""
        self.prompts = []
    def __call__(self, images, text, return_tensors):
        """Record the prompt while returning float pixels and integer text IDs."""
        self.prompts.append(text)
        return BatchFeature({'pixel_values': torch.ones(1, 3, 2, 2), 'input_ids': torch.tensor([[1, 2]])})
    def decode(self, tokens, **kwargs):
        """Return a controlled answer to isolate scientific-loop behavior."""
        return ' blue '


class AnswerModel:
    """Enforce the authors' greedy-generation call with no wrapper length override."""
    def __init__(self):
        """Record call counts and a serializable fake checkpoint configuration."""
        self.calls = 0
        self.generation_config = SimpleNamespace(to_dict=lambda: {'max_length': 20})
    def generate(self, pixel_values, input_ids, **kwargs):
        """Check IDs stay integer and no extra generation settings are injected."""
        assert input_ids.dtype == torch.long and kwargs == {'do_sample': False}
        self.calls += 1
        return torch.tensor([[3, 4]])


class ScriptedAnswerer:
    """Call the real original answer function with tiny fake image/model inputs."""
    def __init__(self, questioner):
        """Link transports so simultaneous GPU-active phases can be detected."""
        self.device = torch.device('cpu')
        self.processor, self.model, self.questioner = AnswerProcessor(), AnswerModel(), questioner
    def answer_original(self, original, image_path, query):
        """Run the unchanged prompt/greedy/decode function after Qwen has been parked."""
        assert not self.questioner.active
        return original(img_path=str(image_path), query=query, model_caps=self.model, processor_caps=self.processor)
    def park(self):
        """Provide the same cleanup interface as the real answerer."""
        pass


class RoundRobinKMeans:
    """Deterministic ten-cluster double for core tests without optional dependencies."""
    def fit_predict(self, vectors):
        """Consume randomness as real KMeans does, then return ten observed clusters."""
        offset = int(torch.randint(10, (1,)))
        return (torch.arange(len(vectors)) + offset) % 10


@pytest.fixture
def case(tmp_path):
    """Build one fully local target, corpus and original-argument configuration."""
    retriever = TinyRetriever()
    image = tmp_path / 'target.jpg'
    Image.new('RGB', (4, 4), 'blue').save(image)
    return {'retriever': retriever, 'image_path': image,
            'captions': {name: [f'a public image number {i}'] for i, name in enumerate(retriever.cache.paths)},
            'settings': LiveSettings(q_n=2, threshold_low=100, recall_hitting=2)}


def wrapped(case, tmp_path, *, questioner=None, n_states=11, kmeans=None, target_id=0):
    """Run the actual wrapper with controlled transports and a selected target."""
    qwen = ScriptedQuestioner() if questioner is None else questioner
    answerer = ScriptedAnswerer(qwen)
    result = generate_live_dialogue(**case, initial_caption='a room', target_id=target_id,
        questioner=qwen, answerer=answerer, n_states=n_states, checkpoint_path=tmp_path / 'case.json',
        run_identity='test', _kmeans=RoundRobinKMeans() if kmeans is None else kmeans)
    return result, qwen, answerer


def original_oracle(case, *, known=False, kmeans=None):
    """Independently compile the unmodified original ten-round loop as a parity oracle.
    This deliberately does not use our loader or loop adapter. Only replace the
    image resolver and model transports; all source statements remain intact.
    """
    source, path = verified_source('generate_dialog.py')
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'idx')
    retriever, settings = case['retriever'], case['settings']
    qwen = ScriptedQuestioner(known)
    model, processor = AnswerModel(), AnswerProcessor()
    namespace = {'torch': torch, 'Optional': Optional, 'normalize': normalize,
        'BlipForImageTextRetrieval': TinyBlip, 'Image': SimpleNamespace(open=lambda _: Image.open(case['image_path'])),
        'SEED': settings.seed, 'q_n': settings.q_n, 'threshold_low': settings.threshold_low,
        'recall_hitting': settings.recall_hitting, 'referring': settings.referring,
        'reconstruct': settings.reconstruct, 'filtering': settings.filtering, 'select': settings.select,
        'openai': SimpleNamespace(ChatCompletion=qwen), 'device': 'cpu',
        'img_embs': retriever.images, 'search_space': retriever.cache.paths,
        'captions': [{'caption': case['captions'][name]} for name in retriever.cache.paths],
        'visdial': [{'img': retriever.cache.paths[0], 'dialog': ['a room']}],
        's_idx': 0, 'e_idx': 1, 'tqdm': lambda items: items,
        'dial_new': [], 'dial_new_recon': [], 'blip2': model, 'processor2': processor,
        'processor': retriever.processor, 'kmeans': RoundRobinKMeans() if kmeans is None else kmeans}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['blip'] = namespace['BlipForRetrieval']()
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(settings.seed)
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, qwen, processor


@pytest.mark.parametrize('known', [False, True])
def test_entire_original_ten_round_loop_matches_wrapper(case, tmp_path, known):
    """Compare every request, chosen question, answer, reformulation and retrieval rank."""
    oracle, original_qwen, processor = original_oracle(case, known=known)
    result, qwen, answerer = wrapped(case, tmp_path, questioner=ScriptedQuestioner(known))
    assert qwen.records == original_qwen.records
    assert result['dialog'] == oracle['dial_new'][0]['dialog']
    assert result['reformulated'] == oracle['dial_new_recon'][0]['dialog']
    assert answerer.processor.prompts == processor.prompts
    scores = case['retriever'].scores(result['reformulated'])
    ranks = scores.argsort(descending=True).eq(0).nonzero()[:, 1] + 1
    assert ranks.tolist() == [s['target_rank'] for s in result['states']]
    assert any(r['stage'] == 'paraphrase' for r in qwen.records)
    if known:
        assert any(r['stage'] == 'generate_questions' for r in qwen.records)
    assert result['source_audit']['loop_otherwise_unchanged']


def test_actual_fast_kmeans_matches_original(case, tmp_path):
    """Run both paths with the real optional clustering library and identical RNG."""
    KMeans = pytest.importorskip('fast_pytorch_kmeans').KMeans
    oracle, oq, _ = original_oracle(case, kmeans=KMeans(n_clusters=10, mode='cosine', verbose=0))
    result, qwen, _ = wrapped(case, tmp_path, kmeans=KMeans(n_clusters=10, mode='cosine', verbose=0))
    assert qwen.records == oq.records
    assert result['reformulated'] == oracle['dial_new_recon'][0]['dialog']


@pytest.mark.parametrize('settings', [
    LiveSettings(q_n=5, threshold_low=100, recall_hitting=2),
    LiveSettings(q_n=1, threshold_low=100, recall_hitting=2, referring=False),
    LiveSettings(q_n=2, threshold_low=100, recall_hitting=2, filtering=False),
    LiveSettings(q_n=2, threshold_low=100, recall_hitting=2, reconstruct=False, select=False),
])
def test_original_configuration_branches_match(case, tmp_path, settings):
    """Keep q_n=5 and original feature switches faithful, including their distinct paths."""
    case['settings'] = settings
    oracle, oq, _ = original_oracle(case)
    result, qwen, _ = wrapped(case, tmp_path)
    assert qwen.records == oq.records
    assert result['dialog'] == oracle['dial_new'][0]['dialog']
    assert result['reformulated'] == oracle['dial_new_recon'][0]['dialog']


def test_completed_prefix_and_interruption_resume(case, tmp_path):
    """Extend a saved prefix and recover an interrupted round without changing later results."""
    full, _, _ = wrapped(case, tmp_path / 'full')
    short, _, _ = wrapped(case, tmp_path / 'short', n_states=3)
    resumed, _, _ = wrapped(case, tmp_path / 'short')
    assert short['states'] == full['states'][:3]
    assert resumed['states'] == full['states']
    with pytest.raises(UpstreamExecutionStop, match='deliberate'):
        wrapped(case, tmp_path / 'interrupted', questioner=ScriptedQuestioner(fail_at=(1, 1)))
    state = json.loads((tmp_path / 'interrupted/case.json').read_text())
    assert len(state['states']) == 2 and state['status'] == 'interrupted'
    completed, _, _ = wrapped(case, tmp_path / 'interrupted')
    assert completed['states'] == full['states']


def test_target_rank_does_not_influence_questions(case, tmp_path):
    """Change the measured target index and prove the generated conversation is identical."""
    first, q1, _ = wrapped(case, tmp_path / 'a', target_id=0, n_states=3)
    second, q2, _ = wrapped(case, tmp_path / 'b', target_id=90, n_states=3)
    assert q1.records == q2.records and first['dialog'] == second['dialog']
    assert [r['target_rank'] for r in first['states']] != [r['target_rank'] for r in second['states']]


def test_live_queries_enter_common_comparison_without_human_answers(case, tmp_path):
    """Evaluate generated states beside saved references without leaking future human text."""
    live, qwen, _ = wrapped(case, tmp_path, n_states=3)
    target = case['retriever'].cache.paths[0]
    pairs = [({'img': target, 'dialog': ['a room', 'HUMAN_SECRET', 'HUMAN_FUTURE']},
              {'img': target, 'dialog': ['a room', 'saved one', 'saved two']})]
    result = evaluate_experiments(case['retriever'], pairs, {'live': generated_query_method([live])}, rounds=3, ks=[1, 10])
    assert set(result['summary']) == {'caption_only', 'raw_dialogue', 'supplied_plugir', 'live'}
    assert 'HUMAN_SECRET' not in json.dumps(qwen.records)


def test_source_checksum_and_catalogue_guard(case, tmp_path):
    """Reject edited original source and preserve all catalogue captions by exact filename."""
    from agentir.plugir_upstream import VENDOR
    (tmp_path / 'generate_dialog.py').write_bytes((VENDOR / 'generate_dialog.py').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        verified_source('generate_dialog.py', folder=tmp_path)
    path = tmp_path / 'captions.json'
    rows = [{'id': name, 'caption': captions} for name, captions in reversed(list(case['captions'].items()))]
    path.write_text(json.dumps(rows))
    assert load_candidate_captions(path, case['retriever'].cache.paths) == case['captions']


def test_metrics_match_independent_original_source():
    """Check indexing and real Recall/Hits/BRI on misses, recoveries and later regressions."""
    source, path = verified_source('eval.py')
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('get_first_hitting_time', 'cumulative_hits_per_round')]
    ranks = torch.tensor([[1, 11, 50000], [20, 10, 100], [50, 15, 1]])
    ns = {'torch': torch, 'args': SimpleNamespace(num_rounds=3)}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), ns)
    rows, bri = metrics_from_ranks(ranks, [1, 10, 100])
    for k in (1, 10, 100):
        hits, recall = ns['cumulative_hits_per_round']((ranks - 1).flatten(), None, None, hitting_recall=k)
        selected = [r for r in rows if r['k'] == k]
        assert [r['recall_percent'] for r in selected] == recall.tolist()
        assert [r['hits_percent'] for r in selected] == hits.tolist()
    best = ranks.cummin(0).values.float().log()
    assert bri == pytest.approx(float(((best[:-1] + best[1:]) / 2).mean()), abs=1e-6)
    assert metrics_from_ranks(ranks[:1], [10])[1] is None


class CausalTokenizer:
    """Expose only the HF tokenization calls used by the Qwen transport."""
    eos_token_id = 99
    def apply_chat_template(self, messages, **kwargs):
        """Remember exactly the messages supplied by upstream, without additions."""
        self.messages = deepcopy(messages)
        return 'prompt'
    def __call__(self, prompt, **kwargs):
        """Produce a three-token prompt for cheap transport tests."""
        return BatchFeature({'input_ids': torch.tensor([[1, 2, 3]])})
    def decode(self, tokens, **kwargs):
        """Return plain completion text, intentionally without JSON."""
        return 'What color is it?'


class CausalModel:
    """Record actual generation options and consume RNG inside the adapter."""
    def __init__(self):
        """Set EOS metadata and a log of all generation calls."""
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.calls = []
    def to(self, device):
        """Simulate model transfer without a physical GPU."""
        return self
    def generate(self, input_ids, **kwargs):
        """Consume randomness and return an output that hits its configured token cap."""
        self.calls.append(kwargs)
        torch.rand(4)
        return torch.cat([input_ids, torch.full((1, kwargs['max_new_tokens']), 4)], dim=1)


def test_qwen_preserves_original_options_slots_and_rng(tmp_path, monkeypatch):
    """Verify transparent transport, bounded raw output, replay and independent sample slots."""
    qwen = Qwen3.__new__(Qwen3)
    qwen.device, qwen.dtype, qwen.folder = torch.device('cpu'), torch.float32, tmp_path
    qwen.max_input_tokens, qwen.seed = 4096, 1021
    qwen.model, qwen.tokenizer, qwen.records = CausalModel(), CausalTokenizer(), []
    qwen.begin_round(0, 'case')
    request = dict(messages=[{'role': 'user', 'content': 'original prompt'}], model='gpt-3.5-turbo',
                   max_tokens=32, temperature=.5, n=1, request_timeout=10)
    rng = torch.get_rng_state().clone()
    first = qwen._create('generate_questions_referring', request)
    assert torch.equal(rng, torch.get_rng_state())
    assert qwen.tokenizer.messages == request['messages']
    assert qwen.model.calls[-1]['max_new_tokens'] == 32 and qwen.model.calls[-1]['temperature'] == .5
    assert qwen.model.calls[-1]['top_k'] == 0 and qwen.model.calls[-1]['top_p'] == 1.
    assert first['choices'][0]['finish_reason'] == 'length'
    qwen._create('generate_questions_referring', request)
    assert len(list(tmp_path.glob('*.json'))) == 2
    qwen.begin_round(0, 'case')
    assert qwen._create('generate_questions_referring', request) == first and len(qwen.model.calls) == 2
    for tokens, temperature in ((10, 0.), (512, .7)):
        qwen._create('test', {**request, 'max_tokens': tokens, 'temperature': temperature, 'top_p': .8})
        assert qwen.model.calls[-1]['max_new_tokens'] == tokens


def test_blip2_calls_original_function_with_checkpoint_defaults(case, tmp_path, monkeypatch):
    """Check exact prompt/greedy options, integer IDs, FP16 pixels and cached answer replay."""
    from agentir.plugir_upstream import load_generator
    namespace = {'torch': torch, 'Optional': Optional, 'BlipForImageTextRetrieval': TinyBlip,
                 'Image': Image, 'device': 'cpu', 'normalize': normalize}
    original, _, _ = load_generator(namespace)
    answerer = Blip2Answerer.__new__(Blip2Answerer)
    answerer.device, answerer.folder = torch.device('cpu'), tmp_path / 'answers'
    answerer.processor, answerer.model, answerer.records = AnswerProcessor(), AnswerModel(), []
    monkeypatch.setattr(answerer, '_activate', lambda: None)
    assert answerer.answer_original(original, case['image_path'], 'a room. What color?') == 'blue'
    assert answerer.processor.prompts == ['Question: a room. What color? Answer:']
    assert answerer.answer_original(original, case['image_path'], 'a room. What color?') == 'blue'
    assert answerer.model.calls == 1 and answerer.records[-1]['reused']


def test_invalid_original_settings_fail_before_models():
    """Explain unsupported original branches and windows instead of changing their algorithm."""
    with pytest.raises(ValueError, match='non-referring'):
        LiveSettings(referring=False).validate(50000)
    with pytest.raises(ValueError, match='Final window'):
        LiveSettings(threshold_low=10).validate(50000)


@pytest.mark.parametrize('mode', ['saved', 'live', 'prepared'])
def test_shared_runner_includes_enabled_methods(case, tmp_path, monkeypatch, mode):
    """Exercise click-Run orchestration, JSON/CSV exports and future method preparation."""
    import agentir.live_models as live_models
    from agentir.evaluation import PreparedExperiment
    runner_path = Path(__file__).resolve().parents[1] / 'run_evaluation.py'
    spec = importlib.util.spec_from_file_location('test_shared_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = tmp_path.parent / ('r_' + mode)
    root.mkdir()
    config = {key: key + '.json' for key in ('cache', 'manifest', 'baseline_queries', 'supplied_queries')}
    (root / 'experiment.json').write_text(json.dumps(config))
    for filename in config.values():
        (root / filename).write_text('[]')
    for filename in ('run_evaluation.py', 'src/agentir/live.py', 'src/agentir/live_models.py',
                     'src/agentir/plugir_upstream.py', 'src/agentir/retrieval.py'):
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture source')
    (root / 'captions.json').write_text(json.dumps([
        {'id': name, 'caption': captions} for name, captions in case['captions'].items()]))
    retriever = case['retriever']
    pairs = [({'img': retriever.cache.paths[0], 'dialog': ['a room', 'HUMAN_SECRET', 'HUMAN_FUTURE']},
              {'img': retriever.cache.paths[0], 'dialog': ['a room', 'supplied one', 'supplied two']})]
    qwen = ScriptedQuestioner()
    answerer = ScriptedAnswerer(qwen)
    original_find = importlib.util.find_spec
    original_version = runner.importlib.metadata.version
    extras = ('accelerate', 'sentencepiece', 'fast_pytorch_kmeans', 'psutil')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: SimpleNamespace() if name in extras else original_find(name))
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: 'fixture' if name in (*extras, 'fast-pytorch-kmeans') else original_version(name))
    monkeypatch.setitem(sys.modules, 'fast_pytorch_kmeans', SimpleNamespace(KMeans=lambda **kwargs: RoundRobinKMeans()))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'fixture-GPU')
    monkeypatch.setattr(live_models, 'require_cuda', lambda device: device)
    monkeypatch.setattr('agentir.live.resolve_target_image', lambda *args, **kwargs: case['image_path'])
    monkeypatch.setattr(runner, 'load_cache', lambda *args: retriever.cache)
    monkeypatch.setattr(runner, 'matched_queries', lambda *args: pairs)
    monkeypatch.setattr(runner, 'BlipRetriever', lambda *args, **kwargs: retriever)
    def no_models(*args, **kwargs):
        """Prove disabled live generation cannot load Qwen or BLIP-2."""
        raise AssertionError('Unexpected model initialization')
    monkeypatch.setattr(live_models, 'Qwen3', (lambda *args, **kwargs: qwen) if mode == 'live' else no_models)
    monkeypatch.setattr(live_models, 'Blip2Answerer', (lambda *args, **kwargs: answerer) if mode == 'live' else no_models)
    for name, value in {'__file__': str(root / 'run_evaluation.py'), 'N_TARGETS': 1, 'N_STATES': 3,
                        'K_VALUES': [1, 10], 'CANDIDATE_CAPTIONS': 'captions.json',
                        'LIVE': case['settings'], 'RUN_LIVE_PLUGIR': mode == 'live'}.items():
        monkeypatch.setattr(runner, name, value)
    def identity(context, model):
        """Provide a simple ordinary experiment that must match saved raw dialogue."""
        return context.current_query
    def prepare_custom(root, model, selected, *, n_states, offline):
        """Check the future setup contract and attach a small generated artifact."""
        assert selected == pairs and n_states == 3
        return PreparedExperiment({'prepared_identity': identity}, artifacts={'setup.json': {'ok': True}})
    monkeypatch.setattr(runner, 'EXPERIMENTS', {'identity': identity})
    monkeypatch.setattr(runner, 'PREPARED_EXPERIMENTS', {'future': prepare_custom} if mode == 'prepared' else {})
    runner.main()
    folder = next((root / 'runs/experiments').iterdir())
    report = json.loads((folder / 'report.json').read_text())
    expected = {'caption_only', 'raw_dialogue', 'supplied_plugir', 'identity'}
    if mode == 'live':
        expected.update({'live_plugir_qwen', 'live_raw_same_dialogue'})
        assert (folder / 'prepared/qwen_live/generated_dialogues.json').is_file()
        assert 'HUMAN_SECRET' not in json.dumps(qwen.records)
    if mode == 'prepared':
        expected.add('prepared_identity')
        assert (folder / 'prepared/future/setup.json').is_file()
    assert set(report['summary']) == expected
    assert all((folder / name).is_file() for name in
               ('per_round.csv', 'rank_comparison.csv', 'per_example.csv', 'final_metrics.csv', 'runner_snapshot.py'))
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires NVIDIA CUDA for model architecture smoke test"))])
def test_installed_transformers_generates_with_tiny_qwen3_and_blip2(device):
    """Check real generation APIs using tiny random architectures, never pretrained downloads.
    This verifies package/device wiring only, not the quality of either full model.
    """
    from transformers import (
        Blip2Config, Blip2ForConditionalGeneration, Blip2QFormerConfig, Blip2VisionConfig,
        Qwen3Config, Qwen3ForCausalLM, T5Config,
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    qwen = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
        eos_token_id=2, pad_token_id=0)).to(device=device, dtype=dtype).eval()
    vision = Blip2VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=8, patch_size=4)
    qformer = Blip2QFormerConfig(hidden_size=16, encoder_hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2)
    text = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                    num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                    pad_token_id=0, eos_token_id=2)
    config = Blip2Config.from_vision_qformer_text_configs(vision, qformer, text,
        num_query_tokens=2, image_token_index=3)
    blip2 = Blip2ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        qwen_out = qwen.generate(input_ids=torch.tensor([[4, 5, 6]], device=device),
            attention_mask=torch.ones(1, 3, device=device, dtype=torch.long), max_new_tokens=2, do_sample=False)
        answer_out = blip2.generate(pixel_values=torch.zeros(1, 3, 8, 8, device=device, dtype=dtype),
            input_ids=torch.tensor([[3, 3, 4, 5]], device=device), max_new_tokens=2, do_sample=False)
    assert qwen_out.shape[0] == answer_out.shape[0] == 1
    assert qwen_out.shape[1] > 3 and answer_out.shape[1] > 1


