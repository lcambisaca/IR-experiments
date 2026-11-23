"""Read-only protocol checks and model loading; no dialogue generation."""
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import torch
from PIL import Image
import run_evaluation as runner
from agentir.cache import load_cache, read_json, digest
from agentir.evaluation import matched_queries
from agentir.live import load_candidate_captions, resolve_target_image, save_json
from agentir.live_models import Qwen3, Blip2Answerer, QWEN_ID, QWEN_REVISION, BLIP2_ID, BLIP2_REVISION
from agentir.multagent_experiment import setup
from agentir.retrieval import BlipRetriever, ranks_for_targets

root = Path(__file__).resolve().parent
evidence = Path.home() / 'pilot-evidence-20261120'
evidence.mkdir(exist_ok=True)
expected = dict(RUN_LIVE_PLUGIR=True, RUN_MULTI_AGENT=True, USE_PILOT_TARGETS=True,
                N_STATES=11, INCLUDE_SAVED_REFERENCES=False, DEVICE='cuda', LIVE_DEVICE='cuda')
assert all(getattr(runner,k)==v for k,v in expected.items())
assert setup.AGENT_MODES == (False, True)
assert runner.SEARCH == 'rewrite' and runner.SELECTION == 'plugir_kl'
config = read_json(root/'experiment.json')
cache = load_cache(root/config['cache'], root/config['manifest'])
pairs = runner.select_target_pairs(matched_queries(read_json(root/config['baseline_queries']),
    read_json(root/config['supplied_queries']),cache.paths))
captions = load_candidate_captions(root/runner.CANDIDATE_CAPTIONS,cache.paths)
targets=[]
for raw,_ in pairs:
    path=resolve_target_image(root/runner.IMAGE_ROOT,raw['img'],download=True)
    with Image.open(path) as image:
        size=image.size
    targets.append(dict(target=raw['img'],initial_caption=raw['dialog'][0],
                        candidate_id=cache.paths.index(raw['img']),jpeg_sha256=digest(path),size=size))
torch.manual_seed(runner.SEED)
retriever=BlipRetriever(cache,device='cuda',batch_size=runner.TEXT_BATCH_SIZE,threads=runner.CPU_THREADS,local_files_only=True)
initial_ranks=ranks_for_targets(retriever.scores([t['initial_caption'] for t in targets]),[t['candidate_id'] for t in targets]).tolist()
for t,r in zip(targets,initial_ranks): t['initial_rank']=r
qwen=Qwen3(evidence/'preflight_unused_qwen_calls',offline=True,max_input_tokens=8192,max_output_tokens=1024)
answerer=Blip2Answerer(evidence/'preflight_unused_blip2_calls',offline=True)
qwen._activate(); qwen.park()
answerer._activate(); answerer.park()
save_json(evidence/'preflight.json',dict(status='passed_no_generation',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    settings=expected,runner_settings={k:getattr(runner,k) for k in ['SEED','MAX_INPUT_TOKENS','SEARCH','SELECTION','OFFLINE','K_VALUES','TEXT_BATCH_SIZE','CPU_THREADS']},
    live=dataclasses.asdict(runner.LIVE),agents=dataclasses.asdict(setup.AGENTS),agent_modes=setup.AGENT_MODES,
    agent_input_tokens=setup.AGENT_INPUT_TOKENS,cache=cache.report,candidate_captions_sha256=digest(root/runner.CANDIDATE_CAPTIONS),
    candidate_captions_count=len(captions),targets=targets,python=platform.python_version(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    models=dict(qwen=[QWEN_ID,QWEN_REVISION],answerer=[BLIP2_ID,BLIP2_REVISION],retriever=[cache.report['model'],cache.revision]),
    packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','tokenizers','huggingface-hub','pillow','numpy','accelerate','sentencepiece','fast-pytorch-kmeans','psutil']},
    existing_checkpoints=[str(p.relative_to(root)) for p in (root/'runs').rglob('*.json')],
    source_sha256={str(p.relative_to(root)):digest(p) for p in [root/'run_evaluation.py',*sorted((root/'src').rglob('*.py'))]},
    peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved()))
print(json.dumps(read_json(evidence/'preflight.json'),indent=2),flush=True)
print('PREFLIGHT_PASSED_NO_GENERATION',flush=True)
