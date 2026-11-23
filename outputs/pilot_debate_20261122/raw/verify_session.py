import ast,dataclasses,gc,hashlib,importlib.metadata,json,pathlib,platform,sys,time
root=pathlib.Path.home()/'pilot_debate_20261122/IR-experiments'
sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'src'))
import torch
import run_evaluation as r
from agentir.cache import load_cache,read_json,digest,MODEL_ID
from agentir.evaluation import matched_queries
from agentir.live import load_candidate_captions,resolve_target_image
from agentir.live_models import Qwen3,Blip2Answerer,QWEN_ID,QWEN_REVISION,BLIP2_ID,BLIP2_REVISION
from agentir.multagent_experiment.setup import AGENT_MODES
from agentir.retrieval import BlipRetriever,ranks_for_targets
from huggingface_hub import snapshot_download,model_info
assert torch.cuda.is_available()
expected=dict(RUN_LIVE_PLUGIR=True,RUN_MULTI_AGENT=True,RUN_MULTI_AGENT_DEBATE=True,USE_PILOT_TARGETS=True,N_STATES=11,INCLUDE_SAVED_REFERENCES=False,DEVICE='cuda',LIVE_DEVICE='cuda',SEARCH='rewrite',SELECTION='plugir_kl',MAX_INPUT_TOKENS=8192)
assert all(getattr(r,k)==v for k,v in expected.items()); assert AGENT_MODES==(True,)
m=read_json(root/'TRANSFER_MANIFEST.json')
assert all(digest(root/n)==h for n,h in m['files'].items())
cfg=read_json(root/'experiment.json'); cache=load_cache(root/cfg['cache'],root/cfg['manifest'])
pairs=r.select_target_pairs(matched_queries(read_json(root/cfg['baseline_queries']),read_json(root/cfg['supplied_queries']),cache.paths))
catalogue=load_candidate_captions(root/r.CANDIDATE_CAPTIONS,cache.paths)
images=[]
for raw,_ in pairs:
 image=resolve_target_image(root/r.IMAGE_ROOT,raw['img'],download=True)
 from PIL import Image
 with Image.open(image) as im: size=im.size
 images.append(dict(target=raw['img'],caption=raw['dialog'][0],sha256=digest(image),size=size))
print('CACHE_AND_TARGETS_VERIFIED',json.dumps(images),flush=True)
models=[(MODEL_ID,cache.revision),(QWEN_ID,QWEN_REVISION),(BLIP2_ID,BLIP2_REVISION)]
for name,rev in models:
 print('DOWNLOADING_PINNED_MODEL',name,rev,flush=True)
 files=[x.rfilename for x in model_info(name,revision=rev).siblings]
 safes=any(x.endswith('.safetensors') for x in files)
 allow=[x for x in files if x.endswith(('.json','.txt','.model','.tiktoken','.jinja')) or x.endswith('.safetensors') or (x.endswith('.bin') and not safes)]
 snapshot_download(name,revision=rev,allow_patterns=allow,max_workers=4)
print('MODEL_FILES_READY',flush=True)
retriever=BlipRetriever(cache,device='cuda',local_files_only=True)
ranks=ranks_for_targets(retriever.scores([raw['dialog'][0] for raw,_ in pairs]),[cache.paths.index(raw['img']) for raw,_ in pairs]).tolist()
del retriever; gc.collect(); torch.cuda.empty_cache()
fits={}
for cls,name in [(Qwen3,'qwen'),(Blip2Answerer,'blip2')]:
 torch.cuda.reset_peak_memory_stats(); obj=cls(root.parent/'preflight_empty_calls'/name,device='cuda',offline=True); obj._activate(); fits[name]=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());obj.park(); del obj;gc.collect();torch.cuda.empty_cache()
info=dict(status='ready',source=m,settings=expected,agent_modes=AGENT_MODES,images=images,initial_ranks=ranks,cache=cache.report,catalogue_count=len(catalogue),models=models,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,python=platform.python_version(),model_fit=fits,versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','tokenizers','huggingface-hub','pillow','numpy','accelerate','sentencepiece','fast-pytorch-kmeans','psutil')},prior_checkpoints=list(map(str,(root/'runs').glob('**/dialogues/*.json'))) if (root/'runs').exists() else [])
(root.parent/'preflight.json').write_text(json.dumps(info,indent=2));print('PREFLIGHT_READY',json.dumps(fits),flush=True)
