"""Local model transports for original PlugIR calls; no prompts or algorithms live here."""
import gc
import inspect
from pathlib import Path
import time

import torch

from .cache import digest, read_json
from .live import save_json, signature
from .plugir_upstream import DeviceProcessor, GENERATOR_SHA256, UpstreamExecutionStop

QWEN_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BLIP2_ID = "Salesforce/blip2-flan-t5-xl"
BLIP2_REVISION = "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"


def require_cuda(device):
    """Stop before downloads if the selected NVIDIA device is unavailable."""
    device = torch.device(device)
    if device.type == "mps" and torch.backends.mps.is_available():  # Apple MPS support
        return device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Live Qwen/BLIP-2 needs a working CUDA PyTorch environment on the GPU machine.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("Selected CUDA device does not exist.")
    return device


class Qwen3:
    """OpenAI-shaped local transport preserving upstream messages, options and single calls.
    No JSON instructions, batched questions, content repairs or increased token caps.
    """

    def __init__(self, folder, *, device="cuda", offline=False, max_input_tokens=4096, seed=1021):
        """Remember pinned model/cache settings; weights load only for an uncached call."""
        self.device = require_cuda(device)
        self.folder, self.offline = Path(folder), offline
        self.max_input_tokens, self.seed = max_input_tokens, seed
        if self.device.type == "mps":  # Apple MPS support
            self.dtype = torch.bfloat16
        else:
            with torch.cuda.device(self.device):
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = self.model = None
        self.case_id, self.round_index, self.ordinal = "", 0, 0
        self.records = []

    def begin_round(self, round_index, case_id):
        """Reset reproducible call slots; repeated identical prompts remain separate samples.
        Replaying a partial round reuses the matching slot's saved completion.
        """
        self.round_index, self.case_id, self.ordinal = round_index, case_id, 0

    def _activate(self):
        """Load pinned full Qwen once, then move its parked weights to the GPU."""
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading {QWEN_ID}; first use downloads several GB.", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                                          local_files_only=self.offline)
            self.model = AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                torch_dtype=self.dtype, low_cpu_mem_usage=True, local_files_only=self.offline).eval()
        self.model.to(self.device)

    def park(self):
        """Release Qwen VRAM before image answering, retaining weights in host RAM."""
        if self.model is not None:
            self.model.to("cpu")
            gc.collect()
            torch.mps.empty_cache() if self.device.type == "mps" else torch.cuda.empty_cache()  # Apple MPS support

    def create(self, **kwargs):
        """Replace original ChatCompletion.create with local inference, preserving request text.
        Failures escape the original unbounded network retry handlers through a
        dedicated stop signal; the outer wrapper saves progress and cleans up.
        """
        stage = inspect.currentframe().f_back.f_code.co_name
        try:
            return self._create(stage, kwargs)
        except Exception as error:
            raise UpstreamExecutionStop(f"Local Qwen call failed in {stage}: {error}") from error

    def _create(self, stage, kwargs):
        """Generate/cache one original request and return its OpenAI-shaped response.
        Honour original temperature/max_tokens/top_p/seed. Isolate sampling RNG
        so a Qwen call cannot alter the KMeans random initialization sequence.
        """
        from transformers import __version__ as transformers_version
        if kwargs.get("n", 1) != 1 or not 1 <= kwargs["max_tokens"] <= 512:
            raise ValueError("Unsupported upstream completion count or token cap.")
        slot = {"case": self.case_id, "round": self.round_index, "ordinal": self.ordinal}
        self.ordinal += 1
        effective_seed = kwargs.get("seed", (self.seed + int(signature(slot)[:8], 16)) % (2**31))
        request = {"upstream_function": stage, "original_request": kwargs, "slot": slot,
                   "effective_seed": effective_seed, "model": QWEN_ID, "revision": QWEN_REVISION,
                   "dtype": str(self.dtype), "torch": torch.__version__, "transformers": transformers_version,
                   "max_input_tokens": self.max_input_tokens}
        path = self.folder / (signature(request) + ".json")
        if path.is_file():
            record = read_json(path)
            if record["request"] != request:
                raise ValueError("Cached request identity mismatch.")
            reused = True
        else:
            self._activate()
            prompt = self.tokenizer.apply_chat_template(kwargs["messages"], tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            length = inputs["input_ids"].shape[1]
            if length > self.max_input_tokens:
                raise ValueError("Original prompt exceeds MAX_INPUT_TOKENS; no silent truncation or rewrite.")
            temperature = kwargs.get("temperature", 1.)
            options = {"do_sample": temperature > 0, "max_new_tokens": kwargs["max_tokens"],
                       "num_beams": 1, "pad_token_id": self.tokenizer.eos_token_id, "repetition_penalty": 1.}
            if temperature > 0:
                options.update(temperature=temperature, top_p=kwargs.get("top_p", 1.), top_k=0)
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            started = time.perf_counter()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(effective_seed)
                output = self.model.generate(**inputs, **options)[0, length:]
            content = self.tokenizer.decode(output, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            truncated = len(output) >= kwargs["max_tokens"] and int(output[-1]) not in eos
            response = {"choices": [{"message": {"content": content},
                          "finish_reason": "length" if truncated else "stop"}], "model": QWEN_ID}
            record = {"request": request, "response": response, "input_tokens": length,
                      "output_tokens": len(output), "seconds": time.perf_counter() - started}
            save_json(path, record)
            del inputs, output
            reused = False
        # Upstream parses length-limited text itself; do not repair or reject it.
        self.records.append({**record, "reused": reused})
        return record["response"]


class Blip2Answerer:
    """Load the paper's answerer checkpoint and call its actual generate_answer function."""

    def __init__(self, folder, *, device="cuda", offline=False):
        """Remember settings; preserve the pinned checkpoint's generation defaults."""
        self.device = require_cuda(device)
        self.folder, self.offline = Path(folder), offline
        self.processor = self.model = None
        self.records = []

    def _activate(self):
        """Load pinned FP16 BLIP-2/processor once, checking image-token compatibility."""
        if self.model is None:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            print(f"Loading {BLIP2_ID}; first use downloads several GB.", flush=True)
            self.processor = Blip2Processor.from_pretrained(BLIP2_ID, revision=BLIP2_REVISION,
                                                           local_files_only=self.offline)
            self.model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_ID, revision=BLIP2_REVISION,
                torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=self.offline).eval()
            image_index = getattr(self.model.config, "image_token_index", None)
            if image_index is not None:
                self.processor.num_query_tokens = self.model.config.num_query_tokens
                token = self.processor.image_token
                token = token.content if hasattr(token, "content") else str(token)
                if self.processor.tokenizer.convert_tokens_to_ids(token) != image_index:
                    raise RuntimeError("Pinned BLIP-2 processor/model image-token mismatch.")
        self.model.to(self.device)

    def park(self):
        """Free BLIP-2 VRAM before Qwen reconstructs the next description."""
        if self.model is not None:
            self.model.to("cpu")
            gc.collect()
            torch.mps.empty_cache() if self.device.type == "mps" else torch.cuda.empty_cache()  # Apple MPS support

    def answer_original(self, original_function, image_path, query):
        """Call upstream generate_answer with FP16 pixels or replay its saved answer.
        Preserve the prompt, decoding and greedy checkpoint defaults; our former
        extra 32-token output cap and answer-content repairs are removed.
        """
        from transformers import __version__ as transformers_version
        request = {"image_sha256": digest(image_path), "query": query, "source_sha256": GENERATOR_SHA256,
                   "model": BLIP2_ID, "revision": BLIP2_REVISION, "torch": torch.__version__,
                   "transformers": transformers_version, "generation_policy": "upstream greedy/checkpoint defaults"}
        path = self.folder / (signature(request) + ".json")
        if path.is_file():
            record = read_json(path)
            if record["request"] != request:
                raise ValueError("Cached answer identity mismatch.")
            reused = True
        else:
            self._activate()
            processor = DeviceProcessor(self.processor, self.device, dtype=torch.float16)
            with torch.inference_mode():
                answer = original_function(img_path=str(image_path), query=query,
                                           model_caps=self.model, processor_caps=processor)
            record = {"request": request, "answer": answer,
                      "generation_config": self.model.generation_config.to_dict()}
            save_json(path, record)
            reused = False
        self.records.append({**record, "reused": reused})
        return record["answer"]


VLM_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_MAX_PIXELS = 256 * 28 * 28  # About 256 image tokens per photo; enough for short answers.


class VisionAnswerer:
    """The agent's own vision model (deliberately not BLIP-2, which plays the user).
    Answers one question about a batch of candidate photos in a few words, greedy.
    Answers are cached per (photo, question), so interrupted runs resume without recomputation.
    """

    def __init__(self, folder, *, device="cuda", offline=False):
        self.device = require_cuda(device)
        self.folder, self.offline = Path(folder), offline
        self.model = self.processor = None
        self.revision = None
        self.records = []

    def _activate(self):
        if self.model is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            print(f"Loading {VLM_ID}; first use downloads several GB.", flush=True)
            local = snapshot_download(VLM_ID, local_files_only=self.offline)
            self.revision = Path(local).name
            self.processor = AutoProcessor.from_pretrained(local, max_pixels=VLM_MAX_PIXELS)
            self.processor.tokenizer.padding_side = "left"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        self.model.to(self.device)

    def park(self):
        if self.model is not None:
            self.model.to("cpu")
            gc.collect()
            torch.mps.empty_cache() if self.device.type == "mps" else torch.cuda.empty_cache()  # Apple MPS support

    def answer(self, image_paths, question):
        """Short answers (lowercase) for each photo, in order."""
        from PIL import Image
        keys = [signature({"image": Path(p).name, "question": question, "model": VLM_ID}) for p in image_paths]
        answers, missing = {}, []
        for key, path in zip(keys, image_paths):
            cached = self.folder / (key + ".json")
            if cached.is_file():
                answers[key] = read_json(cached)["answer"]
            else:
                missing.append((key, path))
        if missing:
            self._activate()
            prompt = f"{question} Answer in one to three words."
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images = [Image.open(path).convert("RGB") for _, path in missing]
            inputs = self.processor(text=[text] * len(images), images=images, padding=True,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            decoded = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for (key, path), reply in zip(missing, decoded):
                answers[key] = reply.strip().lower()
                save_json(self.folder / (key + ".json"), {"image": Path(path).name, "question": question,
                                                          "model": VLM_ID, "revision": self.revision,
                                                          "answer": answers[key]})
        result = [answers[key] for key in keys]
        self.records.append({"question": question, "images": [Path(p).name for p in image_paths], "answers": result})
        return result
