"""Exercise the two inference contexts used by the shared retrieval runner."""

import sys
from types import SimpleNamespace

import pytest
import torch

from agentir.live_models import Qwen3, QwenVLAnswerer


@pytest.mark.parametrize("plugir_first", [True, False])
@pytest.mark.parametrize("transport_type", [Qwen3, QwenVLAnswerer])
def test_qwen_load_and_transfers_keep_version_counters(monkeypatch, transport_type, plugir_first):
    class CopyingProjection(torch.nn.Linear):
        def to(self, *args, **kwargs):
            # A real device change allocates new tensors. Force a copy on CPU too.
            self.weight = torch.nn.Parameter(self.weight.detach().clone())
            return super().to(*args, **kwargs)

    def load_projection(*args, **kwargs):
        assert not torch.is_grad_enabled()
        return CopyingProjection(4, 4)

    mock_loader = SimpleNamespace(from_pretrained=load_projection)
    mock_processor = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=mock_loader, Qwen2_5_VLForConditionalGeneration=mock_loader,
        AutoTokenizer=mock_processor, AutoProcessor=mock_processor))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda *args, **kwargs: "local-model"))
    monkeypatch.setattr("agentir.live_models.require_cuda", lambda device: torch.device("cpu"))
    if transport_type is Qwen3:
        # Bypass the constructor's CUDA dtype selection for the CPU regression.
        transport = Qwen3.__new__(Qwen3)
        transport.device, transport.offline, transport.dtype = torch.device("cpu"), True, torch.float32
        transport.model = transport.tokenizer = None
    else:
        transport = QwenVLAnswerer("unused", offline=True)

    for outer_mode in (plugir_first, not plugir_first):
        with torch.inference_mode(outer_mode):
            transport._activate()
            assert not transport.model.weight.is_inference()
            assert transport.model.weight._version >= 0
            transport.model(torch.ones(1, 4))
            transport.park()
            assert transport.model.weight._version >= 0
            assert torch.is_inference_mode_enabled() == outer_mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires NVIDIA CUDA for weight transfer")
@pytest.mark.parametrize("plugir_first", [True, False])
def test_shared_qwen_generates_after_real_cuda_transfers(monkeypatch, tmp_path, plugir_first):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=64, eos_token_id=2, pad_token_id=0)

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, prompt, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[4, 5, 6]]),
                                  "attention_mask": torch.ones(1, 3, dtype=torch.long)})

        def decode(self, output, **kwargs):
            return " ".join(map(str, output.tolist()))

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: Qwen3ForCausalLM(config))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    transport = Qwen3(tmp_path, offline=True)
    for outer_mode in (plugir_first, not plugir_first):
        method = "plugir" if outer_mode else "debate"
        transport.begin_round(0, method)
        with torch.inference_mode(outer_mode):
            transport.create(messages=[{"role": "user", "content": method}], max_tokens=2, temperature=0)
            projection = transport.model.model.layers[0].self_attn.q_proj
            assert projection.weight.device.type == "cuda" and projection.weight._version >= 0
            transport.park()
            assert projection.weight.device.type == "cpu" and projection.weight._version >= 0
    assert len(transport.records) == 2
    assert all(not record["reused"] for record in transport.records)
