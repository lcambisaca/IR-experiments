"""Qwen2.5-VL simulated user: cached answers, identity and runner flag (CPU, no model weights)."""
from pathlib import Path

from PIL import Image

from agentir.cache import digest
from agentir.live import save_json, signature
from agentir.live_models import QwenVLAnswerer


def test_cached_answer_is_returned_without_loading_the_model(tmp_path):
    image = tmp_path / "target.jpg"
    Image.new("RGB", (8, 8)).save(image)
    answerer = object.__new__(QwenVLAnswerer)  # Skip the CUDA check; nothing may load.
    answerer.folder, answerer.records, answerer.model = tmp_path / "calls", [], None
    answerer._activate = lambda: (_ for _ in ()).throw(AssertionError("model must not load"))
    request = {"image_sha256": digest(image), "query": "What color is the car?",
               "identity": list(QwenVLAnswerer.IDENTITY), "prompt": QwenVLAnswerer.PROMPT}
    save_json(answerer.folder / (signature(request) + ".json"), {"request": request, "answer": "red"})
    assert answerer.answer_original(None, image, "What color is the car?") == "red"
    assert answerer.records[-1]["reused"]


def test_prompt_never_mentions_dialogue_and_runner_exposes_flag():
    assert "{question}" in QwenVLAnswerer.PROMPT and "caption" not in QwenVLAnswerer.PROMPT.lower()
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_plugir_live.py").read_text()
    assert '"--answerer"' in source and "runner.ANSWERER" in source
