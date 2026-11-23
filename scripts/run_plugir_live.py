"""Live PlugIR (unchanged, minimum-KL selection) on a block of matched targets, fully offline.

    PYTHONPATH=src python scripts/run_plugir_live.py --start 150 --targets 100                  # targets 151-250
    PYTHONPATH=src python scripts/run_plugir_live.py --start 150 --targets 100 --search facts   # end to end
    PYTHONPATH=src python scripts/run_plugir_live.py --start 30 --targets 120 --selection probe_vlm  # answer-entropy agent

Same settings as the 119-target PlugIR run; rerunning the same command resumes from checkpoints.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_evaluation as runner  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True, help="first matched target index (0 = target 1)")
parser.add_argument("--targets", type=int, required=True)
parser.add_argument("--search", choices=["rewrite", "facts"], default="rewrite",
                    help="rewrite = original PlugIR; facts = fact fusion inside the live loop")
parser.add_argument("--selection", choices=["plugir_kl", "probe_vlm"], default="plugir_kl",
                    help="plugir_kl = original PlugIR question choice; probe_vlm = answer-entropy agent")
parser.add_argument("--answerer", choices=["blip2", "qwen_vl"], default="blip2",
                    help="simulated user: blip2 = PlugIR's; qwen_vl = Qwen2.5-VL-3B shown the target (new baseline)")
args = parser.parse_args()

runner.START_TARGET, runner.N_TARGETS, runner.N_STATES = args.start, args.targets, 11
runner.DEVICE = runner.LIVE_DEVICE = "cuda"
runner.RUN_LIVE_PLUGIR, runner.OFFLINE = True, True
runner.SEARCH, runner.SELECTION, runner.ANSWERER = args.search, args.selection, args.answerer
agent = ("" if args.selection == "plugir_kl" else f"{args.selection}_") + ("" if args.answerer == "blip2" else "qwenuser_")
runner.RUN_LABEL = f"plugir_live_{agent}{args.search}_{args.start + 1}_{args.start + args.targets}"
runner.main()
