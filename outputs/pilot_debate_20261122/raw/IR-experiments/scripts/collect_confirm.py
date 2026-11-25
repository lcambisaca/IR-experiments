"""Copy the finished live dialogues of the confirmation targets into one folder for fact_fusion.py."""
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[1]
wanted = set((root / "assets/protocol/confirm_targets_151_250.txt").read_text().split())
out = root / "runs/confirm_151_250/dialogues"
out.mkdir(parents=True, exist_ok=True)
found = {}
for p in (root / "runs/live_upstream").glob("*/dialogues/*.json"):
    s = json.loads(p.read_text())
    t = s.get("identity", {}).get("target")
    if (t in wanted and s.get("status") == "complete" and len(s.get("states", [])) >= 11 and "agent" not in s["identity"]
            and s["states"][1].get("selection") in (None, "plugir_kl")):  # PlugIR minimum-KL only
        found[t] = p
for t, p in found.items():
    shutil.copy2(p, out / p.name)
missing = sorted(wanted - set(found))
print(f"{len(found)}/{len(wanted)} complete confirmation dialogues collected; incomplete: {missing}")
sys.exit(0 if found else 1)
