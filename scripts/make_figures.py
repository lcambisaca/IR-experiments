"""Figures, example tables and data for the live experiments (Nov 9 update).

Usage (repo root): python scripts/make_figures.py results/live_experiments results/live_experiments/saved_plugir_gpt35_ranks.json
Needs results/agent_v2_vlm_probe/per_target_ranks.csv and summary.json (from final_results.py), the Stage A
summary in vm_results/stageA/ and the raw conversations in vm_results/final_pull/ (local only).
"""
import csv
import glob
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
SAVED = json.load(open(sys.argv[2]))
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(exist_ok=True)

# Validated categorical palette (light surface), fixed order; gray for the reference line.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, SURFACE, GRID = "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.titlelocation": "left", "lines.linewidth": 2})


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- data ----------
ranks = {}
for row in list(csv.reader(open(REPO / "results/agent_v2_vlm_probe/per_target_ranks.csv")))[1:]:
    ranks.setdefault(row[1], {})[row[0]] = list(map(int, row[2:]))
targets = SAVED["targets"]  # Matched-target order (31..134).
ranks["saved"] = dict(zip(targets, SAVED["saved_gpt35_ranks"]))
summary = json.load(open(REPO / "results/agent_v2_vlm_probe/summary.json"))
n = len(targets)
r10 = lambda m, r: 100 * sum(ranks[m][t][r] <= 10 for t in targets) / n
h10 = lambda m, r: 100 * sum(any(x <= 10 for x in ranks[m][t][:r + 1]) for t in targets) / n

# ---------- 1. setup diagram ----------
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.set_axis_off(); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4)
def box(x, y, w, h, text, edge=GRID):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#ffffff", ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.35)
def arrow(x1, y1, x2, y2, **kw):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, **kw))
box(0.1, 2.35, 1.6, 0.9, "Start: the\nphoto's caption")
box(2.0, 2.35, 1.8, 0.9, "Qwen3-4B writes\n5 candidate questions")
box(4.2, 3.0, 3.3, 0.85, "PlugIR: keep the question that changes\nthe current top results least (min KL)", edge=BLUE)
box(4.2, 1.65, 3.3, 1.1, "Agent v2: its own vision model answers\neach question on the current top 20;\nask the one with the most varied answers", edge=ORANGE)
box(7.9, 2.35, 1.75, 0.9, "BLIP-2 answers,\nlooking at the\ntarget photo")
box(10.0, 2.35, 1.85, 0.9, "Qwen rewrites\nthe description")
box(10.0, 0.35, 1.85, 0.9, "BLIP searches\n50,000 photos")
box(5.4, 0.25, 4.0, 1.1, "What the user is shown (top 10):\n- rewrite alone (PlugIR default)\n- caption mix: 0.7 x rewrite + 0.3 x caption", edge=AQUA)
arrow(1.7, 2.8, 2.0, 2.8); arrow(3.8, 2.8, 4.2, 3.4); arrow(3.8, 2.8, 4.2, 2.2)
arrow(7.5, 3.4, 7.9, 3.0); arrow(7.5, 2.2, 7.9, 2.6); arrow(9.65, 2.8, 10.0, 2.8)
arrow(10.92, 2.35, 10.92, 1.25); arrow(10.0, 0.8, 9.4, 0.8)
ax.annotate("", (2.9, 2.35), (5.4, 0.8), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, ls="--", connectionstyle="arc3,rad=-0.3"))
ax.text(2.2, 1.0, "next round\n(10 rounds)", fontsize=9, color=INK2, ha="center")
ax.text(0.1, 4.2, "Live pipeline: identical for both methods except the question-choice step", fontsize=12, weight="bold", color=INK)
save(fig, "fig1_setup.png")

# ---------- 2. Matan's strategies (Stage A, 30 live PlugIR conversations) ----------
stage = [r for r in csv.DictReader(open(glob.glob(str(REPO / "vm_results/stageA/*/summary.csv"))[0])) if r["round"] == "10"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"wspace": 0.28})
names = [r["strategy"] for r in stage]
x = range(len(names)); w = 0.26
orig = float(stage[0]["recall@10_original"])
for i, (key, color, label) in enumerate([("recall@10_original", BLUE, "original search"),
                                         ("recall@10_max", ORANGE, "merged by max (NN) score"),
                                         ("recall@10_mean", AQUA, "merged by mean score")]):
    vals = [float(r[key]) for r in stage]
    bars = a1.bar([j + (i - 1) * w for j in x], vals, w - 0.03, color=color, label=label)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
a1.set_xticks(list(x), names); a1.set_ylim(0, 100); a1.set_ylabel("Recall@10 at round 10 (%)")
a1.set_title("Merged top 10 vs original top 10", pad=34); a1.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1, columnspacing=1)
for i, (key, color, label) in enumerate([("any_top10", ORANGE, "target in ANY of the 6 top-10s\n(\"winning version\")"),
                                         ("control_same_count", BLUE, "original search shown with the\nsame number of photos")]):
    vals = [float(r[key]) for r in stage]
    bars = a2.bar([j + (i - 0.5) * 0.36 for j in x], vals, 0.33, color=color, label=label)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
pool = [float(r["mean_pool_size"]) for r in stage]
a2.set_xticks(list(x), [f"{s}\n(~{p:.0f} photos)" for s, p in zip(names, pool)]); a2.set_ylim(0, 105)
a2.set_title("Winning version vs fair control", pad=34); a2.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1, columnspacing=1)

save(fig, "fig2_extra_search_strategies.png")

# ---------- 3. Recall@10 and Hits@10 by round ----------
series = [("plugir", "PlugIR (live, Qwen)", BLUE, "-"), ("agent_v2", "Agent v2", ORANGE, "-"),
          ("plugir+mix", "PlugIR + caption mix", AQUA, "-"), ("agent_v2+mix", "Agent v2 + caption mix", YELLOW, "-"),
          ("saved", "Authors' saved PlugIR (GPT-3.5)", GRAY, "--")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.42})
for ax_, fn, title in ((a1, r10, "Recall@10 by round"), (a2, h10, "Hits@10 by round")):
    ends = []
    for key, label, color, style in series:
        ys = [fn(key, r) for r in range(11)]
        ax_.plot(range(11), ys, style, color=color, label=label, marker="o", markersize=3.5)
        ends.append([ys[-1], ys[-1], color, label.split(" (")[0]])
    merged = {}
    for v, y, color, label in ends:
        merged.setdefault(round(v, 1), []).append(label)
    ends = sorted([[v, v] for v in merged])
    gap = 1.4 if ax_ is a1 else 1.0
    for i in range(1, len(ends)):  # Nudge end labels apart; equal values share one label.
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for v, y in ends:
        ax_.text(10.25, y, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
    ax_.set_xticks(range(11)); ax_.set_xlabel("Round (0 = caption only)"); ax_.set_ylabel("%")
    ax_.set_title(title); ax_.set_xlim(-0.3, 11)
a1.set_ylim(55, 90); a2.set_ylim(70, 96)
handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=8.8)
fig.suptitle(f"Same {n} fresh targets for every line", x=0.07, y=1.03, ha="left", weight="bold", fontsize=12)
save(fig, "fig3_recall_hits_by_round.png")

# ---------- 4. Agent's lead as targets accumulate ----------
rng = random.Random(42)
diff = [(ranks["agent_v2"][t][10] <= 10) - (ranks["plugir"][t][10] <= 10) for t in targets]
xs, mean, lo, hi = [], [], [], []
for k in range(10, n + 1):
    d = diff[:k]
    boots = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(2000))
    xs.append(k); mean.append(100 * sum(d) / k); lo.append(100 * boots[50]); hi.append(100 * boots[1949])
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.fill_between(xs, lo, hi, color=ORANGE, alpha=0.15, lw=0, label="95% range")
ax.plot(xs, mean, color=ORANGE, label="Agent v2 − PlugIR")
ax.axhline(0, color=INK2, lw=1)
ax.axvline(30, color=GRAY, lw=1, ls="--"); ax.text(31, 26, "first check\n(30 targets): +6.7", fontsize=8.5, color=INK2, va="top")
ax.text(n, mean[-1] - 4, f"{n} targets: {mean[-1]:+.1f}", ha="right", fontsize=8.5, color=INK2)
ax.set_xlabel("Number of fresh targets included (in run order)"); ax.set_ylabel("Recall@10 difference at round 10 (points)")
ax.set_title("The agent's early lead was noise"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "fig4_agent_lead_over_targets.png")

# ---------- 5. Effect sizes with 95% ranges ----------
d = summary["differences_round_10"]
rows = [("Agent v2 − PlugIR", d["agent_v2 - plugir"]["recall@10"], ORANGE),
        ("With mix: Agent v2 − PlugIR", d["agent_v2+mix - plugir+mix"]["recall@10"], ORANGE),
        ("PlugIR: mix − no mix", d["plugir+mix - plugir"]["recall@10"], AQUA),
        ("Agent v2: mix − no mix", d["agent_v2+mix - agent_v2"]["recall@10"], AQUA)]
fig, ax = plt.subplots(figsize=(8.5, 3.0))
for i, (label, (m, l, h), color) in enumerate(rows[::-1]):
    ax.plot([l, h], [i, i], color=color, lw=2.5, solid_capstyle="round")
    ax.plot(m, i, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(h + 0.5, i, f"{m:+.1f}  [{l:+.1f}, {h:+.1f}]", va="center", fontsize=9, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]]); ax.set_xlim(-13, 19); ax.grid(axis="y", visible=False)
ax.set_xlabel("Recall@10 change at round 10 (points), paired, 95% bootstrap range")
ax.set_title(f"What changes Recall@10 ({n} targets)")
save(fig, "fig5_effect_sizes.png")

# ---------- 6. Examples: target photo + rank per round ----------
convs = {}
for name, folders in {"plugir": ("0364674fdf0478954425", "355f786724fdc410f0f7"),
                      "agent_v2": ("10db503e990c5e70feb9", "5064c43d1a55fc1d6877")}.items():
    for f in folders:
        for p in glob.glob(str(REPO / f"vm_results/final_pull/runs/live_upstream/{f}/dialogues/*.json")):
            c = json.load(open(p))
            if c.get("status") == "complete":
                convs[(c["identity"]["target"], name)] = c
examples = [("ex1_microwave", "unlabeled2017/000000239836.jpg", "Drift: the microwave is outdoors on grass, but the rewrite moves it into a kitchen", ("plugir", "plugir+mix")),
            ("ex2_toilet", "unlabeled2017/000000222337.jpg", "Agent keeps the target; PlugIR's rewrite invents a second toilet", ("plugir", "agent_v2")),
            ("ex3_apples", "unlabeled2017/000000358981.jpg", "Agent loses the target at round 4: BLIP-2 says the background is \"white\" (it is a street)", ("plugir", "agent_v2"))]
for stem, target, title, show in examples:
    fig = plt.figure(figsize=(11, 3.6))
    ia = fig.add_axes([0.0, 0.05, 0.3, 0.82]); ia.imshow(Image.open(REPO / "data/images" / target)); ia.set_axis_off()
    ia.set_title("Target photo", fontsize=10, loc="center", weight="normal", color=INK2)
    ax = fig.add_axes([0.38, 0.14, 0.5, 0.72])
    for key, label, color, style in [x for x in series if x[0] in show]:
        ax.plot(range(11), ranks[key][target], style, color=color, marker="o", markersize=3.5, label=label)
    ax.set_yscale("log"); ax.invert_yaxis(); ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"]); ax.minorticks_off(); ax.axhline(10, color=GRAY, lw=1, ls="--")
    ax.text(0, 10, " top-10 line", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(11)); ax.set_xlabel("Round"); ax.set_ylabel("Target rank (log, 1 = top)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(title, x=0.01, ha="left", weight="bold", fontsize=12)
    save(fig, f"{stem}.png")

# ---------- 7. Example conversations as text ----------
lines = ["# Example conversations\n",
         "Ranks are out of 50,000 photos (1 = top). \"mix\" = the rank the user would see with the caption mix.\n"]
for stem, target, title, show in examples:
    lines.append(f"\n## {title}\n\nTarget: `{target}` (photo: `figures/{stem}.png`)\n")
    for key, label in (("plugir", "PlugIR (live, Qwen)"), ("agent_v2", "Agent v2")):
        c = convs[(target, key)]
        lines.append(f"\n### {label}\n\nCaption: *{c['states'][0]['query']}* (rank {ranks[key][target][0]})\n")
        lines.append("| Round | Question | BLIP-2 answer | Rank | Rank with mix |\n|---|---|---|---|---|")
        for i, qa in enumerate(c["dialogue"], 1):
            q = qa["question"].replace("**Question:**", "").replace("|", "/").strip()
            lines.append(f"| {i} | {q} | {qa['answer'].replace('|', '/')} | {ranks[key][target][i]} | {ranks[key + '+mix'][target][i]} |")
        lines.append(f"\nFinal description: *{c['states'][10]['query']}*\n")
(OUT / "example_conversations.md").write_text("\n".join(lines))

# ---------- data tables ----------
with open(OUT / "data/recall_hits_by_round.csv", "w", newline="") as stream:
    w_ = csv.writer(stream)
    w_.writerow(["method", "metric"] + [f"round_{r}" for r in range(11)])
    for key, label, *_ in series:
        w_.writerow([label, "recall@10"] + [round(r10(key, r), 1) for r in range(11)])
        w_.writerow([label, "hits@10"] + [round(h10(key, r), 1) for r in range(11)])
        w_.writerow([label, "recall@1"] + [round(100 * sum(ranks[key][t][r] <= 1 for t in targets) / n, 1) for r in range(11)])
print("saved R@10/H@10 at round 10:", r10("saved", 10), h10("saved", 10))
