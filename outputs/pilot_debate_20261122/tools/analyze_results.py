"""Analyze preserved pilot checkpoints without any generation or retrieval calls."""
import collections
import json
import pathlib
import re
import sys

import torch
from agentir.evaluation import metrics_from_ranks

base = pathlib.Path(sys.argv[1])
runs = base / "IR-experiments" / "runs"
dest = pathlib.Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

def key(text):
    return re.sub(r"[^\w]+", " ", (text or "").casefold()).strip()

histories = []
patterns = {
    "agents_critique": "three_agents/**/critique_selector_retry_once/*.json",
    "agents_debate": "three_agents_debate/**/dialogues/*.json",
    "live_plugir_qwen": "live_upstream/**/dialogues/*.json",
}
for method, pattern in patterns.items():
    for path in sorted(runs.glob(pattern)):
        raw = json.loads(path.read_text(encoding="utf-8"))
        states = raw["states"]
        ranks = [s["target_rank"] for s in states]
        metrics, bri = metrics_from_ranks(torch.tensor(ranks)[:, None], [1, 10, 100])
        owners = collections.Counter()
        filters = collections.Counter()
        repeated = []
        seen = {}
        literal_seen = {}
        literal_repeats = []
        rows = []
        for i, state in enumerate(states):
            audit = state.get("audit", {})
            question = state.get("question")
            if question:
                if question in literal_seen:
                    literal_repeats.append([literal_seen[question], i, question])
                literal_seen.setdefault(question, i)
                if key(question) in seen:
                    repeated.append([seen[key(question)], i, question])
                seen.setdefault(key(question), i)
            for proposal in audit.get("final_proposals", []):
                if proposal["question"] == question:
                    owners[proposal["owner"]] += 1
            for attempt in audit.get("attempts", []):
                filters[attempt["filter"]["status"]] += 1
            transcript = audit.get("transcript", [])
            matches = [item["role"] for item in transcript if question and key(item["content"]["question"]) == key(question)]
            rows.append({"state": i, "question": question, "answer": state.get("answer"),
                         "query": state["query"], "rank": ranks[i],
                         "improvement": None if i == 0 else ranks[i-1] - ranks[i],
                         "window": audit.get("shared_retrieval", {}).get("window_size"),
                         "skip": state.get("skip_reason"),
                         "regeneration": audit.get("proposal_regeneration_used", False),
                         "reviewer_attempts": len(audit.get("critique_attempts", [])),
                         "coordinator_attempts": len(audit.get("coordinator_attempts", [])),
                         "coordinator_exact_matches": matches,
                         "llm_calls": state.get("llm_calls") if isinstance(state.get("llm_calls"), int) else len(state.get("llm_calls", []))})
        histories.append({"method": method, "target": raw.get("target", raw.get("identity", {}).get("target")),
                          "checkpoint": str(path.relative_to(base)), "status": raw.get("status", "partial"),
                          "last_error": raw.get("last_error"), "states": len(states),
                          "ranks": ranks, "final_rank": ranks[-1], "best_rank": min(ranks),
                          "BRI": bri, "metrics": metrics, "owners": dict(owners),
                          "filters": dict(filters), "normalized_repeats": repeated,
                          "literal_repeats": literal_repeats, "rows": rows})

events_path = base / "history_events.jsonl"
events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
aggregate = []
for method in patterns:
    selected = [h for h in histories if h["method"] == method]
    if len(selected) != 2:
        continue
    n = min(h["states"] for h in selected)
    metrics, bri = metrics_from_ranks(torch.tensor([h["ranks"][:n] for h in selected]).T, [1, 10, 100])
    aggregate.append({"method": method, "shared_states": n, "BRI": bri, "metrics": metrics})

out = {"histories": histories, "events": events, "aggregate": aggregate}
(dest / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for h in histories:
    print(h["method"], h["target"], h["status"], h["ranks"], "BRI", h["BRI"], "owners", h["owners"])
