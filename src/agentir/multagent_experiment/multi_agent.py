"""Three specialist questioners, shared retrieval evidence, one shared Q&A stream.

This is a new experiment. It calls checksum-verified original PlugIR clustering,
entropy and KL helpers, but its prompts, filters and orchestration are new.
Official per-state rankings and shared retrieval use one neutral reformulation.
No fusion, spectral clustering, training or image re-encoding.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Optional

from PIL import Image
import torch
from torch.nn.functional import normalize

from .agent_prompts import PROMPT_VERSION, ROLES, messages_for
from ..cache import read_json
from ..live import (LiveSettings, generated_query_method, load_candidate_captions,
                   resolve_target_image, save_json, signature)
from ..plugir_upstream import bind_original_text_encoder, load_generator
from ..retrieval import ranks_for_targets


@dataclass(frozen=True)
class AgentSettings:
    """Small explicit generation budget; a state is caption plus completed Q&A rounds."""
    seed: int = 1021
    threshold_low: int = 500 #Window Size 
    exclude_top_k: int = 10
    max_calls_per_round: int = 40
    max_targets: int = 2 # Two for now cuz were looking for cake and giraffe
    questions_per_agent: int = 5
    exclude_unjudged: bool = False

    def validate(self, pool_size, n_states):
        """Keep upstream's ten clusters and KL slice valid and all budgets bounded."""
        LiveSettings(seed=self.seed, q_n=self.questions_per_agent, threshold_low=self.threshold_low,
                     recall_hitting= self.exclude_top_k).validate(pool_size, n_states)
        if type(self.questions_per_agent) is not int or not 1 <= self.questions_per_agent <= 5:
            raise ValueError("Use one to five question slots per agent.")
        if any(type(v) is not int or v < 1 for v in
               (self.max_calls_per_round, self.max_targets)):
            raise ValueError("Agent budgets must be positive integers.")
        if type(self.exclude_unjudged) is not bool:
            raise ValueError("exclude_unjudged must be a boolean.")


def question_key(text):
    """Collapse exact surface duplicates; semantic near-duplicates remain an audited limitation."""
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def valid_question(text):
    """Reject abstentions, overlong text and multiple questions before an LLM filter."""
    return (isinstance(text, str) and 4 <= len(text.strip()) <= 240
            and text.strip().endswith("?") and text.count("?") == 1)


class AgentChat:
    """Use the existing pinned, cached Qwen transport with new structured prompts."""

    def __init__(self, transport, settings):
        """Keep the transport separate from target-aware answer generation and metric logging."""
        self.transport, self.settings = transport, settings
        self.audit, self.calls, self.sampling_context = [], 0, ""

    def begin(self, record, interaction):
        """Seed proposals by confirmed context, independent of critique/no-critique condition."""
        self.calls, self.audit = 0, []
        self.sampling_context = signature({"record": record, "interaction": interaction})
        self.transport.begin_round(interaction, self.sampling_context)

    def complete(self, stage, role, payload):
        """Make one bounded JSON call; never silently repair malformed model output."""
        if self.calls >= self.settings.max_calls_per_round:
            raise RuntimeError("Agent per-round call budget exhausted; checkpoint retained.")
        self.calls += 1
        messages = messages_for(stage, role, payload)
        seed = int(signature({"seed": self.settings.seed, "context": self.sampling_context,
                              "stage": stage, "role": role, "payload": payload})[:8], 16) % (2**31)
        max_tokens = 384 if stage == "critique" else 512
        if stage == "propose_batch":
            max_tokens = min(512, max(192, 112 * payload["question_count"]))
        elif stage == "filter_batch":
            max_tokens = max(256, 64 * len(payload["proposals"]))
        kwargs = {"messages": messages, "model": "local-qwen", "n": 1,
                  "max_tokens": max_tokens,
                  "temperature": .5 if stage == "propose_batch" else 0., "seed": seed}
        response = self.transport.create(**kwargs)
        choice = response["choices"][0]
        text = choice["message"]["content"]
        item = {"stage": stage, "role": role, "payload": payload, "response": text}
        self.audit.append(item)
        if choice.get("finish_reason") == "length":
            raise ValueError("Agent output hit the token cap; increase settings explicitly, not silently.")
        try:
            result = json.loads(text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid JSON in {role}/{stage}; no automatic repair loop.") from error
        if not isinstance(result, dict):
            raise ValueError("Agent output must be one JSON object.")
        if stage == "propose_batch":
            proposals = result.get("proposals")
            if (not isinstance(proposals, list) or len(proposals) > payload["question_count"]
                    or any(not isinstance(p, dict) for p in proposals)):
                raise ValueError("Invalid proposal batch or question count.")
            fields = ("question",)
            if any(any(not isinstance(p.get(k), str) or not p[k].strip() for k in fields)
                   or not isinstance(p.get("evidence_ids"), list) for p in proposals):
                raise ValueError("Missing batch proposal fields.")
            return result
        if stage == "filter_batch":
            decisions = result.get("decisions")
            expected = {p["slot"] for p in payload["proposals"]}
            if (not isinstance(decisions, list)
                    or any(not isinstance(d, dict) or type(d.get("slot")) is not int
                           or d.get("status") not in ("unknown", "already_known", "unsupported", "ambiguous")
                           or not isinstance(d.get("reason"), str) or not d["reason"].strip()
                           for d in decisions)
                    or len({d["slot"] for d in decisions}) != len(decisions)
                    or not {d["slot"] for d in decisions}.issubset(expected)):
                raise ValueError("Batch filter must return one valid decision per slot.")
            missing = sorted(expected - {d["slot"] for d in decisions})
            if missing and not self.settings.exclude_unjudged:
                raise ValueError("Batch filter must return one valid decision per slot.")
            if missing:
                # Fail closed per omitted question: do not invent a model judgement or retry.
                item["filter_omissions"] = missing
                decisions = decisions + [{"slot": slot, "status": "ambiguous",
                    "reason": "Missing model decision; excluded without retry."} for slot in missing]
            result = {**result, "decisions": decisions, "omitted_slots": missing}
            return result
        if stage == "critique":
            slot = result.get("slot")
            reason = result.get("reason")
            if (type(slot) is not int or not 0 <= slot < len(payload["proposals"])
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError("Critique must choose one eligible proposal slot and give a reason.")
            return result
        required_text = ("query", "coverage_issue")
        if any(not isinstance(result.get(key), str) or not result[key].strip() for key in required_text):
            raise ValueError(f"Missing required text fields in {role}/{stage}.")
        if stage == "reformulate" and not isinstance(result.get("emphasis"), str):
            raise ValueError("Reformulation emphasis metadata must be a string, possibly empty.")
        return result


class RetrievalContext:
    """Reuse original PlugIR scientific helpers with the existing image-vector cache."""

    def __init__(self, retriever, captions, settings, *, kmeans=None):
        """Bind original BLIP functions and ten-cluster extraction; inject doubles only in tests."""
        if kmeans is None:
            from fast_pytorch_kmeans import KMeans
            kmeans = KMeans(n_clusters=10, mode="cosine", verbose=0)
        self.retriever, self.settings = retriever, settings
        self.catalogue = [captions[name] for name in retriever.cache.paths]
        self.namespace = {"torch": torch, "normalize": normalize, "Optional": Optional,
            "BlipForImageTextRetrieval": type(retriever.model), "Image": Image,
            "device": retriever.device, "threshold_low": settings.threshold_low,
            "img_embs": retriever.images, "kmeans": kmeans,
            "captions": [{"caption": c} for c in self.catalogue]}
        self.original_answer, _, _ = load_generator(self.namespace)
        bind_original_text_encoder(self.namespace, retriever)
        original_features = self.namespace["get_text_features"]
        self.features = {}

        def cached_features(text):
            """Memoize exact original text features on CPU; no changes to scoring or preprocessing."""
            key = (text,) if isinstance(text, str) else tuple(text)
            if key not in self.features:
                with torch.inference_mode():
                    self.features[key] = original_features(text).detach().cpu()
            return self.features[key].to(retriever.device)

        self.namespace["get_text_features"] = cached_features

    def scores(self, query):
        """Score candidate images with the original generation feature normalization."""
        with torch.inference_mode():
            return (normalize(self.namespace["get_text_features"](query), dim=-1)
                    @ self.retriever.images.T).squeeze(0).cpu()

    def candidates(self, query, interaction, role, record):
        """Call original entropy/KMeans selection with deterministic context-specific RNG.

        Returned caption strings are linked to matching members of the current
        window. Identical catalogue captions use the highest-ranked matching ID;
        this is a caption evidence ID, not a claim to recover the original representative index.
        """
        scores = self.scores(query)
        window = int(self.settings.threshold_low - interaction * (self.settings.threshold_low / 10.))
        top = scores.argsort()[-window:]
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        seed = int(signature({"record": record, "role": role, "round": interaction,
                              "seed": self.settings.seed})[:8], 16) % (2**31)
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            selected = self.namespace["get_related_captions"](query, round=interaction + 1)
        matches = {}
        for index in reversed(top.tolist()):
            matches.setdefault(self.catalogue[index][0], index)
        candidates = [{"id": matches[caption], "caption": caption} for caption in selected]
        return {"captions": candidates, "pool_ids": list(reversed(top.tolist())), "window_size": window,
                "query": query}

    def select(self, neutral, proposals, interaction):
        """Choose minimum KL on one common neutral-query slice, with PlugIR's KL formula."""
        questions = [p["question"] for p in proposals]
        if not questions:
            return None, []
        scores = self.scores(neutral).to(self.retriever.device)
        window = int(self.settings.threshold_low - interaction * self.settings.threshold_low / 10.)
        indices = scores.argsort()[-window:-self.settings.exclude_top_k]
        p = torch.softmax(scores[indices], dim=0)
        log = []
        for proposal in proposals:
            temporary = self.scores(neutral + ", " + proposal["question"]).to(scores.device)
            q = torch.softmax(temporary[indices], dim=0)
            log.append({"question": proposal["question"], "owner": proposal["owner"],
                        "kl": float((p * (p.log() - q.log())).sum())})
        # Match the original helper's float32 scores and argsort tie handling.
        winner = torch.tensor([row["kl"] for row in log], dtype=torch.float32).argsort()[0].item()
        return questions[winner], log


def rewrite_query(chat, record, role):
    """Generate one search caption and validate its format without a second model call."""
    payload = {"record": record}
    result = chat.complete("reformulate", role, payload)
    query = result.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 3000:
        raise ValueError("Missing/overlong role query.")
    if "?" in query:
        raise ValueError("Role query must be a declarative image description, not a question.")
    return query.strip()


def precheck_proposal(record, proposal, candidates):
    """Reject malformed, asked or ungrounded questions without calling the model."""
    question, ids = proposal.get("question"), proposal.get("evidence_ids")
    allowed = {c["id"] for c in candidates}
    if not valid_question(question):
        return {"status": "abstain" if question == "abstain" else "malformed", "reason": "No single valid question."}
    if question_key(question) in {question_key(qa["question"]) for qa in record["dialogue"]}:
        return {"status": "already_asked", "reason": "This question was already asked; its answer may be uncertain."}
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in allowed for i in ids):
        return {"status": "unsupported", "reason": "Missing or out-of-scope evidence IDs."}
    return None


def unique_proposals(proposals):
    """Retain one exact normalized question and record all proposing roles without extra votes."""
    found = {}
    for proposal in proposals:
        key = question_key(proposal["question"])
        if key in found:
            found[key]["duplicate_owners"].append(proposal["owner"])
        else:
            found[key] = {**proposal, "duplicate_owners": []}
    return list(found.values())


def question_round(chat, context, record, neutral, interaction, *, critique, prior_attempts=()):
    """Generate one question batch per role, batch-filter, then select by critique or minimum KL.

    No target identity, pixels, target ranks, saved human answers or future states
    enter this function. Returns all candidate/critique evidence for later audits.
    """
    batches, attempts, eligible, pending = {}, [], [], []
    omitted_slots = []
    # Only previously rejected questions are blocked. A useful unselected question
    # may still be useful next round; it has not been answered or confirmed.
    blocked = {question_key(p["question"]): p for p in prior_attempts
               if isinstance(p.get("question"), str)}
    seen = {}
    roles = list(ROLES)
    # One search, one clustering pass and identical caption evidence per round.
    shared = context.candidates(neutral, interaction, "shared", record)
    for role in roles:
        batches[role] = chat.complete("propose_batch", role, {"record": record, "query": neutral,
            "agent_role": role,
            "batch_id": roles.index(role),
            "candidates": shared["captions"], "question_count": chat.settings.questions_per_agent,
            "attempted_proposals": list(prior_attempts)})
    # Finish all independent searches before any question-pool filtering/review.
    for role in roles:
        batch = batches[role]
        for slot, raw in enumerate(batch["proposals"]):
            proposal = {**raw, "owner": role, "slot": slot}
            key = question_key(proposal["question"])
            gate = precheck_proposal(record, proposal, shared["captions"])
            if gate is None and key in blocked:
                gate = {"status": "previously_rejected", "reason": blocked[key]["reason"]}
            duplicate = seen.get(key) if gate is None else None
            if duplicate is not None:
                gate = {"status": "duplicate", "reason": "Question already proposed this round."}
            item = {"owner": role, "slot": slot, "attempt": 0, "proposal": proposal, "filter": gate}
            if duplicate is not None:
                item["duplicate_of"] = {"owner": duplicate["owner"], "slot": duplicate["slot"]}
            attempts.append(item)
            if gate is None:
                seen[key] = item
                pending.append(item)
    if pending:
        # Globally unique filter slots prevent role-local slots (0..4) colliding.
        filter_result = chat.complete("filter_batch", "full", {"record": record,
            "proposals": [{"question":item["proposal"]["question"],
                           "evidence_ids":item["proposal"]["evidence_ids"], "slot":slot}
                          for slot,item in enumerate(pending)],
            "candidates": shared["captions"]})
        decisions = filter_result["decisions"]
        omitted_slots = filter_result["omitted_slots"]
        by_slot = {d["slot"]: d for d in decisions}
        for slot, item in enumerate(pending):
            decision = by_slot[slot]
            item["filter"] = {"status": decision["status"], "reason": decision["reason"]}
            if chat.settings.exclude_unjudged:
                item["filter"]["model_omitted"] = slot in omitted_slots
            if decision["status"] == "unknown":
                eligible.append(item["proposal"])
    final = unique_proposals(eligible)
    decision = None
    if critique:
        selected, kl = None, []
        if final:
            decision = chat.complete("critique", "full", {
                "record": record, "query": neutral, "candidates": shared["captions"],
                "proposals": [{"slot": slot, "question": p["question"],
                               "evidence_ids": p["evidence_ids"]}
                              for slot, p in enumerate(final)],
            })
            selected = final[decision["slot"]]["question"]
    else:
        selected, kl = context.select(neutral, final, interaction)
    selected_key = question_key(selected) if selected is not None else None
    final_keys = {question_key(p["question"]) for p in final}
    memory = []
    for item in attempts:
        question = item["proposal"]["question"]
        key = question_key(question)
        status = item["filter"]["status"]
        # A duplicate of a still-unasked eligible question is not permanently
        # suppressed: deduplication does not provide its missing answer.
        effective = seen[key]["filter"] if status == "duplicate" else item["filter"]
        blocked_next = key == selected_key or (key not in final_keys and
                       effective["status"] != "unknown")
        memory.append({"question": question, "owner": item["owner"], "status": status,
                       "reason": "Already asked; see the shared dialogue for its answer."
                                 if key == selected_key else effective["reason"], "blocked": blocked_next,
                       "selected": key == selected_key})
    return {"shared_retrieval": shared, "role_preferences": dict(ROLES),
            "filter_omitted_slots": omitted_slots,
            "attempts": attempts, "before_critique": final,
            "question_memory": memory,
            "selector": "critique" if critique else "min_kl",
            "critique_decision": decision, "final_proposals": final,
            "selected_question": selected, "kl": kl}


def generate_agent_dialogue(retriever, *, initial_caption, target_id, image_path, captions,
                            questioner, answerer, settings, n_states, checkpoint_path,
                            critique=True, _kmeans=None):
    """Run/resume independent critique or no-critique histories; ranks never feed decisions."""
    settings.validate(len(retriever.cache.paths), n_states)
    path = Path(checkpoint_path)
    context = RetrievalContext(retriever, captions, settings, kmeans=_kmeans)
    chat = AgentChat(questioner, settings)
    if path.is_file():
        state = read_json(path)
    else:
        state = {"target": retriever.cache.paths[target_id], "dialogue": [], "states": [{"round": 0, "query": initial_caption,
            "raw_query": initial_caption, "target_rank": int(ranks_for_targets(retriever.scores([initial_caption]), [target_id])[0])}],
            "status": "partial"}
        save_json(path, state)
    try:
        for interaction in range(len(state["states"]) - 1, n_states - 1):
            record = {"initial_caption": initial_caption, "dialogue": state["dialogue"]}
            neutral = state["states"][-1]["query"]
            chat.begin(record, interaction)
            transport_start = len(questioner.records)
            prior_attempts = list({question_key(entry["question"]):
                                  {"question": entry["question"], "reason": entry["reason"]}
                for previous in state["states"][1:]
                for entry in previous["audit"]["question_memory"] if entry["blocked"]}.values())
            audit = question_round(chat, context, record, neutral, interaction,
                                   critique=critique, prior_attempts=prior_attempts)
            question, answer = audit["selected_question"], None
            next_dialogue = list(state["dialogue"])
            if question is not None:
                questioner.park()
                try:
                    # Upstream wraps this input in Question: ... Answer:; pass only the selected question.
                    answer = answerer.answer_original(context.original_answer, image_path, question)
                finally:
                    answerer.park()
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Answerer returned no answer; do not update confirmed context.")
                next_dialogue.append({"question": question, "answer": answer.strip()})
                neutral = rewrite_query(chat, {"initial_caption": initial_caption, "dialogue": next_dialogue}, "full")
            raw = ", ".join([initial_caption] + [qa["question"] + " " + qa["answer"] for qa in next_dialogue])
            # Target ranks are added ONLY after selection/answering/reformulation.
            measured = int(ranks_for_targets(retriever.scores([neutral]), [target_id])[0])
            state["dialogue"] = next_dialogue
            state["question_memory"] = [entry for previous in state["states"][1:]
                for entry in previous["audit"]["question_memory"]] + audit["question_memory"]
            state["states"].append({"round": interaction + 1, "query": neutral, "raw_query": raw,
                "target_rank": measured, "question": question, "answer": answer,
                "no_question": question is None, "audit": audit, "llm_calls": chat.calls,
                "llm_audit": chat.audit, "transport_records": questioner.records[transport_start:]})
            state["status"] = "partial"
            state.pop("last_error", None)
            state.pop("partial_round_audit", None)
            save_json(path, state)
            print(f"Agents critique={critique}, state {interaction+1}: "
                  f"question={question!r}, answer={answer!r}, rank={measured}, "
                  f"filter_omissions={len(audit['filter_omitted_slots'])}", flush=True)
        state["status"] = "complete"
        save_json(path, state)
        return {**state, "states": state["states"][:n_states],
                "dialogue": state["dialogue"][:sum(s["question"] is not None for s in state["states"][1:n_states])]}
    except BaseException as error:
        state["status"], state["last_error"] = "interrupted", str(error)
        state["partial_round_audit"] = chat.audit
        save_json(path, state)
        raise
    finally:
        questioner.park()
        answerer.park()


def prepare_agent_experiment(root, retriever, pairs, *, n_states, settings, device="cuda", offline=False,
        candidate_captions="assets/protocol/visdial_captions.json", image_root="data/images",
        image_archive=None, download_missing=True, max_input_tokens=8192,
        modes=(False, True), run_name="shared_agents_simple_v1"):
    """Prepare both conditions for run_evaluation.py using identical initial targets and cached images."""
    import importlib.util
    from ..evaluation import PreparedExperiment
    from ..live_models import Qwen3, Blip2Answerer, require_cuda
    require_cuda(device)
    settings.validate(len(retriever.cache.paths), n_states)
    if not pairs or len(pairs) > settings.max_targets:
        raise ValueError(f"Agent pilot allows at most {settings.max_targets} targets; edit settings explicitly to scale.")
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("Agent input token limit must be positive.")
    if not modes or len(set(modes)) != len(modes) or any(type(m) is not bool for m in modes):
        raise ValueError("Modes must contain distinct booleans (False=no critique, True=critique).")
    if modes and True in modes and settings.max_calls_per_round < 6:
        raise ValueError("Critique selection requires a budget of at least six calls per round.")
    extras = ("accelerate", "sentencepiece", "fast_pytorch_kmeans", "psutil")
    if any(importlib.util.find_spec(name) is None for name in extras):
        raise RuntimeError('Install live extras in the GPU environment: python -m pip install -e ".[live,test]"')
    root = Path(root)
    catalogue_path = root / candidate_captions
    catalogue = load_candidate_captions(catalogue_path, retriever.cache.paths)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name must use only letters, digits, underscores or hyphens.")
    folder = root / "runs" / "three_agents" / run_name
    questioner = Qwen3(folder / "qwen_calls", device=device, offline=offline,
                       max_input_tokens=max_input_tokens, seed=settings.seed, max_output_tokens=1024)
    answerer = Blip2Answerer(folder / "blip2_calls", device=device, offline=offline)
    target_ids = {name: i for i, name in enumerate(retriever.cache.paths)}
    conversations, methods, elapsed = {}, {}, {}
    try:
        for critique in modes:
            name = "agents_critique" if critique else "agents_no_critique"
            started = time.perf_counter()
            conversations[name] = []
            for raw, _ in pairs:
                print(f"Target {len(conversations[name])+1}/{len(pairs)}: {raw['img']}; method={name}", flush=True)
                image = resolve_target_image(root / image_root, raw["img"],
                    archive=None if image_archive is None else root / image_archive,
                    download=download_missing and not offline)
                conversations[name].append(generate_agent_dialogue(retriever,
                    initial_caption=raw["dialog"][0], target_id=target_ids[raw["img"]], image_path=image,
                    captions=catalogue, questioner=questioner, answerer=answerer, settings=settings,
                    n_states=n_states, checkpoint_path=folder / ("critique_selector" if critique else name) / (Path(raw["img"]).stem + ".json"),
                    critique=critique))
            methods[name] = generated_query_method(conversations[name])
            elapsed[name] = time.perf_counter() - started
    finally:
        questioner.park()
        answerer.park()
    from .agent_prompts import EXAMPLES, NEUTRAL_EXAMPLE, QA_GROUNDING_EXAMPLE, SHARED, TASKS
    return PreparedExperiment(methods=methods,
        metadata={"run_name": run_name, "settings": asdict(settings), "checkpoint_folder": str(folder),
                  "condition_wall_seconds": elapsed,
                  "official_ranking": "neutral reformulation of original caption plus shared Q&A; no fusion",
                  "retrieval_evidence": "one neutral query, shared candidate pool and ten original representatives per round",
                  "limitations": "New agent prompts/filter/orchestration; original clustering and PlugIR's KL formula. "
                      "Rewrites can drift; the standard rewrite has no separate fidelity check. Exact duplicate detection only. "
                      "Conditions share initial proposals at the same context, then histories can diverge. "
                      "Model quality, speed and GPU fit need a real pilot."},
        artifacts={"agent_dialogues.json": conversations,
                   "prompt_pack.json": {"version": PROMPT_VERSION, "roles": ROLES,
                                        "examples": EXAMPLES, "neutral_example": NEUTRAL_EXAMPLE,
                                        "qa_grounding_example": QA_GROUNDING_EXAMPLE,
                                        "shared": SHARED, "tasks": TASKS}})
