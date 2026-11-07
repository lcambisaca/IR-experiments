"""Agent question selection: probe the corpus with guessed answers, ask the question that splits it.

PlugIR picks the candidate question whose wording changes the candidate scores least (minimum KL).
Here the agent instead, for each candidate question:
  1. asks Qwen for up to three short answers the question could plausibly have, given the
     current description and the candidate captions PlugIR already shows it;
  2. probes: searches with "<description>, <question> <answer>" for each guessed answer;
  3. lets every current candidate image "vote" for the answer whose probe fits it best.
The question whose answers split the candidates most evenly is asked. Votes are soft, so an image
that fits all answers equally (as when the probes barely differ) counts as undecided; the score is
the mutual information between the answer and the candidate image, in bits. Identical probes score
0; a clean, even split into n answers scores log2(n).

Only the selection step changes. Question generation, filtering, the BLIP-2 answer and the rewrite
stay PlugIR's. The agent never sees the target image or its rank, and probes are not shown to the
simulated user or scored.
"""
from collections import Counter
import random
import re

import torch
from torch.nn.functional import normalize

SELECTIONS = ("plugir_kl", "probe_split", "probe_vlm", "random")
VLM_PHOTOS = 20  # probe_vlm: how many of the current top results the agent's vision model looks at.
FILLER = re.compile(r"^(?:a|an|the|some|it is|it's|they are|there is|there are)\s+")
# Questions about meaning or opinion have no factual answer in a photo; their answers vary from guessing.
OPINION = re.compile(r"\b(symboli[sz]e|symbol|represent|might|mood|feel|feeling|suggest|why|think|emotion|atmosphere|meaning)\b", re.I)
UNSURE = ("unknown", "not sure", "cannot", "can't", "unclear", "not visible", "no idea")
MAX_ANSWERS = 3
VOTE_TEMPERATURE = 0.01  # Fixed before any pilot; a typical contrastive temperature for BLIP-style scores.
GUESS_SYSTEM = ("You help an image search system. Given a description of the image the user wants, "
                "descriptions of images currently in the search results, and a question the system may "
                "ask the user, list up to three different short answers the user could give. Each answer "
                "is 1-4 words. Write one answer per line with no numbering and nothing else.")


def parse_answers(text):
    """Turn Qwen's reply into up to three distinct short answers.
    Qwen sometimes lists answers on one line ("laptop, printer, keyboard"); split those on commas.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = lines[0].split(",")
    answers = []
    for line in lines:
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip(".").lower()
        line = re.sub(r"^(?:and|or)\s+", "", line).strip()
        if line and len(line.split()) <= 6 and line not in answers:
            answers.append(line)
    return answers[:MAX_ANSWERS]


def split_score(probe_vectors, window_vectors, temperature=VOTE_TEMPERATURE):
    """Mutual information (bits) between the answer and the candidate image under soft votes.
    probe_vectors [answers, d] and window_vectors [images, d] are normalized.
    """
    votes = torch.softmax(probe_vectors @ window_vectors.T / temperature, dim=0)  # [answers, images]
    share = votes.mean(1)
    entropy = lambda p: -(p * p.clamp_min(1e-12).log2()).sum(0)
    return float(entropy(share) - entropy(votes).mean()), [round(float(s), 3) for s in share]


def bucket(answer):
    """Group answers that mean the same: lowercase, no punctuation or filler words; unsure answers together."""
    if any(word in answer.lower() for word in UNSURE):
        return "(unsure)"
    answer = re.sub(r"[^\w\s]", " ", answer.lower()).strip()
    answer = FILLER.sub("", " ".join(answer.split()))
    return answer or "(unsure)"


def answer_entropy(answers):
    """How varied the answers are, in bits: 0 when all agree; log2(n) when n answers are equally common."""
    counts = torch.tensor(list(Counter(bucket(a) for a in answers).values()), dtype=torch.float)
    share = counts / counts.sum()
    return float(-(share * share.log2()).sum())


def make_selector(selection, namespace, questioner, case_seed, vlm=None, resolve_image=None):
    """Return a drop-in replacement for upstream select_question, recording each decision."""
    original = namespace["select_question"]

    def guess_answers(description, candidates, question):
        """One Qwen call per candidate question; deterministic (temperature 0)."""
        messages = [{"role": "system", "content": GUESS_SYSTEM},
                    {"role": "user", "content": f"Description: {description}\n\nImages in the current results:\n"
                                                f"{candidates}\n\nQuestion: {question}\n\nAnswers:"}]
        response = questioner.create(model="local", messages=messages, temperature=0, max_tokens=40, n=1)
        return parse_answers(response["choices"][0]["message"]["content"])

    def select_question(caption_recon="", questions=[], cossim_prev=None, k=10, img_embs=None,
                        threshold=500, round=1):
        unique = list(dict.fromkeys(questions))
        if selection == "random":
            chosen = random.Random(f"{case_seed}-{round}").choice(unique)
            namespace["selection_log"] = {"selection": selection, "questions": unique, "chosen": chosen}
            return chosen
        if selection == "probe_vlm":
            # Ask the agent's own vision model each question about the current top results (never told
            # which, if any, is the target) and ask the user the question whose answers vary most.
            factual = [q for q in unique if not OPINION.search(q)]
            if not factual:
                chosen = original(caption_recon=caption_recon, questions=questions, cossim_prev=cossim_prev, k=k,
                                  img_embs=img_embs, threshold=threshold, round=round)
                namespace["selection_log"] = {"selection": selection, "candidates": [], "skipped_opinion": unique,
                                              "chosen": chosen, "fallback_to_plugir": True}
                return chosen
            top = cossim_prev.argsort(descending=True)[:VLM_PHOTOS].tolist()
            paths = [resolve_image(namespace["search_space"][i]) for i in top]
            questioner.park()
            rows = []
            try:
                for question in factual:
                    answers = vlm.answer(paths, question)
                    rows.append({"question": question, "score": answer_entropy(answers),
                                 "answers": dict(Counter(bucket(a) for a in answers))})
            finally:
                vlm.park()
            best = max(rows, key=lambda row: row["score"])
            fallback = best["score"] == 0
            chosen = original(caption_recon=caption_recon, questions=questions, cossim_prev=cossim_prev, k=k,
                              img_embs=img_embs, threshold=threshold, round=round) if fallback else best["question"]
            namespace["selection_log"] = {"selection": selection, "photos": VLM_PHOTOS, "candidates": rows,
                                          "skipped_opinion": [q for q in unique if q not in factual],
                                          "chosen": chosen, "fallback_to_plugir": fallback}
            return chosen
        size = int(threshold - (round - 1) * (threshold / 10.0))
        window = img_embs[cossim_prev.argsort()[-size:]]
        rows = []
        for question in unique:
            answers = guess_answers(caption_recon, namespace.get("prompt_fake", ""), question)
            row = {"question": question, "answers": answers, "score": 0.0, "shares": []}
            if len(answers) >= 2:
                texts = [f"{caption_recon}, {question} {answer}" for answer in answers]
                probes = normalize(namespace["get_text_features"](texts), dim=-1)
                row["score"], row["shares"] = split_score(probes, window)
            rows.append(row)
        best = max(rows, key=lambda row: row["score"])
        if best["score"] > 0:
            chosen, fallback = best["question"], False
        else:  # No question produced two usable answers: fall back to PlugIR's own choice.
            chosen, fallback = original(caption_recon=caption_recon, questions=questions, cossim_prev=cossim_prev,
                                        k=k, img_embs=img_embs, threshold=threshold, round=round), True
        namespace["selection_log"] = {"selection": selection, "window": size, "candidates": rows,
                                      "chosen": chosen, "fallback_to_plugir": fallback}
        return chosen

    return select_question
