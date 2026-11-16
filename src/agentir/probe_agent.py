"""Answer-entropy question selection (agent v2): ask the question whose answers vary most.

PlugIR picks the candidate question whose wording changes the candidate scores least (minimum KL).
This agent instead simulates the answer before asking:
  1. its own vision model (Qwen2.5-VL-3B, never told which photo, if any, is the target) answers each
     candidate question on the current top VLM_PHOTOS results;
  2. answers that mean the same are grouped (`bucket`), and their spread is measured as entropy in bits
     (`answer_entropy`): 0 when every photo gives the same answer, log2(n) for n equally common answers;
  3. the question with the highest entropy is asked, because the user's answer should rule out the most
     candidates.
Opinion questions ("what might this symbolize?") are skipped, since they have no factual answer. If no
question gets differing answers, the agent falls back to PlugIR's own choice.

Only the selection step changes. Question generation, filtering, the BLIP-2 answer and the rewrite stay
PlugIR's.
"""
from collections import Counter
import re

import torch

SELECTIONS = ("plugir_kl", "probe_vlm")
VLM_PHOTOS = 20  # How many of the current top results the agent's vision model looks at.
FILLER = re.compile(r"^(?:a|an|the|some|it is|it's|they are|there is|there are)\s+")
# Questions about meaning or opinion have no factual answer in a photo; their answers vary from guessing.
OPINION = re.compile(r"\b(symboli[sz]e|symbol|represent|might|mood|feel|feeling|suggest|why|think|emotion|atmosphere|meaning)\b", re.I)
UNSURE = ("unknown", "not sure", "cannot", "can't", "unclear", "not visible", "no idea")


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


def make_selector(selection, namespace, questioner, vlm, resolve_image):
    """Return a drop-in replacement for upstream select_question, recording each decision."""
    if selection != "probe_vlm":
        raise ValueError(f"Unknown selection {selection!r}.")
    original = namespace["select_question"]

    def select_question(caption_recon="", questions=[], cossim_prev=None, k=10, img_embs=None,
                        threshold=500, round=1):
        plugir_choice = lambda: original(caption_recon=caption_recon, questions=questions, cossim_prev=cossim_prev,
                                         k=k, img_embs=img_embs, threshold=threshold, round=round)
        unique = list(dict.fromkeys(questions))
        factual = [q for q in unique if not OPINION.search(q)]
        if not factual:
            chosen = plugir_choice()
            namespace["selection_log"] = {"selection": selection, "candidates": [], "skipped_opinion": unique,
                                          "chosen": chosen, "fallback_to_plugir": True}
            return chosen
        top = cossim_prev.argsort(descending=True)[:VLM_PHOTOS].tolist()
        paths = [resolve_image(namespace["search_space"][i]) for i in top]
        questioner.park()  # Free GPU memory for the vision model.
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
        chosen = plugir_choice() if fallback else best["question"]
        namespace["selection_log"] = {"selection": selection, "photos": VLM_PHOTOS, "candidates": rows,
                                      "skipped_opinion": [q for q in unique if q not in factual],
                                      "chosen": chosen, "fallback_to_plugir": fallback}
        return chosen

    return select_question
