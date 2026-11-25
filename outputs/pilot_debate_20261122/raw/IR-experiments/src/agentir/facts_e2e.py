"""End-to-end facts-only search inside the unchanged PlugIR loop.

PlugIR searches through one function, `get_text_features(text)`, at three places in every round:
  1. `search_imgs(caption_recon)`            the ranking (and the KL reference distribution),
  2. `get_related_captions(caption_recon)`   the candidate images whose captions seed new questions,
  3. `select_question`                       "caption_recon, <candidate question>" for minimum KL.
This module wraps that one function. When PlugIR asks for the description's vector it receives the
facts-only vector instead; for "description, question" it receives the facts-only vector with that question
added as one more caption-anchored query. Every other text (candidate captions) is encoded unchanged.

facts-only vector = normalize(mean(normalize(BLIP(caption)), normalize(BLIP("caption, fact_i")) ...))

Facts come from the same Qwen (one greedy call per answered round) with the prompt frozen in
fact_fusion.py; non-answers, echoes and negative answers give no fact. Qwen's rewritten description is
still produced and still used as TEXT (question prompts, filter, BLIP-2 answer prompt), never for search.
"""
import re

import torch
from torch.nn.functional import normalize

# Frozen copies of fact_fusion.py's prompt and cleaning (a test checks they are identical).
FACT_PROMPT = (
    "You turn one question and answer about a photo into a short visual fact for an image search engine. "
    "Reply with one short noun phrase (3 to 8 words) that names the object together with the new detail. "
    "Example: Question: What color is the wetsuit? Answer: white -> a white wetsuit. "
    "Reply NONE if the answer does not answer the question, only repeats the description, says something "
    "is absent or not visible, or is unsure. Reply with the phrase or NONE only.")
NEGATION = re.compile(r"\b(no|not|none|nothing|without|neither|nor|absent|n't)\b", re.I)
FACT_TOKENS = 24


def clean_fact(text):
    text = text.strip().strip('"').splitlines()[0].strip().strip('"').rstrip(".").strip() if text.strip() else ""
    return None if not text or text.upper().startswith("NONE") or NEGATION.search(text) else text


def install(namespace, questioner, caption, facts):
    """Wrap PlugIR's text encoder in `namespace`. `facts` is the live list of confirmed facts (mutated
    by `extract_fact`). Returns (extract_fact, rewrite_features)."""
    original = namespace["get_text_features"]
    cache = {}

    def unit(text):
        if text not in cache:
            cache[text] = normalize(original(text), dim=-1).reshape(-1)
        return cache[text]

    def facts_vector(extra=None):
        texts = [caption] + [f"{caption}, {f}" for f in facts] + ([f"{caption}, {extra}"] if extra else [])
        return normalize(torch.stack([unit(t) for t in texts]).mean(0), dim=-1)[None]

    def get_text_features(text):
        description = namespace.get("caption_recon")
        if isinstance(text, str) and description is not None:
            if text == description:
                return facts_vector()
            if text.startswith(description + ", "):  # select_question's "description, question"
                return facts_vector(extra=text[len(description) + 2:])
        return original(text)

    def extract_fact(question, answer):
        """One greedy Qwen call; the transport records this function's name as the stage."""
        response = questioner.create(model="local", temperature=0, max_tokens=FACT_TOKENS, n=1, messages=[
            {"role": "system", "content": FACT_PROMPT},
            {"role": "user", "content": f"Description: {caption}\nQuestion: {question}\nAnswer: {answer or ''}"}])
        fact = clean_fact(response["choices"][0]["message"]["content"])
        if fact:
            facts.append(fact)
        return fact

    namespace["get_text_features"] = get_text_features
    return extract_fact, original
