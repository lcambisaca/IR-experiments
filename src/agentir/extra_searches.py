"""Matan's extra-search strategies and ways to merge their results (Nov 2 email).

Phase 1, extra searches: each round, the agent turns the current description into variant searches.
    details  same subject, a different kind of detail highlighted in each variant
    focus    same scene, a different main subject in each variant
    split    short single-focus searches, one part of the scene each
The original description is always kept as search 0, so k = 1 + N_VARIANTS = 6 searches.

Phase 2, merging: pool the top 10 of every search (up to 10 * k photos) and rank the pool by
    max   each photo's best similarity across the k searches (Matan's "NN score")
    mean  each photo's average similarity across the k searches (Matan's "mean score")
The merged top 10 is what the user would see, so it is compared like-for-like with the original top 10.

The agent sees only the current description, never the target photo, its name or its rank.
"""
import json
import re

import torch

N_VARIANTS = 5
TOP = 10
STRATEGIES = ("details", "focus", "split")
MERGES = ("max", "mean")
DETAIL_KEYS = ("colors", "setting", "action", "count", "background")

PROMPTS = {
    "details": (
        "You help an image search engine. Given a photo description, imagine the photo and write 5 new, "
        "fuller photo captions of it, each keeping the main subject and adding one kind of detail: "
        "'colors' adds colors and materials, 'setting' adds the place, 'action' adds what is happening or "
        "the state of things, 'count' adds how many things there are and where, 'background' adds what is "
        "behind. Add only details that are very likely true; never change or contradict a fact. Write full "
        "captions, not labels. At most 25 words each. Reply with only a JSON object with the keys "
        '"colors", "setting", "action", "count", "background".'
    ),
    "focus": (
        "You help an image search engine. Given a photo description, imagine the photo. First list 5 "
        "different things visible in it (objects, people, animals, parts of things, the place, the "
        "background) in 'subjects'. Then, in 'queries', write one new photo caption per subject that starts "
        "with that subject, written like a normal third-person photo caption. Example: 'a ginger cat "
        "sleeping in a busy street, people in the background' -> 'a man and woman standing in a busy street "
        "with a ginger cat in front of them'. Keep every fact and never contradict one. Each caption must "
        "differ from the original description. At most 25 words each. Reply with only a JSON object with "
        'the keys "subjects" (5 strings) and "queries" (5 strings).'
    ),
    "split": (
        "You turn an image description into search queries for an image search engine. Write exactly 5 "
        "short queries. Each query describes ONE part of the scene (one object, person or animal, or one "
        "relationship between two things) as a short photo caption of 4 to 12 words. Example: 'a ginger cat "
        "sleeping in a busy street, people in the background' -> 'a ginger cat in the street', 'people in a "
        "busy street', 'a cat sleeping on the ground'. Together the queries cover the whole description. "
        "Use only facts from the description and add nothing new. If there are fewer than 5 parts, combine "
        'two parts in the extra queries. Reply with only a JSON object with the key "queries" (5 strings).'
    ),
}


def messages(strategy, description):
    return [{"role": "system", "content": PROMPTS[strategy]},
            {"role": "user", "content": f'Photo description: "{description}"'}]


def parse_variants(strategy, text):
    """Pull up to N_VARIANTS distinct non-empty searches from Qwen's JSON reply; [] if unreadable."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return []
    items = [data.get(key) for key in DETAIL_KEYS] if strategy == "details" else data.get("queries") or []
    items = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    return list(dict.fromkeys(items))[:N_VARIANTS]


def round_outcome(scores, target, top=TOP):
    """Where the target lands for one round, given scores [k, images] with row 0 = original description.
    Returns ranks (1 = best; None = not in the pooled candidates) for the original search, each merge,
    plus Matan's "winning version" (target in any search's top 10) and its fair control.
    """
    k = scores.shape[0]
    original_rank = int((scores[0] > scores[0, target]).sum()) + 1
    pool = torch.unique(scores.topk(top, dim=1).indices.flatten())
    outcome = {"original": original_rank, "k": k, "pool_size": len(pool),
               "any_top10": bool((pool == target).any()),
               # Fair control: the original search allowed to show as many photos as the pool holds.
               "control_same_count": original_rank <= len(pool)}
    for merge in MERGES:
        pooled = scores[:, pool]
        value = pooled.max(0).values if merge == "max" else pooled.mean(0)
        order = pool[value.argsort(descending=True)]
        hit = (order == target).nonzero()
        outcome[merge] = int(hit[0, 0]) + 1 if len(hit) else None
    return outcome
