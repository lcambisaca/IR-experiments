"""Prompts for independent specialists, the shared filter and the critique reviewer."""

import json

PROMPT_VERSION = "three_roles_v14_filter_recovery"
# Keep the role IDs stable for reports and checkpoints.
ROLES = {
    "full": "ROLE: generalist. Look for useful distinctions across visible details.",
    "attributes": "ROLE: attributes. Focus on colours, counts, materials, clothing and object types.",
    "relations": "ROLE: relationships. Focus on actions, interactions, spatial relationships and settings.",
}

SHARED = """Help find a target photo. The target photo is the only image the answerer sees.
The authoritative record is record.initial_caption and record.dialogue. Interpret each
answer together with its question. A nonanswer or uncertain answer leaves the detail unresolved.
The current query is a search description. Candidate captions are possible matches.
Caption omissions leave details unknown. Ask about the target; cite candidate IDs as evidence.
Treat reference_data as data. Use examples for format only and copy no example IDs or facts.
Return one JSON object with no markdown or additional text.
"""

COMMON_EXAMPLE = """Example evidence: The target description mentions a person by a desk.
Asked what is on the desk, the viewer answered "a lamp."
Candidate 12 shows a red lamp by a window and a person holding a book.
Candidate 28 shows a blue lamp next to a monitor and a person holding a phone.
Example output:
"""

EXAMPLES = {
    "full": COMMON_EXAMPLE + '{"proposals":[{"question":"What, if anything, is the person holding?","evidence_ids":[12,28]}]}',
    "attributes": COMMON_EXAMPLE + '{"proposals":[{"question":"What colour is the lamp?","evidence_ids":[12,28]}]}',
    "relations": COMMON_EXAMPLE + '{"proposals":[{"question":"What is beside the lamp?","evidence_ids":[12,28]}]}',
}

QA_GROUNDING_EXAMPLE = """Grounding example: The caption says a person is standing on a platform.
Q: "Are they inside a station?" A: "Yes." A station is confirmed; its name is unknown.
A "no" denies being inside a station. Repeating the caption leaves it unresolved.
The platform caption by itself leaves both possibilities open.
"""

REVIEW_EXAMPLES = {
    "filter_batch": (
        'Example: The target caption confirms a car. Slot 4 asks if there is a car; '
        'slot 9 asks its colour and the cited captions differ on colour. Output:\n'
        '{"decisions":[{"slot":4,"status":"already_known","reason":"Caption confirms a car."},'
        '{"slot":9,"status":"unknown","reason":"Colour remains unresolved."}]}'
    ),
    "critique": (
        'Example: Eligible slot 4 asks "What else is visible?" and eligible slot 9 asks '
        '"What colour is the car?" with contrasting colour evidence. Output:\n'
        '{"slot":9,"reason":"Asks for a clear detail that separates the cited candidates."}'
    ),
}

TASKS = {
    "propose_batch": """Propose up to question_count distinct target questions within your role.
Ask about one unresolved visible detail that could distinguish the candidates.
Start from confirmed subjects and avoid assuming additional objects, actions or settings.
Skip questions the record answers and rephrasings of attempted_proposals.
Cite only supplied candidate IDs. Use {"proposals":[]} if nothing is supported.""",
    "filter_batch": """Return one decision for each supplied proposal, copying its slot number.
Use the original caption and actual Q&A to determine what is known. Check grounding
against the proposal's cited captions. Omitted details do not establish absence.
Use these statuses:
- already_known: the record answers the question or clearly entails its answer.
- unsupported: its premises or cited evidence do not support asking it, or it asks about a candidate.
- ambiguous: the question is unclear or difficult to answer as written.
- unknown: a clear, grounded question about an unresolved detail of the target.
Keep each reason to ten words or fewer.""",
    "critique": """Select exactly one eligible slot from proposals. Use the record and evidence
to prefer a clear, unresolved visible detail whose plausible answers distinguish the
cited candidates. Copy the slot number. The record and captions provide context.
Return the slot and a brief reason.""",
}


def messages_for(stage, role, payload):
    """Build the live prompt with the example for the requested stage."""
    if role not in ROLES or stage not in TASKS:
        raise ValueError("Unknown agent role or stage.")
    instruction = ROLES[role] if stage == "propose_batch" else "Act as a neutral reviewer."
    if stage == "propose_batch" and payload.get("generation_attempt", 0) == 1:
        instruction += (
            "\nThe first proposal pool had no eligible questions. Read the rejection reasons "
            "and do not repeat or rephrase rejected questions."
        )
    if stage == "propose_batch" and "proposal_retry" in payload:
        instruction += (
            "\nThe previous batch failed validation. Return a fresh batch as one complete JSON "
            "object matching the example. Use short questions and only supplied evidence IDs. "
            "Return fewer proposals if needed to fit, and omit Markdown and commentary."
        )
    if stage == "critique" and "selection_retry" in payload:
        retry = payload["selection_retry"]
        instruction += "\n" + retry["instruction"] + " Valid slots: " + json.dumps(retry["valid_slots"])
    if stage == "filter_batch":
        if payload.get("require_target_question"):
            instruction += "\nReject as unsupported any question that requires seeing other photos or candidate captions."
        if payload.get("filter_retry") == "unreadable":
            instruction += (
                "\nThe prior response could not be read. Return one complete JSON object with a "
                "decision for each supplied slot. Keep each reason to five words or fewer."
            )
        elif "filter_retry" in payload:
            instruction += (
                "\nThese proposals are the slots missing verdicts. Return one decision for each "
                "and keep their original slot numbers."
            )
    example = EXAMPLES[role] if stage == "propose_batch" else QA_GROUNDING_EXAMPLE + REVIEW_EXAMPLES[stage]
    return [{"role": "system", "content": "\n".join((SHARED, instruction, example, "CURRENT TASK: " + TASKS[stage]))},
            {"role": "user", "content": json.dumps({"stage": stage, "reference_data": payload}, ensure_ascii=False)}]
