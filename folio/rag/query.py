"""Query understanding: question -> section filter + search rewrite + HyDE + sub-questions.

One call to the fast model when a key is set; otherwise rules over the document's own outline
and acronym table. Two lessons from the guide are encoded in the prompt and the validation:
  * a section is set only when the question names it (or a place inside it) - never inferred
    from the topic, because a wrong filter hides the answer
  * any section name the model returns that is not in the outline is dropped
The filter is applied softly downstream (filtered and unfiltered results are fused).
"""
from __future__ import annotations

import json
import re

from .llm import chat, text_of

PROMPT = """You turn a question about the document "{title}" into search instructions.
Passages in the document are tagged with the section they live in. Routable sections:
{outline}

Reply with JSON only:
{{"sections": [<section names copied exactly from the list, or empty>],
  "search_query": "<the question rewritten for search>",
  "hypothetical_answer": "<one sentence the document might contain that answers it; invent no numbers, use X>",
  "sub_questions": [<2-3 simpler questions if the question has several parts, else empty>]}}

Rules:
- Add a section only when the question names it, or names a place, entity or unit that belongs to it
  (for example a country inside a region section). If the question is global, multi-section, or only
  about a topic, return no sections. A wrong section hides the answer; a missing one costs little.
- search_query: spell out abbreviations, use the document's own vocabulary, keep every number and year.
Question: {question}"""


def _outline_names(meta, max_items=70):
    """Routable sections: top two levels, and only names that are unambiguous. A heading such as
    "Outlook" that recurs under every region cannot identify a place, so it is never a filter."""
    outline = meta.get("outline", [])
    count = {}
    for o in outline:
        count[o["name"].lower()] = count.get(o["name"].lower(), 0) + 1
    items = [o for o in outline if o["depth"] <= 1 and count[o["name"].lower()] == 1
             and (len(o["name"].split()) >= 2 or re.match(r"(chapter|box|part|annex)\b", o["name"], re.I))]
    return [o["name"] for o in items[:max_items]]


def rules(question: str, meta: dict) -> dict:
    q = question.lower()
    names = _outline_names(meta)
    aliases = meta.get("aliases", {})
    expanded = q
    for short, long in aliases.items():
        if re.search(rf"\b{re.escape(short)}\b", q):
            expanded += " " + long
    secs = []
    for n in names:
        m = re.match(r"(chapter|box|part|annex)\s+(\d+(\.\d+)*)", n.lower())
        if m:
            # numbered units are routed only when the question says "Chapter 3" / "Box 1.1";
            # their titles are topics ("Global Outlook") and would filter on topic
            if re.search(rf"\b{m.group(1)}\s+{re.escape(m.group(2))}\b", q):
                secs.append(n)
            continue
        if len(n) >= 4 and re.search(rf"(?<![a-z]){re.escape(n.lower())}(?![a-z])", expanded):
            secs.append(n)
    # the most specific match wins ("South Asia" over "Chapter 2: Regional Outlooks")
    secs = sorted(set(secs), key=lambda s: -len(s))[:2]
    return {"sections": secs, "search_query": question, "hypothetical_answer": "",
            "sub_questions": [], "method": "rules"}


def understand(question: str, meta: dict, routing: bool) -> dict:
    llm = chat("fast")
    base = rules(question, meta)
    if not routing:
        base["sections"] = []
    if llm is None:
        return base
    names = _outline_names(meta)
    try:
        reply = text_of(llm.invoke(PROMPT.format(title=meta.get("title", "the document"),
                                                 outline="\n".join(f"- {n}" for n in names) if routing else "(none: do not set sections)",
                                                 question=question)))
        data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
    except Exception as exc:
        base["error"] = str(exc)[:200]
        return base
    secs = [s for s in data.get("sections") or [] if s in names] if routing else []
    subs = [s for s in data.get("sub_questions") or [] if isinstance(s, str) and len(s) > 8][:3]
    return {"sections": secs[:2], "search_query": data.get("search_query") or question,
            "hypothetical_answer": data.get("hypothetical_answer") or "", "sub_questions": subs if len(subs) >= 2 else [],
            "method": "llm"}
