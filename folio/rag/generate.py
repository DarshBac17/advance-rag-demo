"""Grounded answers: numbered passages in, cited answer (or an explicit refusal) out,
then a deterministic check that every citation exists and every number is in a cited passage."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import chat, text_of

REFUSAL = "I could not find this in the document."

SYSTEM = f"""You answer questions about documents using ONLY the numbered context passages.
- Use only facts stated in the passages. Never add outside knowledge.
- Quote numbers, names, units and dates exactly as written.
- Tables are written as "row: column = value". Read the value where the asked row and column meet.
- Lines starting with [Note n] are the document's own footnotes for that passage. Use them when the
  question asks how something is measured, defined, scoped or qualified.
- Chart passages list labels and printed values. You may do simple arithmetic on printed values; show it.
- Cite the passages you used with square brackets only, like [1] or [2][3]. Plain text, no bold, no headings.
- If passages give different figures, report each one with its context.
- If the passages do not contain the answer, reply exactly: {REFUSAL}
- Be brief: one to four sentences."""


def passage_header(i: int, p: dict) -> str:
    pages = p["pages"]
    pg = f"p. {pages[0]}" if len(pages) == 1 else f"pp. {pages[0]}–{pages[-1]}"
    path = " > ".join(p["section_path"][-3:]) or "—"
    return f"[{i}] {p['doc_title']} · {pg} · {path} · {p['kind']}"


def build_context(passages: list[dict]) -> str:
    return "\n\n".join(f"{passage_header(i, p)}\n{p['text']}" for i, p in enumerate(passages, 1))


_FULLWIDTH_CITE = re.compile(r"【\s*(\d+)[^】]*】")


def clean_answer(text: str) -> str:
    """gpt-oss cites as 【1】 or 【1†L9-L11】 and bolds with **; the prompt asks for [1] and plain text."""
    text = _FULLWIDTH_CITE.sub(r"[\1]", text)
    return text.replace("**", "").replace(" ", " ")


def generate(question: str, passages: list[dict], strict: bool = False) -> tuple[str, str]:
    """Returns (answer, method)."""
    if not passages:
        return REFUSAL, "no-context"
    llm = chat("answer")
    if llm is None:
        return extractive(question, passages), "extractive"
    extra = ("\nYour previous draft cited numbers that are not in the passages. "
             "Use only numbers printed in the passages.") if strict else ""
    msgs = [SystemMessage(SYSTEM + extra),
            HumanMessage(f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}")]
    try:
        return clean_answer(text_of(llm.invoke(msgs))).strip(), "llm"
    except Exception as exc:
        return extractive(question, passages) + f"\n\n(LLM unavailable: {str(exc)[:120]})", "extractive"


def extractive(question: str, passages: list[dict]) -> str:
    """No-LLM fallback: the sentences that overlap the question most, with citations.
    Refuses when the question names a year or figure that no passage contains (e.g. FY 2020)."""
    pool = " ".join(p["text"] for p in passages)
    for n in re.findall(r"\b(?:19|20)\d\d\b|\b\d[\d,.]*\d\b", question):
        if n not in pool:
            return REFUSAL
    qt = set(re.findall(r"[a-z0-9.]{3,}", question.lower()))
    best = []
    for i, p in enumerate(passages, 1):
        for s in re.split(r"(?<=[.!?])\s+|\n", p["text"]):
            if s.startswith("[Note") or s.startswith("Figure text") or s.startswith("Visual description"):
                continue
            st = set(re.findall(r"[a-z0-9.]{3,}", s.lower()))
            score = len(qt & st) + 0.5 * bool(re.search(r"\d", s))
            if len(s) > 25:
                best.append((score, -i, s.strip(), i))
    best.sort(reverse=True)
    picked = best[:2]
    if not picked or picked[0][0] < 2:
        return REFUSAL
    return " ".join(f"{s} [{i}]" for _, _, s, i in picked)


# --------------------------------------------------------------------------- verification
_NUM = re.compile(r"(?<![\w.])[-–−]?\$?\d[\d,]*(?:\.\d+)?%?")


def _norm_num(x: str) -> str:
    return x.replace(",", "").replace("−", "-").replace("–", "-").lstrip("$").rstrip("%").lstrip("-")


def verify(answer: str, passages: list[dict]) -> dict:
    refusal = REFUSAL.lower().rstrip(".") in answer.lower()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    bad_cites = [n for n in cited if n < 1 or n > len(passages)]
    body = re.sub(r"\[\d+\]", " ", answer)
    nums = {_norm_num(m.group(0)) for m in _NUM.finditer(body)}
    nums = {n for n in nums if n and not (n.isdigit() and int(n) <= 3)}       # "two", list counters
    pool_ids = [n for n in cited if 1 <= n <= len(passages)] or list(range(1, len(passages) + 1))
    pool = " ".join(passages[n - 1]["text"] for n in pool_ids)
    pool_nums = {_norm_num(m.group(0)) for m in _NUM.finditer(pool)}
    unsupported = sorted(n for n in nums if n not in pool_nums and n not in pool.replace(",", ""))
    grounded = (not refusal) and bool(cited) and not bad_cites and not unsupported
    return {"refusal": refusal, "cited": cited, "bad_citations": bad_cites,
            "numbers": sorted(nums), "unsupported_numbers": unsupported,
            "grounded": grounded,
            "verdict": "refused" if refusal else ("grounded" if grounded else
                                                  ("uncited" if not cited else "check numbers"))}
