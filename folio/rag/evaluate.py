"""Evaluation harness.

Retrieval: rank of the first passage *the LLM actually sees* that contains the evidence
(any of `evidence`, or all of `evidence_all`). hit@1/3/5 and MRR over answerable questions.
Answers: every answer group must be matched by one of its alternatives; refusal questions
are correct only when the pipeline refuses. Matching is normalised (case, spacing, dashes,
footnote markers, line-end hyphens) so formatting differences do not count as misses.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

from .config import settings
from .llm import status as llm_status

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text()) if QUESTIONS.exists() else []


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\[\^?\s*note\s*\d+\]|\[\^\d+\]", " ", s)
    s = s.replace("–", "-").replace("−", "-").replace("\u2011", "-").replace("•", " ")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)               # "per- cent" -> "percent"
    return re.sub(r"\s+", " ", s).strip()


def evidence_rank(q: dict, passages: list[dict]) -> int | None:
    for i, p in enumerate(passages, 1):
        t = norm(p["text"])
        if q.get("evidence_all"):
            if all(norm(e) in t for e in q["evidence_all"]):
                return i
        elif any(norm(e) in t for e in q.get("evidence", [])):
            return i
    return None


def answer_correct(q: dict, result: dict) -> bool:
    refused = result["verification"]["refusal"]
    if q.get("refusal"):
        return refused
    if refused:
        return False
    a = norm(result["answer"])
    return all(any(norm(alt) in a for alt in group) for group in q["answer"])


def metrics(rows: list[dict]) -> dict:
    ans = [r for r in rows if not r["refusal"]]
    ranks = [r["rank"] for r in ans]
    n = max(len(ranks), 1)
    return {"hit@1": sum(1 for r in ranks if r == 1) / n, "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
            "MRR": sum(1 / r for r in ranks if r) / n,
            "answers": sum(1 for r in rows if r["correct"]) / max(len(rows), 1), "n": len(rows)}


def run_eval(docs: dict, modes=("naive", "advanced"), progress=None) -> dict:
    """docs: {doc_id: filename}. Returns per-document metrics and per-question rows."""
    from .workflow import ask, meta_for
    qs = load_questions()
    pause = float(__import__("os").environ.get("EVAL_SLEEP", "1.0" if llm_status()["answer_llm"] else "0"))
    out = []
    for doc_id, filename in docs.items():
        mine = [q for q in qs if re.search(q["file_pattern"], filename, re.I)]
        if not mine:
            continue
        rows = {m: [] for m in modes}
        qrows = []
        for q in mine:
            qr = {"id": q["id"], "question": q["question"], "type": q["type"], "gold_pages": q["gold_pages"]}
            for m in modes:
                res = ask(q["question"], [doc_id], m)
                rank = None if q.get("refusal") else evidence_rank(q, res["passages"])
                ok = answer_correct(q, res)
                rows[m].append({"rank": rank, "correct": ok, "refusal": q.get("refusal", False)})
                qr[m] = {"rank": rank, "correct": ok, "answer": res["answer"][:400],
                         "verdict": res["verification"]["verdict"], "pages": [p["pages"] for p in res["passages"]]}
                if pause:
                    time.sleep(pause)
            qrows.append(qr)
            if progress:
                progress(q["id"], qr)
        meta = meta_for(doc_id)
        out.append({"doc_id": doc_id, "title": meta.get("title", filename), "filename": filename,
                    "profile": (meta.get("strategy") or {}).get("label", meta.get("profile")), "n": len(mine),
                    "metrics": {m: metrics(rows[m]) for m in modes}, "questions": qrows})
    note = None
    if not llm_status()["answer_llm"]:
        note = ("No GROQ_API_KEY: answers are extractive, so answer accuracy is a lower bound. "
                "Retrieval metrics are unaffected.")
    from .embeddings import embedder_info
    if not embedder_info()["semantic"]:
        note = (note + " " if note else "") + ("Embeddings are the offline hashed fallback; both pipelines "
                                               "score higher with the real embedding model.")
    return {"documents": out, "note": note, "llm": llm_status(), "embedder": embedder_info()}


def to_markdown(res: dict) -> str:
    lines = ["# Folio evaluation", ""]
    if res.get("note"):
        lines += [f"> {res['note']}", ""]
    lines += ["| Document | Questions | MRR basic | MRR Folio | hit@1 basic | hit@1 Folio | Answers basic | Answers Folio |",
              "|---|---|---|---|---|---|---|---|"]
    for d in res["documents"]:
        n, a = d["metrics"].get("naive", {}), d["metrics"].get("advanced", {})
        f = lambda x, k, p=False: "–" if not x else (f"{x[k]*100:.0f}%" if p else f"{x[k]:.2f}")
        lines.append(f"| {d['title']} | {d['n']} | {f(n,'MRR')} | {f(a,'MRR')} | {f(n,'hit@1',1)} | {f(a,'hit@1',1)} "
                     f"| {f(n,'answers',1)} | {f(a,'answers',1)} |")
    for d in res["documents"]:
        lines += ["", f"## {d['title']}", "", "| Question | Type | Basic rank | Folio rank | Basic ✓ | Folio ✓ |", "|---|---|---|---|---|---|"]
        for q in d["questions"]:
            g = lambda m, k: (q.get(m) or {}).get(k)
            mark = lambda v: "✓" if v else "✗"
            lines.append(f"| {q['question']} | {q['type']} | {g('naive','rank') or '–'} | {g('advanced','rank') or '–'} "
                         f"| {mark(g('naive','correct'))} | {mark(g('advanced','correct'))} |")
    return "\n".join(lines) + "\n"
