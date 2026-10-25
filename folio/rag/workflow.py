"""Online pipeline as a LangGraph state machine.

  understand -> retrieve -> graph_expand -> rerank -> assemble -> generate -> verify
                   ^                                                            |
                   +------------ reformulate  <-- (refused / ungrounded, once) -+--> finish

Every node appends to `trace`, which the UI shows step by step.
"""
from __future__ import annotations

import operator
import re
import time
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from .chunking import tokens
from .config import settings
from .generate import generate, verify, REFUSAL
from .graph import DocGraph, load_graph
from .pipeline import load_meta
from .query import understand
from .rerank import rerank
from .store import get_index


class RAGState(TypedDict, total=False):
    question: str
    doc_ids: list
    attempt: int
    analysis: dict
    hits: list
    passages: list
    answer: str
    method: str
    verification: dict
    attempts: list
    trace: Annotated[list, operator.add]


# --------------------------------------------------------------------------- resources
@lru_cache(maxsize=32)
def _graph(doc_id: str, stamp: float) -> DocGraph | None:
    f = get_index()._dir(doc_id) / "graph.json"
    d = get_index().load(doc_id, "advanced")
    if not f.exists() or not d:
        return None
    return DocGraph(load_graph(f), d["chunks"])


def graph_for(doc_id):
    f = get_index()._dir(doc_id) / "graph.json"
    return _graph(doc_id, f.stat().st_mtime if f.exists() else 0)


def meta_for(doc_id):
    return load_meta(doc_id) or {}


def _step(stage, text, t0, **extra):
    return [{"stage": stage, "text": text, "ms": int((time.time() - t0) * 1000), **extra}]


# --------------------------------------------------------------------------- nodes
def n_understand(s: RAGState):
    t0, analysis, notes = time.time(), {}, []
    for d in s["doc_ids"]:
        m = meta_for(d)
        a = understand(s["question"], m, routing=bool(m.get("strategy", {}).get("routing")))
        analysis[d] = a
        bits = []
        if a["sections"]:
            bits.append("section: " + " / ".join(a["sections"]))
        if a["search_query"] != s["question"]:
            bits.append(f'rewrite: "{a["search_query"]}"')
        if a["sub_questions"]:
            bits.append(f"{len(a['sub_questions'])} sub-questions")
        g = graph_for(d)
        ents = [f"{n} ({how})" if how != "named" else n for n, how in (g.query_entities(s["question"]) if g else [])][:5]
        if ents:
            bits.append("entities: " + ", ".join(ents))
        notes.append(f"{m.get('title', d)[:40]} — {a['method']}: " + ("; ".join(bits) or "no filter, search as asked"))
    return {"analysis": analysis, "attempt": s.get("attempt", 0), "trace": _step("Understand", " | ".join(notes), t0)}


def n_retrieve(s: RAGState):
    t0, idx, attempt, pool = time.time(), get_index(), s.get("attempt", 0), settings.pool
    all_hits, notes = [], []
    for d in s["doc_ids"]:
        a, g = s["analysis"][d], graph_for(d)
        if attempt == 0:
            queries, sections = [a["search_query"]] + a.get("sub_questions", []), a["sections"]
        else:   # corrective pass: original wording + hypothetical answer, no section filter
            queries, sections = [s["question"]] + ([a["hypothetical_answer"]] if a.get("hypothetical_answer") else []), []
        lists, weights = {}, {}
        for i, q in enumerate(queries):
            vec = idx.emb.embed_query(q)
            bq = g.expand_query(q) if g else q
            lists[f"dense{i}"] = [(x, c) for x, c, _ in idx.dense(q, [d], "advanced", None, pool, qvec=vec)]
            lists[f"bm25_{i}"] = [(x, c) for x, c, _ in idx.bm25(bq, d, "advanced", None, pool)]
            if sections:
                lists[f"dense{i}@section"] = [(x, c) for x, c, _ in idx.dense(q, [d], "advanced", sections, pool, qvec=vec)]
                lists[f"bm25_{i}@section"] = [(x, c) for x, c, _ in idx.bm25(bq, d, "advanced", sections, pool)]
                weights[f"dense{i}@section"] = weights[f"bm25_{i}@section"] = 2.0
        if g:
            eh = g.entity_hits(s["question"], sections or None, pool)
            if eh:
                lists["graph"] = [(d, c) for c, _ in eh]
                weights["graph"] = 0.8
        hits = idx.fuse(lists, "advanced", weights)
        # kind prior: in prose documents, chart labels and photo captions are fragments that match
        # many queries and answer few (the guide's "fragments of tables" failure); demote them
        # unless the question is about a figure/chart/table
        prof = meta_for(d).get("profile", "narrative")
        asks_visual = re.search(r"\b(figure|chart|graph|table|caption|photo|image)\b", s["question"], re.I)
        if prof != "visual" and not asks_visual:
            for h in hits:
                if h["chunk"]["kind"] in ("figure", "caption"):
                    h["score"] = round(h["score"] * 0.55, 5)
                    h["demoted"] = True
            hits.sort(key=lambda h: -h["score"])
        hits = hits[:pool]
        all_hits += hits
        notes.append(f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} × dense + BM25"
                     + (" (+ section-filtered copies)" if sections else "")
                     + (f" + graph list ({len(lists['graph'])})" if "graph" in lists else "")
                     + f" → RRF shortlist of {len(hits)}"
                     + (f"; {sum(1 for h in hits if h.get('demoted'))} chart-label/caption chunks demoted"
                        if any(h.get("demoted") for h in hits) else ""))
    all_hits.sort(key=lambda h: -h["score"])
    for h in all_hits:
        h["via"] = "search"
    label = "Retrieve" if attempt == 0 else "Retrieve (retry)"
    return {"hits": all_hits, "trace": _step(label, " | ".join(notes), t0)}


def n_graph_expand(s: RAGState):
    t0, hits, added = time.time(), s["hits"], {"previous": 0, "next": 0, "same page": 0}
    for h in hits[:8]:
        g = graph_for(h["doc_id"])
        prof = meta_for(h["doc_id"]).get("profile", "narrative")
        if not g:
            continue
        comps = []
        for cid, rel in g.companions(h["chunk"]["id"], prof):
            c = g.chunks.get(cid)
            if c:
                comps.append({"chunk": c, "rel": rel})
                added["same page" if rel.startswith("same page") else rel] += 1
        h["companions"] = comps
    txt = (f"{added['same page']} same-page tables/charts, "
           f"{added['previous'] + added['next']} neighbouring passages linked for the top hits")
    return {"hits": hits, "trace": _step("Graph", txt, t0)}


def n_rerank(s: RAGState):
    t0 = time.time()
    hits = s["hits"]
    profiles = {meta_for(d).get("profile") for d in s["doc_ids"]}
    want = all(meta_for(d).get("strategy", {}).get("rerank") for d in s["doc_ids"])
    keep = settings.top_k
    if not want:
        return {"hits": hits[:keep], "trace": _step(
            "Rerank", "Skipped: the cross-encoder is trained on prose and under-rates table/chart chunks", t0)}
    top, used = rerank(s["question"], hits[:20], keep)
    msg = (f"Cross-encoder reread the top {min(20, len(hits))}, kept {len(top)}" if used else
           "Reranker model unavailable here; kept fused order")
    return {"hits": top, "trace": _step("Rerank", msg, t0)}


def _dedupe_join(texts: list[str]) -> str:
    seen, out = set(), []
    for t in texts:
        for sent in re.split(r"(?<=[.!?])\s+", t):
            k = sent.strip()[:80]
            if k and k not in seen:
                seen.add(k)
                out.append(sent.strip())
    return " ".join(out)


def n_assemble(s: RAGState):
    t0, idx = time.time(), get_index()
    passages, used_parents, budget, notes = [], set(), settings.max_context_tokens, []
    swapped = 0
    for h in s["hits"]:
        c, d = h["chunk"], h["doc_id"]
        m = meta_for(d)
        prof, title = m.get("profile", "narrative"), m.get("title", d)
        base = {"doc_id": d, "doc_title": title, "via": h.get("via", "search"), "ranks": h.get("ranks", {}),
                "rerank": h.get("rerank"), "score": h.get("score")}
        comps = h.get("companions", [])
        if prof == "long_structured" and c.get("parent_id"):
            pid = c["parent_id"]
            if (d, pid) in used_parents:
                continue
            par = idx.load(d, "advanced")["parents"].get(pid)
            if par:
                used_parents.add((d, pid))
                swapped += 1
                p = {**base, "chunk_ids": [c["id"]], "text": par["text"], "pages": par["pages"],
                     "section_path": par["section_path"], "kind": "section"}
            else:
                p = {**base, "chunk_ids": [c["id"]], "text": c["text"], "pages": c["pages"],
                     "section_path": c["section_path"], "kind": c["kind"]}
        elif prof == "narrative" and c["kind"] == "text":
            prev = [x["chunk"] for x in comps if x["rel"] == "previous"]
            nxt = [x["chunk"] for x in comps if x["rel"] == "next"]
            win = prev + [c] + nxt
            p = {**base, "chunk_ids": [x["id"] for x in win], "text": _dedupe_join([x["text"] for x in win]),
                 "pages": sorted({pg for x in win for pg in x["pages"]}), "section_path": c["section_path"],
                 "kind": "text"}
        else:
            p = {**base, "chunk_ids": [c["id"]], "text": c["text"], "pages": c["pages"],
                 "section_path": c["section_path"], "kind": c["kind"]}
        if any(set(p["chunk_ids"]) & set(q["chunk_ids"]) and p["doc_id"] == q["doc_id"] for q in passages):
            continue
        if tokens(p["text"]) > budget and passages:
            break
        budget -= tokens(p["text"])
        passages.append(p)
        # visual documents: the chart/table that sits on the same page travels with the hit
        if prof == "visual":
            for x in comps:
                if x["rel"].startswith("same page") and budget > 0 and len(passages) < settings.top_k + 3:
                    cc = x["chunk"]
                    if any(cc["id"] in q["chunk_ids"] and q["doc_id"] == d for q in passages):
                        continue
                    budget -= tokens(cc["text"])
                    passages.append({**base, "via": f"graph: {x['rel']}", "chunk_ids": [cc["id"]], "text": cc["text"],
                                     "pages": cc["pages"], "section_path": cc["section_path"], "kind": cc["kind"]})
    if swapped:
        notes.append(f"swapped small chunks for {swapped} parent sections")
    wins = sum(1 for p in passages if len(p["chunk_ids"]) > 1)
    if wins:
        notes.append(f"{wins} passages widened with their neighbours")
    extra = sum(1 for p in passages if p["via"].startswith("graph"))
    if extra:
        notes.append(f"{extra} same-page tables/charts added by the graph")
    used = settings.max_context_tokens - budget
    return {"passages": passages, "trace": _step(
        "Assemble", f"{len(passages)} numbered passages, ~{used} tokens" + ("; " + "; ".join(notes) if notes else ""), t0)}


def n_generate(s: RAGState):
    t0 = time.time()
    strict = bool(s.get("attempts")) and bool(s["attempts"][-1]["verification"]["unsupported_numbers"])
    ans, method = generate(s["question"], s["passages"], strict=strict)
    msg = {"llm": "Grounded prompt: cite passages or refuse", "extractive": "No LLM key: extractive answer from top passages",
           "no-context": "Nothing retrieved: refused"}[method]
    return {"answer": ans, "method": method, "trace": _step("Generate", msg, t0)}


def n_verify(s: RAGState):
    t0 = time.time()
    v = verify(s["answer"], s["passages"])
    attempts = list(s.get("attempts", [])) + [{"answer": s["answer"], "passages": s["passages"],
                                               "verification": v, "method": s.get("method")}]
    parts = [f"verdict: {v['verdict']}", f"cites {v['cited'] or 'none'}"]
    if v["unsupported_numbers"]:
        parts.append("numbers not in cited passages: " + ", ".join(v["unsupported_numbers"][:5]))
    return {"verification": v, "attempts": attempts, "trace": _step("Verify", "; ".join(parts), t0)}


def route_after_verify(s: RAGState):
    v = s["verification"]
    if s.get("attempt", 0) == 0 and s.get("method") != "extractive" and (v["refusal"] or not v["grounded"]):
        return "reformulate"
    return "finish"


def n_reformulate(s: RAGState):
    return {"attempt": 1, "trace": _step("Correct", "Answer was not grounded: retrying once with the original "
                                         "wording plus a hypothetical answer, section filter dropped", time.time())}


def n_finish(s: RAGState):
    def rank(a):
        v = a["verification"]
        return (v["grounded"], not v["refusal"], -len(v["unsupported_numbers"]), len(v["cited"]))
    best = max(s["attempts"], key=rank)
    return {"answer": best["answer"], "passages": best["passages"], "verification": best["verification"],
            "method": best["method"]}


@lru_cache(maxsize=1)
def compiled():
    g = StateGraph(RAGState)
    for name, fn in [("understand", n_understand), ("retrieve", n_retrieve), ("graph_expand", n_graph_expand),
                     ("rerank", n_rerank), ("assemble", n_assemble), ("generate", n_generate),
                     ("verify", n_verify), ("reformulate", n_reformulate), ("finish", n_finish)]:
        g.add_node(name, fn)
    g.set_entry_point("understand")
    g.add_edge("understand", "retrieve")
    g.add_edge("retrieve", "graph_expand")
    g.add_edge("graph_expand", "rerank")
    g.add_edge("rerank", "assemble")
    g.add_edge("assemble", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"reformulate": "reformulate", "finish": "finish"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("finish", END)
    return g.compile()


# --------------------------------------------------------------------------- public API
def _public_passages(passages):
    return [{k: p.get(k) for k in ("doc_id", "doc_title", "chunk_ids", "text", "pages", "section_path", "kind",
                                    "via", "ranks", "rerank", "score")} for p in passages]


def ask_advanced(question: str, doc_ids: list[str]) -> dict:
    t0 = time.time()
    out = compiled().invoke({"question": question, "doc_ids": doc_ids, "attempt": 0, "attempts": [], "trace": []})
    graph_view = {"nodes": [], "edges": []}
    for d in doc_ids:
        g = graph_for(d)
        if g:
            ids = [cid for p in out["passages"] if p["doc_id"] == d for cid in p["chunk_ids"]]
            sub = g.subgraph(ids, question)
            graph_view["nodes"] += sub["nodes"]
            graph_view["edges"] += sub["edges"]
    return {"pipeline": "advanced", "answer": out["answer"], "method": out["method"],
            "verification": out["verification"], "passages": _public_passages(out["passages"]),
            "trace": out["trace"], "analysis": out.get("analysis", {}), "graph": graph_view,
            "attempts": len(out.get("attempts", [])), "ms": int((time.time() - t0) * 1000)}


def ask_naive(question: str, doc_ids: list[str]) -> dict:
    """The guide's baseline: default extraction, 1,000-char chunks, dense top-5, same prompt."""
    t0, idx = time.time(), get_index()
    hits = idx.dense(question, doc_ids, "naive", None, settings.top_k)
    passages = []
    for d, cid, score in hits:
        c = idx.chunk(d, "naive", cid)
        if c:
            passages.append({"doc_id": d, "doc_title": meta_for(d).get("title", d), "chunk_ids": [cid],
                             "text": c["text"], "pages": c["pages"], "section_path": [], "kind": "naive",
                             "via": "dense", "ranks": {"dense": len(passages) + 1}, "rerank": None,
                             "score": round(score, 4)})
    t1 = time.time()
    ans, method = generate(question, passages)
    v = verify(ans, passages)
    trace = [{"stage": "Retrieve", "text": f"Dense search over 1,000-character chunks: top {len(passages)}",
              "ms": int((t1 - t0) * 1000)},
             {"stage": "Generate", "text": "Same grounded prompt", "ms": int((time.time() - t1) * 1000)},
             {"stage": "Verify", "text": f"verdict: {v['verdict']}", "ms": 0}]
    return {"pipeline": "naive", "answer": ans, "method": method, "verification": v,
            "passages": _public_passages(passages), "trace": trace, "analysis": {}, "graph": {"nodes": [], "edges": []},
            "attempts": 1, "ms": int((time.time() - t0) * 1000)}


def ask(question: str, doc_ids: list[str], mode: str = "advanced") -> dict:
    if mode == "naive":
        return ask_naive(question, doc_ids)
    if mode == "compare":
        return {"advanced": ask_advanced(question, doc_ids), "naive": ask_naive(question, doc_ids)}
    return ask_advanced(question, doc_ids)
