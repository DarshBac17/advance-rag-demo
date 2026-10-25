"""Offline pipeline: index one PDF. Runs once per document (in a background thread from Django)."""
from __future__ import annotations

import json
import re
import time
from collections import Counter, OrderedDict

from .chunking import build_chunks, naive_chunks, Chunk, with_notes
from .config import settings
from .embeddings import embedder_info
from .graph import build_graph, save_graph
from .ingest.parser import parse_pdf
from .ingest.profile import profile_document, STRATEGIES
from .ingest.vision import pick_pages, describe_pages
from .llm import status as llm_status
from .store import get_index


def _outline(chunks) -> list[dict]:
    """Section names a question can be routed to: the first two levels of the heading tree."""
    seen: OrderedDict = OrderedDict()
    for c in chunks:
        for depth, name in enumerate(c.section_path[:3]):
            key = (depth, name)
            if key not in seen:
                seen[key] = {"name": name, "depth": depth, "chunks": 0,
                             "parent": c.section_path[depth - 1] if depth else None}
            seen[key]["chunks"] += 1
    return [v for v in seen.values() if v["chunks"] >= 2 and len(v["name"]) <= 90]


def index_document(path: str, doc_id: str, profile: str | None = None, progress=None) -> dict:
    t0 = time.time()
    say = progress or (lambda stage, pct, msg="": None)
    idx = get_index()

    say("parsing", 5, "Reading layout, fonts, tables and footnotes")
    pd = parse_pdf(path)
    prof = profile_document(pd)
    chosen = profile if profile in STRATEGIES else prof["profile"]
    say("chunking", 30, f"{STRATEGIES[chosen]['label']} strategy")
    cs = build_chunks(pd, chosen)
    chunks: list[Chunk] = list(cs.chunks)

    vision_pages, vision_note = [], None
    if cs.strategy.get("vision"):
        pages = pick_pages(pd.page_info, settings.vision_max_pages)
        if not llm_status()["vision"]:
            vision_note = "Vision is off: set GOOGLE_API_KEY to describe charts. Chart labels are still indexed."
        else:
            say("vision", 40, f"Describing {len(pages)} chart pages")
            described = describe_pages(path, pages, lambda i, n: say("vision", 40 + int(15 * i / n),
                                                                        f"Chart page {i}/{n}"))
            page_path = {}
            for c in chunks:
                page_path.setdefault(c.page, c.section_path)
            for pg, txt in described:
                txt = re.sub(r"\[note (\d+)\]", r"[^\1]", txt)
                body, nums = with_notes(txt, pg, pd)
                label = f"Visual description of page {pg} (vision model)"
                sp = page_path.get(pg, [])
                chunks.append(Chunk(id=f"v{pg}", text=f"{label}\n{body}", embed_text=" > ".join(sp) + "\n" + body,
                                    kind="vision", page=pg, pages=[pg], section_path=sp, notes=nums,
                                    order=len(chunks)))
                vision_pages.append(pg)

    say("embedding", 58, f"Embedding {len(chunks)} chunks")
    idx.delete(doc_id)
    idx.add(doc_id, "advanced", chunks, cs.parents,
            progress=lambda i, n: say("embedding", 58 + int(25 * i / n), f"Embedded {i}/{n} chunks"))
    naive = []
    if settings.build_naive:
        naive = naive_chunks(path)
        say("baseline", 85, f"Embedding {len(naive)} naive baseline chunks")
        idx.add(doc_id, "naive", naive, {})

    say("graph", 93, "Building knowledge graph")
    G = build_graph(chunks, cs.parents, pd)
    d = settings.data_dir / "index" / doc_id
    save_graph(G, d / "graph.json")

    kinds = Counter(c.kind for c in chunks)
    meta = {
        "doc_id": doc_id, "title": pd.title, "pages": pd.n_pages, "profile": chosen,
        "detected_profile": prof["profile"], "reasons": prof["reasons"], "features": prof["features"],
        "strategy": {k: v for k, v in cs.strategy.items()}, "outline": _outline(chunks),
        "aliases": G.graph.get("aliases", {}),
        "stats": {"chunks": len(chunks), "parents": len(cs.parents), "naive_chunks": len(naive),
                  "kinds": dict(kinds), "tables": kinds.get("table", 0), "figures": kinds.get("figure", 0),
                  "vision": len(vision_pages), "notes": len(pd.notes), "skipped": cs.skipped,
                  "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
                  "entities": sum(1 for _, x in G.nodes(data=True) if x.get("type") == "entity"),
                  "ligature_fixes": pd.stats.get("ligature_fixes", 0), "seconds": round(time.time() - t0, 1)},
        "vision_pages": vision_pages, "vision_note": vision_note, "embedder": embedder_info(),
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    say("ready", 100, "Ready")
    return meta


def load_meta(doc_id: str) -> dict | None:
    f = settings.data_dir / "index" / doc_id / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None
