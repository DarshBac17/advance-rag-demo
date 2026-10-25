"""Choose a RAG strategy from what the parsed document actually looks like.

The profile is a label for a bundle of switches (see STRATEGIES). Every switch is also
listed with the measured reason, so the UI can explain *why* a document got a strategy.
"""
from __future__ import annotations

from .parser import ParsedDoc

STRATEGIES = {
    "narrative": {
        "label": "Plain text",
        "chunker": "sentence", "target_tokens": 220, "overlap_tokens": 45,
        "parents": "neighbours", "routing": False, "rerank": True, "vision": False,
        "why": "Prose dominates: whole-sentence chunks, hybrid search and a reranker work best.",
    },
    "long_structured": {
        "label": "Long, structured",
        "chunker": "hierarchical", "child_tokens": 280, "parent_tokens": 900,
        "parents": "section", "routing": True, "rerank": True, "vision": False,
        "why": "Many sections share vocabulary: location (heading path) has to travel with "
               "each passage, questions are routed to sections, and the model reads parent sections.",
    },
    "visual": {
        "label": "Visual",
        "chunker": "element", "target_tokens": 220,
        "parents": "companions", "routing": False, "rerank": False, "vision": True,
        "why": "Facts live in tables, charts and footnotes: one chunk per element, footnotes "
               "attached, charts described by a vision model; the prose-trained reranker is skipped.",
    },
}


def profile_document(pd: ParsedDoc) -> dict:
    s = pd.stats
    pages = max(pd.n_pages, 1)
    words = s.get("words", 0)
    wpp = words / pages
    visual_pages = sum(1 for p in pd.page_info
                       if p["drawings"] >= 25 or p["image_area"] >= 0.2 or p["tables"])
    markers = sum(len(e.markers) for e in pd.elements)
    features = {
        "pages": pd.n_pages, "words": words, "words_per_page": round(wpp),
        "headings": s.get("headings", 0), "toc_entries": s.get("toc_entries", 0),
        "tables": s.get("tables", 0), "footnote_markers": markers, "notes": s.get("notes", 0),
        "visual_pages": visual_pages, "ligature_fixes": s.get("ligature_fixes", 0),
    }
    reasons = []
    if pages <= 40 and wpp < 450 and (s.get("tables", 0) >= 1 or markers >= 5
                                      or visual_pages / pages >= 0.3):
        prof = "visual"
        reasons.append(f"{pages} pages, {round(wpp)} words/page")
        if s.get("tables"):
            reasons.append(f"{s['tables']} tables")
        if markers:
            reasons.append(f"{markers} footnote markers")
        reasons.append(f"{visual_pages} pages with charts or images")
    elif words >= 60000 or (pages >= 120 and s.get("headings", 0) >= 20):
        prof = "long_structured"
        reasons.append(f"{words:,} words over {pages} pages")
        reasons.append(f"{s.get('headings', 0)} headings, {s.get('toc_entries', 0)} bookmarks")
    else:
        prof = "narrative"
        reasons.append(f"{words:,} words, prose-dominant")
    return {"profile": prof, "reasons": reasons, "features": features, "strategy": STRATEGIES[prof]}
