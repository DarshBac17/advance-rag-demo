"""Charts tell a story text extraction cannot: 445 kg minus 99 kg of reductions gives 346 kg.
One vision call per visual page, cached on disk by (file hash, page), so re-indexing is free."""
from __future__ import annotations

import base64
import hashlib
import logging
import re

import pymupdf
from langchain_core.messages import HumanMessage

from ..config import settings
from ..llm import vision, text_of

log = logging.getLogger(__name__)

PROMPT = """You are preparing one page of a report for a search index.
Write plain text capturing everything a reader learns from the CHARTS, TABLES, ICONS and CALL-OUTS.
- Transcribe every number with its label, unit, and the column or group it belongs to.
- For a chart, say what each bar or segment represents and how the values relate
  (for example, which values add up to or are subtracted from which).
- Mention footnote numbers where they appear, like [note 7].
- Do not guess values that are not printed. Ignore photos of people or products. Under 250 words."""


def pick_pages(page_info: list[dict], limit: int) -> list[int]:
    scored = [(p["drawings"] + 200 * p["image_area"] + 50 * p["tables"], p["page"]) for p in page_info
              if p["drawings"] >= 25 or p["image_area"] >= 0.2 or p["tables"]]
    return sorted(pg for _, pg in sorted(scored, reverse=True)[:limit])


def describe_pages(path: str, pages: list[int], progress=None) -> list[tuple[int, str]]:
    model = vision()
    if model is None or not pages:
        return []
    digest = hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]
    cache = settings.data_dir / "cache" / "vision"
    cache.mkdir(parents=True, exist_ok=True)
    doc, out = pymupdf.open(path), []
    for i, pg in enumerate(pages):
        f = cache / f"{digest}_{pg}.txt"
        if f.exists():
            out.append((pg, f.read_text()))
            continue
        try:
            png = doc[pg - 1].get_pixmap(dpi=110).tobytes("png")
            msg = HumanMessage(content=[{"type": "text", "text": PROMPT},
                                        {"type": "image_url",
                                         "image_url": "data:image/png;base64," + base64.b64encode(png).decode()}])
            txt = text_of(model.invoke([msg])).strip()
            txt = re.sub(r"\*\*|__|^#+\s*", "", txt, flags=re.M)
            if txt:
                f.write_text(txt)
                out.append((pg, txt))
        except Exception as exc:
            log.warning("vision failed on page %s: %s", pg, exc)
        if progress:
            progress(i + 1, len(pages))
    return out
