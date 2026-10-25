"""Cross-encoder reranker: reads question and passage together. Local and free (fastembed ONNX).
Returns None when the model cannot be loaded; callers record the skip in the trace."""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def cross_encoder():
    if not settings.rerank_enabled:
        return None
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return TextCrossEncoder(settings.rerank_model)
    except Exception as exc:
        log.warning("reranker unavailable: %s", exc)
        return None


def rerank(question: str, hits: list[dict], keep: int) -> tuple[list[dict], bool]:
    ce = cross_encoder()
    if ce is None or not hits:
        return hits[:keep], False
    scores = list(ce.rerank(question, [h["chunk"]["text"][:2000] for h in hits]))
    for h, s in zip(hits, scores):
        h["rerank"] = round(float(s), 3)
    return sorted(hits, key=lambda h: -h["rerank"])[:keep], True
