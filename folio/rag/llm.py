"""Chat models via LangChain. Every caller must handle `None` (no key) with a fallback."""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def chat(role: str = "answer"):
    """role: 'answer' (gpt-oss-120b), 'fast' (gpt-oss-20b, query analysis)."""
    if not settings.groq_key:
        return None
    from langchain_groq import ChatGroq
    model = settings.answer_model if role == "answer" else settings.fast_model
    return ChatGroq(model=model, api_key=settings.groq_key, temperature=0, max_retries=2,
                    request_timeout=60, reasoning_effort="low" if role == "fast" else "medium")


@lru_cache(maxsize=1)
def vision():
    if not settings.google_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=settings.vision_model, google_api_key=settings.google_key,
                                  temperature=0, max_retries=2)


def text_of(msg) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def status() -> dict:
    return {"answer_llm": settings.answer_model if settings.groq_key else None,
            "fast_llm": settings.fast_model if settings.groq_key else None,
            "vision": settings.vision_model if settings.google_key else None}
