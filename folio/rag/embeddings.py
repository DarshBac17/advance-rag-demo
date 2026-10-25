"""Embeddings behind LangChain's `Embeddings` interface.

Providers, tried in order of the configured preference:
  fastembed  BAAI/bge-small-en-v1.5 via ONNX, local and free (default; ~130 MB download once)
  gemini     Google text-embedding (free tier, needs GOOGLE_API_KEY)
  hash       deterministic hashed bag-of-words+bigrams. NOT semantic: exists so the app and the
             tests run with no network. The UI shows a warning whenever it is active.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from .config import settings

log = logging.getLogger(__name__)
_TOK = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    name, dim = "hash-1024", 1024

    def _vec(self, text: str):
        toks = _TOK.findall(text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        v = [math.copysign(math.log1p(abs(x)), x) for x in v]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model)
        self.name = model.split("/")[-1]
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(list(texts), batch_size=32)]

    def embed_query(self, text):
        # bge models expect an instruction prefix on queries; fastembed's query_embed adds it
        return next(iter(self.model.query_embed(text))).tolist()


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str, key: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.inner = GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)
        self.name = model.split("/")[-1]
        self.dim = len(self.inner.embed_query("probe"))

    def embed_documents(self, texts):
        return self.inner.embed_documents(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self.inner.embed_query(text, task_type="RETRIEVAL_QUERY")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    pref = settings.embeddings
    order = {"fastembed": ["fastembed", "gemini"], "gemini": ["gemini", "fastembed"],
             "hash": []}.get(pref, ["fastembed"])
    for p in order:
        try:
            if p == "fastembed":
                return FastEmbedEmbeddings(settings.embed_model)
            if p == "gemini" and settings.google_key:
                return GeminiEmbeddings(settings.gemini_embed_model, settings.google_key)
        except Exception as exc:          # model download blocked, no key, ...
            log.warning("embeddings provider %s unavailable: %s", p, exc)
    if pref != "hash" and not settings.allow_hash_fallback:
        raise RuntimeError("No embedding provider available")
    log.warning("Using hashed bag-of-words embeddings (not semantic).")
    return HashEmbeddings()


def embedder_info() -> dict:
    e = get_embeddings()
    return {"name": e.name, "dim": e.dim, "semantic": not isinstance(e, HashEmbeddings)}
