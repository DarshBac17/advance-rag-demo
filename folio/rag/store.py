"""One vector database, two pipelines.

Dense vectors live in Qdrant (embedded on-disk mode by default, or a server via QDRANT_URL).
BM25 runs in-process over the same chunks. Chunk records, parent sections and the knowledge
graph are stored as JSON next to the vectors, one folder per document.

Hybrid search = dense list + BM25 list (+ optional extra ranked lists, e.g. from the graph),
merged with Reciprocal Rank Fusion: every list contributes w / (k + rank). Only ranks matter,
so cosine similarities and BM25 scores never need a common scale.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import get_embeddings

STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
will with what which who how why when where does did do about into than then there their these those
report document according say says""".split())


def stem(t: str) -> str:
    """Tiny suffix stripper: enough to join employ/employed/employees, forecast/forecasts."""
    for suf in ("ations", "ation", "ings", "ing", "ees", "edly", "ed", "es", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4 and not t.endswith("ss"):
            return t[: -len(suf)]
    return t


def bm25_tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*%?", text.lower()):
        t = t.rstrip("%")
        if t in STOP:
            continue
        out.append(t)
        if "," in t:
            out.append(t.replace(",", ""))          # 802,834 also matches 802834
        if not t[0].isdigit():
            st = stem(t)
            if st != t:
                out.append(st)                       # "employed", "employees" also match "employ"
    return out


def rrf(lists: list[list[str]], k: int = 60, weights: list[float] | None = None):
    scores: dict[str, float] = {}
    for li, lst in enumerate(lists):
        w = weights[li] if weights else 1.0
        for r, key in enumerate(lst, start=1):
            scores[key] = scores.get(key, 0.0) + w / (k + r)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Index:
    _lock = threading.RLock()

    def __init__(self, path: str | None = None, url: str | None = None):
        self._path, self._url = path, url if url is not None else settings.qdrant_url
        self._client = None
        self._cache: dict = {}

    # -- plumbing ------------------------------------------------------------------------
    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(url=self._url) if self._url else \
                    QdrantClient(path=self._path or settings.qdrant_path)
            except RuntimeError as exc:
                if "already accessed" in str(exc):
                    raise RuntimeError(
                        "The embedded vector store is open in another process (usually the running web "
                        "server). Stop it before running management commands, or run a Qdrant server and "
                        "set QDRANT_URL so several processes can share it.") from exc
                raise
        return self._client

    @property
    def emb(self):
        return get_embeddings()

    @property
    def collection(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.emb.name.lower())
        return f"folio_{slug}_{self.emb.dim}"

    def _ensure(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=self.emb.dim, distance=models.Distance.COSINE))
            for field in ("doc_id", "pipeline", "section", "kind") if self._url else ():
                try:
                    self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass

    def _dir(self, doc_id) -> Path:
        d = settings.data_dir / "index" / doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- write ---------------------------------------------------------------------------
    def add(self, doc_id: str, pipeline: str, chunks, parents: dict, progress=None):
        recs = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in chunks]
        toks = [bm25_tokens(r["embed_text"]) for r in recs]
        (self._dir(doc_id) / f"{pipeline}.json").write_text(json.dumps(
            {"chunks": recs, "parents": parents, "tokens": toks}, ensure_ascii=False))
        with self._lock:
            self._ensure()
            B = 64
            for i in range(0, len(recs), B):
                batch = recs[i:i + B]
                vecs = self.emb.embed_documents([r["embed_text"] for r in batch])
                pts = [models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}/{pipeline}/{r['id']}")), vector=v,
                    payload={"doc_id": doc_id, "pipeline": pipeline, "chunk_id": r["id"], "kind": r["kind"],
                             "page": r["page"], "section": [s for s in r["section_path"]]})
                    for r, v in zip(batch, vecs)]
                self.client.upsert(self.collection, pts)
                if progress:
                    progress(min(i + B, len(recs)), len(recs))
        self._cache.pop((doc_id, pipeline), None)

    def delete(self, doc_id: str):
        with self._lock:
            if self.client.collection_exists(self.collection):
                self.client.delete(self.collection, models.FilterSelector(filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])))
        shutil.rmtree(settings.data_dir / "index" / doc_id, ignore_errors=True)
        for k in [k for k in self._cache if k[0] == doc_id]:
            self._cache.pop(k)

    # -- read ----------------------------------------------------------------------------
    def load(self, doc_id: str, pipeline: str) -> dict | None:
        key = (doc_id, pipeline)
        if key not in self._cache:
            f = settings.data_dir / "index" / doc_id / f"{pipeline}.json"
            if not f.exists():
                return None
            d = json.loads(f.read_text())
            d["by_id"] = {c["id"]: c for c in d["chunks"]}
            d["bm25"] = BM25Okapi(d["tokens"]) if d["tokens"] else None
            self._cache[key] = d
        return self._cache[key]

    @staticmethod
    def _in_sections(chunk, sections) -> bool:
        return not sections or any(s in chunk["section_path"] for s in sections)

    def dense(self, query, doc_ids, pipeline, sections=None, limit=30, qvec=None):
        with self._lock:
            if not self.client.collection_exists(self.collection):
                return []
            must = [models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))),
                    models.FieldCondition(key="pipeline", match=models.MatchValue(value=pipeline))]
            if sections:
                must.append(models.FieldCondition(key="section", match=models.MatchAny(any=list(sections))))
            res = self.client.query_points(self.collection, query=qvec or self.emb.embed_query(query),
                                           query_filter=models.Filter(must=must), limit=limit,
                                           with_payload=True).points
        return [(p.payload["doc_id"], p.payload["chunk_id"], p.score) for p in res]

    def bm25(self, query, doc_id, pipeline, sections=None, limit=30):
        d = self.load(doc_id, pipeline)
        if not d or not d["bm25"]:
            return []
        scores = d["bm25"].get_scores(bm25_tokens(query))
        ranked = sorted(((s, c) for s, c in zip(scores, d["chunks"]) if s > 0 and self._in_sections(c, sections)),
                        key=lambda x: -x[0])[:limit]
        return [(doc_id, c["id"], float(s)) for s, c in ranked]

    def chunk(self, doc_id, pipeline, chunk_id):
        d = self.load(doc_id, pipeline)
        return d["by_id"].get(chunk_id) if d else None

    def hybrid(self, query, doc_ids, pipeline, sections=None, pool=None, use_bm25=True,
               extra: dict[str, list] | None = None, weights: dict | None = None, qvec=None) -> list[dict]:
        """Returns fused hits: {doc_id, chunk, score, ranks:{dense,bm25,...}} best first."""
        pool = pool or settings.pool
        lists = {"dense": [(d, c) for d, c, _ in self.dense(query, doc_ids, pipeline, sections, pool, qvec)]}
        if use_bm25:
            lists["bm25"] = [(d, c) for doc in doc_ids for d, c, _ in self.bm25(query, doc, pipeline, sections, pool)]
            if len(doc_ids) > 1:     # re-rank BM25 across documents by score
                scored = [x for doc in doc_ids for x in self.bm25(query, doc, pipeline, sections, pool)]
                lists["bm25"] = [(d, c) for d, c, _ in sorted(scored, key=lambda x: -x[2])][:pool]
        for name, lst in (extra or {}).items():
            lists[name] = [tuple(x) for x in lst]
        return self.fuse(lists, pipeline, weights)

    def fuse(self, lists: dict[str, list], pipeline: str, weights: dict | None = None) -> list[dict]:
        """RRF over named ranked lists of (doc_id, chunk_id); records each hit's rank per list."""
        names = [n for n in lists if lists[n]]
        w = [(weights or {}).get(n, 1.0) for n in names]
        pos = {n: {tuple(x): i + 1 for i, x in reversed(list(enumerate(lists[n])))} for n in names}
        fused = rrf([[f"{d}::{c}" for d, c in lists[n]] for n in names], k=settings.rrf_k, weights=w)
        out = []
        for key, score in fused:
            d, c = key.split("::", 1)
            ch = self.chunk(d, pipeline, c)
            if not ch:
                continue
            ranks = {n: pos[n][(d, c)] for n in names if (d, c) in pos[n]}
            out.append({"doc_id": d, "chunk": ch, "score": round(score, 5), "ranks": ranks})
        return out


_index: Index | None = None


def get_index() -> Index:
    global _index
    if _index is None:
        import atexit
        _index = Index()
        atexit.register(lambda: _index._client and _index._client.close())
    return _index
