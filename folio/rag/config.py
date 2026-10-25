"""All tunables come from the environment (.env is loaded by Django settings)."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def env(name, default=None, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes", "on")
    return cast(v)


class Settings:
    @property
    def data_dir(self) -> Path:
        p = Path(env("FOLIO_DATA_DIR", str(BASE / "data")))
        p.mkdir(parents=True, exist_ok=True)
        return p

    # embeddings / rerank (local, free)
    embeddings = property(lambda s: env("EMBEDDINGS_PROVIDER", "fastembed"))       # fastembed|gemini|hash
    embed_model = property(lambda s: env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    gemini_embed_model = property(lambda s: env("GEMINI_EMBED_MODEL", "models/text-embedding-004"))
    allow_hash_fallback = property(lambda s: env("ALLOW_HASH_FALLBACK", True, bool))
    rerank_enabled = property(lambda s: env("RERANK", True, bool))
    rerank_model = property(lambda s: env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))

    # vector DB
    qdrant_url = property(lambda s: env("QDRANT_URL"))                              # empty -> embedded
    qdrant_path = property(lambda s: str(s.data_dir / "qdrant"))

    # LLMs (free tiers)
    groq_key = property(lambda s: env("GROQ_API_KEY"))
    answer_model = property(lambda s: env("GROQ_MODEL", "openai/gpt-oss-120b"))
    fast_model = property(lambda s: env("GROQ_FAST_MODEL", "openai/gpt-oss-20b"))
    google_key = property(lambda s: env("GOOGLE_API_KEY") or env("GEMINI_API_KEY"))
    vision_model = property(lambda s: env("VISION_MODEL", "gemini-flash-lite-latest"))
    vision_max_pages = property(lambda s: env("VISION_MAX_PAGES", 12, int))

    # retrieval
    pool = property(lambda s: env("RETRIEVAL_POOL", 30, int))
    top_k = property(lambda s: env("TOP_K", 5, int))
    rrf_k = property(lambda s: env("RRF_K", 60, int))
    max_context_tokens = property(lambda s: env("MAX_CONTEXT_TOKENS", 3500, int))
    build_naive = property(lambda s: env("BUILD_NAIVE_BASELINE", True, bool))


settings = Settings()
