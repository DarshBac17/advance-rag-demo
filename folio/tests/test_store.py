import os
import pytest

os.environ.setdefault("EMBEDDINGS_PROVIDER", "hash")

from rag.chunking import Chunk, build_chunks
from rag.store import Index, bm25_tokens, rrf
from tests.conftest import parsed, needs_samples


def _c(i, text, path, page=1, kind="text"):
    return Chunk(id=f"c{i}", text=text, embed_text=" > ".join(path) + "\n" + text, kind=kind,
                 page=page, pages=[page], section_path=path, order=i)


@pytest.fixture()
def index(tmp_path):
    os.environ["FOLIO_DATA_DIR"] = str(tmp_path)
    return Index(path=str(tmp_path / "q"))


def test_bm25_tokens_keep_numbers():
    assert {"6.3", "802,834", "802834", "sar"} <= set(bm25_tokens("SAR grew 6.3 percent; 802,834 copies"))


def test_rrf_merges_by_rank_only():
    fused = rrf([["a", "b", "c"], ["c", "a"]], k=60)
    assert [x for x, _ in fused][:2] == ["a", "c"]


def test_roundtrip_and_section_filter(index):
    chunks = [_c(0, "Growth in SAR is expected to soften to 6.3 percent in 2026.", ["Ch 2", "South Asia", "Outlook"]),
              _c(1, "Growth in the region is projected at 4.2 percent in 2026.", ["Ch 2", "Sub-Saharan Africa", "Outlook"]),
              _c(2, "Box on artificial intelligence and productivity.", ["Ch 1", "Box 1.1"])]
    index.add("docA", "advanced", chunks, parents={})
    hits = index.hybrid("growth outlook 2026", ["docA"], "advanced", sections=["South Asia"])
    assert hits[0]["chunk"]["id"] == "c0"
    assert all("South Asia" in h["chunk"]["section_path"] for h in hits)
    hits = index.hybrid("6.3 percent", ["docA"], "advanced")
    assert hits[0]["chunk"]["id"] == "c0" and hits[0]["ranks"]["bm25"] == 1
    index.delete("docA")
    assert index.hybrid("growth", ["docA"], "advanced") == []


@needs_samples
def test_imac_chart_question_finds_page_2(index):
    cs = build_chunks(parsed("imac"), "visual")
    index.add("imac", "advanced", cs.chunks, cs.parents)
    hits = index.hybrid("baseline emissions kg reductions", ["imac"], "advanced")
    assert 2 in [h["chunk"]["page"] for h in hits[:3]]


def test_bm25_stemming_joins_word_forms():
    assert "employ" in bm25_tokens("employed") and "employ" in bm25_tokens("employees")
