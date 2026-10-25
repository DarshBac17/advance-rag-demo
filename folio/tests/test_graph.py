import pytest

from rag.graph import best_long_form, mine_aliases, build_graph, DocGraph
from rag.chunking import build_chunks
from tests.conftest import parsed, needs_samples


def test_schwartz_hearst():
    assert best_long_form("MNA", "the Middle East, North Africa, Afghanistan, and Pakistan") == \
        "Middle East, North Africa, Afghanistan, and Pakistan"
    assert best_long_form("EAP", "In East Asia and Pacific") == "East Asia and Pacific"
    assert best_long_form("RHS", "Agriculture") is None


def test_mine_aliases_validates_pairs():
    txt = ("Growth in East Asia and Pacific (EAP) slowed. Region: South Asia (SAR) expanded. "
           "Index values (RHS) are shown. Emerging market and developing economies (EMDEs) grew.")
    a = mine_aliases(txt)
    assert a["eap"] == "east asia and pacific"
    assert a["sar"] == "south asia"                       # initials + one trailing letter ("Region")
    assert a["emdes"] == "emerging market and developing economies"
    assert "rhs" not in a


@pytest.fixture(scope="module")
def gep_graph(tmp_path_factory):
    pd = parsed("gep")
    cs = build_chunks(pd, "long_structured")
    g = build_graph(cs.chunks, cs.parents, pd)
    return DocGraph(g, cs.chunks), cs


@needs_samples
def test_gep_alias_bridges_sar(gep_graph):
    g, cs = gep_graph
    ents = [e for e, _ in g.query_entities("What is the growth outlook for South Asia?")]
    assert "south asia" in ents and "sar" in ents
    hits = g.entity_hits("What is the growth outlook for South Asia?", limit=40)
    texts = [g.chunks[c]["text"] for c, _ in hits]
    assert any("Growth in SAR is expected to soften" in t for t in texts)


@needs_samples
def test_visual_companions_same_page():
    pd = parsed("imac")
    cs = build_chunks(pd, "visual")
    g = DocGraph(build_graph(cs.chunks, cs.parents, pd), cs.chunks)
    prose = next(c for c in cs.chunks if c.page == 2 and c.kind == "text" and "reduced emissions" in c.text)
    comp = g.companions(prose.id, "visual")
    assert any(g.chunks[c]["kind"] == "figure" and g.chunks[c]["page"] == 2 for c, _ in comp)


@needs_samples
def test_narrative_neighbours():
    pd = parsed("gpo")
    cs = build_chunks(pd, "narrative")
    g = DocGraph(build_graph(cs.chunks, cs.parents, pd), cs.chunks)
    mid = next(c for c in cs.chunks if c.kind == "text" and "802,834 copies" in c.text)
    rel = {r for _, r in g.companions(mid.id, "narrative")}
    assert "previous" in rel or "next" in rel
