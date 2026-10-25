import re

from rag.ingest.profile import profile_document
from rag.chunking import build_chunks, naive_chunks, split_sentences
from tests.conftest import parsed, sample, needs_samples


def test_split_sentences_respects_abbreviations():
    s = split_sentences("GPO serves the U.S. Congress. It printed 47 million pages in FY 2018. Done.")
    assert s == ["GPO serves the U.S. Congress.", "It printed 47 million pages in FY 2018.", "Done."]


@needs_samples
def test_profiles():
    assert profile_document(parsed("gpo"))["profile"] == "narrative"
    assert profile_document(parsed("gep"))["profile"] == "long_structured"
    assert profile_document(parsed("imac"))["profile"] == "visual"


@needs_samples
def test_narrative_sentence_chunks():
    res = build_chunks(parsed("gpo"), "narrative")
    text = [c for c in res.chunks if c.kind == "text"]
    ends = sum(1 for c in text if re.search(r"[.!?”\"')\]]$", c.text.strip()))
    assert ends / len(text) >= 0.8
    assert any("802,834 copies of 4,211 titles" in c.text for c in res.chunks)
    # a chunk never spans two sections
    assert all(len({tuple(c.section_path)}) == 1 for c in text)


@needs_samples
def test_long_structured_parent_child():
    res = build_chunks(parsed("gep"), "long_structured")
    kids = [c for c in res.chunks if "Growth in SAR is expected to soften to 6.3" in c.text]
    assert kids
    k = kids[0]
    assert "South Asia" in k.embed_text and "Outlook" in k.embed_text
    assert k.parent_id in res.parents
    assert "Growth in SAR is expected to soften" in res.parents[k.parent_id]["text"]
    # bibliography is not indexed
    assert not any(c.section_path and c.section_path[-1].lower() == "references" for c in res.chunks)


@needs_samples
def test_visual_elements_carry_footnotes():
    res = build_chunks(parsed("imac"), "visual")
    ghg = [c for c in res.chunks if c.kind == "table" and "Total product footprint" in c.text]
    assert len(ghg) == 1 and "[Note 7]" in ghg[0].text and "business-as-usual" in ghg[0].text
    energy = [c for c in res.chunks if "58 percent less energy" in c.text]
    assert energy and all("ENERGY STAR Program Requirements" in c.text for c in energy)
    renew = [c for c in res.chunks if "Over 20%" in c.text and "renewable" in c.text]
    assert renew and "prior fiscal year" in renew[0].text


@needs_samples
def test_naive_baseline_is_fixed_size():
    ch = naive_chunks(sample("gpo"))
    assert len(ch) > 50 and all(len(c.text) <= 1000 for c in ch)


@needs_samples
def test_long_tables_split_with_header():
    res = build_chunks(parsed("gep"), "long_structured")
    t11 = [c for c in res.chunks if c.kind == "table" and "TABLE 1.1 Real GDP" in c.text]
    assert len(t11) > 1
    assert all(c.text.startswith("Table: TABLE 1.1 Real GDP") for c in t11)
    assert any("Brazil: 2023 = 3.2" in c.text for c in t11)
    assert max(len(c.text.split()) for c in res.chunks if c.kind == "table") < 400
