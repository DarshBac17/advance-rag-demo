import os
import pytest

from rag.ingest.parser import reading_order, parse_pdf

UP = "/mnt/user-data/uploads"
IMAC = os.path.join(UP, "iMac_PER_Oct2024.pdf")
GPO = os.path.join(UP, "GPO-Anual-report.pdf")
GEP = os.path.join(UP, "GEP-Jun-2026.pdf")
have = pytest.mark.skipif(not os.path.exists(IMAC), reason="sample PDFs not present")


def _b(x0, y0, x1, y1, t):
    return {"bbox": (x0, y0, x1, y1), "text": t}


def test_reading_order_left_column_before_right():
    W = 600
    blocks = [_b(320, 100, 580, 150, "R1"), _b(20, 100, 280, 150, "L1"),
              _b(20, 160, 280, 200, "L2"), _b(320, 160, 580, 200, "R2"),
              _b(20, 20, 580, 60, "TITLE")]
    assert [b["text"] for b in reading_order(blocks, W)] == ["TITLE", "L1", "L2", "R1", "R2"]


def test_reading_order_wide_block_splits_bands():
    W = 600
    blocks = [_b(20, 100, 280, 150, "L1"), _b(320, 100, 580, 150, "R1"),
              _b(20, 200, 580, 260, "WIDE"),
              _b(20, 300, 280, 350, "L2"), _b(320, 300, 580, 350, "R2")]
    assert [b["text"] for b in reading_order(blocks, W)] == ["L1", "R1", "WIDE", "L2", "R2"]


from tests.conftest import parsed


@pytest.fixture(scope="module")
def imac():
    return parsed("imac")


@have
def test_imac_header_bar_table_rebuilt(imac):
    tables = [e for e in imac.elements if e.kind == "table"]
    ghg = [t for t in tables if "Total product footprint" in t.text]
    assert ghg, "GHG table not found"
    t = ghg[0].text
    assert "iMac (two ports) 256GB" in t
    assert "Total product footprint: 346 kg CO2e" in t
    assert "GHG reductions achieved[^7]" in t


@have
def test_imac_two_tier_header_table(imac):
    t = next(e.text for e in imac.elements if e.kind == "table" and "Power consumption" in e.text)
    assert "Off: 100V = 0.27W; 115V = 0.27W; 230V = 0.29W" in t
    assert "Power adapter efficiency: 100V = 91.1%" in t


@have
def test_imac_superscripts_become_markers_subscripts_do_not(imac):
    text = " ".join(e.text for e in imac.elements if e.page == 7)
    assert "ENERGY STAR.[^18]" in text or "ENERGY STAR[^18]" in text
    assert "CO2" in " ".join(e.text for e in imac.elements if e.page == 10)


@have
def test_imac_endnotes_collected(imac):
    assert "ENERGY STAR Program Requirements" in imac.note_for(18, 7)
    assert "prior fiscal year" in imac.note_for(2, 1)          # continuation lines joined
    # endnote definitions must not also be indexed as prose
    assert not any(e.kind == "text" and e.text.startswith("Product recycled or renewable content")
                   for e in imac.elements)


@pytest.fixture(scope="module")
def gpo():
    return parsed("gpo")


@have
def test_gpo_caption_not_spliced_into_message(gpo):
    p4 = [e for e in gpo.elements if e.page == 4 and e.kind == "text"]
    joined = " ".join(e.text for e in p4)
    first = joined.find("Acting Deputy Director of GPO")
    assert first != -1
    assert "binding line" not in joined          # the photo caption is a caption element
    assert any(e.kind == "caption" and "binding line" in e.text for e in gpo.elements if e.page == 4)


@pytest.fixture(scope="module")
def gep():
    return parsed("gep")


@have
def test_gep_ligature_repaired(gep):
    es = " ".join(e.text for e in gep.elements if e.page in (19, 20))
    assert "The global economy" in es and "Te global" not in es


@have
def test_gep_sar_outlook_heading_path(gep):
    hit = [e for e in gep.elements if "Growth in SAR is expected to soften to 6.3" in e.text]
    assert hit
    path = " > ".join(hit[0].section_path)
    assert "South Asia" in path and path.endswith("Outlook")
    assert "Regional Outlooks" in path


@have
def test_gep_box_closed_after_box_pages(gep):
    # chapter-1 prose after Box 1.1 must not be labelled as part of the box
    later = [e for e in gep.elements if e.page == 38 and e.kind == "text"]
    assert later and all("Box 1.1" not in " ".join(e.section_path) for e in later)


@have
def test_imac_callouts_are_prose_not_figure_labels(imac):
    p5 = [e for e in imac.elements if e.page == 5]
    callout = [e for e in p5 if "safer cleaners and degreasers" in e.text]
    assert callout and callout[0].kind == "text"
    assert "supplier low-carbon energy projects" in " ".join(e.text for e in p5)


@have
def test_imac_chart_labels_stay_figure(imac):
    assert any(e.kind == "figure_text" and "Baseline emissions" in e.text for e in imac.elements if e.page == 2)
