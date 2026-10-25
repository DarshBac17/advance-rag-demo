"""End-to-end over real indexes. Uses the offline hash embedder, so these test the *pipeline*
(routing, graph hops, parent swap, footnotes, verification, retry), not embedding quality."""
import os

import pytest

os.environ["EMBEDDINGS_PROVIDER"] = "hash"
os.environ["FOLIO_DATA_DIR"] = os.environ.get("FOLIO_TEST_DATA", "/tmp/folio_data")
os.environ.pop("GROQ_API_KEY", None)
os.environ["RERANK"] = "0"

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import rag.query as query_mod
import rag.generate as gen_mod
from rag.pipeline import index_document, load_meta
from rag.generate import verify, REFUSAL
from rag.workflow import ask_advanced, ask_naive, compiled
from tests.conftest import sample, needs_samples


@pytest.fixture(scope="module", autouse=True)
def indexed():
    if not os.path.exists(sample("imac")):
        pytest.skip("samples missing")
    for name in ("imac", "gpo", "gep"):
        if not load_meta(name):
            index_document(sample(name), name)
    compiled.cache_clear()


def stages(res):
    return [t["stage"] for t in res["trace"]]


def test_verify_flags_invented_numbers():
    ps = [{"text": "Growth in SAR is expected to soften to 6.3 percent in 2026."}]
    assert verify("Growth will be 6.3 percent in 2026 [1].", ps)["grounded"]
    v = verify("Growth will be 7.1 percent [1].", ps)
    assert not v["grounded"] and v["unsupported_numbers"] == ["7.1"]
    assert verify(REFUSAL, ps)["refusal"]
    assert verify("Growth is 6.3 percent [4].", ps)["bad_citations"] == [4]


@needs_samples
def test_gep_south_asia_offline():
    r = ask_advanced("What is the growth outlook for South Asia in this report?", ["gep"])
    assert stages(r)[:6] == ["Understand", "Retrieve", "Graph", "Rerank", "Assemble", "Generate"]
    assert r["analysis"]["gep"]["sections"] == ["South Asia"]               # rules routing
    assert any("Growth in SAR is expected to soften to 6.3" in p["text"] for p in r["passages"])
    assert any(p["kind"] == "section" for p in r["passages"])              # parent swap


@needs_samples
def test_imac_footnote_question_offline():
    r = ask_advanced("What share of iMac manufacturing electricity comes from renewable sources, "
                     "and how is that measured?", ["imac"])
    assert any("Over 20%" in p["text"] and "prior fiscal year" in p["text"] for p in r["passages"])
    assert "Skipped" in next(t["text"] for t in r["trace"] if t["stage"] == "Rerank")


@needs_samples
def test_naive_baseline_runs():
    r = ask_naive("How many copies of how many titles did GPO distribute to depository libraries?", ["gpo"])
    assert r["pipeline"] == "naive" and len(r["passages"]) == 5


@needs_samples
def test_llm_path_and_corrective_retry(monkeypatch):
    understand_reply = '{"sections": [], "search_query": "GPO total revenue FY 2020", ' \
                       '"hypothetical_answer": "GPO revenue in FY 2020 was X", "sub_questions": []}'
    fast = GenericFakeChatModel(messages=iter([AIMessage(understand_reply)]))
    answer = GenericFakeChatModel(messages=iter([AIMessage(REFUSAL), AIMessage(REFUSAL)]))
    monkeypatch.setattr(query_mod, "chat", lambda role="fast": fast)
    monkeypatch.setattr(gen_mod, "chat", lambda role="answer": answer)
    r = ask_advanced("What was GPO's total revenue in FY 2020?", ["gpo"])
    assert r["answer"] == REFUSAL and r["attempts"] == 2
    assert "Correct" in stages(r) and "Retrieve (retry)" in stages(r)


@needs_samples
def test_llm_grounded_answer_passes_verification(monkeypatch):
    fast = GenericFakeChatModel(messages=iter([AIMessage('{"sections": ["South Asia"], "search_query": '
                                                         '"South Asia growth outlook 2026", "hypothetical_answer": "", '
                                                         '"sub_questions": []}')]))
    answer = GenericFakeChatModel(messages=iter([AIMessage("Growth in South Asia is expected to soften to "
                                                           "6.3 percent in 2026 [1].")] * 2))
    monkeypatch.setattr(query_mod, "chat", lambda role="fast": fast)
    monkeypatch.setattr(gen_mod, "chat", lambda role="answer": answer)
    r = ask_advanced("What is the growth outlook for South Asia?", ["gep"])
    assert r["analysis"]["gep"]["method"] == "llm"
    if "6.3" in r["passages"][0]["text"]:
        assert r["verification"]["grounded"] and r["attempts"] == 1
