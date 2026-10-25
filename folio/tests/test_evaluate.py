from rag.evaluate import norm, evidence_rank, answer_correct, metrics


def test_norm_handles_markers_dashes_hyphens():
    assert norm("–99 kg [note 8] per- cent") == "-99 kg per cent".replace("per cent", "percent")


def test_evidence_rank_any_and_all():
    ps = [{"text": "Over 20% of electricity [note 2]"}, {"text": "Over 20% of electricity\n[Note 2] ... prior fiscal year"}]
    assert evidence_rank({"evidence": ["Over 20%"]}, ps) == 1
    assert evidence_rank({"evidence_all": ["Over 20%", "prior fiscal year"]}, ps) == 2
    assert evidence_rank({"evidence": ["missing"]}, ps) is None


def test_answer_groups_and_refusals():
    res = lambda a, r=False: {"answer": a, "verification": {"refusal": r}}
    q = {"answer": [["445"], ["99"]]}
    assert answer_correct(q, res("Baseline 445 kg; reductions removed 99 kg [1]."))
    assert not answer_correct(q, res("Baseline 445 kg [1]."))
    assert answer_correct({"refusal": True}, res("I could not find this in the document.", True))
    assert not answer_correct(q, res("I could not find this in the document.", True))


def test_metrics_ignore_refusal_rows_for_retrieval():
    m = metrics([{"rank": 1, "correct": True, "refusal": False}, {"rank": 2, "correct": False, "refusal": False},
                 {"rank": None, "correct": True, "refusal": True}])
    assert m["hit@1"] == 0.5 and m["MRR"] == 0.75 and abs(m["answers"] - 2 / 3) < 1e-9
