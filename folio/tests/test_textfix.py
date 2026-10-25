from rag.ingest.textfix import TextRepair, normalize_chars


def test_normalize_ligature_glyphs():
    assert normalize_chars("conﬂict and artiﬁcial eﬀects") == "conflict and artificial effects"


def _repair():
    corpus = "the this these they there that their then ten to tax term production one quarter"
    return TextRepair.from_corpus(corpus)


def test_dropped_th_ligature_restored():
    r = _repair()
    assert r.fix_words("Te conflict. Tis is it. Tese rose. Tey said. Tere was. Teir view.") == \
        "The conflict. This is it. These rose. They said. There was. Their view."


def test_real_t_words_untouched():
    r = _repair()
    # 'ten', 'to', 'tax', 'term' exist in lowercase, so the capitalised forms are real words
    assert r.fix_words("Ten years. To be. Tax rose. Term ends. Togo grew.") == \
        "Ten years. To be. Tax rose. Term ends. Togo grew."


def test_hyphen_join_when_word_known():
    r = _repair()
    assert r.join_hyphen("produc", "tion") == "production"


def test_hyphen_kept_for_compound():
    r = _repair()
    assert r.join_hyphen("one", "quarter") == "one-quarter"


def test_hyphen_join_when_prefix_not_a_word():
    r = _repair()
    assert r.join_hyphen("reposi", "tory") == "repository"
