"""Repair damage that PDF text extraction does before chunking ever sees the text.

Two failure modes seen in real reports:
  * ligature glyphs kept as single code points ("ﬂ", "ﬁ") -> fixed by NFKC
  * a font that drops the "Th" ligature entirely ("Te conflict", "Tis is") -> fixed
    with the document's own vocabulary: "Te" becomes "The" only if the lowercase
    "te" never occurs in the document while "the" does.
Line-end hyphenation is undone the same way: join "produc-" + "tion" when the joined
word appears elsewhere in the document, keep "one-quarter" when both halves are words.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD = re.compile(r"[^\W\d_]+")      # unicode letters: "Côte" stays one word
_LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
        "\u00ad": "", "\u00a0": " ", "\u2009": " ", "\u202f": " "}


def normalize_chars(text: str) -> str:
    for k, v in _LIG.items():
        text = text.replace(k, v)
    return unicodedata.normalize("NFKC", text)


class TextRepair:
    def __init__(self, case_sensitive: Counter, lower: Counter):
        self.cs = case_sensitive      # exact-case counts
        self.lower = lower            # lowercase counts
        self._cache: dict[str, str] = {}

    @classmethod
    def from_corpus(cls, text: str) -> "TextRepair":
        text = normalize_chars(text)
        words = _WORD.findall(text)
        return cls(Counter(words), Counter(w.lower() for w in words))

    # -- dropped "Th" ligature -------------------------------------------------
    def _fix_token(self, tok: str) -> str:
        if tok in self._cache:
            return self._cache[tok]
        out = tok
        if len(tok) >= 2 and tok[0] == "T" and tok[1].islower():
            rest = tok[1:]
            lower_t = "t" + rest              # e.g. "te"   - must NOT occur mid-sentence
            lower_th = "th" + rest            # e.g. "the"  - must occur
            # damaged if "the"-form dominates the "te"-form at least 40:1 in lowercase
            if self.cs.get(lower_th, 0) > 0 and self.cs.get(lower_t, 0) * 40 < self.cs.get(lower_th, 0):
                out = "Th" + rest
        self._cache[tok] = out
        return out

    def fix_words(self, text: str) -> str:
        return re.sub(r"\bT[a-z]+\b", lambda m: self._fix_token(m.group(0)), text)

    # -- hyphenation at line ends ---------------------------------------------
    def known(self, w: str) -> bool:
        return self.lower.get(w.lower(), 0) > 0

    def join_hyphen(self, left: str, right: str) -> str:
        joined = left + right
        if self.known(joined):
            return joined
        if self.known(left) and self.known(right):
            return f"{left}-{right}"
        return joined

    def fix_line_hyphens(self, text: str) -> str:
        """Fix 'produc- tion' artefacts that survive inside already-joined text."""
        def rep(m):
            return self.join_hyphen(m.group(1), m.group(2))
        return re.sub(r"\b([A-Za-z]{2,})- ([a-z]{2,})\b", rep, text)

    def repair(self, text: str) -> str:
        return self.fix_line_hyphens(self.fix_words(normalize_chars(text)))
