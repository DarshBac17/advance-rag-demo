"""Three chunking strategies plus the naive baseline.

Shared rules
  * chunks never cross a section boundary (the heading path is part of the chunk's identity)
  * prose is cut on whole sentences; a footnote marker after a full stop stays with its sentence
  * footnote markers [^n] become "[note n]" and the note text is appended to the chunk,
    so a caveat printed pages away travels with the claim it qualifies
  * what is embedded = heading path + text; what the model reads = text (path shown in context)
  * tables and figures are always one chunk each, never cut by character count
  * bibliography / contents sections are not indexed (they match every query and answer none)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from .ingest.parser import ParsedDoc, Element
from .ingest.profile import STRATEGIES

ABBR = {"u.s", "u.k", "e.g", "i.e", "inc", "no", "nos", "fig", "figs", "vs", "mr", "mrs", "ms", "dr",
        "st", "jr", "co", "corp", "ltd", "approx", "al", "p", "pp", "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "sept", "oct", "nov", "dec", "dept", "est", "govt", "ch", "vol"}
EXCLUDE = re.compile(r"^(references|bibliography|works cited|(table of )?contents|summary of contents|"
                     r"acknowledge?ments|index)$", re.I)
BOUNDARY = re.compile(r"[.!?][\"”’')\]]*(?:\[\^\d+\])*\s+(?=[A-Z0-9“\"(\[•])")


def tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in BOUNDARY.finditer(text):
        piece = text[start:m.end()].strip()
        last = re.findall(r"(\S+)[.!?][\"”’')\]]*(?:\[\^\d+\])*$", piece)
        lw = last[0].lower().rstrip(".") if last else ""
        if lw in ABBR or re.fullmatch(r"[A-Z]", last[0] if last else "x"):
            continue                       # "U.S. Congress", "J. Smith": not a boundary
        out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


@dataclass
class Chunk:
    id: str
    text: str
    embed_text: str
    kind: str                     # text | table | figure | caption | vision | naive
    page: int
    pages: list
    section_path: list
    parent_id: str | None = None
    notes: list = field(default_factory=list)
    order: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class ChunkSet:
    chunks: list
    parents: dict
    strategy: dict
    skipped: dict


# --------------------------------------------------------------------------- footnotes
def with_notes(text: str, page: int, pd: ParsedDoc | None) -> tuple[str, list[int]]:
    nums = sorted({int(n) for n in re.findall(r"\[\^(\d+)\]", text)})
    body = re.sub(r"\[\^(\d+)\]", r" [note \1]", text).replace("  [note", " [note")
    body = re.sub(r"\s+\[note", " [note", body)
    if not pd or not nums:
        return body, nums
    extra = []
    for n in nums:
        note = pd.note_for(n, page)
        if note:
            extra.append(f"[Note {n}] {note}")
    return body + ("\n" + "\n".join(extra) if extra else ""), nums


def _path_str(path) -> str:
    return " > ".join(path)


def _embed(path, text) -> str:
    return (f"{_path_str(path)}\n{text}" if path else text)


def _toc_like(text: str) -> bool:
    return bool(re.search(r"\.{6,}", text)) or \
        len(re.findall(r"\b\d{1,3}\b", text)) > 0.25 * max(len(text.split()), 1) and len(text.split()) > 12


# --------------------------------------------------------------------------- runs
def _runs(pd: ParsedDoc, skipped: dict):
    """Yield ('prose', path, [elements]) or ('element', path, element) in document order."""
    run, run_path = [], None
    for e in pd.elements:
        path = e.section_path
        if path and EXCLUDE.match(path[-1].strip()):
            skipped["excluded_sections"] = skipped.get("excluded_sections", 0) + 1
            continue
        if e.kind == "heading":
            continue
        if e.kind == "text":
            if _toc_like(e.text):
                skipped["toc_like"] = skipped.get("toc_like", 0) + 1
                continue
            if run and path != run_path:
                yield "prose", run_path, run
                run = []
            run_path = path
            run.append(e)
            continue
        if run:
            yield "prose", run_path, run
            run = []
        yield "element", path, e
    if run:
        yield "prose", run_path, run


def _sentences_with_pages(els: list[Element]):
    """Join the run first, so a sentence that flows across a column or page break is whole;
    each sentence is attributed to the page where it starts."""
    text, offsets = "", []
    for e in els:
        if text and text.endswith("-") and e.text[:1].islower():
            text = text[:-1]
        elif text:
            text += " "
        offsets.append((len(text), e.page))
        text += e.text
    out, pos = [], 0
    for s in split_sentences(text):
        at = text.find(s[:40], pos)
        at = pos if at < 0 else at
        pos = at + len(s)
        out.append((s, max(pg for off, pg in offsets if off <= at)))
    return out


def _windows(sents, target, overlap):
    """Greedy whole-sentence windows of ~target tokens, carrying ~overlap tokens forward."""
    chunks, cur, size = [], [], 0
    for s, pg in sents:
        t = tokens(s)
        if cur and size + t > target:
            chunks.append(cur)
            keep, ksize = [], 0
            for x in reversed(cur):
                if ksize + tokens(x[0]) > overlap:
                    break
                keep.insert(0, x)
                ksize += tokens(x[0])
            cur, size = keep, ksize
        cur.append((s, pg))
        size += t
    if cur and (not chunks or cur != chunks[-1][-len(cur):]):
        chunks.append(cur)
    return chunks


class _Builder:
    def __init__(self, pd, strategy):
        self.pd, self.strategy = pd, strategy
        self.chunks: list[Chunk] = []
        self.parents: dict = {}

    def add(self, text, kind, path, pages, parent_id=None, meta=None, label=None):
        # a marker on the heading ("Smarter chemistry[^5]") qualifies everything under it
        head_marks = "".join(f"[^{n}]" for n in re.findall(r"\[\^(\d+)\]", path[-1] if path else "")
                             if f"[^{n}]" not in text)
        path = [re.sub(r"\[\^\d+\]", "", x).strip() for x in path]
        body, nums = with_notes(text + head_marks, pages[0], self.pd)
        shown = f"{label}\n{body}" if label else body
        c = Chunk(id=f"c{len(self.chunks)}", text=shown, embed_text=_embed(path, shown), kind=kind,
                  page=pages[0], pages=sorted(set(pages)), section_path=list(path),
                  parent_id=parent_id, notes=nums, order=len(self.chunks), meta=meta or {})
        self.chunks.append(c)
        return c

    def element(self, path, e: Element):
        if e.kind == "table":
            lines = e.text.split("\n")
            head, rows = lines[0], lines[1:]
            if tokens(e.text) <= 260 or len(rows) < 4:
                self.add(e.text, "table", path, [e.page], meta=e.meta)
                return
            # a 40-row forecast table is one chunk per ~200 tokens of rows, each repeating
            # the title and column header so every row keeps its meaning
            part, size, parts = [], 0, []
            for r in rows:
                if part and size + tokens(r) > 200:
                    parts.append(part)
                    part, size = [], 0
                part.append(r)
                size += tokens(r)
            if part:
                parts.append(part)
            done = 0
            for i, rs in enumerate(parts, 1):
                label = f"(rows {done + 1}–{done + len(rs)} of {len(rows)})"
                self.add(head + f" {label}\n" + "\n".join(rs), "table", path, [e.page],
                         meta={**e.meta, "part": i, "parts": len(parts)})
                done += len(rs)
        elif e.kind == "figure_text":
            label = f"Figure text, page {e.page}" + (" (chart labels)" if e.meta.get("chart") else "")
            self.add(e.text.replace("\n", " / "), "figure", path, [e.page], label=label)
        elif e.kind == "caption":
            self.add(e.text, "caption", path, [e.page], label="Caption")


def build_chunks(pd: ParsedDoc, profile: str, strategy: dict | None = None) -> ChunkSet:
    st = dict(STRATEGIES[profile], **(strategy or {}))
    b, skipped = _Builder(pd, st), {}
    for kind, path, item in _runs(pd, skipped):
        if kind == "element":
            b.element(path, item)
            continue
        sents = _sentences_with_pages(item)
        if st["chunker"] == "hierarchical":
            for pw in _windows(sents, st["parent_tokens"], 0):
                pid = f"p{len(b.parents)}"
                ptext = " ".join(s for s, _ in pw)
                pbody, _ = with_notes(ptext, pw[0][1], pd)
                b.parents[pid] = {"id": pid, "text": pbody, "pages": sorted({p for _, p in pw}),
                                  "section_path": list(path), "children": []}
                for cw in _windows(pw, st["child_tokens"], 40):
                    c = b.add(" ".join(s for s, _ in cw), "text", path, [p for _, p in cw], parent_id=pid)
                    b.parents[pid]["children"].append(c.id)
        else:
            target = st.get("target_tokens", 220)
            overlap = st.get("overlap_tokens", 45) if st["chunker"] == "sentence" else 0
            for w in _windows(sents, target, overlap):
                b.add(" ".join(s for s, _ in w), "text", path, [p for _, p in w])
    return ChunkSet(b.chunks, b.parents, st, skipped)


def naive_chunks(path: str, size: int = 1000) -> list[Chunk]:
    """The baseline from the guide: default text extraction, fixed 1,000-character windows."""
    doc = pymupdf.open(path)
    text, starts = "", []
    for p in doc:
        starts.append((len(text), p.number + 1))
        text += p.get_text() + "\n"
    out = []
    for i in range(0, len(text), size):
        seg = text[i:i + size]
        if not seg.strip():
            continue
        page = max(pg for off, pg in starts if off <= i)
        out.append(Chunk(id=f"n{len(out)}", text=seg.strip(), embed_text=seg.strip(), kind="naive",
                         page=page, pages=[page], section_path=[], order=len(out)))
    return out
