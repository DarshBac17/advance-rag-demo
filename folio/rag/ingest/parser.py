"""Layout-aware PDF parsing with PyMuPDF.

Turns a PDF into typed *elements* (heading, text, table, caption, figure_text) in human
reading order, each carrying its heading path, plus a dictionary of footnote/endnote
definitions. Nothing here is specific to one PDF; the rules are driven by signals measured
on the page (fonts, fills, positions) and by the document's own statistics.

Signals used
  * reading order    wide blocks split the page into bands; inside a band, columns are
                     read left to right, each top to bottom
  * running headers  short text in the top/bottom margin that repeats across pages
  * headings         short blocks set larger than body text, or in a different face/weight
  * structure        PDF bookmarks (if sane) anchor the top levels, fonts give the rest
  * boxes            "BOX 1.1" sections close when the main body style resumes
  * tables           (a) dark header bars with light text -> rebuild rows from word positions
                     (b) PyMuPDF's ruled-table finder as a fallback, validated
  * footnotes        raised small digits become [^n]; lines that start with one are definitions
  * captions         small text touching an image
  * figure text      small text that is not a caption (chart labels, sources)
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pymupdf

from .textfix import TextRepair, normalize_chars

FIG_TITLE = re.compile(r"^\s*(FIGURE|TABLE|CHART|EXHIBIT|MAP)\s+[A-Z]?\d+(\.\d+)*[A-Z.]?\b", re.I)
BOX_TITLE = re.compile(r"^\s*BOX\s+\d+(\.\d+)*", re.I)
CONTINUED = re.compile(r"\(\s*continued\s*\)?\s*$", re.I)
MARK = re.compile(r"\[\^(\d+)\]")


@dataclass
class Element:
    kind: str                      # heading | text | table | caption | figure_text
    text: str
    page: int                      # 1-based
    bbox: tuple
    level: int = 0
    style: tuple = ()
    section_path: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def markers(self) -> list[int]:
        return sorted({int(n) for n in MARK.findall(self.text)})


@dataclass
class ParsedDoc:
    path: str
    title: str
    n_pages: int
    elements: list[Element]
    notes: dict                    # n -> [(page, text)]
    toc: list
    page_info: list[dict]
    stats: dict

    def note_for(self, n: int, page: int) -> str | None:
        """A marker on `page` refers to the nearest definition at or after it (page footnotes),
        otherwise to the last one before it (endnotes collected at the back)."""
        cands = self.notes.get(n) or []
        if not cands:
            return None
        after = [c for c in cands if c[0] >= page]
        return (min(after, key=lambda c: c[0]) if after else cands[-1])[1]


# --------------------------------------------------------------------------- helpers
def _family(font: str) -> str:
    font = font.split("+")[-1]
    return re.split(r"[-,]", font)[0]


def _is_bold(span) -> bool:
    return bool(span["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold|medium|-md", span["font"], re.I))


def _lum(rgb) -> float:
    if rgb is None:
        return 1.0
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_lum(c: int) -> float:
    """Luminance of a packed sRGB integer colour (PyMuPDF span colour)."""
    return _lum((((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255))


def reading_order(blocks: list[dict], page_width: float, wide_frac: float = 0.55) -> list[dict]:
    """Bands split by wide blocks; within a band, columns left-to-right, each top-to-bottom."""
    if not blocks:
        return []
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    wide = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) >= wide_frac * page_width]
    out, band = [], []

    def flush():
        if not band:
            return
        # cluster columns by left edge
        cols: list[list[dict]] = []
        for b in sorted(band, key=lambda b: b["bbox"][0]):
            if cols and b["bbox"][0] - min(x["bbox"][0] for x in cols[-1]) < 0.15 * page_width:
                cols[-1].append(b)
            else:
                cols.append([b])
        for c in cols:
            out.extend(sorted(c, key=lambda b: b["bbox"][1]))
        band.clear()

    for b in blocks:
        if b in wide:
            # a wide block closes the band only once every narrow block above it is placed
            flush()
            out.append(b)
        else:
            band.append(b)
    flush()
    return out


# --------------------------------------------------------------------------- spans -> text
def _line_text(line: dict) -> tuple[str, dict]:
    spans = [s for s in line["spans"] if s["text"] != ""]
    if not spans:
        return "", {}
    main = max(spans, key=lambda s: len(s["text"].strip()) * s["size"])
    dom, base = main["size"], main["origin"][1]
    parts = []
    for s in spans:
        t = s["text"]
        st = t.strip()
        small = s["size"] < 0.8 * dom
        if small and st and re.fullmatch(r"\d{1,3}(,\d{1,3})*", st):
            if s["origin"][1] < base - 0.6:          # raised -> footnote marker(s)
                parts.append("".join(f"[^{n}]" for n in st.split(",")) + (" " if t.endswith((" ", "\t")) else ""))
                continue
            parts.append(st)                         # lowered -> subscript (CO2), keep inline
            continue
        parts.append(t.replace("\t", " "))
    text = "".join(parts)
    text = re.sub(r"[ \u00a0]+", " ", text)
    style = (_family(main["font"]), round(main["size"], 1), _is_bold(main))
    return text, {"style": style, "size": dom, "color": main["color"], "bold": _is_bold(main),
                  "font": main["font"]}


# --------------------------------------------------------------------------- parser
class PDFParser:
    def __init__(self, path: str):
        self.path = path
        self.doc = pymupdf.open(path)
        self.W = self.doc[0].rect.width if len(self.doc) else 612

    # -- pass 1: raw lines + statistics ---------------------------------------------------
    def _raw_pages(self):
        pages = []
        for pno, page in enumerate(self.doc):
            d = page.get_text("dict")
            lines = []
            for bi, b in enumerate(d["blocks"]):
                if b["type"] != 0:
                    continue
                for li, l in enumerate(b["lines"]):
                    text, info = _line_text(l)
                    if not text.strip():
                        continue
                    lines.append({"block": bi, "bbox": tuple(l["bbox"]), "text": text, **info,
                                  "spans": l["spans"]})
            images = [tuple(b["bbox"]) for b in d["blocks"] if b["type"] == 1]
            try:
                for info in page.get_image_info():
                    bb = tuple(info["bbox"])
                    if bb not in images:
                        images.append(bb)
            except Exception:
                pass
            pages.append({"page": page, "pno": pno, "lines": lines, "images": images,
                          "height": page.rect.height, "width": page.rect.width})
        return pages

    def _margins(self, pages):
        """Text in the top/bottom 9% that repeats on >=3 pages (digits ignored) is chrome."""
        sig = Counter()
        for p in pages:
            H = p["height"]
            for l in p["lines"]:
                if l["bbox"][1] < 0.09 * H or l["bbox"][3] > 0.91 * H:
                    sig[re.sub(r"[\d\s|]+", "", l["text"]).lower()] += 1
        rep = {k for k, v in sig.items() if v >= 3 or k == ""}
        return rep

    # -- chart regions: clusters of filled vector shapes -----------------------------------
    @staticmethod
    def _graphic_regions(p, table_boxes) -> list:
        page, W, H = p["page"], p["width"], p["height"]
        rects = []
        for d in page.get_drawings():
            r, fill = d.get("rect"), d.get("fill")
            if r is None or fill is None or r.width < 3 or r.height < 3:
                continue
            if _lum(fill) > 0.97 or r.width * r.height > 0.6 * W * H:
                continue
            if any(pymupdf.Rect(t).intersects(r) for t in table_boxes):
                continue
            rects.append(pymupdf.Rect(r))
        clusters: list[pymupdf.Rect] = []
        counts: list[int] = []
        for r in rects:
            g = pymupdf.Rect(r.x0 - 12, r.y0 - 12, r.x1 + 12, r.y1 + 12)
            hit = [i for i, c in enumerate(clusters) if c.intersects(g)]
            new, n = pymupdf.Rect(r), 1
            for i in hit:
                new |= clusters[i]
                n += counts[i]
            clusters = [c for i, c in enumerate(clusters) if i not in hit] + [new]
            counts = [c for i, c in enumerate(counts) if i not in hit] + [n]
        # side-by-side clusters (bars of one chart) become one region
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    a, b = clusters[i], clusters[j]
                    v_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
                    h_gap = max(a.x0, b.x0) - min(a.x1, b.x1)
                    if v_overlap > 0.5 * min(a.height, b.height) and h_gap < 0.3 * W:
                        clusters[i] = a | b
                        counts[i] += counts[j]
                        del clusters[j], counts[j]
                        merged = True
                        break
                if merged:
                    break
        return [c for c, n in zip(clusters, counts) if n >= 3 and c.width * c.height >= 0.02 * W * H]

    # -- tables under dark header bars ---------------------------------------------------
    def _header_bar_tables(self, p) -> tuple[list[Element], set]:
        page = p["page"]
        cells = []
        for d in page.get_drawings():
            r, fill = d.get("rect"), d.get("fill")
            if fill is None or r is None:
                continue
            if _lum(fill) < 0.45 and 8 <= r.height <= 45 and r.width >= 30 and r.width < 0.98 * p["width"]:
                cells.append(pymupdf.Rect(r))
        if not cells:
            return [], set()
        # header regions = connected groups of touching dark cells (handles two-tier headers)
        regions: list[list] = []
        for c in sorted(cells, key=lambda r: (r.y0, r.x0)):
            grow = pymupdf.Rect(c.x0 - 1.5, c.y0 - 1.5, c.x1 + 1.5, c.y1 + 1.5)
            hit = [g for g in regions if any(grow.intersects(r) for r in g)]
            merged = [c] + [r for g in hit for r in g]
            regions = [g for g in regions if g not in hit] + [merged]
        regions.sort(key=lambda g: min(r.y0 for r in g))
        lines = p["lines"]
        used, out = set(), []

        def text_in(rect):
            return [i for i, l in enumerate(lines) if _color_lum(l["color"]) > 0.7 and
                    rect.contains(pymupdf.Point((l["bbox"][0] + l["bbox"][2]) / 2, (l["bbox"][1] + l["bbox"][3]) / 2))]

        hrows = regions
        for hi, region in enumerate(regions):
            ry1 = max(r.y1 for r in region)
            leaves = sorted([r for r in region if r.y1 >= ry1 - 2], key=lambda r: r.x0)
            spans_ = [r for r in region if r.y1 < ry1 - 2]
            x0, x1 = min(r.x0 for r in region), max(r.x1 for r in region)
            hy0, hy1 = min(r.y0 for r in region), ry1
            header_lines = sorted({i for r in region for i in text_in(r)})
            if not header_lines:
                continue
            hrow = leaves
            bounds = [r.x0 for r in leaves] + [x1]
            headers = [" ".join(lines[i]["text"].strip() for i in text_in(r)) for r in leaves]
            caption = " ".join(" ".join(lines[i]["text"].strip() for i in text_in(r)) for r in spans_).strip()
            next_y = min(r.y0 for r in hrows[hi + 1]) if hi + 1 < len(hrows) else 1e9
            body = sorted([i for i, l in enumerate(lines) if i not in header_lines
                           and l["bbox"][1] >= hy1 - 1 and l["bbox"][3] <= next_y + 1
                           and l["bbox"][0] >= x0 - 6 and l["bbox"][2] <= x1 + 6],
                          key=lambda i: lines[i]["bbox"][1])
            rows, last_y, size0 = [], hy1, None
            for i in body:
                l = lines[i]
                if size0 is None:
                    size0 = l["size"]
                gap = l["bbox"][1] - last_y
                if abs(l["size"] - size0) > 0.4 and gap > 4:
                    break
                if gap > 3.2 * max(size0, 5) and rows:
                    break
                rows.append(i)
                last_y = max(last_y, l["bbox"][3]) if abs(l["bbox"][1] - last_y) > 3 else last_y
                last_y = l["bbox"][3]
            if not rows:
                continue
            # group lines into rows by vertical position, assign to columns by x
            grid: list[dict] = []
            for i in rows:
                l = lines[i]
                yc = (l["bbox"][1] + l["bbox"][3]) / 2
                row = next((g for g in grid if abs(g["y"] - yc) < 4), None)
                if row is None:
                    row = {"y": yc, "cols": defaultdict(list)}
                    grid.append(row)
                xc = l["bbox"][0]
                ci = max([k for k in range(len(hrow)) if bounds[k] - 6 <= xc] or [0])
                row["cols"][ci].append(l["text"].strip())
            grid.sort(key=lambda g: g["y"])
            if caption:
                lines_out = [f"Table: {caption}. Rows: {headers[0]}. Columns: " + ", ".join(h for h in headers[1:] if h) + "."]
            else:
                lines_out = [f"Table: {headers[0] or 'Table'}." + "".join(f" Column: {h}." for h in headers[1:] if h)]
            for g in grid:
                cells_txt = [" ".join(g["cols"].get(k, [])).strip() for k in range(len(hrow))]
                label, vals = cells_txt[0], cells_txt[1:]
                if len(vals) == 1:
                    lines_out.append(f"{label}: {vals[0]}" if vals[0] else label)
                else:
                    lines_out.append(label + ": " + "; ".join(f"{h} = {v}" for h, v in zip(headers[1:], vals) if v))
            bbox = (x0, hy0, x1, max(lines[i]["bbox"][3] for i in rows))
            out.append(Element("table", "\n".join(lines_out).replace(" [^", "[^"), p["pno"] + 1, bbox,
                               meta={"method": "header-bar", "columns": headers, "caption": caption, "rows": len(grid)}))
            used.update(header_lines)
            used.update(rows)
        return out, used

    # -- ruled tables (fallback) --------------------------------------------------------
    def _ruled_tables(self, p) -> tuple[list[Element], set]:
        page = p["page"]
        if len(page.get_drawings()) < 12:
            return [], set()
        try:
            found = page.find_tables(strategy="lines")
        except Exception:
            return [], set()
        words = page.get_text("words")
        out, used = [], set()
        for t in found.tables:
            grid = []
            for row in t.rows:
                cells = []
                for cb in row.cells:
                    if cb is None:
                        cells.append("")
                        continue
                    r = pymupdf.Rect(cb)
                    ws = [w for w in words if r.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))]
                    ws.sort(key=lambda w: (round(w[1] / 3), w[0]))
                    cells.append(normalize_chars(" ".join(w[4] for w in ws)).strip())
                grid.append(cells)
            if len(grid) < 3 or not (2 <= len(grid[0]) <= 14):
                continue
            flat = [c for r in grid for c in r]
            if sum(1 for c in flat if c) / max(len(flat), 1) < 0.45:
                continue
            year = lambda c: bool(re.fullmatch(r"(FY\s?)?(19|20)\d\d([/-]\d\d)?[ef]?", c.strip()))
            numeric = lambda c: not year(c) and bool(re.fullmatch(
                r"[-–−+$€£(]?[\d.,]+%?\)?[a-z]?(\s*(million|billion|trillion|thousand))?", c.strip().lower()))
            # header rows: leading rows without numeric data cells
            n_head = 0
            for r in grid:
                if sum(numeric(c) for c in r[1:]) >= max(1, (len(r) - 1) // 2):
                    break
                n_head += 1
            n_head = min(n_head, 3)
            head = [" ".join(grid[i][j] for i in range(n_head) if grid[i][j]).strip() for j in range(len(grid[0]))]
            title = next((c for c in flat if FIG_TITLE.match(c)), "")
            rows = []
            for r in grid[n_head:]:
                if not any(r):
                    continue
                label, vals = r[0], r[1:]
                if any(head[1:]):
                    rows.append(f"{label}: " + "; ".join(f"{h or 'value'} = {v}" for h, v in zip(head[1:], vals) if v))
                else:
                    rows.append(f"{label}: " + " | ".join(v for v in vals if v))
            bb = pymupdf.Rect(t.bbox)
            hdr = f"Table: {title or head[0] or 'untitled'}." + (f" Columns: {', '.join(h for h in head[1:] if h)}." if any(head[1:]) else "")
            out.append(Element("table", hdr + "\n" + "\n".join(rows), p["pno"] + 1, tuple(bb),
                               meta={"method": "ruled", "rows": len(rows), "columns": head}))
            for i, l in enumerate(p["lines"]):
                if bb.contains(pymupdf.Rect(l["bbox"])):
                    used.add(i)
        return out, used

    # -- main ---------------------------------------------------------------------------
    def parse(self) -> ParsedDoc:
        pages = self._raw_pages()
        corpus = " ".join(l["text"] for p in pages for l in p["lines"])
        fixer = TextRepair.from_corpus(corpus)
        chrome = self._margins(pages)

        # body styles: char-weighted over paragraph lines (multi-line blocks, readable size)
        style_chars = Counter()
        for p in pages:
            per_block = Counter(l["block"] for l in p["lines"])
            for l in p["lines"]:
                if len(l["text"]) > 40 and per_block[l["block"]] >= 2 and l["size"] >= 7.5:
                    style_chars[l["style"]] += len(l["text"])
        total = sum(style_chars.values()) or 1
        body_styles = [s for s, c in style_chars.most_common() if c / total >= 0.08] or \
            ([style_chars.most_common(1)[0][0]] if style_chars else [("", 10.0, False)])
        primary = body_styles[0]
        body_families = {s[0] for s in body_styles}
        body_set = set(body_styles)
        for p in pages:          # each page is judged against its own dominant body style
            c = Counter()
            for l in p["lines"]:
                if l["style"] in body_set:
                    c[l["style"][1]] += len(l["text"])
            p["body_size"] = c.most_common(1)[0][0] if c else primary[1]

        elements: list[Element] = []
        notes: dict[int, list] = defaultdict(list)
        page_info = []
        for p in pages:
            pno, H = p["pno"] + 1, p["height"]
            tables, used = self._header_bar_tables(p)
            if not tables:
                tables, used = self._ruled_tables(p)
            lines = [l for i, l in enumerate(p["lines"]) if i not in used]
            lines = [l for l in lines if not ((l["bbox"][1] < 0.09 * H or l["bbox"][3] > 0.91 * H)
                                              and re.sub(r"[\d\s|]+", "", l["text"]).lower() in chrome)]
            # rebuild blocks from surviving lines
            by_block = defaultdict(list)
            for l in lines:
                by_block[l["block"]].append(l)
            blocks = []
            for bi, ls in by_block.items():
                ls.sort(key=lambda l: (round(l["bbox"][1]), l["bbox"][0]))
                bb = (min(l["bbox"][0] for l in ls), min(l["bbox"][1] for l in ls),
                      max(l["bbox"][2] for l in ls), max(l["bbox"][3] for l in ls))
                blocks.append({"bbox": bb, "lines": ls, "text": ""})
            blocks = reading_order(blocks, p["width"])
            p["charts"] = self._graphic_regions(p, [t.bbox for t in tables])
            page_els = list(tables)
            note_state = {"open": None}
            for b in blocks:
                page_els.extend(self._classify_block(b, pno, p, fixer, p["body_size"], body_families, notes, note_state))
            # blocks are already in reading order; slot each table in at its vertical position
            order = [e for e in page_els if e.kind != "table"]
            for t in tables:
                idx = next((i for i, e in enumerate(order) if e.bbox[1] > t.bbox[1] and
                            abs((e.bbox[0] + e.bbox[2]) / 2 - (t.bbox[0] + t.bbox[2]) / 2) < p["width"] / 2), len(order))
                order.insert(idx, t)
            merged = self._merge_figure_text(self._merge_lines(self._merge_dropcaps(order), fixer))
            elements.extend(self._reclassify_callouts(merged, fixer))
            pg = p["page"]
            img_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in p["images"]) / (p["width"] * H)
            page_info.append({"page": pno, "images": len(p["images"]), "image_area": round(min(img_area, 1), 3),
                              "drawings": len(pg.get_drawings()), "words": len(pg.get_text("words")),
                              "tables": len(tables)})

        toc = self._clean_toc()
        self._assign_sections(elements, toc, primary)
        title = (self.doc.metadata or {}).get("title") or next((e.text for e in elements if e.kind == "heading"), "")
        n_words = sum(len(e.text.split()) for e in elements)
        stats = {"body_styles": [list(s) for s in body_styles], "words": n_words,
                 "headings": sum(e.kind == "heading" for e in elements),
                 "tables": sum(e.kind == "table" for e in elements),
                 "notes": len(notes), "captions": sum(e.kind == "caption" for e in elements),
                 "figure_text": sum(e.kind == "figure_text" for e in elements),
                 "ligature_fixes": sum(1 for k, v in fixer._cache.items() if k != v),
                 "toc_entries": len(toc)}
        return ParsedDoc(self.path, title, len(self.doc), elements, dict(notes), toc, page_info, stats)

    def _classify_block(self, b, pno, p, fixer, min_body, body_families, notes, note_state) -> list[Element]:
        ls = b["lines"]
        # join lines, undoing line-end hyphenation
        text = ""
        for l in ls:
            t = l["text"].strip()
            if text.endswith("-") and t[:1].islower():
                left = re.search(r"([A-Za-z]+)-$", text)
                right = re.match(r"([a-z]+)", t)
                if left and right:
                    text = text[: left.start()] + fixer.join_hyphen(left.group(1), right.group(1)) + t[right.end():]
                    continue
            text = (text + " " + t) if text else t
        text = fixer.repair(text).strip()
        if not text:
            return []
        main = max(ls, key=lambda l: len(l["text"]))
        size, style = main["size"], main["style"]
        bb = b["bbox"]
        n_words = len(text.split())

        # footnote / endnote definitions: block starts with a marker
        m = re.match(r"^\[\^(\d+)\]\s*", text)
        small = size < 0.86 * min_body
        if not m and small and bb[1] > 0.7 * p["height"]:
            m = re.match(r"^(\d{1,3})[.)]?\s+(?=[A-Z])", text)
        if not m and note_state["open"] is not None:
            n, nstyle = note_state["open"]
            if abs(style[1] - nstyle[1]) < 0.3 and not text.startswith("[^"):
                notes[n][-1] = (notes[n][-1][0], notes[n][-1][1] + " " + text)   # continuation line
                return []
            note_state["open"] = None
        if m:
            parts = re.split(r"(?:^|\s)\[\^(\d+)\]\s*", " " + text) if text.startswith("[^") else ["", m.group(1), text[m.end():]]
            for i in range(1, len(parts) - 1, 2):
                body = parts[i + 1].strip()
                if body:
                    notes[int(parts[i])].append((pno, body))
                    note_state["open"] = (int(parts[i]), style)
            return []

        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        in_chart = n_words <= 12 and any(pymupdf.Rect(c.x0 - 20, c.y0 - 45, c.x1 + 20, c.y1 + 30).contains(pymupdf.Point(cx, cy))
                                         for c in p.get("charts", []))
        if in_chart:
            return [Element("figure_text", text, pno, bb, style=style, meta={"chart": True})]
        if FIG_TITLE.match(text) and n_words <= 30:
            return [Element("figure_text", text, pno, bb, style=style, meta={"figure_title": text})]

        near_img = any(bb[1] - 30 <= img[3] <= bb[1] + 4 or bb[3] - 4 <= img[1] <= bb[3] + 30
                       for img in p["images"] if not (bb[2] < img[0] or bb[0] > img[2]))
        letters = sum(ch.isalpha() for ch in text)
        if letters <= 2 and size >= 1.8 * min_body and len(text.strip()) <= 2:
            return [Element("dropcap", text.strip(), pno, bb, style=style)]
        is_heading = (n_words <= 14 and len(text) <= 130 and letters >= 3 and letters >= 0.5 * len(text.replace(" ", ""))
                      and len(ls) <= 3 and not text.rstrip().endswith((".", ",", ";"))
                      and not re.match(r"^(source|sources|note|notes)\s*:", text, re.I)
                      and (size >= 1.15 * min_body or
                           (size >= 0.95 * min_body and (main["bold"] or style[0] not in body_families
                                                          or _color_lum(main["color"]) < 0.9 and main["color"] != 0))))
        if is_heading and not CONTINUED.search(text):
            return [Element("heading", text, pno, bb, style=style, meta={"size": size, "bold": main["bold"]})]
        if is_heading:      # "BOX 1.1 ... (continued)" repeats on every page of the box
            return []
        if small:
            sentence_like = n_words >= 12 and re.search(r"[a-z][.!?](\s|$|\[\^)", text)
            if sentence_like and not near_img and not re.match(r"^(sources?|notes?)\b", text, re.I):
                return [Element("text", text, pno, bb, style=style, meta={"callout": True})]
            kind = "caption" if near_img else "figure_text"
            return [Element(kind, text, pno, bb, style=style)]
        return [Element("text", text, pno, bb, style=style)]

    @staticmethod
    def _reclassify_callouts(els: list[Element], fixer: TextRepair) -> list[Element]:
        """Small text that was split line-by-line can only be judged once merged: if the
        joined lines read as sentences and are not chart labels or source notes, it is prose."""
        for e in els:
            if e.kind != "figure_text" or e.meta.get("chart") or e.meta.get("figure_title"):
                continue
            joined = ""
            for line in e.text.split("\n"):
                line = line.strip()
                m = re.search(r"([A-Za-z]+)-$", joined)
                r = re.match(r"([a-z]+)", line)
                if m and r:
                    joined = joined[: m.start()] + fixer.join_hyphen(m.group(1), r.group(1)) + line[r.end():]
                else:
                    joined = (joined + " " + line).strip()
            if (len(joined.split()) >= 10 and re.search(r"[a-z][.!?](\s|$|\[\^)", joined)
                    and not re.match(r"^(sources?|notes?)\b", joined, re.I)):
                e.kind, e.text = "text", joined
                e.meta["callout"] = True
        return els

    @staticmethod
    def _merge_lines(els: list[Element], fixer: TextRepair | None = None) -> list[Element]:
        """Blocks that are really one paragraph (or one heading) split line by line are rejoined:
        same kind and style, same left edge or centred together, small vertical gap."""
        out: list[Element] = []
        for e in els:
            prev = out[-1] if out else None
            mergeable = e.kind in ("text", "heading") or (e.kind == "figure_text" and not e.meta.get("chart")
                                                         and not e.meta.get("figure_title"))
            if (prev and prev.kind == e.kind and mergeable and prev.style == e.style
                    and not prev.meta.get("chart") and not prev.meta.get("figure_title")):
                size = e.style[1] if e.style else 10
                gap = e.bbox[1] - prev.bbox[3]
                aligned = abs(prev.bbox[0] - e.bbox[0]) < 4 or \
                    abs((prev.bbox[0] + prev.bbox[2]) / 2 - (e.bbox[0] + e.bbox[2]) / 2) < 6
                if -2 < gap < 0.75 * size and aligned:
                    left = re.search(r"([A-Za-z]+)-$", prev.text)
                    right = re.match(r"([a-z]+)", e.text)
                    if left and right and fixer:
                        prev.text = prev.text[: left.start()] + fixer.join_hyphen(left.group(1), right.group(1)) + e.text[right.end():]
                    else:
                        prev.text = prev.text + " " + e.text
                    prev.bbox = (min(prev.bbox[0], e.bbox[0]), prev.bbox[1], max(prev.bbox[2], e.bbox[2]), e.bbox[3])
                    continue
            out.append(e)
        for e in out:     # small text that turned out to be sentences is a call-out, i.e. prose
            if e.kind == "figure_text" and not e.meta.get("chart") and not e.meta.get("figure_title") \
                    and len(e.text.split()) >= 12 and re.search(r"[a-z][.!?](\s|$|\[\^)", e.text) \
                    and not re.match(r"^(sources?|notes?)\b", e.text, re.I):
                e.kind, e.meta["callout"] = "text", True
        return out

    @staticmethod
    def _merge_dropcaps(els: list[Element]) -> list[Element]:
        """A decorative initial ("T" + "he United States...") is glued back onto its paragraph."""
        out, pending = [], None
        for e in els:
            if e.kind == "dropcap":
                pending = e.text
                continue
            if pending and e.kind == "text":
                e.text = pending + e.text
                pending = None
            out.append(e)
        return out

    @staticmethod
    def _merge_figure_text(els: list[Element]) -> list[Element]:
        """Consecutive figure_text blocks on a page become one figure element."""
        out: list[Element] = []
        for e in els:
            if e.kind == "figure_text" and out and out[-1].kind == "figure_text" and out[-1].page == e.page:
                prev = out[-1]
                if e.meta.get("figure_title") and prev.meta.get("figure_title"):
                    out.append(e)
                    continue
                prev.text += "\n" + e.text
                prev.meta["chart"] = prev.meta.get("chart") or e.meta.get("chart")
                prev.bbox = (min(prev.bbox[0], e.bbox[0]), min(prev.bbox[1], e.bbox[1]),
                             max(prev.bbox[2], e.bbox[2]), max(prev.bbox[3], e.bbox[3]))
            else:
                out.append(e)
        return out

    # -- structure ------------------------------------------------------------------------
    def _clean_toc(self):
        raw = [t for t in self.doc.get_toc() if t[2] >= 1]
        raw.sort(key=lambda t: t[2])
        if len(raw) < 3:
            return []
        cnt = Counter(t[1].strip() for t in raw)
        out = []
        for lvl, title, page in raw:
            if cnt[title.strip()] > 1:
                continue
            nested = "--" in title
            t = normalize_chars(re.sub(r"\s+", " ", title.replace("--", " "))).strip()
            t = re.sub(r"^Ch\.\s*(\d+)\s*", r"Chapter \1: ", t)
            if len(t) > 120 or len(t) < 2:
                continue
            depth = lvl - 1 + (1 if (nested or BOX_TITLE.match(t)) else 0)
            out.append({"title": t, "page": page, "depth": depth, "box": bool(BOX_TITLE.match(t))})
        if len(out) < 4 or out[0]["page"] > max(3, 0.15 * len(self.doc)):
            return []      # bookmarks that only cover part of the document mislead more than help
        mind = min(o["depth"] for o in out)
        for o in out:
            o["depth"] -= mind
        out.sort(key=lambda o: (o["page"], o["depth"]))
        return out

    @staticmethod
    def _sim(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z]{3,}", a.lower()))
        tb = set(re.findall(r"[a-z]{3,}", b.lower()))
        return len(ta & tb) / max(1, min(len(ta), len(tb)))

    def _assign_sections(self, els: list[Element], toc: list, primary_style: tuple):
        # heading levels from size ranks
        sizes = sorted({round(e.meta.get("size", 0)) for e in els if e.kind == "heading"}, reverse=True)
        rank = {s: min(i, 4) for i, s in enumerate(sizes)}
        for e in els:
            if e.kind == "heading":
                e.level = rank.get(round(e.meta.get("size", 0)), 4)

        # where each TOC entry starts: matching heading on its page, else page start
        starts = {}
        for ti, t in enumerate(toc):
            cand = [i for i, e in enumerate(els) if e.page == t["page"] and e.kind == "heading"
                    and self._sim(e.text, t["title"]) >= 0.6]
            if not cand:
                cand = [i for i, e in enumerate(els) if e.page >= t["page"]][:1]
            if cand:
                starts.setdefault(cand[0], []).append(ti)

        toc_stack: list[dict] = []
        font_stack: list[tuple[int, str]] = []
        box = None          # {"saved_toc", "saved_font", "style"}
        for i, e in enumerate(els):
            for ti in starts.get(i, []):
                t = toc[ti]
                toc_stack = [x for x in toc_stack if x["depth"] < t["depth"]] + [t]
                font_stack = []
                box = {"style": None, "saved_toc": [x for x in toc_stack if not x["box"]],
                       "saved_font": []} if t["box"] else None
            if e.kind == "heading":
                last = toc_stack[-1]["title"] if toc_stack else ""
                if last and self._sim(e.text, last) >= 0.6:
                    e.section_path = [x["title"] for x in toc_stack]
                    continue
                if BOX_TITLE.match(e.text) and box is None:
                    box = {"style": None, "saved_toc": list(toc_stack), "saved_font": list(font_stack)}
                    font_stack = [(-1, e.text)]
                else:
                    font_stack = [h for h in font_stack if h[0] < e.level] + [(e.level, e.text)]
            elif e.kind == "text" and box is not None and len(e.text) > 120:
                if box["style"] is None:
                    box["style"] = e.style
                elif e.style == primary_style and box["style"] != primary_style:
                    toc_stack, font_stack, box = box["saved_toc"], box["saved_font"], None
            e.section_path = [x["title"] for x in toc_stack] + [h[1] for h in font_stack]


def parse_pdf(path: str) -> ParsedDoc:
    return PDFParser(path).parse()
