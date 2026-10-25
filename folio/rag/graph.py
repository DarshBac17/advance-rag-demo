"""The document knowledge graph behind graph-based retrieval.

Built at index time with no LLM calls (so it costs nothing against free-tier limits).

Nodes                          Edges
  section  "sec:<path>"          section  -has_section->  section
  chunk    "chunk:<id>"          section  -contains->     chunk
  parent   "parent:<id>"         chunk    -next->         chunk         (reading order, same section)
  note     "note:<n>"            chunk    -part_of->      parent
  page     "page:<n>"            chunk    -cites->        note
  entity   "ent:<name>"          chunk    -on_page->      page
                                 chunk    -mentions->     entity
                                 entity   -alias_of->     entity        (SAR <-> South Asia)

At question time the graph does three jobs:
  1. entity hits   chunks that mention the question's entities or their aliases, ranked by
                   IDF; fused with dense and BM25 as a third ranked list
  2. companions    evidence that belongs with a hit: the chart or table on the same page,
                   the sentences before/after, the parent section
  3. explanation   the hops taken, shown in the UI trace
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import networkx as nx

GENERIC = {"chapter", "box", "figure", "table", "annex", "section", "page", "note", "notes", "source", "sources",
           "percent", "report", "summary", "introduction", "conclusion", "appendix"}
STOP = {"the", "and", "of", "for", "in", "on", "a", "an", "to", "with", "by", "at", "or", "its", "their"}
NOT_ACRONYM = {"I", "A", "II", "III", "IV", "VI", "OK", "AM", "PM", "RHS", "LHS", "NA", "TBD", "FY", "Q1", "Q2", "Q3", "Q4"}
_ACR = re.compile(r"\b([A-Z][A-Z0-9&]{1,7})(s)?\b")
_DEF = re.compile(r"([^()]{3,140})\(\s*([A-Z][A-Za-z0-9&]{1,9})\s*\)")
_CAPS = re.compile(r"\b([A-Z][a-z]+(?:[-’'][A-Za-z]+)?(?:\s+(?:of|and|the|for|de)?\s*[A-Z][a-z]+(?:[-’'][A-Za-z]+)?){1,5})\b")
_WORDCAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{3,})\b")


# --------------------------------------------------------------------------- aliases
def best_long_form(short: str, long: str) -> str | None:
    """Schwartz & Hearst (2003): match the short form's letters right-to-left inside the
    candidate long form; the first letter must start a word."""
    s = short.rstrip("s") if short.endswith("s") and short[:-1].isupper() else short
    si, li = len(s) - 1, len(long) - 1
    while si >= 0:
        ch = s[si].lower()
        if not ch.isalnum():
            si -= 1
            continue
        while li >= 0 and (long[li].lower() != ch or (si == 0 and li > 0 and long[li - 1].isalnum())):
            li -= 1
        if li < 0:
            return None
        li -= 1
        si -= 1
    start = long.rfind(" ", 0, li + 1) + 1
    lf = long[start:].strip(" ,;:")
    while lf.split() and lf.split()[0].lower() in STOP:      # "of the Federal Register" -> "Federal Register"
        lf = lf.split(" ", 1)[1] if " " in lf else ""
    words = lf.split()
    if len(words) < 2 or len(words) > len(s) + 5 or "\n" in lf:     # S&H window: |A| + 5 words
        return None
    return lf


def _prefix_initials(short: str, long: str) -> str | None:
    """'South Asia (SAR)': word initials form the short form minus one trailing letter."""
    s = short.rstrip("s") if short.endswith("s") and short[:-1].isupper() else short
    words = [w.strip(",:;") for w in long.split()]
    for k in range(min(len(s), 6), 1, -1):
        tail = [w for w in words[-(k + 3):] if w]
        content = [w for w in tail if w.lower() not in STOP]
        content = content[-k:]
        if len(content) == k and all(w[:1].isupper() for w in content):
            ini = "".join(w[0] for w in content).upper()
            if s.upper().startswith(ini) and len(s) - len(ini) == 1:
                first = tail.index(content[0])
                return " ".join(tail[first:])
    return None


def mine_aliases(text: str) -> dict[str, str]:
    out: dict[str, Counter] = defaultdict(Counter)
    for m in _DEF.finditer(text):
        before, short = m.group(1), m.group(2)
        if short in NOT_ACRONYM or not re.search(r"[A-Z].*[A-Z]", short):
            continue
        lf = best_long_form(short, before) or _prefix_initials(short, before)
        if lf:
            out[short.lower()][lf.lower()] += 1
    return {s: c.most_common(1)[0][0] for s, c in out.items()}


# --------------------------------------------------------------------------- entities
def _norm(s: str) -> str:
    return re.sub(r"[’']s$", "", s.strip().lower())


def extract_entities(text: str, aliases: dict, long_forms: set) -> set[str]:
    ents = set()
    for m in _ACR.finditer(text):
        a = m.group(1) + (m.group(2) or "")
        if a not in NOT_ACRONYM and not a.isdigit() and len(a) >= 2:
            ents.add(a.lower())
    low = text.lower()
    for lf in long_forms:
        if lf in low:
            ents.add(lf)
    for m in _CAPS.finditer(text):
        ents.add(_norm(m.group(1)))
    for m in _WORDCAP.finditer(text):
        ents.add(_norm(m.group(1)))
    return ents


def build_graph(chunks, parents: dict, pd=None) -> nx.DiGraph:
    G = nx.DiGraph()
    recs = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]
    corpus = " ".join(e.text for e in pd.elements) if pd else " ".join(r["text"] for r in recs)
    aliases = mine_aliases(corpus)
    long_forms = set(aliases.values())
    G.graph["aliases"] = aliases

    # sections
    for r in recs:
        path = r["section_path"]
        for i in range(len(path)):
            sid = "sec:" + " > ".join(path[: i + 1])
            G.add_node(sid, type="section", title=path[i], depth=i)
            if i:
                G.add_edge("sec:" + " > ".join(path[:i]), sid, rel="has_section")
    for pid, p in parents.items():
        G.add_node(f"parent:{pid}", type="parent", pages=p["pages"])

    # chunks + structure
    ents_by_chunk = {}
    prev_by_section = {}
    for r in sorted(recs, key=lambda r: r["order"]):
        cid = f"chunk:{r['id']}"
        G.add_node(cid, type="chunk", kind=r["kind"], page=r["page"])
        sec = "sec:" + " > ".join(r["section_path"]) if r["section_path"] else None
        if sec:
            G.add_edge(sec, cid, rel="contains")
        key = tuple(r["section_path"])
        if r["kind"] == "text":
            if key in prev_by_section:
                G.add_edge(prev_by_section[key], cid, rel="next")
            prev_by_section[key] = cid
        if r.get("parent_id"):
            G.add_edge(cid, f"parent:{r['parent_id']}", rel="part_of")
        for n in r.get("notes", []):
            G.add_edge(cid, f"note:{n}", rel="cites")
            G.nodes[f"note:{n}"]["type"] = "note"
        G.add_edge(cid, f"page:{r['page']}", rel="on_page")
        G.nodes[f"page:{r['page']}"]["type"] = "page"
        ents_by_chunk[cid] = extract_entities(r["text"], aliases, long_forms)

    # entities: keep the informative ones (seen in >=2 chunks, <=25% of chunks) + all aliases
    df = Counter(e for es in ents_by_chunk.values() for e in es)
    n = max(len(recs), 1)
    keep = {e for e, c in df.items() if 2 <= c <= max(3, 0.25 * n)} | set(aliases) | long_forms
    for cid, es in ents_by_chunk.items():
        for e in es & keep:
            eid = f"ent:{e}"
            if eid not in G:
                G.add_node(eid, type="entity", name=e, df=df.get(e, 1),
                           idf=round(math.log((n + 1) / (df.get(e, 0) + 1)) + 1, 3))
            G.add_edge(cid, eid, rel="mentions")
    for s, lf in aliases.items():
        for a, b in ((s, lf), (lf, s)):
            for x in (a, b):
                if f"ent:{x}" not in G:
                    G.add_node(f"ent:{x}", type="entity", name=x, df=0, idf=1.0)
            G.add_edge(f"ent:{a}", f"ent:{b}", rel="alias_of")
    return G


def save_graph(G: nx.DiGraph, path):
    data = nx.node_link_data(G, edges="links")
    path.write_text(json.dumps(data))


def load_graph(path) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(path.read_text()), edges="links")


# --------------------------------------------------------------------------- query time
class DocGraph:
    def __init__(self, G: nx.DiGraph, chunks):
        self.G = G
        recs = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]
        self.chunks = {r["id"]: r for r in recs}
        self.aliases = G.graph.get("aliases", {})
        self._entities = {d["name"]: d for n, d in G.nodes(data=True) if d.get("type") == "entity"}

    def query_entities(self, question: str) -> list[tuple[str, str]]:
        """Entities named in the question, plus their aliases. Returns (entity, how)."""
        q = question.lower()
        found = {}
        for name in self._entities:
            if len(name) < 2 or name in GENERIC:
                continue
            if " " in name:            # multi-word names match case-insensitively
                hit = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", q)
            else:                      # single words only as written proper nouns / acronyms
                hit = re.search(rf"(?<![A-Za-z0-9])({re.escape(name.capitalize())}|{re.escape(name.upper())}"
                                rf"|{re.escape(name.upper()[:-1]) + 's' if name.endswith('s') else '#'})(?![A-Za-z0-9])",
                                question)
                if hit and hit.start() == 0 and not name.upper() == hit.group(0):
                    hit = None         # sentence-initial capital is not evidence of a proper noun
            if hit:
                found[name] = "named"
        for name in list(found):
            for _, other, d in self.G.out_edges(f"ent:{name}", data=True):
                if d.get("rel") == "alias_of":
                    found.setdefault(self.G.nodes[other]["name"], f"alias of {name}")
        # prefer the longest matches: drop entities contained in a longer named one
        names = sorted(found, key=len, reverse=True)
        keep = [n for n in names if found[n] != "named" or not any(n != m and n in m and found[m] == "named" for m in names)]
        return [(n, found[n]) for n in keep]

    def section_hits(self, names: list[str]) -> dict[str, str]:
        """Graph hop entity -> section with that title -> every chunk inside it."""
        out = {}
        for node, d in self.G.nodes(data=True):
            if d.get("type") == "section" and re.sub(r"\s+", " ", d["title"]).lower().strip() in names:
                for desc in nx.descendants(self.G, node):
                    if desc.startswith("chunk:"):
                        out[desc.split(":", 1)[1]] = d["title"]
        return out

    def entity_hits(self, question: str, sections=None, limit=20) -> list[tuple[str, float]]:
        scores = Counter()
        ents = self.query_entities(question)
        in_named_section = self.section_hits([n for n, _ in ents])
        for cid in in_named_section:
            scores[cid] += 1.0
        for name, _ in ents:
            idf = self._entities.get(name, {}).get("idf", 1.0)
            for src, _, d in self.G.in_edges(f"ent:{name}", data=True):
                if d.get("rel") != "mentions":
                    continue
                cid = src.split(":", 1)[1]
                c = self.chunks.get(cid)
                if c and (not sections or any(s in c["section_path"] for s in sections)):
                    scores[cid] += idf
        return scores.most_common(limit)

    def expand_query(self, question: str) -> str:
        """Add alias surface forms so BM25 can match 'SAR' when the question says 'South Asia'."""
        extra = [n for n, how in self.query_entities(question) if how.startswith("alias")]
        return question + (" " + " ".join(extra) if extra else "")

    def companions(self, chunk_id: str, profile: str, limit=3) -> list[tuple[str, str]]:
        node = f"chunk:{chunk_id}"
        if node not in self.G:
            return []
        out = []
        if profile in ("narrative", "long_structured"):
            for p, _, d in self.G.in_edges(node, data=True):
                if d.get("rel") == "next":
                    out.append((p.split(":", 1)[1], "previous"))
            for _, n, d in self.G.out_edges(node, data=True):
                if d.get("rel") == "next":
                    out.append((n.split(":", 1)[1], "next"))
        page = self.chunks[chunk_id]["page"]
        kind = self.chunks[chunk_id]["kind"]
        for src, _, d in self.G.in_edges(f"page:{page}", data=True):
            cid = src.split(":", 1)[1]
            if cid == chunk_id:
                continue
            ck = self.chunks.get(cid, {}).get("kind")
            # prose pulls in the page's chart/table; a chart pulls in the page's prose
            if (kind == "text" and ck in ("table", "figure", "vision")) or \
               (kind in ("table", "figure", "vision") and ck in ("text", "vision", "figure", "table")):
                if profile == "visual" or ck in ("table", "vision"):
                    out.append((cid, f"same page ({ck})"))
        return out[:limit]

    def subgraph(self, chunk_ids: list[str], question: str) -> dict:
        """Small explanation graph for the UI: query entities -> chunks -> sections/notes."""
        nodes, edges = {}, []
        for name, how in self.query_entities(question)[:6]:
            nodes[f"ent:{name}"] = {"label": name, "type": "entity", "how": how}
        for cid in chunk_ids:
            n = f"chunk:{cid}"
            if n not in self.G:
                continue
            c = self.chunks[cid]
            nodes[n] = {"label": f"p.{c['page']} {c['kind']}", "type": "chunk"}
            for _, t, d in self.G.out_edges(n, data=True):
                if d["rel"] == "mentions" and t in nodes:
                    edges.append([t, n, "mentioned in"])
                elif d["rel"] == "cites":
                    nodes[t] = {"label": t.replace("note:", "note "), "type": "note"}
                    edges.append([n, t, "cites"])
            for s, _, d in self.G.in_edges(n, data=True):
                if d["rel"] == "contains":
                    nodes[s] = {"label": self.G.nodes[s].get("title", s), "type": "section"}
                    edges.append([s, n, "contains"])
        return {"nodes": [{"id": k, **v} for k, v in nodes.items()], "edges": edges}
