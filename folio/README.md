# Folio: document-aware graph RAG

A Django chatbot that answers questions about uploaded PDFs with **numbered citations you can check**, and refuses when the document is silent.

It does not run one RAG recipe everywhere. Each document is profiled on upload, and the profile switches individual techniques on or off. Every technique is triggered by something measured in the PDF (fonts, fill colours, page density, bookmarks), never by the filename.

| | GPO annual report | World Bank GEP | Apple iMac report |
|---|---|---|---|
| Detected as | Plain text (24k words, prose) | Long, structured (112k words, 200 pages) | Visual (12 pages, 3 tables, 27 footnote markers) |
| How basic RAG breaks | Columns interleaved, chunks cut mid-sentence | Similar sections everywhere ("growth forecasts" in six regions); retrieval returns table fragments | Values separated from their columns; caveats in endnotes; the key fact is in a chart |
| Parsing fix | Column reading order; captions split out; drop caps rejoined | "Te" → "The" ligature repair; heading path from bookmarks and fonts; boxes closed when body style resumes | Tables rebuilt under their header bars; footnote markers → notes; chart regions grouped |
| Chunking | Whole sentences, about 220 tokens, never crossing a section | Children about 280 tokens for search; parents about 900 tokens for the answer; long tables split by rows with the header repeated | One chunk per table, chart or call-out; footnote text attached |
| Retrieval | Hybrid (dense + BM25) + graph entities, RRF, cross-encoder | Same, plus section routing (soft filter) and chart-label demotion | Hybrid + graph; reranker off (prose-trained, under-rates tables) |
| Context for the LLM | Chunk widened with its neighbours | Parent section | Element plus same-page chart or table, via the graph |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY (free); GOOGLE_API_KEY optional (charts)
python manage.py migrate
python manage.py runserver      # http://127.0.0.1:8000
```

Upload PDFs in the left pane. They index in the background, with live progress. The first run downloads the local embedding model and reranker (about 150 MB, once).

To load the three challenge PDFs from the command line and score them, **stop the web server first** (see Vector DB below):

```bash
python manage.py ingest GPO-Anual-report.pdf GEP-Jun-2026.pdf iMac_PER_Oct2024.pdf
python manage.py evaluate       # writes eval/results.md and eval/results.json
pytest && python manage.py test chat
```

**Keys are optional.** With no Groq key, query routing uses rules and answers are extractive (the best-matching cited sentences). Add the key for real answers.

**Vector DB.** Embedded Qdrant allows one process at a time, and `ingest` and `evaluate` give a clear error message if the server holds it. To run both at once, start a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`) and set `QDRANT_URL`.

## Using it

- **Ask.** Tick documents to scope a question, or leave all unticked to ask everything. Pick one of three modes:
  - **Folio pipeline** runs the full workflow.
  - **Basic RAG** runs the guide's baseline: 1,000-character chunks, dense top-5, same prompt.
  - **Compare** runs both side by side.
- **Evidence pane.**
  - **Passages** shows each passage with its type, page, section path, how it was found and its per-list ranks. The answer's numbers are highlighted inside the passage, and "Show page" renders the PDF page.
  - **Pipeline** shows every step with its timing, plus the query analysis.
  - **Graph** shows the question's entities, the passages, sections and footnotes used.
  - **Document** shows why the document got its strategy, its stats, the abbreviations found and the routable sections. You can re-index with another strategy from here.
- **Evaluate.** Basic vs Folio, per document, with per-question results.

## Architecture

```
OFFLINE (once per PDF)                          ONLINE (LangGraph, every question)
parse ─ profile ─ chunk ─ [vision] ─ embed      understand → retrieve → graph_expand → rerank
  │                          │                       → assemble → generate → verify ─┐
  └──── knowledge graph ─────┘                           ▲                            │
            ▼                                            └── reformulate (once) ◄─────┘
  Qdrant (dense) + BM25 + chunks/parents/graph JSON
```

| File | Role |
|---|---|
| `rag/ingest/parser.py` | Layout-aware parsing into typed elements (heading, text, table, figure_text, caption) in reading order, with heading paths and footnote definitions |
| `rag/ingest/textfix.py` | Ligature, dropped-"Th" and hyphenation repair, using the document's own vocabulary |
| `rag/ingest/profile.py` | Measured features → strategy |
| `rag/ingest/vision.py` | Gemini describes chart pages, cached per (file, page) |
| `rag/chunking.py` | Sentence, hierarchical and element chunkers; footnotes attached; naive baseline |
| `rag/embeddings.py` | LangChain `Embeddings`: fastembed bge-small (default), Gemini, or an offline hash fallback |
| `rag/store.py` | Qdrant + BM25 (number-aware, light stemming); RRF fusion over any named ranked lists |
| `rag/graph.py` | Knowledge graph: build, Schwartz–Hearst abbreviation mining, entity hits, companions |
| `rag/query.py` | Query understanding (fast LLM, or rules over the outline) |
| `rag/rerank.py` | MiniLM cross-encoder (local) |
| `rag/generate.py` | Grounded prompt, context assembly, verification, extractive fallback |
| `rag/workflow.py` | LangGraph state machine and the naive baseline |
| `rag/evaluate.py` | Metrics and harness |
| `chat/` | Django models, function-based JSON views, background ingestion, UI, management commands |

### Graph-based RAG

The knowledge graph is built at index time with no LLM calls, so it costs nothing against free-tier limits.

**Nodes:** sections, chunks, parents, footnotes, pages, and entities (acronyms, proper-noun phrases, abbreviation long forms).

**Edges:**
- `has_section`, `contains`, `next` and `part_of` carry the document's structure and reading order.
- `cites` links a chunk to its footnote.
- `on_page` links a chunk to its page.
- `mentions` links a chunk to an entity.
- `alias_of` links an abbreviation to its long form (SAR ↔ South Asia).

At question time it does three jobs:
1. **Entity hits.** Chunks that mention the question's entities, or their aliases, or sit in a section named after one, become a third ranked list in the fusion. This is how "South Asia" reaches a passage that only says "SAR".
2. **Companions.** It pulls in the chart or table on the same page as a prose hit (visual documents), the neighbouring sentences (plain text), or the parent section (long documents).
3. **Explanation.** The hops taken are shown in the Graph tab.

### Hallucination controls

1. The prompt allows answers from the numbered passages only, with citations required. The exact refusal text is "I could not find this in the document."
2. Verification is deterministic: every `[n]` must exist, and every number in the answer must appear in a cited passage.
3. If the model refuses or the answer fails verification, the workflow retries once. The retry uses the original wording plus a hypothetical answer (HyDE), with the section filter dropped. The better-grounded attempt wins.
4. Section filters are soft (fused with unfiltered results) and only set when the question names a section. "Chapter N" must be literal, and ambiguous headings such as "Outlook" are never filters. Both rules follow the guide's lesson that a wrong filter hides the answer.

## Results

37 hand-checked questions, with gold pages and answer facts verified against the PDF text, are in `eval/questions.json`.

Retrieval is scored on **what the LLM actually sees**: the rank of the first passage containing the evidence.

`eval/results.md` was produced in the build sandbox, which could not download models or reach Groq. Both pipelines therefore ran with **hashed bag-of-words embeddings, no reranker, and extractive answers**:

| Document | MRR basic → Folio | hit@1 basic → Folio | Answers basic → Folio |
|---|---|---|---|
| GPO (plain text) | 0.72 → **0.80** | 60% → **70%** | 27% → **64%** |
| GEP (long, structured) | 0.34 → **0.53** | 23% → **31%** | 36% → **43%** |
| iMac (visual) | 0.52 → **0.88** | 36% → **82%** | 17% → **42%** |

The gap is largest where documents are hardest, as the guide found. Run `python manage.py evaluate` with real embeddings and a Groq key to get your own numbers. With about 12 questions per document, treat differences under about 0.1 as noise.

## Known limits and next steps

- **Thresholds are tuned on these three PDFs.** Heading sizes, header-bar luminance and the chart-region heuristics transfer in principle, but measure on your own documents. The strategy can be overridden on upload or re-index.
- **Vision is off without `GOOGLE_API_KEY`.** Chart labels are still indexed, but the relationship "445 − 99 = 346" comes from the vision description.
- **Groq free tier is about 30 requests per minute.** A question costs 2 calls (4 on a retry). The evaluation pauses 1 s between questions (`EVAL_SLEEP`).
- **Next steps:**
  - An LLM-as-judge for answers.
  - LLM-extracted relations in the graph, for cross-document questions.
  - Multi-query and decomposition beyond the current sub-questions.
  - Qdrant server mode for multi-worker deployments.

Build log and task schedule: `TASKS.md`.
