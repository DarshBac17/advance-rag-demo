# Folio build schedule

Status: [x] done · [~] in progress · [ ] todo

| #  | Task | Depends on | Exit criteria |
|----|------|-----------|---------------|
| T1 ✅ | Layout-aware parser (finish: call-out reclassification) | — | parser tests green, iMac p5 call-outs are prose |
| T2 ✅ | Document profiler + three chunking strategies | T1 | GPO→narrative, GEP→long_structured, iMac→visual; chunk tests green |
| T3 ✅ | Embeddings, Qdrant store, BM25, hybrid RRF, naive baseline | T2 | index + search round-trip test green |
| T4 ✅ | Knowledge graph build + graph expansion | T2 | alias "SAR"↔"South Asia" mined; footnote/page/parent hops tested |
| T5 ✅ | Query understanding, LangGraph workflow, generation, verification | T3, T4 | end-to-end answer with citations; refusal on FY 2020 |
| T6 ✅ | Django project, JSON views, background ingestion | T5 | upload → ready → ask via HTTP test client |
| T7 ✅ | UI (library, chat, evidence + trace, compare, eval tab) | T6 | renders, uploads, shows citations and trace |
| T8 ✅ | Evaluation set + `evaluate` command + UI scorecard | T5, T7 | naive vs advanced table for all three PDFs |
| T9 ✅ | README, .env.example, packaging | all | zip delivered |
