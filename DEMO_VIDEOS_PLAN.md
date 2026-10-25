# Folio demo videos: plan and narration scripts

**Goal:** demonstrate the solution and explain in detail how it works. One video per document, about 6 minutes each. The viewers are technical, they watch the videos alone, and the narration is synthetic (no human voice).

| # | Document | Strategy Folio picks | What the video demonstrates |
|---|---|---|---|
| 1 | GPO 2018 Annual Report (82 pp) | narrative | Accurate answers from plain text, with citations you can check, and a refusal when the document is silent |
| 2 | World Bank GEP, June 2026 (200 pp) | long_structured | Finding the right region and section in a long structured report |
| 3 | Apple iMac Environmental Report (11 pp) | visual | Pulling facts out of tables, charts and footnotes |

## Format

Every video follows the same five parts:

| Part | Time | On screen | Purpose |
|---|---|---|---|
| 1. The problem | ~0:45 | The PDF, then a basic-RAG answer that fails | Show why this document type is hard |
| 2. Upload and profile | ~1:15 | Upload, live progress, **Document** tab | Explain how the strategy is chosen and what it changes |
| 3. Ask and answer | ~2:00 | **Compare** mode, then **Passages** tab | Show Folio against basic RAG, and the evidence behind each answer |
| 4. How it works | ~1:30 | **Pipeline** and **Graph** tabs, plus a diagram slide | Explain each step the question went through |
| 5. Refusal and recap | ~0:45 | A question the document can't answer, then a summary slide | Show grounding, then restate the techniques used |

Video 1 opens with a 45-second architecture slide, because viewers have no repo. Videos 2 and 3 open with a 10-second recap of it.

## Production

- **Screen:** record in 1920×1080 at 125% browser zoom, with the bookmarks bar hidden. Record silent clips, one per narration beat.
- **Narration:** `edge-tts` (already installed at `~/.local/bin/edge-tts`, free, no key). Use one voice for all three videos, such as `en-US-AndrewNeural` (check `edge-tts --list-voices`).
  - Generate one MP3 and one `.vtt` caption file per beat: `edge-tts --voice en-US-AndrewNeural --rate=-5% --text "..." --write-media b01.mp3 --write-subtitles b01.vtt`.
  - Trim or stretch each silent clip to its MP3's length, then join everything with `ffmpeg`. The `.vtt` files become burned-in captions.
  - Write numbers and abbreviations as they should be spoken ("G P O", "S S A"), or the voice will mangle them.
- **Diagram slide:** redraw the README's architecture diagram as a clean image (offline: parse, profile, chunk, vision, embed, graph; online: understand, retrieve, graph expand, rerank, assemble, generate, verify, reformulate once).
- **Environment:** use a fresh `data/` directory so the library shows only the demo document. Don't open `folio/.env` on camera. Embedded Qdrant allows one process, so upload through the UI rather than running `ingest` while the server is up.
- **Before recording:** set `GROQ_API_KEY` (and `GOOGLE_API_KEY` for the iMac chart pages) and rehearse every question in Compare mode twice. The repo's eval run used hashed embeddings and extractive answers, so some questions that failed there may pass with real models and some may not. Use only questions that give the right answer.

---

## Video 1: GPO annual report

Questions: `gpo-b1` (copies and titles), `gpo-s4` (govinfo), `gpo-r1` (FY 2020 refusal). Backup: `gpo-s3` (printing procured).

**Part 1: The problem**

> N0. "Before the demo, here is the whole system on one slide. Folio has two paths. The offline path runs once per PDF. It parses the layout into typed elements, profiles the document, chunks it according to that profile, optionally asks a vision model to describe charts, embeds the chunks with a local model, and builds a knowledge graph. Dense vectors go to Qdrant, keyword search uses BM25, and the chunks and the graph are saved as files. The online path is a LangGraph workflow that runs for every question. It understands the question, retrieves with dense and keyword search fused by rank, expands through the graph, reranks, assembles the context, generates an answer from numbered passages only, and then verifies it. If the answer refuses or fails verification, the workflow reformulates once. Every technique is switched on by what is measured in the PDF, never by its filename."
>
> N1. "This is the 2018 annual report of the U.S. Government Publishing Office. Eighty-two pages of mostly plain narrative text, with numbers written straight into the sentences. It is the baseline case, the one most teams meet first. Basic retrieval often does reasonably well here, so the question for this video is what Folio adds on top."
>
> N2. [PDF, two-column page] "Look at the layout. Two columns, drop caps, and a caption in the middle of the text. A basic pipeline cuts the text into fixed one-thousand-character chunks. That slices sentences in half and can mix the two columns together, so the number you need ends up separated from the words that explain it."
>
> N3. [Basic-RAG answer to "How many copies of how many titles did GPO distribute to depository libraries in FY 2018?"] "Here is a basic RAG answer to a simple question. It finds the right numbers. Keep that in mind, because for plain text the difference is not in the answer. It is in the evidence behind it, and in what happens when the document is silent."

**Part 2: Upload and profile**

> N4. [Upload; progress bar] "I upload the PDF. Indexing runs in the background and the progress is live: parsing, profiling, chunking, embedding, and building the knowledge graph."
>
> N5. [Document tab] "Folio doesn't run one recipe everywhere. It measures the PDF first: word count, page density, fonts, fill colours, bookmarks and tables. Here it finds about twenty-four thousand words of prose and almost no tables, so it labels the document narrative. The tab shows the measurements behind that decision, and I could re-index with a different strategy from here."
>
> N6. "That label sets three things. Parsing uses column reading order, so the columns don't interleave, and rejoins drop caps. Captions are split out as their own elements. And chunking works on whole sentences, about two hundred twenty tokens each, and never crosses a section boundary."

**Part 3: Ask and answer**

> N7. [Compare mode; ask the same question] "Now the same question in Compare mode. Basic RAG on the left, the Folio pipeline on the right."
>
> N8. "Folio gives the same numbers: eight hundred two thousand, eight hundred thirty-four copies of four thousand, two hundred eleven titles. Each claim carries a numbered citation."
>
> N9. [Passages tab] "The Passages tab shows each cited passage with its type, page, section path, how it was found, and its rank in each search list. The numbers from the answer are highlighted in the passage."
>
> N10. [Click Show page] "And 'Show page' renders the actual PDF page, so you can check the number against the source instead of trusting the model."

**Part 4: How it works**

> N11. [Diagram slide, online path highlighted] "Here is what happened to the question. First it's analysed. A fast model, or rules if there's no key, pulls out entities and decides whether to filter by section."
>
> N12. [Pipeline tab] "Retrieval runs two searches over the same chunks. Dense search finds passages with similar meaning, using local bge-small embeddings stored in Qdrant. BM25 finds exact words, and it's number-aware, so a figure like eight hundred two thousand is a searchable token. The two lists are combined with reciprocal rank fusion, which uses ranks rather than scores, so neither search needs tuning against the other."
>
> N13. [Graph tab] "A third list comes from the knowledge graph. It's built at index time with no LLM calls. Nodes are sections, chunks, pages, footnotes and entities. Chunks that mention the question's entities join the fusion. For a narrative document, the graph also widens each hit with its neighbouring sentences, which is how the full context reaches the model."
>
> N14. [Pipeline tab, rerank step] "A cross-encoder then reranks the top candidates by reading the question and each passage together. That's slower than the first-stage search, which is why it only sees a short list."
>
> N15. "Generation is grounded: the prompt allows answers only from the numbered passages and requires citations. Then verification runs. It's deterministic code, not another model. Every citation number must exist, and every number in the answer must appear in a cited passage. If that fails, the workflow reformulates once, using the original wording plus a hypothetical answer, and keeps the better-grounded attempt."

**Part 5: Refusal and recap**

> N16. [Ask: "How many FDLP Academy webinars did GPO host, and how many participants did the programs serve?"] "A multi-part question: how many F D L P Academy webinars did GPO host, and how many participants did the programs serve? Folio answers both parts, eighty-five webinars and more than ten thousand participants, and cites the passage."
>
> N17. [Ask: "What was GPO's total revenue in FY 2020?"] "Now a question this document can't answer. It's the 2018 report. Folio answers with the exact refusal text: 'I could not find this in the document.' It declines instead of guessing."
>
> N18. [Recap slide] "To recap: profile the document, chunk by whole sentences, search with dense, keyword and graph signals fused together, rerank, answer only from numbered evidence, and verify. Next, the same system on a two-hundred-page report."

---

## Video 2: World Bank Global Economic Prospects

Questions (checked against the live app with real models): `gep-b1` (South Asia), `gep-s4` (East Asia and Pacific), `gep-b2` (the region that includes Pakistan), `gep-s8` (Box 1.1), `gep-r1` (S&P 500 refusal).

**Part 1: The problem**

> N0. "A quick reminder of the architecture. An offline path profiles each PDF and builds the index and a knowledge graph. An online path retrieves, expands through the graph, reranks, generates from numbered passages, and verifies. The strategy for each document is chosen by what is measured in it."
>
> N1. "This is the World Bank's Global Economic Prospects report from June 2026. Two hundred pages, organised into chapters, regions, sub-sections, boxes and a statistics appendix. Here the structure is the information."
>
> N2. [PDF: bookmarks panel] "Growth forecasts appear in six regional chapters. A question like 'what is the growth outlook for South Asia' matches all of them, because similar text sits in every section. Plain similarity search picks whichever happens to score highest, so you can get East Asia's numbers in a South Asia answer."
>
> N3. [Basic-RAG answer] "Here is basic RAG's answer to the same question, using one thousand character chunks and a dense top five. In this run it is correct. Nothing in that pipeline guarantees it, though: it has no section path, no verification, and no way to say why it picked these passages."

**Part 2: Upload and profile**

> N4. [Upload; Document tab] "Folio measures this PDF at two hundred pages and one hundred twelve thousand words, with a bookmark outline, and labels it long structured."
>
> N5. "Parsing handles three problems in this particular file. Headings come from the bookmarks and font sizes, so every passage carries a full heading path such as chapter, region, section. Boxed articles are closed when the body style resumes, so box text doesn't leak into the next section. And a ligature problem turns 'The' into 'Te' in extracted text, so a repair step uses the document's own vocabulary to restore it."
>
> N6. [Document tab: abbreviations and routable sections] "The tab lists the abbreviations it mined, using the Schwartz–Hearst method: SAR means South Asia, SSA means Sub-Saharan Africa. It also lists the sections the system can route to."
>
> N7. "Chunking is two-level. Small child chunks of about two hundred eighty tokens are searched, because short text matches precisely. When a child is retrieved, the answer is written from its parent section, about nine hundred tokens, so the model sees the whole argument. Long tables are split by rows with the header repeated in each piece, so a value never loses its column names."

**Part 3: Ask and answer**

> N8. [Compare; "What is the growth outlook for South Asia in this report?"] "Same question, both pipelines. Basic RAG's answer is also six point three percent here, so the answer alone does not separate them. Watch what Folio adds: the section it found, how each passage was ranked, and a verified citation."
>
> N9. [Passages tab] "Each passage shows its section path. Here it's the South Asia region, not a lookalike. The rank columns show which search found it: dense, keyword, or graph."
>
> N10. [Ask: "How fast is growth in East Asia and Pacific expected to be in 2026?"] "Now a different region. Folio switches sections and doesn't mix the two."
>
> N11. [Ask: "What growth is projected in 2026 for the region that includes Pakistan?"] "Now a question that names a country, not a region. In this report, Pakistan sits in the Middle East, North Africa, Afghanistan and Pakistan region. Folio finds that region's section and returns one point six percent for twenty twenty-six, with a citation."

**Part 4: How it works**

> N12. [Graph tab] "How does it find the right section? The graph links entities and aliases, for example SAR and South Asia, and chunks that sit in a section named after an entity join the fusion as a third ranked list. The Graph tab shows the hops it took."
>
> N13. [Pipeline tab, query analysis] "Section routing is a soft filter. When a question names a section, Folio fuses filtered results with unfiltered ones, so a wrong filter can't hide the answer. It only sets a filter in two cases. 'Chapter N' must be written literally in the question. And ambiguous headings such as 'Outlook' are never used as filters, because they appear in every chapter."
>
> N14. "On long documents, Folio also demotes chart-label text, which matches queries well but carries no explanation. The reranker then orders what's left."
>
> N15. [Ask a question that triggers the retry, if you have a reliable one] "If the first answer is a refusal or fails verification, the workflow retries once with the original wording plus a hypothetical answer, a technique called HyDE, and with the section filter dropped. The better-grounded attempt wins."

**Part 5: Refusal and recap**

> N16. [Ask: "In Box 1.1, by how much could AI raise productivity growth per year under optimistic estimates?"] "Boxes are first-class sections. Folio goes straight to Box 1.1."
>
> N17. [Ask: "What level does the report forecast for the S&P 500 at the end of 2026?"] "A question outside the report. Folio refuses."
>
> N18. [Recap slide] "To recap: heading paths from bookmarks and fonts, parent and child chunks, abbreviation mining, soft section routing, graph aliases, and a one-time HyDE retry. Next, the shortest document and the hardest to read."

---

## Video 3: Apple iMac Environmental Report

Questions: `imac-s1` (346 total footprint), `imac-s3` (sleep power 0.57), `imac-b3` (ENERGY STAR and its footnote), `imac-s7` (does the recycled copper apply to every model), `imac-r1` (battery refusal). Confirm each with real models, since several of these missed in the repo's own eval.

**Part 1: The problem**

> N0. "A quick reminder of the architecture. An offline path profiles each PDF and builds the index and a knowledge graph. An online path retrieves, expands through the graph, reranks, generates from numbered passages, and verifies. The strategy for each document is chosen by what is measured in it."
>
> N1. "This is Apple's environmental report for the iMac. Eleven pages, but almost none of the facts are in sentences."
>
> N2. [PDF: carbon chart, power table, icon call-outs, footnote markers] "They're in a carbon emissions bar chart, a power consumption table, icon call-outs like 'one hundred percent recycled aluminum', and numbered footnotes that limit what each claim means. A plain text extractor reads a table as a pile of numbers with no column names, and drops footnotes to the bottom of the page."
>
> N3. [Basic-RAG answer to "What was the baseline emissions figure for iMac, and how many kilograms did the reductions remove?"] "Basic RAG answers: I could not find this in the document. The two figures are printed in the carbon chart on page two."

**Part 2: Upload and profile**

> N4. [Upload; Document tab] "Folio detects a visual document: twelve pages, three tables, and twenty-seven footnote markers. That profile switches on four techniques."
>
> N5. "First, tables are rebuilt under their header bars. Folio reads the fill colours and drawing positions in the PDF to find the shaded header row, then keeps every value attached to its column. Second, footnote markers are linked to their note text, and the note is attached to the chunk that cites it. Third, chart regions are grouped into one element. Fourth, chunking is one chunk per table, chart or call-out, so nothing is split."
>
> N6. [Document tab] "The reranker is turned off for this document. It's trained on prose and under-rates tables, so Folio leaves ranking to fusion."
>
> N7. "Chart pages can also be described by a vision model when a Google key is set. This demo runs without it, and the chart's printed labels are still indexed, which is enough for the question you are about to see."

**Part 3: Ask and answer**

> N8. [Compare; same question] "Same question again. Folio returns four hundred forty-five kilograms of C O two e as the baseline, and ninety-nine kilograms removed by the reductions, with a citation."
>
> N9. [Passages tab → Show page] "The passage is the chart's own printed labels, kept together as one element. Show page opens the page, so you can check both numbers against the chart."
>
> N10. [Ask: "What is the carbon footprint of the iMac (four ports) 512GB configuration?"] "A value inside a table. The iMac with four ports and five hundred twelve gigabytes has a footprint of three hundred ninety-one kilograms. Basic RAG refuses. Folio keeps the value attached to its configuration, with a citation."
>
> N11. [Ask: "How much less energy than the ENERGY STAR requirement does iMac use, and what is that figure based on?"] "A two-part question. Fifty-eight percent, and the footnote explains the program requirements it's measured against. The footnote text is part of the same passage."
>
> N12. [Ask: "Does the recycled copper in the thermal module apply to every iMac model?"] "Footnotes that limit a claim matter. Here the note says it applies to the four-port models only."

**Part 4: How it works**

> N13. [Graph tab] "The graph does the connecting. A footnote edge links a chunk to its note, a page edge links chunks that share a page, and a mentions edge links chunks to entities."
>
> N14. "For a visual document, a prose hit also pulls in the chart or table on the same page, through the page edge. That's why the answer to a prose-style question can still come with the table value."
>
> N15. [Pipeline tab] "Dense search, BM25 with number-aware tokens, and graph hits are fused by reciprocal rank. With no reranker, the fused order is what the model sees."

**Part 5: Refusal and recap**

> N16. [Ask: "What is the battery capacity of the iMac?"] "The report doesn't say. Folio refuses."
>
> N17. [Recap slide] "To recap: tables rebuilt under their headers, footnotes attached to the claims they limit, charts grouped and optionally described, one chunk per element, graph links between elements on the same page, and the reranker switched off where it hurts. That's three document types, one system, and the strategy is chosen by what's measured in the PDF."

---

## Schedule

| Step | Output | Time |
|---|---|---|
| Keys, rehearsal, final question list | Confirmed questions per video | 0.5 day |
| Diagram and recap slides | 1 architecture slide, 3 recap slides | 0.5 day |
| Generate narration | MP3 and VTT per beat, 3 videos | 0.5 day |
| Record screen clips | Silent clips per beat | 1 day |
| Assemble and caption | 3 final MP4s | 1 day |

## Open decisions

- Which `edge-tts` voice to use (I can generate samples).
- Whether to add a short title card per video, or start straight at part 1.
