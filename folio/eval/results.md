# Folio evaluation

> Produced in the build sandbox: hashed bag-of-words embeddings (no model download), no reranker, no LLM key (extractive answers). Re-run `python manage.py evaluate` with real models for representative numbers.

> No GROQ_API_KEY: answers are extractive, so answer accuracy is a lower bound. Retrieval metrics are unaffected. Embeddings are the offline hashed fallback; both pipelines score higher with the real embedding model.

| Document | Questions | MRR basic | MRR Folio | hit@1 basic | hit@1 Folio | Answers basic | Answers Folio |
|---|---|---|---|---|---|---|---|
| U.S. Government Publishing Office 2018 Annual Report | 11 | 0.72 | 0.80 | 60% | 70% | 27% | 64% |
| Global Economic Prospects -- June 2026 | 14 | 0.34 | 0.53 | 23% | 31% | 36% | 43% |
| iMac Product Environmental Report | 12 | 0.52 | 0.88 | 36% | 82% | 17% | 42% |

## U.S. Government Publishing Office 2018 Annual Report

| Question | Type | Basic rank | Folio rank | Basic ✓ | Folio ✓ |
|---|---|---|---|---|---|
| How many copies of how many titles did GPO distribute to depository libraries in FY 2018? | fact | 1 | 1 | ✗ | ✓ |
| How many people did GPO employ at the end of FY 2018? | fact | 1 | 2 | ✓ | ✗ |
| What was GPO's net income from operations in FY 2018? | fact | 1 | 1 | ✗ | ✓ |
| What were GPO's total revenues in FY 2018? | fact | 4 | 2 | ✓ | ✓ |
| How many U.S. passports did GPO produce for the State Department in FY 2018? | fact | 1 | 1 | ✓ | ✓ |
| How much printing did GPO procure on behalf of Federal agencies in FY 2018? | fact | 2 | 1 | ✗ | ✓ |
| What is govinfo, and what did it replace? | fact | – | – | ✗ | ✓ |
| How many FDLP Academy webinars did GPO host, and how many participants did the programs serve? | multi-part | 1 | 1 | ✗ | ✗ |
| How many Preservation Steward agreements did GPO sign in FY 2018? | fact | 1 | 1 | ✗ | ✓ |
| By how much has total GPO employment fallen since 1980? | fact | 2 | 1 | ✗ | ✗ |
| What was GPO's total revenue in FY 2020? | refusal | – | – | ✗ | ✗ |

## Global Economic Prospects -- June 2026

| Question | Type | Basic rank | Folio rank | Basic ✓ | Folio ✓ |
|---|---|---|---|---|---|
| What is the growth outlook for South Asia in this report? | routing | 3 | 2 | ✗ | ✗ |
| What growth is projected in 2026 for the region that includes Pakistan? | routing | – | 1 | ✗ | ✓ |
| According to the executive summary, what are the main risks to the global outlook? | routing | – | 1 | ✗ | ✗ |
| What is global growth projected to be in 2026? | fact | 2 | 1 | ✓ | ✓ |
| What is India's projected growth for fiscal year 2026/27? | routing | 1 | 1 | ✓ | ✓ |
| What is the outlook for SSA? | routing | 3 | 3 | ✗ | ✗ |
| How fast is growth in East Asia and Pacific expected to be in 2026? | routing | – | 2 | ✗ | ✗ |
| What is the 2026 growth forecast for Europe and Central Asia? | routing | – | 2 | ✗ | ✗ |
| What is the 2026 growth projection for Latin America and the Caribbean? | routing | – | – | ✗ | ✗ |
| What does the report forecast for Brazil's growth in 2026? | table | – | 4 | ✗ | ✗ |
| In Box 1.1, by how much could AI raise productivity growth per year under optimistic estimates? | routing | 4 | – | ✗ | ✗ |
| In the downside scenario, what would global GDP growth fall to this year? | fact | 1 | 3 | ✓ | ✓ |
| What was global growth in 2025 according to the report? | fact | 1 | 2 | ✓ | ✓ |
| What level does the report forecast for the S&P 500 index at the end of 2026? | refusal | – | – | ✓ | ✓ |

## iMac Product Environmental Report

| Question | Type | Basic rank | Folio rank | Basic ✓ | Folio ✓ |
|---|---|---|---|---|---|
| What was the baseline emissions figure for iMac, and how many kilograms did the reductions remove? | chart | – | 7 | ✗ | ✓ |
| What share of iMac manufacturing electricity comes from renewable sources, and what does the footnote say about how that is measured? | footnote | – | 1 | ✗ | ✗ |
| How much less energy than the ENERGY STAR requirement does iMac use, and what is that figure based on? | footnote | – | 1 | ✗ | ✗ |
| What is the total product carbon footprint of iMac (two ports) 256GB? | table | 2 | 1 | ✗ | ✗ |
| What is the carbon footprint of the iMac (four ports) 512GB configuration? | table | 1 | 1 | ✗ | ✓ |
| What is the iMac's power consumption in sleep mode at 115V? | table | 3 | 1 | ✗ | ✗ |
| What share of the iMac's life cycle emissions comes from product use? | table | 2 | 1 | ✗ | ✗ |
| What percentage of the iMac packaging is recycled content? | fact | 1 | 2 | ✓ | ✓ |
| In how many components does iMac use at least 50 percent recycled plastic? | fact | 1 | 1 | ✗ | ✓ |
| Does the recycled copper in the thermal module apply to every iMac model? | footnote | 3 | 1 | ✓ | ✗ |
| What share of the device's rare earth elements do the recycled magnets represent? | fact | 1 | 1 | ✗ | ✓ |
| What is the battery capacity of the iMac? | refusal | – | – | ✗ | ✗ |
