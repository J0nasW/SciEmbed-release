# Data sources and licenses

SciEmbed trains only on open-access scientific corpora. No proprietary or
paywalled content. The training data lives in a local DuckDB snapshot; it is not
redistributable, but every source below is publicly downloadable.

## Sources

| Stage | Source | Snapshot | License |
|---|---|---|---|
| Stage 1 MLM | PMC-OAS, S2ORC, arXiv, peS2o | 2026-02 / 2026-03-01 | CC-BY(-NC) / ODC-BY |
| Stage 2 Signal A/B | S2AG citations + citation contexts | 2026-03-01 | ODC-BY |
| Stage 2 Signal D | S2AG `s2fieldsofstudy` (field labels) | 2026-03-01 | ODC-BY |
| Ceiling probe | OpenAlex related-works, S2AG co-citation / `tldrs` | 2026-02 / 03 | CC0 / ODC-BY |
| BFR | S2ORC full text | 2026-03-01 | ODC-BY |
| Eval | SciRepEval, MTEB, LongEmbed | via HF / mteb 2.12.15 | per-task |

Downloads: S2AG <https://api.semanticscholar.org/datasets> · S2ORC
<https://github.com/allenai/s2orc> · PMC-OAS
<https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/> · arXiv (Kaggle) · peS2o
<https://huggingface.co/datasets/allenai/peS2o> · OpenAlex
<https://docs.openalex.org/download-all-data>.

## Construction filters

**Stage 1 MLM (13.4M papers, 8.3B tokens).** Body 1,000–500,000 chars;
LaTeX/JATS stripping, reference removal, NFKC normalization; ModernBERT BPE,
8,192-token chunks, 128-token stride.

**Signal A — citation edges (~50M raw, ~30M dedup).** Two-tier sampler: 16M
influential + 34M general (≥5 citations both sides). Pair = `(citing
title+abstract, cited title+abstract)`. MD5 dedup on normalized pairs.

**Signal B — citation contexts (~20M from ~1B).** Context 50–1,000 chars, ≥3
words, ≥0.5 alphabetic ratio; influential tier only.

**Signal D — structured hard negatives (50/30/20).** Same-field non-cited (from
the S2AG `s2fieldsofstudy` category, 24 broad disciplines), forward-neighbour
(cited by the positive, not the anchor), random.

**BFR diagnostic.** 9,749 S2ORC papers with title + abstract ≥50 chars + body
8,000–80,000 chars + citation count in [1, 4] (the [1,4] band guarantees
exclusion from Stage-2 training, which required ≥5). 1,000 queries: one sentence
per paper from the body middle third, rejected if word-3-gram overlap with the
abstract > 20% (answer is body-only, not abstract-leaked).

Exact table schemas are in `src/sciembed/data/datalake.py`.

## Licensing of released artifacts

Weights, training pipeline, and the BFR diagnostic are MIT, consistent with the
predominantly ODC-BY / CC-BY sources. PMC-OAS article licenses vary; respect
per-article terms when redistributing derived text.

## Coverage bias

S2AG over-represents English-language, North-American/European, and
biomedical/STEM publications; Signal B is denser in biomedicine and CS. These
biases propagate into the embeddings. Do not use SciEmbed as the sole signal in
high-stakes settings (reviewer assignment, hiring, funding) without
fairness-aware controls. Also discussed in the paper's Ethics Statement.
