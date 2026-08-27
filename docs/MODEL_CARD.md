# SciEmbed model card

Accompanies the weights on Hugging Face (`J0nasW` during review).

## Summary

| | |
|---|---|
| Architecture | ModernBERT-base, 149M params, 8,192-token context |
| Pooling | mean |
| Output dim | 768 (Matryoshka-truncatable to 512/256/128) |
| Tokenizer | ModernBERT BPE (50,368 vocab) |
| Precision | BF16 train; fp16/bf16/fp32 inference |
| License | MIT |

## Variants

| Repo | Notes |
|---|---|
| `J0nasW/sciembed-full` | **Headline.** DAPT + A+B on ~30M-pair pool, 1 epoch. Seeds 123/456/789. |
| `J0nasW/sciembed-ctx` | A+B, 7M-pair subsample, 3 epochs. |
| `J0nasW/sciembed-base` | Signal A only, 7M pairs, 3 epochs. |
| `J0nasW/sciembed-nodapt-ctx` | CTX with Stage 1 skipped. |
| `J0nasW/sciembed-ctx-8192` | Long-context variant (`max_seq_length=8192`). Recommended for long inputs. |
| `J0nasW/sciembed-ctx-2048` | Intermediate long-context variant. |

## SciRepEval (4-cat macro, 3 seeds)

| Variant | Classif. | Regr. | Prox. | Search | Overall |
|---|---|---|---|---|---|
| SciEmbed-BASE | 75.3 | 26.8 | 80.2 | 82.2 | 66.1 ± 0.09 |
| SciEmbed-CTX | 75.5 | **28.3** | 80.9 | 82.5 | 66.8 ± 0.02 |
| SciEmbed-NoDAPT-CTX | 75.3 | 28.2 | 80.8 | 82.6 | 66.7 ± 0.07 |
| **SciEmbed-FULL** | 75.6 | 28.2 | 80.9 | 82.7 | **66.85 ± 0.38** |

## Intended use

Scientific document retrieval, classification (field/MeSH via linear probe), and
impact-correlated regression (citation count, h-index, peer-review score) — the
bucket where SciEmbed-FULL leads the matched-architecture Granite R2 by +2.3.

## Out of scope

- High-stakes decisions (reviewer assignment, hiring, funding) without
  fairness-aware controls — inherits S2AG coverage bias (`DATA.md`).
- General (non-scientific) long-context retrieval — Granite R2 leads LongEmbed
  by ~28 points. SciEmbed is a *scientific* long-context retriever; use
  `sciembed-ctx-8192` for long scientific inputs.
- Low-latency single-doc serving — ModernBERT's launch path gives ~11 ms/doc at
  batch=1 on H100; it shines at batch ≥64 (5,363 docs/sec at bs=128, fp16).

## Training data

- **Stage 1 (MLM):** 13.4M papers (PMC-OAS 4.6M, S2ORC 7.4M, arXiv 0.75M,
  peS2o 1.1M; 8.3B tokens).
- **Stage 2 (contrastive):** ~30M deduplicated S2AG pairs — Signal A (citation
  edges), Signal B (citation contexts), Signal D (structured hard negatives).

Full lineage, snapshots, filters, licenses in `DATA.md`.

## Evaluation

SciRepEval (22 tasks, official harness), MTEB science subset (9 tasks, mteb
2.12.15), LongEmbed (6 tasks), and the Body-Fact Retrieval diagnostic
(`J0nasW/bfr-diagnostic`).

## Limitations

- English only.
- Trails Granite R2 by 0.6 on SciRepEval overall (leads regression by +2.3).
- Trails general long-context models on LongEmbed by ~28 points.
- Stage 1 DAPT adds only +0.1 once Signals A+B are in the mix (at the 8.3B-token
  budget).
- BFR is a self-constructed diagnostic, reported with BM25 controls plus
  same-field and paraphrase robustness checks.

## Compute

Stage 1 ~64 GPU-h (4×H200); each Stage 2 variant ~48 GPU-h (1×H100); full
pipeline ≈700 GPU-h.

## Citation

See top-level `README.md`.
