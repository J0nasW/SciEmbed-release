# SciEmbed

[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-J0nasW-FFD21E)](https://huggingface.co/J0nasW)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-bfr--diagnostic-FFD21E)](https://huggingface.co/datasets/J0nasW/bfr-diagnostic)
[![SciRepEval](https://img.shields.io/badge/SciRepEval-66.85-success)](#results-scirepeval-4-category-macro)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A 149M-parameter scientific document embedder (ModernBERT-base, 8K context)
trained with **citation context sentences** as the primary contrastive signal.
Paper: *SciEmbed: Citation-Context Supervision for Scientific Document
Embeddings*, Findings of the Association for Computational Linguistics: EMNLP 2026.

## Results (SciRepEval, 4-category macro)

| Model | Params | Classif. | Regr. | Prox. | Search | Overall |
|---|---|---|---|---|---|---|
| SPECTER2 Base | 110M | 74.9 | 27.7 | 81.4 | 81.7 | 66.5 |
| SciNCL | 110M | 74.1 | 25.5 | 81.5 | 80.7 | 65.4 |
| Nomic ModernBERT | 149M | 75.9 | 26.7 | 79.3 | 83.2 | 66.3 |
| BGE-large-en-v1.5 | 335M | 76.1 | 25.6 | 80.5 | 83.9 | 66.5 |
| Granite Embedding R2 | 149M | **76.3** | 25.9 | **82.7** | **84.9** | **67.4** |
| **SciEmbed-CTX** (A+B, 7M pairs) | 149M | 75.5 | **28.3** | 80.9 | 82.5 | 66.8 ± 0.02 |
| **SciEmbed-FULL** (A+B, ~30M pairs) | 149M | 75.6 | 28.2 | 80.9 | 82.7 | 66.85 ± 0.38 |

SciEmbed-FULL trails Granite R2 by 0.6 overall but leads the regression cluster
by +2.3 — citation-graph vs. general-retrieval supervision specialise in
opposite directions. LongEmbed, Body-Fact Retrieval, the ceiling probe, and the
contamination audit are in `results/` and the paper appendix.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[eval]
```

## Models

All weights are on the Hugging Face Hub under
[`J0nasW`](https://huggingface.co/J0nasW) hosts the released checkpoints:

| Model | Context | Use it for |
|---|---|---|
| [`sciembed-full`](https://huggingface.co/J0nasW/sciembed-full) | 512 | **headline** — A+B on the ~30M-pair pool |
| [`sciembed-ctx`](https://huggingface.co/J0nasW/sciembed-ctx) | 512 | best ablation — A+B, 7M pairs |
| [`sciembed-base`](https://huggingface.co/J0nasW/sciembed-base) | 512 | Signal A only (citation edges) |
| [`sciembed-nodapt-ctx`](https://huggingface.co/J0nasW/sciembed-nodapt-ctx) | 512 | CTX with Stage 1 skipped |
| [`sciembed-ctx-2048`](https://huggingface.co/J0nasW/sciembed-ctx-2048) | 2048 | medium-length scientific inputs |
| [`sciembed-ctx-8192`](https://huggingface.co/J0nasW/sciembed-ctx-8192) | 8192 | long scientific inputs |
| [`bfr-diagnostic`](https://huggingface.co/datasets/J0nasW/bfr-diagnostic) | — | BFR candidate pool (dataset) |

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("J0nasW/sciembed-full")
emb = model.encode(["citation-context supervision for scientific embeddings"],
                   normalize_embeddings=True)
```

## Reproduce the headline table

```bash
python scripts/aggregate_eval_results.py results/scirepeval/
```

Full pipeline (data → Stage 1 MLM → Stage 2 contrastive → eval) in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Layout

```
src/sciembed/   package: data/ (signals, MLM corpus), train/ (stage 1+2), eval/, inference/
configs/        OmegaConf YAML (data/, train/, eval/)
scripts/        data prep, eval, figure generation; hpc/ SLURM templates
tests/          pytest suite
results/        curated eval JSONs cited in the paper (scirepeval/, mteb/, longembed/, bfr/, ...)
docs/           REPRODUCTION, DATA, MODEL_CARD
```

## Reproducibility

- **Seeds:** headline 3-seed mean uses {123, 456, 789}; long-context, ceiling, and
  scaling runs use seed 42 (encoded in `results/scirepeval/*.json` filenames).
- **Hardware:** Stage 1 MLM 4×H200, 16.1 h; each Stage 2 variant 1×H100 (~1 day
  ablation, ~2 days FULL).
- **Software:** pinned in `uv.lock` (PyTorch 2.7, transformers 4.57,
  sentence-transformers 5.3, Composer 0.32, mteb 2.12.15).
- **SciRepEval:** pinned to `allenai/scirepeval@e04594e` + a ~10-line `SE_SEED`
  patch for multi-seed runs (applied inline in
  `scripts/hpc/template_evaluate_scirepeval.sbatch`; also shown in `REPRODUCTION.md`).
- **Data:** local DuckDB snapshot of S2AG + full-text sources, 2026-03-01
  (`docs/DATA.md`).

## License

MIT (code and released weights).

## Citation

```bibtex
@inproceedings{wilinski2026sciembed,
  title={SciEmbed: Citation-Context Supervision for Scientific Document Embeddings},
  author={Wilinski, Jonas and F{\"a}rber, Michael},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026}
}
```
