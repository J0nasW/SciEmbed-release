# Scripts

Standalone utilities. Most are thin wrappers around `src/sciembed/`, which is the
source of truth. All take OmegaConf YAML via `--config` with `key=value`
overrides; paths default under `${WORK:-./output}`.

## Data construction

| Script | Purpose |
|---|---|
| `build_mlm_corpus.py` | Stage 1 MLM corpus (13.4M papers → MDS shard) |
| `build_citation_triplets.py` | Signal A: citation-edge triplets |
| `build_fullpaper_context_retrieval.py` | Build the BFR probe (9,749 candidates, 1,000 queries) |
| `build_related_works_pairs.py` | Ceiling probe: OpenAlex related-works pairs |
| `build_signal_e_co_citation.py` | Ceiling probe: co-citation pairs |
| `build_signal_g_tldr.py` | Ceiling probe: TLDR↔abstract pairs |
| `build_signal_h_improves_triplets.py` | Ceiling probe: intent-filtered improvement citations |
| `merge_triplet_shards.py`, `merge_mds_shards.py`, `generate_mds_index.py` | Shard plumbing |

## Evaluation

| Script | Purpose |
|---|---|
| `download_scirepeval.py` | Prefetch SciRepEval HF datasets |
| `scirepeval_granite_runner.py` | SciRepEval runner for Granite R2 |
| `run_scirepeval_matryoshka.py` | SciRepEval at each Matryoshka dim |
| `run_mteb_subset.py` | MTEB 9-task science subset |
| `run_mteb_official.py` | Full MTEB(eng, v2) |
| `eval_fullpaper_context_retrieval.py` | BFR evaluation (full / same-field / paraphrased) |
| `eval_bfr_bm25.py` | BFR BM25 lexical floor/ceiling |
| `paraphrase_bfr_queries.py` | Qwen3-32B query paraphrasing for BFR robustness |
| `build_cold_filter.py`, `eval_scirepeval_cold.py` | Cold-paper contamination audit |
| `contamination_overlap_check.py` | Per-task overlap (SciRepEval ∩ training pool) |
| `benchmark_inference.py` | Throughput / latency / VRAM |
| `aggregate_eval_results.py` | SciRepEval JSONs → 4-category macro |
| `aggregate_mteb_science.py`, `aggregate_matryoshka.py`, `aggregate_singh_check.py` | Aggregators |
| `analyze_task_variance.py` | Per-task variance (ceiling probe) |

## Figures

| Script | Purpose |
|---|---|
| `generate_paper_figures.py` | Stage 1 loss curve and related figures |
| `generate_analysis_figures.py` | Inference-throughput and analysis figures |
| `build_qualitative_analysis.py` | UMAP + qualitative citation-context examples |

Figures already ship as PDFs under `paper/latex/figures/generated/`; these
scripts are for transparency and read from local logs / `results/`.
