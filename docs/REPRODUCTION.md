# Reproducing SciEmbed

Pipeline: data construction → Stage 1 MLM → Stage 2 contrastive → evaluation.
All training data comes from a local DuckDB snapshot of S2AG + full-text corpora
(see `DATA.md`); you must supply your own. Set `DATALAKE_DB` and `WORK` (output
root) env vars, or edit the config paths.

## Environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[train,eval]   # train extras: flash-attn + composer
```

Python 3.12. Exact pins in `uv.lock`. SciRepEval needs a small `SE_SEED` patch
for multi-seed runs (shown inline in step 4).

## 1. Data

```bash
# Stage 1 MLM corpus (13.4M papers, 8.3B tokens -> MDS shard)
python scripts/build_mlm_corpus.py --config configs/data/mlm_corpus.yaml

# Signal A: citation-edge triplets (~50M raw, ~30M dedup)
python scripts/build_citation_triplets.py --config configs/data/citation_triplets.yaml

# Signal B: citation-context queries (~20M sampled from ~1B)
python -m sciembed.data.citation_contexts --config configs/data/citation_contexts.yaml

# Mix A+B for the FULL recipe
python -m sciembed.data.data_mixer --config configs/data/data_mixer_full.yaml
```

Ceiling-probe signals: `build_related_works_pairs.py`,
`build_signal_e_co_citation.py`, `build_signal_g_tldr.py`,
`build_signal_h_improves_triplets.py`.

## 2. Stage 1 MLM

```bash
python -m sciembed.cli train stage1 --config configs/train/stage1_mlm.yaml
```

4×H200, 16.1 h, eval CE 1.079 → 1.041. SLURM: `scripts/hpc/template_stage1_mlm.sbatch`.

> Use `torch.optim.AdamW`, **not** Composer's `DecoupledAdamW` — the latter
> (with `wd=0.01`) destroys the pretrained weights during continued pretraining
> (~46% quality drop in 500 steps). Composer still runs the loop; only the
> optimizer class changes.

## 3. Stage 2 contrastive

```bash
# Ablations (BASE, CTX, NoDAPT-CTX): 7M-pair subsample, 3 epochs
python -m sciembed.cli train stage2 --config configs/train/stage2_ctx.yaml

# FULL: ~30M dedup pool, 1 epoch, 3 seeds for the headline number
for SEED in 123 456 789; do
    python -m sciembed.cli train stage2 --config configs/train/stage2_full.yaml seed=$SEED
done
```

Loss (Matryoshka + InfoNCE + structured hard negatives) lives in
`src/sciembed/train/contrastive.py`; hyperparameters in the YAML configs.
SLURM: `scripts/hpc/template_stage2_contrastive.sbatch`.

## 4. SciRepEval (official)

```bash
git clone https://github.com/allenai/scirepeval.git
cd scirepeval && git checkout e04594e
```

The harness hardcodes seed 42 in two spots; patch both to read `SE_SEED`
(defaults to 42, so single-seed runs stay identical to upstream):

```diff
--- a/scirepeval.py
+++ b/scirepeval.py
-    pl.seed_everything(42, workers=True)
+    import os; pl.seed_everything(int(os.environ.get('SE_SEED', 42)), workers=True)
--- a/evaluation/evaluator.py
+++ b/evaluation/evaluator.py
-RANDOM_STATE = 42
+import os; RANDOM_STATE = int(os.environ.get('SE_SEED', 42))
```

```bash
SE_SEED=123 python scirepeval.py -m J0nasW/sciembed-full \
    --pooling-mode mean --batch-size 64 \
    --output results/scirepeval/sciembed_full_seed123.json
```

Aggregate to the 4-category macro:

```bash
python scripts/aggregate_eval_results.py results/scirepeval/
```

## 5. MTEB / LongEmbed

```bash
# 9-task science subset
python scripts/run_mteb_subset.py --model J0nasW/sciembed-full --output results/mteb/

# LongEmbed (6 tasks) via the upstream mteb runner
python -m mteb eval --model J0nasW/sciembed-ctx-8192 \
    --tasks LEMBNarrativeQARetrieval LEMBQMSumRetrieval LEMBSummScreenFDRetrieval \
            LEMBWikimQARetrieval LEMBNeedleRetrieval LEMBPasskeyRetrieval \
    --output_folder results/longembed/
```

## 6. Body-Fact Retrieval (BFR)

Candidate pool (`candidates.parquet`, ~100 MB) is hosted on HF Datasets at
`J0nasW/bfr-diagnostic`; `queries.parquet` ships in `results/bfr/`.

```bash
# build the probe (transparency; needs the data lake)
python scripts/build_fullpaper_context_retrieval.py --snapshot-date 2026-03-01 --output results/bfr/

# evaluate (full pool); same-field and paraphrased are the other two output dirs
python scripts/eval_fullpaper_context_retrieval.py \
    --models configs/eval/bfr_models.json \
    --candidates J0nasW/bfr-diagnostic \
    --queries results/bfr/queries.parquet \
    --output results/bfr/full_pool/
python scripts/eval_bfr_bm25.py            # BM25 lexical floor/ceiling
python scripts/paraphrase_bfr_queries.py   # Qwen3-32B paraphrase robustness
```

## 7. Inference benchmark

```bash
python scripts/benchmark_inference.py \
    --models J0nasW/sciembed-full BAAI/bge-large-en-v1.5 allenai/specter2_base \
    --precisions fp16 fp32 bf16 --batch-sizes 1 8 64 128 \
    --output results/inference_benchmark/h100.json
```

## Troubleshooting

- **Stage 1 weight collapse:** use `torch.optim.AdamW`, not Composer's
  `DecoupledAdamW` (see Stage 1 note above).
- **Garbage embeddings / failing token tests:** never hardcode special-token
  IDs — ModernBERT uses CLS=50281, SEP=50282, PAD=50283, MASK=50284 (not BERT's
  101/102/0/103). Read them from the tokenizer.
- **Stage 1 loss looks wrong after a code change:** the MDS shard is
  pre-tokenized; delete and rebuild it after any tokenizer/chunking/cleaning
  change in `src/sciembed/data/mlm_corpus.py`.
- **HF datasets cache corruption with parallel jobs:** chain with SLURM
  `--dependency=afterok`, or give each job its own `HF_DATASETS_CACHE`.
