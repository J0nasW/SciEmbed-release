# Results

Curated evaluation JSONs cited in the paper, all from official upstream harnesses
(no custom evaluators).

```
scirepeval/    official allenai/scirepeval (22 tasks). SciEmbed variants × seeds,
               baselines, specter2_adapters (reference), and the 28 ceiling-probe configs.
mteb/          9-task MTEB science subset (mteb 2.12.15)
longembed/     6-task LongEmbed (mteb 2.12.15)
bfr/           Body-Fact Retrieval: full_pool/, samefield/ (S2AG fields-of-study-restricted),
               paraphrased/ (Qwen3-32B), and queries.parquet (1,000 body-middle sentences)
inference_benchmark/   throughput / latency / VRAM (H100 + workstation GPU)
qualitative/   citation-context examples + length distributions
```

## Reproduce the headline table

```bash
python scripts/aggregate_eval_results.py results/scirepeval/
```

Renders per-bucket means, the 4-category macro (the headline number), and the
22-task mean cross-check. Each `scirepeval/<model>.json` is keyed by task name
with metric sub-keys (e.g. `{"Citation Count": {"kendalltau": 0.34}, ...}`).

## Notes

- The 102 MB BFR `candidates.parquet` is **not** here — it is on HF Datasets
  (`J0nasW/bfr-diagnostic`). `queries.parquet` ships under `bfr/`.
- Paper table → source mapping: `tab:main_results` ← `scirepeval/sciembed_*_seed*`
  + baselines; `tab:ceiling_configs` ← the `sciembed_signal_*`/`sciembed_sweep_*`
  JSONs; `tab:mteb` ← `mteb/`; `tab:longembed`/`tab:needle` ← `longembed/`;
  `tab:bfr*` ← `bfr/`; `tab:inference` ← `inference_benchmark/`.
