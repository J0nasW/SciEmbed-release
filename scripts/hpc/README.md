# SLURM templates

Sanitized templates of the submission scripts for the headline experiments. Not
runnable as-is — fill in your account, partition, mail address, and paths.

## Resources

| Stage | GPUs | Wall time | Mem |
|---|---|---|---|
| Stage 1 MLM | 4× H200 (or H100) | 16.1 h | 180 GB |
| Stage 2 contrastive | 1× H100 | ~1 day ablation / ~2 days FULL | 128 GB |
| SciRepEval eval | 1× H100 | ~2.5 h / model | 128 GB |
| BFR eval | 1× H100 | ~1 h / model | 64 GB |

## Placeholders to replace

`<YOUR_ACCOUNT>`, `<YOUR_PARTITION>`, `<YOUR_EMAIL>`, `<WORKSPACE>`,
`<VENV_PATH>`, `<HF_CACHE>` (keep the HF cache off a quota'd home dir). Module
names will differ per cluster.

## Useful patterns

- **Multi-seed:** submit one job per seed; backfill runs them in parallel, so
  total wall time ≈ single-seed time.
  ```bash
  for SEED in 123 456 789; do
      sbatch --export=ALL,SE_SEED=$SEED template_evaluate_scirepeval.sbatch
  done
  ```
- **Backup job:** primary at tight wall + a `--dependency=afternotok` backup at
  longer wall that only fires if the primary fails.
- **HF cache races:** chain parallel jobs with `afterok`, or give each its own
  `HF_DATASETS_CACHE`.
- All four templates skip cleanly if their output already looks complete, so
  they are safe to re-run.
