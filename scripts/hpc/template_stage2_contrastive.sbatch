#!/bin/bash
# Template: Stage 2 contrastive fine-tuning (one seed of one variant).
# Headline: 1×H100, ~48h per variant. Pass SE_SEED=123|456|789 via --export.
#
# Submit one job per (variant, seed) pair in parallel; total wall = single-job
# wall, not seeds × wall.
#
# Replace <YOUR_ACCOUNT>, <YOUR_PARTITION>, <YOUR_EMAIL>, <WORKSPACE>, etc.
# Pass variant via VARIANT={base|ctx|nodapt_ctx|full|ctx_2048|ctx_8192}.
#
# Submission example (3 seeds × FULL):
#   for SEED in 123 456 789; do
#       sbatch --export=ALL,VARIANT=full,SE_SEED=$SEED \
#              template_stage2_contrastive.sbatch
#   done
#SBATCH --job-name=sciembed-stage2
#SBATCH --output=<WORKSPACE>/logs/stage2_%j.out
#SBATCH --error=<WORKSPACE>/logs/stage2_%j.err
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

: "${VARIANT:?VARIANT must be set, e.g. sbatch --export=ALL,VARIANT=full,SE_SEED=123 ...}"
: "${SE_SEED:?SE_SEED must be set}"

WORKSPACE=<WORKSPACE>
VENV=<VENV_PATH>
HF_CACHE=<HF_CACHE>

module load CUDA/12.4 GCC/13.2.0 Python/3.11.5

export HF_HOME="$HF_CACHE/hf"
export TORCH_HOME="$HF_CACHE/torch"
export TRITON_CACHE_DIR="$HF_CACHE/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$WORKSPACE"
export PATH="$VENV/bin:$PATH"

# Map variant name to config
case "$VARIANT" in
    base)        CFG="configs/train/stage2_base.yaml" ;;
    ctx)         CFG="configs/train/stage2_ctx.yaml" ;;
    nodapt_ctx)  CFG="configs/train/stage2_ctx.yaml" ;;  # same config; override model_name_or_path below
    full)        CFG="configs/train/stage2_full.yaml" ;;
    ctx_2048)    CFG="configs/train/stage2_ctx.yaml" ;;  # override max_seq_length below
    ctx_8192)    CFG="configs/train/stage2_ctx.yaml" ;;
    *) echo "Unknown VARIANT=$VARIANT"; exit 2 ;;
esac

OUT_DIR="$WORKSPACE/output/stage2_${VARIANT}_seed${SE_SEED}"
mkdir -p "$OUT_DIR"

# Variant-specific overrides
OVERRIDES=()
case "$VARIANT" in
    nodapt_ctx)
        # Skip Stage 1 checkpoint; init from raw ModernBERT-base
        OVERRIDES+=("model_name_or_path=answerdotai/ModernBERT-base")
        ;;
    ctx_2048)
        OVERRIDES+=("max_seq_length=2048")
        ;;
    ctx_8192)
        OVERRIDES+=("max_seq_length=8192")
        ;;
esac

python -m sciembed.cli train stage2 \
    --config "$CFG" \
    "output_dir=$OUT_DIR" \
    "seed=$SE_SEED" \
    "run_name=sciembed-${VARIANT}-seed${SE_SEED}" \
    "${OVERRIDES[@]}"

echo "Done $(date)"
