#!/bin/bash
# Template: Body-Fact Retrieval (BFR) evaluation.
# Wall: ~1h per model on H100. Pass MODEL via --export.
#
# Submission example:
#   sbatch --export=ALL,MODEL=J0nasW/sciembed-ctx-8192,NAME=sciembed_ctx_8192 \
#          template_evaluate_bfr.sbatch
#
# Replace <WORKSPACE>, <YOUR_ACCOUNT>, etc.
#SBATCH --job-name=bfr-eval
#SBATCH --output=<WORKSPACE>/logs/bfr_%j.out
#SBATCH --error=<WORKSPACE>/logs/bfr_%j.err
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

: "${MODEL:?MODEL must be set (HF id or local path)}"
: "${NAME:?NAME must be set (used in output filename)}"
: "${VARIANT:=full_pool}"   # full_pool | samefield | paraphrased

WORKSPACE=<WORKSPACE>
VENV=<VENV_PATH>
HF_CACHE=<HF_CACHE>

# BFR candidates parquet is ~100MB; either fetch from HF Datasets or use local cache.
# The HF dataset is: J0nasW/bfr-diagnostic
BFR_CANDIDATES="${BFR_CANDIDATES:-J0nasW/bfr-diagnostic}"
QUERIES_FILE="$WORKSPACE/results/bfr/queries.parquet"
OUTPUT_DIR="$WORKSPACE/results/bfr/$VARIANT"
mkdir -p "$OUTPUT_DIR"

module load CUDA/12.4 GCC/13.2.0 Python/3.11.5

export HF_HOME="$HF_CACHE/hf"
export TORCH_HOME="$HF_CACHE/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$WORKSPACE"
export PATH="$VENV/bin:$PATH"

OUTFILE="$OUTPUT_DIR/${NAME}.json"
if [ -f "$OUTFILE" ]; then
    echo "SKIP: $OUTFILE exists"
    exit 0
fi

# scripts/run_bfr.py wraps:
#   1. Load candidates + queries (parquet)
#   2. Encode title+abstract+body with the model's native max_seq_length
#   3. Optionally apply paraphrase (Qwen3-32B) or same-field topic filter
#   4. Compute Recall@{1,10,100}, nDCG@10
python scripts/run_bfr.py \
    --model "$MODEL" \
    --variant "$VARIANT" \
    --candidates "$BFR_CANDIDATES" \
    --queries "$QUERIES_FILE" \
    --output "$OUTFILE"

echo "Done $(date)"
cat "$OUTFILE"
