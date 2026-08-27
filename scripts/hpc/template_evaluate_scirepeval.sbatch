#!/bin/bash
# Template: Official SciRepEval evaluation for one model at one seed.
# Wall: ~2.5h per (model, seed). Pass MODEL and SE_SEED via --export.
#
# Submission example (3 seeds of SciNCL):
#   for SEED in 123 456 789; do
#       sbatch --export=ALL,MODEL=malteos/scincl,POOLING=cls,NAME=scincl,SE_SEED=$SEED \
#              template_evaluate_scirepeval.sbatch
#   done
#
# This script applies a small ~10-line patch to scirepeval.py + evaluator.py to
# honor SE_SEED. The patch is restored on EXIT (trap), so the upstream tree is
# left clean for sequential / interleaved runs.
#
# Replace <WORKSPACE>, <YOUR_ACCOUNT>, etc.
#SBATCH --job-name=scirepeval
#SBATCH --output=<WORKSPACE>/logs/scirepeval_%j.out
#SBATCH --error=<WORKSPACE>/logs/scirepeval_%j.err
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

: "${MODEL:?MODEL must be set (HF id or local path)}"
: "${POOLING:?POOLING must be set (mean|cls)}"
: "${NAME:?NAME must be set (used in output filename)}"
: "${SE_SEED:?SE_SEED must be set}"

WORKSPACE=<WORKSPACE>
VENV=<VENV_PATH>
HF_CACHE=<HF_CACHE>
SCIREPEVAL_DIR="$WORKSPACE/eval/scirepeval"     # your SciRepEval clone at pinned commit e04594e
OUTPUT_DIR="$WORKSPACE/results/scirepeval"
EMB_DIR="$SCIREPEVAL_DIR/embeddings_${NAME}_seed${SE_SEED}"  # per-seed to avoid races

mkdir -p "$OUTPUT_DIR" "$EMB_DIR"

module load CUDA/12.4 GCC/13.2.0 Python/3.11.5

export HF_HOME="$HF_CACHE/hf"
export TORCH_HOME="$HF_CACHE/torch"
export TRITON_CACHE_DIR="$HF_CACHE/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$SCIREPEVAL_DIR"
export PATH="$VENV/bin:$PATH"

OUTFILE="$OUTPUT_DIR/${NAME}_seed${SE_SEED}.json"
if [ -f "$OUTFILE" ] && python -c "import json; assert len(json.load(open('$OUTFILE'))) >= 20" 2>/dev/null; then
    echo "SKIP: $OUTFILE already has >=20 tasks"
    exit 0
fi

# ---------- Apply SE_SEED patch (restored on EXIT) ----------
export SCIREP_PY="$SCIREPEVAL_DIR/scirepeval.py"
export EVAL_PY="$SCIREPEVAL_DIR/evaluation/evaluator.py"
cp "$SCIREP_PY" "$SCIREP_PY.unpatched.$SLURM_JOB_ID"
cp "$EVAL_PY"   "$EVAL_PY.unpatched.$SLURM_JOB_ID"
trap 'mv "$SCIREP_PY.unpatched.$SLURM_JOB_ID" "$SCIREP_PY" 2>/dev/null || true;
      mv "$EVAL_PY.unpatched.$SLURM_JOB_ID"   "$EVAL_PY"   2>/dev/null || true' EXIT

python - <<'PY'
import os, pathlib
sci = pathlib.Path(os.environ["SCIREP_PY"])
ev  = pathlib.Path(os.environ["EVAL_PY"])
s = sci.read_text()
if "pl.seed_everything(42, workers=True)" in s:
    s = s.replace(
        "pl.seed_everything(42, workers=True)",
        "import os as _seedos; pl.seed_everything(int(_seedos.environ.get('SE_SEED', 42)), workers=True)",
    )
    sci.write_text(s)
e = ev.read_text()
if "RANDOM_STATE = 42" in e:
    e = e.replace(
        "RANDOM_STATE = 42",
        "import os as _seedos; RANDOM_STATE = int(_seedos.environ.get('SE_SEED', 42))",
    )
    ev.write_text(e)
print("Patched")
PY

# ---------- Run ----------
echo "Evaluating $NAME ($MODEL) seed=$SE_SEED pooling=$POOLING"
SE_SEED="$SE_SEED" python scirepeval.py \
    -m "$MODEL" \
    --pooling-mode "$POOLING" \
    --batch-size 64 \
    --output "$OUTFILE" \
    --embeddings-save-path "$EMB_DIR"

echo "Done $(date). Tasks completed:"
python -c "import json; print(len(json.load(open('$OUTFILE'))))"
