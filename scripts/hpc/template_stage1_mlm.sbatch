#!/bin/bash
# Template: Stage 1 domain-adaptive MLM pretraining on 8.3B tokens.
# Headline run: 4×H200, 16.1 h, eval CE 1.079 → 1.041 (-3.5%).
# Replace <YOUR_ACCOUNT>, <YOUR_PARTITION>, <YOUR_EMAIL>, <WORKSPACE>, etc.
#SBATCH --job-name=sciembed-stage1
#SBATCH --output=<WORKSPACE>/logs/stage1_%j.out
#SBATCH --error=<WORKSPACE>/logs/stage1_%j.err
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:4              # or gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

WORKSPACE=<WORKSPACE>
VENV=<VENV_PATH>
HF_CACHE=<HF_CACHE>

# Load cluster modules; adjust names for your environment
module load CUDA/12.4 GCC/13.2.0 Python/3.11.5

# Caches MUST live outside any quota-bounded home dir
export HF_HOME="$HF_CACHE/hf"
export TORCH_HOME="$HF_CACHE/torch"
export TRITON_CACHE_DIR="$HF_CACHE/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "$WORKSPACE"
export PATH="$VENV/bin:$PATH"

# Verify GPU visibility
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Stage 1 MLM training
# - Optimizer: torch.optim.AdamW (NOT composer.optim.DecoupledAdamW, which destroys pretrained weights)
# - Special tokens: read from tokenizer (NEVER hardcode IDs)
# - Mask rate: 30% (BioClinical ModernBERT recipe)
python -m sciembed.cli train stage1 \
    --config configs/train/stage1_mlm.yaml \
    "output_dir=output/stage1_mlm" \
    "mds_corpus=$WORKSPACE/data/mlm_corpus.mds"

echo "Done $(date)"
