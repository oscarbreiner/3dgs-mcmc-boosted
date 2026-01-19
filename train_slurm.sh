#!/bin/bash
#SBATCH --job-name=3dgs-mcmc-train
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs-mcmc-env

TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

# Set CUDA environment variables
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"
echo "Activated conda env: $CONDA_DEFAULT_ENV"


# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"


# ✅ Your Dataset Structure

# /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/
# ├── tandt/          # Tanks & Temples
# │   ├── train/      ← Ready to use!
# │   └── truck/      ← Ready to use!
# └── db/             # Deep Blending  
#     ├── drjohnson/  ← Ready to use!
#     └── playroom/   ← Ready to use!

# Training parameters
SOURCE_PATH=${SOURCE_PATH:-"/dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/tandt/truck"}
CONFIG=${CONFIG:-"configs/truck.json"}
EXTRA_ARGS=${EXTRA_ARGS:-""}

# Set W&B API key (replace with your key or use wandb login on login node)
# export WANDB_API_KEY=your_api_key_here

# Optional: Custom W&B run name with SLURM job ID
# export WANDB_RUN_NAME="${SLURM_JOB_ID}_$(basename $SOURCE_PATH)"

echo "Training with:"
echo "  Source: $SOURCE_PATH"
echo "  Config: $CONFIG"
echo ""

# Run training
python train.py \
    --source_path "$SOURCE_PATH" \
    --config "$CONFIG" \
    --eval \
    ${EXTRA_ARGS}

echo ""
echo "End time: $(date)"
echo "Training complete!"


## Examples:

# # Tanks & Temples scenes
# sbatch --export=SOURCE_PATH=/dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/tandt/train,CONFIG=configs/train.json train_slurm.sh
# sbatch --export=SOURCE_PATH=/dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/tandt/truck,CONFIG=configs/truck.json train_slurm.sh

# # Deep Blending scenes  
# sbatch --export=SOURCE_PATH=/dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/db/drjohnson,CONFIG=configs/drjohnson.json train_slurm.sh
# sbatch --export=SOURCE_PATH=/dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/breiner/datasets/db/playroom,CONFIG=configs/playroom.json train_slurm.sh


# Flag Options:
# ,EXTRA_ARGS="--reloc_sampling importance --importance_ema 0.9"
# ,EXTRA_ARGS="--reloc_sampling error --error_ema 0.9"
# EXTRA_ARGS="--reloc_sampling hybrid --importance_ema 0.9 --error_ema 0.9"


# # Check if job is running
# squeue -u $USER

# # Watch the output
# tail -f logs/slurm-{JOB_ID}.out
