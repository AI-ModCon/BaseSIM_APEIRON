#!/bin/bash -l
#SBATCH -J modcon_basesim
#SBATCH -t 0:20:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -o output/mnist_example.o%j
#SBATCH -e output/mnist_example.e%j

# Load required modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.4.2

# ROCm/MIOpen flags
ACCOUNT=$(sacct -j $SLURM_JOB_ID --format=Account --noheader | head -1 | tr -d ' ')
mkdir -p $MEMBERWORK/$ACCOUNT/miopen
export MIOPEN_USER_DB_PATH=$MEMBERWORK/$ACCOUNT/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MEMBERWORK/$ACCOUNT/miopen
export WANDB_MODE=offline

# Print environment info
echo "=============================================="
echo "MNIST Example"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "ROCM_PATH: ${ROCM_PATH}"
echo "=============================================="

# Run example
poetry run python -m src.main --config ./examples/mnist/mnist.toml
