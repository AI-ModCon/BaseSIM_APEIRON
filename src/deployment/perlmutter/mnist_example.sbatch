#!/bin/bash -l
#SBATCH -J modcon_basesim
#SBATCH -t 0:20:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -o output/mnist_example.o%j
#SBATCH -e output/mnist_example.e%j

# Load required modules
#module load PrgEnv-gnu
#module load gcc/12.2.0
#module load rocm/6.4.2

# ROCm/MIOpen flags
#mkdir -p $MEMBERWORK/$SBATCH_ACCOUNT/miopen
#export MIOPEN_USER_DB_PATH=$MEMBERWORK/$SBATCH_ACCOUNT/miopen
#export MIOPEN_CUSTOM_CACHE_DIR=$MEMBERWORK/$SBATCH_ACCOUNT/miopen
export WANDB_MODE=offline

# Print environment info
echo "=============================================="
echo "MNIST Example"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

# Run example
poetry run python -m src.main --config ./examples/mnist/mnist.toml
