#!/bin/bash -l
#SBATCH -J modcon_basesim
#SBATCH -t 0:20:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -o output/mnist_example.o%j
#SBATCH -e output/mnist_example.e%j

#
source .venv/bin/activate

# WANDB flag
export WANDB_MODE=offline

# Print environment info
echo "=============================================="
echo "MNIST Example"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

# Run example
python -m src.main --config ./examples/mnist/mnist.toml
