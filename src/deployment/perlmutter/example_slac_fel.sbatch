#!/bin/bash -l
#SBATCH -J modcon_basesim
#SBATCH -t 0:30:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -o output/slac-fel_example.o%j
#SBATCH -e output/slac-fel_example.e%j

#
source .venv/bin/activate

# WANDB flag
export WANDB_MODE=offline

# Print environment info
echo "=============================================="
echo "MOAT: slac-fel"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=============================================="

# Run example
python -m src.main --config ./examples/slac_fel/slac-fel.toml
