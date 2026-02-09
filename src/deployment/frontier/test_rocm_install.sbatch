#!/bin/bash -l
#SBATCH -J test_rocm_install
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -o output/test_rocm_install.o%j
#SBATCH -e output/test_rocm_install.e%j

# Load required modules
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.4.2

# ROCm/MIOpen flags
mkdir -p $MEMBERWORK/$SBATCH_ACCOUNT/miopen
export MIOPEN_USER_DB_PATH=$MEMBERWORK/$SBATCH_ACCOUNT/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MEMBERWORK/$SBATCH_ACCOUNT/miopen

# WANDBD set to offline; Need to set up proxy connections.
export WANDB_MODE=offline

# Print environment info
echo "=============================================="
echo "ROCm Installation & Harness Test"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "ROCM_PATH: ${ROCM_PATH}"
echo "=============================================="

# Test torch rocm compatability
poetry run pytest tests/deployment/frontier/test_rocm_install.py -v

# Test model harness (loaders, model, etc)
poetry run pytest tests/deployment/frontier/test_model_harness_rocm.py -v

# Test jvp update
poetry run pytest tests/deployment/frontier/test_jvp_update_rocm.py -v
