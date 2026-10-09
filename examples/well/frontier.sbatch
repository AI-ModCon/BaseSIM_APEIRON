#!/bin/bash -l
#SBATCH -J well_unet
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o output/well_unet.o%j
#SBATCH -e output/well_unet.e%j

# The Well at the published width: `unet` (init_features=48), trained from
# scratch by detect-and-adapt over the nine turbulent_radiative_layer_2D
# regimes, then scored on the held-out test split. The Frontier counterpart
# of perlmutter.sbatch -- same experiment, one MI250X GCD.
#
# Setup (once, from the project root -- see
# src/apeiron/deployment/frontier/README.md):
#
#   source ./src/apeiron/deployment/frontier/install_venv.sh
#   mkdir -p output
#
# Submit from the project root:
#
#   sbatch -A <project> examples/well/frontier.sbatch
#
# Compute nodes have no internet access, so fetch everything on the login
# node first. The prefetch lists the regimes (cached to regimes.json),
# fetches the statistics, and converts every split into ./data/well (~7 GB):
#
#   python -m examples.well.datasets --config examples/well/well_trl2d.toml
#
# After that, a run touches the network for nothing. To use an existing copy
# of The Well instead, export WELL_DATA before submitting; nothing is
# downloaded or deleted from it:
#
#   WELL_DATA=$MEMBERWORK/<project>/the_well/datasets sbatch -A <project> ...
#
# MAX_ITER is the gradient steps per drift event, as on Perlmutter (default
# 10000 -> roughly 10 PF against the published 432). Frontier caps 1-node
# jobs in the batch partition at 2 hours; for a budget that needs longer,
# submit to the extended partition instead:
#
#   MAX_ITER=20000 sbatch -A <project> -p extended -t 06:00:00 \
#       examples/well/frontier.sbatch

module load PrgEnv-gnu
module load python/3.13.0
module load gcc/12.2.0
module load rocm/6.4.2

source .venv/bin/activate
export WANDB_MODE=offline

# MIOpen's default cache lives in $HOME, which compute nodes must not write.
ACCOUNT=$(sacct -j $SLURM_JOB_ID --format=Account --noheader | head -1 | tr -d ' ')
mkdir -p $MEMBERWORK/$ACCOUNT/miopen
export MIOPEN_USER_DB_PATH=$MEMBERWORK/$ACCOUNT/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MEMBERWORK/$ACCOUNT/miopen

# The node is exclusive but the run uses one GCD.
export ROCR_VISIBLE_DEVICES=0

MAX_ITER=${MAX_ITER:-10000}
DATA=${WELL_DATA:-hf://datasets/polymathic-ai}
RUN=output/well_unet_${SLURM_JOB_ID}
mkdir -p "${RUN}"

echo "=============================================="
echo "The Well -- unet (init_features=48), from scratch"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "ROCM_PATH: ${ROCM_PATH}"
echo "max_iter: ${MAX_ITER}"
echo "data:     ${DATA}"
echo "run dir:  ${RUN}"
echo "=============================================="

python -m src.main --config examples/well/well_trl2d.toml \
    --set model.name=unet \
    --set data.path="${DATA}" \
    --set train.max_iter="${MAX_ITER}" \
    --set model.ckpts_path="${RUN}/checkpoints" \
    --set logging.metrics_output_path="${RUN}/metrics.csv"

# Score every checkpoint on the test split, beside the trivial baselines and
# the untrained width-48 starting model.
python -m examples.well.evaluate \
    --config examples/well/well_trl2d.toml \
    --set model.name=unet \
    --set data.path="${DATA}" \
    --baselines --random \
    --checkpoints "${RUN}/checkpoints" \
    --out "${RUN}/test_matrix.csv"

# The stream trace with regime boundaries and the drift events.
python -m examples.well.plot \
    --config examples/well/well_trl2d.toml \
    --set model.name=unet \
    --set data.path="${DATA}" \
    --metrics "${RUN}/metrics.csv" \
    --out "${RUN}/drift_detection.png"
