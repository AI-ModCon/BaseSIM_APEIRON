#!/bin/bash -l
#SBATCH -J well_unet
#SBATCH -t 04:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -o output/well_unet.o%j
#SBATCH -e output/well_unet.e%j

# The Well at the published width: `unet` (init_features=48), trained from
# scratch by detect-and-adapt over the nine turbulent_radiative_layer_2D
# regimes, then scored on the held-out test split.
#
# Setup (once, from the project root -- see
# src/apeiron/deployment/perlmutter/README.md):
#
#   source ./src/apeiron/deployment/perlmutter/install_venv.sh
#   mkdir -p output
#
# Submit from the project root:
#
#   sbatch -A <account> examples/well/perlmutter.sbatch
#
# The first run downloads the regimes into ./data/well, one at a time
# (~6 GB for train+valid, ~0.7 GB more for test when scoring starts). To use
# an existing copy of The Well instead, export WELL_DATA before submitting:
#
#   WELL_DATA=/pscratch/.../the_well/datasets sbatch -A <account> ...
#
# MAX_ITER is the gradient steps per drift event. A width-48 training step at
# batch 4 is ~0.5 TFLOP, and at the budgets mapped on the small model the
# detector settles at two firings, so the training envelope is roughly
#
#   2 x MAX_ITER x 0.5 TFLOP      MAX_ITER=10000  ->  ~10 PF  (1/43 of the
#                                 published 432 PF training budget)
#
#   MAX_ITER=20000 sbatch -A <account> examples/well/perlmutter.sbatch

source .venv/bin/activate
export WANDB_MODE=offline

MAX_ITER=${MAX_ITER:-10000}
DATA=${WELL_DATA:-hf://datasets/polymathic-ai}
RUN=output/well_unet_${SLURM_JOB_ID}
mkdir -p "${RUN}"

echo "=============================================="
echo "The Well -- unet (init_features=48), from scratch"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
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
