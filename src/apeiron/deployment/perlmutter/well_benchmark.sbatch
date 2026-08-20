#!/bin/bash -l
#SBATCH -J well_benchmark
#SBATCH -t 0:30:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o output/well_benchmark.o%j
#SBATCH -e output/well_benchmark.e%j

# Data-parallel throughput / scaling run for the Well surrogate (Phase 3).
#
# World size = nodes x ntasks-per-node (one rank per GPU). Sweep it by varying
# --nodes at submit time; each run writes its own CSV keyed by SLURM_NTASKS:
#
#   mkdir -p output
#   sbatch -A <account> src/apeiron/deployment/perlmutter/well_benchmark.sbatch            #  4 ranks
#   sbatch -A <account> --nodes=2 src/apeiron/deployment/perlmutter/well_benchmark.sbatch  #  8 ranks
#   sbatch -A <account> --nodes=4 src/apeiron/deployment/perlmutter/well_benchmark.sbatch  # 16 ranks (use -q regular)
#
# IMPORTANT: build the WindowStore FIRST on a login node -- compute nodes have no
# internet, so the HuggingFace download must happen before you submit:
#
#   poetry run python -m examples.well.convert --dataset turbulent_radiative_layer_2D \
#       --split train --max-files 8 --out $SCRATCH/wellstore --window-steps 24
#
# Do NOT add --gpu-bind / --gpus-per-task: leave all GPUs visible per node so
# each rank binds cuda:LOCAL_RANK (which comm reads from SLURM_LOCALID).

source ./.venv/bin/activate
export WANDB_MODE=offline
mkdir -p output

STORE=${WELL_STORE:-$SCRATCH/wellstore}

echo "=============================================="
echo "Well benchmark: throughput (data-parallel)"
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES  world_size (tasks): $SLURM_NTASKS  store: $STORE"
echo "=============================================="

# Scale --width up until the gradient all-reduce stops dominating compute
# (check the comm_all_reduce_s column relative to wall_s).
srun python -m examples.well.benchmark throughput \
    --store "$STORE" \
    --out "output/well_scale_w${SLURM_NTASKS}.csv" \
    --width 256 \
    --batch-size 32

echo "wrote output/well_scale_w${SLURM_NTASKS}.csv"

# --- Single-process modes (memory / frontier / resume) do NOT use srun ---
# Run them on a login node or a 1-task job, e.g.:
#   poetry run python -m examples.well.benchmark memory  --store $STORE --out output/mem.csv --task-counts 1,10,50,200,500
#   poetry run python -m examples.well.benchmark frontier --store $STORE --out output/frontier.csv --budgets 0,1,3,7,14
#   poetry run python -m examples.well.benchmark resume  --store $STORE --out output/resume.csv --ckpt-dir output/ck
