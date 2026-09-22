#!/bin/bash
#SBATCH -A lrn097
#SBATCH -J xgc_mesh_drift
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Mesh-resolved XGC cross-device drift extraction.
# CPU-only, but I/O heavy: one ITER graphdata_*.pt is ~6.9 GB, so this reads
# tens of GB from Lustre and should not run on a login node.

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT=${OUT:-$REPO/output/xgc_mesh_drift_$(date +%Y%m%d_%H%M%S)}

unset PYTHONPATH
source "${MATEY_ENV:?set MATEY_ENV to the MATEY environment setup script}"
export PYTHONPATH="${MATEY_SRC:-}:$REPO/src:$REPO:${PYTHONPATH:-}"

cd "$REPO"
echo "[submit] out=$OUT"
srun -n1 -c56 python examples/matey/drift_showcase/xgc_mesh_drift.py \
    --frames "${FRAMES:-10}" \
    --nodes "${NODES:-20000}" \
    --out "$OUT"

echo "[submit] done -> $OUT"
