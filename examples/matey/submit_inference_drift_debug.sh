#!/usr/bin/env bash
# Short GPU job on Frontier batch (no dedicated debug partition on OLCF).
# Typical runtime ~5–15 min for max_stream_updates=4–10.

#SBATCH --account=lrn097
#SBATCH --job-name=matey-drift
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-drift-%j.out
#SBATCH --error=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-drift-%j.err

set -euo pipefail

ROOT="/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON"
BASELINE="/lustre/orion/lrn097/scratch/asvillar/mateydata/solps_drift/baseline"
OOD="/lustre/orion/lrn097/scratch/asvillar/mateydata/solps_drift/ood"
CKPT="/lustre/orion/lrn097/scratch/asvillar/mateydata/models/leadtime_1/best_ckpt.tar"

cd "${ROOT}"

./examples/matey/run_inference_drift.sh \
  "${BASELINE}" \
  "${OOD}" \
  "${CKPT}" \
  --set data.dset_type=SOLPS2DwION \
  --set drift_detection.max_stream_updates=4 \
  --set drift_detection.detection_interval=1 \
  --set drift_detection.metric_index=3 \
  --set logging.backend=none \
  --set verbosity=INFO:1

echo "Done. See output/matey_inference_drift_latest/ (symlink to newest run)"
