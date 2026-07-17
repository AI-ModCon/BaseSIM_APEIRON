#!/usr/bin/env bash
# Stage NOT-OOD (Sequence_sin4) vs OOD (noLat_dribble) SOLPS2DwION domains for drift runs.
set -euo pipefail

MATEYDATA="${1:-/lustre/orion/lrn097/scratch/asvillar/mateydata}"
PRE="${2:-/lustre/orion/lrn097/proj-shared/fusionMT-data/SOLPS2DwION/D3D/174310_D/puff2.5e21_ss_Sequence_sin4_308_2d_output/b2time.nc}"
OOD="${3:-${MATEYDATA}/Datasets_notusedinpretraining/D3D/174310_D/puff2.5e21_ss_noLat_dribble_308_2d_output/b2time.nc}"
DRIFT="${4:-${MATEYDATA}/solps_drift}"

for f in "$PRE" "$OOD"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing SOLPS file: $f" >&2
    exit 1
  fi
done

rm -rf "${DRIFT}"
mkdir -p "${DRIFT}/baseline/train" "${DRIFT}/baseline/valid"
mkdir -p "${DRIFT}/ood/train" "${DRIFT}/ood/valid"

ln -sfn "$(readlink -f "$PRE")" "${DRIFT}/baseline/train/d3d_sequence_sin4.nc"
ln -sfn "$(readlink -f "$PRE")" "${DRIFT}/baseline/valid/d3d_sequence_sin4.nc"
ln -sfn "$(readlink -f "$OOD")" "${DRIFT}/ood/train/d3d_noLat_dribble.nc"
ln -sfn "$(readlink -f "$OOD")" "${DRIFT}/ood/valid/d3d_noLat_dribble.nc"

cat <<EOF
Staged SOLPS2DwION drift domains under: ${DRIFT}
  baseline (NOT-OOD): ${PRE}
  ood (shift):        ${OOD}

Drift run (GPU, no wandb):
  export MATEYDATA=${MATEYDATA}
  export BASELINE=${DRIFT}/baseline
  export SHIFT=${DRIFT}/ood
  export CKPT=${MATEYDATA}/models/leadtime_1/best_ckpt.tar
  cd /lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON
  unset ENABLE_WANDB
  ./examples/matey/run_inference_drift.sh "\${BASELINE}" "\${SHIFT}" "\${CKPT}" \\
    --set logging.backend=none \\
    --set data.dset_type=SOLPS2DwION \\
    --set drift_detection.max_stream_updates=4 \\
    --set drift_detection.detection_interval=3 \\
    --set drift_detection.adwin_delta=0.001 \\
    --set verbosity=INFO:1
EOF
