#!/usr/bin/env bash
# Stage a SOLPS2D-compatible shift domain under scratch for inference-drift runs.
#
# The MATEY loader expects train/ and valid/ dirs with *.nc files that have an
# "nt" dimension (solps-kstar_example-*.nc). Raw SOLPS2DwION b2time.nc files
# (D3D/KSTAR/SPARC trees) use a different schema and will fail with KeyError: nt.
#
# This script builds a minimal shift layout from the shared KSTAR example files
# (different file mix than the default pretraining/solps tree).

set -euo pipefail

SRC="${1:-/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps}"
OUT="${2:-/lustre/orion/lrn097/scratch/${USER}/solps_domains/shift}"

if [[ ! -d "${SRC}/train" || ! -d "${SRC}/valid" ]]; then
  echo "ERROR: source must contain train/ and valid/: ${SRC}" >&2
  exit 1
fi

rm -rf "${OUT}"
mkdir -p "${OUT}/train" "${OUT}/valid"

ln -sf "${SRC}/valid/solps-kstar_example-3.nc" "${OUT}/train/kstar_ex3.nc"
ln -sf "${SRC}/train/solps-kstar_example-1.nc" "${OUT}/valid/kstar_ex1.nc"

cat <<EOF
Staged shift domain: ${OUT}
  train/ -> solps-kstar_example-3.nc
  valid/ -> solps-kstar_example-1.nc

Run inference drift (GPU node):

  export BASELINE=${SRC}
  export SHIFT=${OUT}
  export CKPT=/lustre/orion/fus183/proj-shared/MATEY/models/Dev_Fusion_Demo_March2026_Final/demo_nbatchsloc100/training_checkpoints/best_ckpt.tar
  export WANDB_MODE=offline ENABLE_WANDB=1
  ./examples/matey/run_inference_drift.sh "\${BASELINE}" "\${SHIFT}" "\${CKPT}"

Note: this is a weak domain shift (same machine/schema). Expect flat NRMSE and
no ADWIN trigger. Use run_outer_loop.sh to validate the drift pipeline, or
obtain additional SOLPS2D data for a strong shift.
EOF
