#!/usr/bin/env bash
# Outer-loop demo: MATEY loaders + placeholder model + synthetic input-noise drift.
# Runs on Frontier login node (CPU). See apeiron-matey-frontier-setup.md.

set -euo pipefail

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Re-source matey-env if python is not the conda env (common after module load).
if [[ ! "$(command -v python)" == *matey-env* ]]; then
  unset PYTHONPATH
  # shellcheck disable=SC1090
  source "${MATEY_ENV}"
fi

export PYTHONPATH="/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

python -c "import adios2, river, logger; print('env ok:', __import__('sys').executable)" || {
  echo "ERROR: matey-env not active. Run: source ${MATEY_ENV}" >&2
  exit 1
}

cd "${ROOT}"
SOLPS="${1:-/lustre/orion/fus183/proj-shared/MATEY/Datasets_pretraining/solps/train}"

python -m src.main \
  --config examples/matey/matey_outer_loop.toml \
  --set "data.path=${SOLPS}" \
  --set logging.backend=none \
  --set verbosity=INFO:1 \
  --set drift_detection.max_stream_updates=10 \
  --set drift_detection.detection_interval=3 \
  --set drift_detection.adwin_delta=0.001 \
  --set train.max_iter=50 \
  "${@:2}"
