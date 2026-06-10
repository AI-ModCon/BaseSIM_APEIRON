#!/usr/bin/env bash
# Run MATEY ViT inference + domain-shift drift detection via APEIRON.
# GPU node recommended when using a real checkpoint.

set -euo pipefail

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"
BASELINE="${1:-}"
SHIFT="${2:-}"
CHECKPOINT="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

EXTRA=()
if [[ -n "${BASELINE}" ]]; then
  EXTRA+=(--set "data.path=${BASELINE}")
fi
if [[ -n "${SHIFT}" ]]; then
  EXTRA+=(--set "data.alt_path=${SHIFT}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA+=(--set "model.pretrained_path=${CHECKPOINT}")
fi

python -m src.main \
  --config examples/matey/matey_inference_drift.toml \
  --set logging.backend=none \
  --set verbosity=INFO:1 \
  --set drift_detection.max_stream_updates=10 \
  --set drift_detection.detection_interval=5 \
  --set drift_detection.adwin_delta=0.01 \
  --set train.max_iter=50 \
  "${EXTRA[@]}"
