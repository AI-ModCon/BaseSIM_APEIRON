#!/usr/bin/env bash
# Outer-loop demo: MATEY loaders + placeholder model + synthetic input-noise drift.
# Runs on Frontier login node (CPU). See apeiron-matey-frontier-setup.md.

set -euo pipefail

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Always re-source: prompt can show matey-env while PYTHONPATH was cleared (e.g. unset PYTHONPATH).
unset PYTHONPATH
# shellcheck disable=SC1090
source "${MATEY_ENV}"

export PYTHONPATH="/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

python -c "import adios2, river, logger; print('env ok:', __import__('sys').executable)" || {
  echo "ERROR: env check failed after sourcing ${MATEY_ENV}" >&2
  echo "  python: $(command -v python)" >&2
  echo "  If adios2 is missing: pip install adios2==2.11.0.1012" >&2
  echo "  If logger is missing: PYTHONPATH must include ${ROOT}/src" >&2
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
