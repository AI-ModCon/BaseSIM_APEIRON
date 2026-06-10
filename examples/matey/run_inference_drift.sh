#!/usr/bin/env bash
# Run MATEY ViT inference + domain-shift drift detection via APEIRON.
#
# Prerequisites: matey-env active, PYTHONPATH set (see notes/BaseSIM_APEIRON/apeiron-matey-frontier-setup.md).
# GPU node recommended when using a real checkpoint.
#
# Usage:
#   ./examples/matey/run_inference_drift.sh \
#     /path/to/baseline/solps \
#     /path/to/shift/solps \
#     /path/to/checkpoint.tar

set -euo pipefail

BASELINE="${1:-}"
SHIFT="${2:-}"
CHECKPOINT="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
  "${EXTRA[@]}"
