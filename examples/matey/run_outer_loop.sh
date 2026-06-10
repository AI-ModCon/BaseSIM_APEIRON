#!/usr/bin/env bash
# Outer-loop demo: MATEY loaders + placeholder model + synthetic input-noise drift.
# Runs on Frontier login node (CPU). See apeiron-matey-frontier-setup.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
