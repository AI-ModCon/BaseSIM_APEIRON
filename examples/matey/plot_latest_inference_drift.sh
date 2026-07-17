#!/usr/bin/env bash
# Plot dashboard for the most recent timestamped inference drift run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:-${ROOT}/output/matey_inference_drift_latest}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run dir not found: ${RUN_DIR}" >&2
  echo "Run inference first: ./examples/matey/run_inference_drift.sh ..." >&2
  exit 1
fi

CSV="${RUN_DIR}/matey_inference_drift.csv"
if [[ ! -f "${CSV}" ]]; then
  echo "ERROR: CSV not found: ${CSV}" >&2
  exit 1
fi

NUM_STREAMS="${PLOT_NUM_STREAMS:-4}"
python3 "${ROOT}/examples/matey/plot_inference_drift_dashboard.py" \
  --csv "${CSV}" \
  --output "${RUN_DIR}/dashboard.png" \
  --num-streams "${NUM_STREAMS}" \
  --metric-index 3

echo "Saved ${RUN_DIR}/dashboard.png"
