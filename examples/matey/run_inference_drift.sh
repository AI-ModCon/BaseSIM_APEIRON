#!/usr/bin/env bash
# Run MATEY ViT inference + domain-shift drift detection via APEIRON.
# GPU node recommended when using a real checkpoint.

set -euo pipefail

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"
BASELINE="${1:-}"
SHIFT="${2:-}"
CHECKPOINT="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# MIOpen writes kernel cache SQLite DBs; default paths are often read-only on Frontier.
MIOPEN_CACHE="${MIOPEN_CACHE:-${SCRATCH:-/lustre/orion/lrn097/scratch/${USER}}/miopen_cache}"
mkdir -p "${MIOPEN_CACHE}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE}"

unset PYTHONPATH
# shellcheck disable=SC1090
source "${MATEY_ENV}"

USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${USER_SITE}:/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

LOGGING_BACKEND=none
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  LOGGING_BACKEND=wandb
  unset WANDB_DISABLED
  if [[ "${WANDB_MODE:-}" != "offline" ]]; then
    unset WANDB_MODE
  fi
  export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
  if [[ "${WANDB_MODE:-}" == "offline" ]]; then
    echo "wandb: offline mode (sync later: wandb sync wandb/offline-run-*)" >&2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Paste WANDB_API_KEY from https://wandb.ai/authorize (input hidden), then Enter:"
    read -rs WANDB_API_KEY
    echo
    export WANDB_API_KEY
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: ENABLE_WANDB=1 but WANDB_API_KEY is empty." >&2
    exit 1
  fi
  python << 'PY' || {
import os
import sys

key = os.environ.get("WANDB_API_KEY", "")
from wandb.sdk.lib.wbauth.validation import check_api_key

problems = check_api_key(key)
if problems:
    print(f"ERROR: WANDB_API_KEY invalid: {problems}", file=sys.stderr)
    print(f"  length={len(key)} (expect ~86 for wandb_v1 keys)", file=sys.stderr)
    print("  Paste only the key from wandb.ai/authorize — no quotes, spaces, or 'Copy' label.", file=sys.stderr)
    print("  Tip: read -rs WANDB_API_KEY && export WANDB_API_KEY", file=sys.stderr)
    sys.exit(1)
print(f"wandb API key ok (length={len(key)})")
PY
    echo "ERROR: WANDB_API_KEY validation failed." >&2
    exit 1
  }
  python -c "import wandb; v=wandb.__version__; assert tuple(int(x) for x in v.split('.')[:2]) >= (0, 22), f'wandb {v} too old; run examples/matey/setup_wandb.sh'" \
    || { echo "ERROR: wandb>=0.22.3 required. Run: ./examples/matey/setup_wandb.sh" >&2; exit 1; }
else
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true
fi

python -c "import adios2, river, logger; print('env ok:', __import__('sys').executable)" || {
  echo "ERROR: env check failed after sourcing ${MATEY_ENV}" >&2
  echo "  python: $(command -v python)" >&2
  echo "  If adios2 is missing: pip install adios2==2.11.0.1012" >&2
  echo "  If logger is missing: PYTHONPATH must include ${ROOT}/src" >&2
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
if [[ -n "${DEVICE:-}" ]]; then
  EXTRA+=(--set "device=${DEVICE}")
fi

if [[ "$(hostname)" == login* ]] && [[ "${DEVICE:-auto}" == "auto" ]]; then
  echo "NOTE: login node detected — ViT inference needs a GPU compute node (srun/sbatch)." >&2
  echo "      Quick smoke test: DEVICE=cpu ENABLE_WANDB=1 ./examples/matey/run_inference_drift.sh ..." >&2
  echo "      Or submit: srun --gpus=1 --time=1:00:00 ... ENABLE_WANDB=1 ./examples/matey/run_inference_drift.sh ..." >&2
fi

python -m src.main \
  --config examples/matey/matey_inference_drift.toml \
  --set "logging.backend=${LOGGING_BACKEND}" \
  --set verbosity=INFO:1 \
  --set drift_detection.max_stream_updates=10 \
  --set drift_detection.detection_interval=5 \
  --set drift_detection.adwin_delta=0.01 \
  --set train.max_iter=50 \
  "${EXTRA[@]}" \
  "${@:4}"
