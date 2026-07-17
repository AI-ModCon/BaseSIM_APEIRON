#!/usr/bin/env bash
# FusionBench bundle drift: stage FB slices, parity gate, then drift on solps_drift_fb.
#SBATCH --account=lrn097
#SBATCH --job-name=matey-fb-drift
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-fb-drift-%j.out
#SBATCH --error=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-fb-drift-%j.err

set -euo pipefail

ROOT="/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON"
MATEYDATA="/lustre/orion/lrn097/scratch/asvillar/mateydata"
FB_ROOT="${MATEYDATA}/solps_drift_fb"
BASELINE="${FB_ROOT}/baseline"
OOD="${FB_ROOT}/ood"
CKPT="${MATEYDATA}/models/leadtime_1/best_ckpt.tar"

cd "${ROOT}"
mkdir -p output

MATEY_ENV="/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh"
MIOPEN_CACHE="${MIOPEN_CACHE:-${SCRATCH:-/lustre/orion/lrn097/scratch/${USER}}/miopen_cache}"
mkdir -p "${MIOPEN_CACHE}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE}"

unset PYTHONPATH
# shellcheck disable=SC1090
source "${MATEY_ENV}"
USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${USER_SITE}:/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

echo "=== Stage FusionBench SOLPS bundles ==="
chmod +x examples/matey/stage_solps_fusionbench_bundles.sh
./examples/matey/stage_solps_fusionbench_bundles.sh "${MATEYDATA}" "${FB_ROOT}"

echo "=== FusionBench parity gate (baseline sin4) ==="
python3 examples/matey/verify_fusionbench_parity.py \
  --baseline "${BASELINE}" \
  --checkpoint "${CKPT}"

echo "=== Drift run on FusionBench bundles ==="
./examples/matey/run_inference_drift.sh \
  "${BASELINE}" \
  "${OOD}" \
  "${CKPT}" \
  --set data.dset_type=SOLPS2DwION \
  --set eval.max_val_batches=15 \
  --set eval.leadtime=1 \
  --set eval.use_step_inference=true \
  --set drift_detection.max_stream_updates=2 \
  --set drift_detection.detection_interval=1 \
  --set drift_detection.metric_index=3 \
  --set drift_detection.adwin_delta=0.005 \
  --set logging.backend=none \
  --set verbosity=INFO:1

RUN_DIR="$(readlink -f output/matey_inference_drift_latest)"
echo ""
echo "=== Summary ==="
echo "Run dir: ${RUN_DIR}"
if [[ -f "${RUN_DIR}/matey_inference_drift.csv" ]]; then
  python3 <<PY
import csv
from pathlib import Path

run_dir = Path("${RUN_DIR}")
rows = list(csv.DictReader(run_dir.joinpath("matey_inference_drift.csv").open()))
eval_rows = [r for r in rows if r.get("stage") == "eval" and r.get("metric") == "nrmse_mean"]
drift_rows = [r for r in rows if r.get("stage") == "drift"]
print(f"eval nrmse_mean rows: {len(eval_rows)}")
if eval_rows:
    vals = [float(r["value"]) for r in eval_rows[:20]]
    print(f"  first eval values: {vals[:10]}")
if drift_rows:
    print(f"drift rows: {len(drift_rows)}")
    for r in drift_rows[:5]:
        print(f"  detected={r.get('detected')} score={r.get('score')} metric_3={r.get('metric_3')}")
PY
fi

echo "Done. Compare NRMSE/ADWIN vs prior full-b2time run under output/matey_inference_drift_20260717_181725/"
