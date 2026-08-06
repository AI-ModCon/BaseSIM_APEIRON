#!/usr/bin/env bash
# Figure 3's upper-bound arm, in two stages:
#
#   1. fine-tune the pre-trained checkpoint on the train splits of all 32
#      arrivals jointly (train_joint_oracle.py);
#   2. stream that frozen checkpoint through the same 32 arrivals with
#      update_mode=none, writing stream_oracle.csv beside the cl/nocl arms.
#
# Stage 2 is deliberately identical to the nocl arm except for which weights it
# loads -- same config, same stream, same detector, no adaptation. Any
# difference between the two curves is therefore the weights and nothing else.
#
# Pass the run directory holding the existing stream_cl.csv / stream_nocl.csv as
# $1 so all three arms land together and plot_cl_gain.py pairs them:
#
#   sbatch examples/matey/submit_joint_oracle.sh output/matey_stream_20260729_053418
#
#SBATCH --account=lrn097
#SBATCH --job-name=matey-joint-oracle
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-oracle-%j.out
#SBATCH --error=/lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON/output/slurm-matey-oracle-%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATEY_ENV="${MATEY_ENV:?set MATEY_ENV}"
STREAM="${STREAM:?set STREAM}"
CKPT="${CKPT:?set CKPT}"
ORACLE_DIR="${ORACLE_DIR:-/lustre/orion/lrn097/scratch/asvillar/mateydata/models/joint_oracle}"

OUTDIR="${1:-${OUTDIR:-}}"
if [[ -z "${OUTDIR}" ]]; then
  echo "usage: $0 <run-dir-with-stream_cl.csv>" >&2
  exit 2
fi

EPOCHS="${EPOCHS:-3}"
# The staged train split is ~35 usable samples per arrival, so 40 means "one
# full pass over this arrival" rather than a cap. 3 epochs x 32 arrivals x ~35
# is ~3400 gradient steps, comparable to the CL arm's ~6 events x 500 iters.
STEPS_PER_ARRIVAL="${STEPS_PER_ARRIVAL:-40}"

cd "${ROOT}"
mkdir -p "${OUTDIR}"

MIOPEN_CACHE="${MIOPEN_CACHE:-/lustre/orion/lrn097/scratch/${USER}/miopen_cache}"
mkdir -p "${MIOPEN_CACHE}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE}"

unset PYTHONPATH
# shellcheck disable=SC1090
source "${MATEY_ENV}"
USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${USER_SITE}:${ROOT}/src:${ROOT}:/lustre/orion/lrn097/proj-shared/fusionMT/MATEY:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

N_ARRIVALS="$(python -c "
import json
print(json.load(open('${STREAM}/stream_manifest.json'))['n_arrivals'])
")"
# MAX_UPDATES caps the evaluated stream; MAX_ARRIVALS caps what the oracle
# is trained on. They must agree, or the oracle sees data the arms do not.
MAX_UPDATES="${MAX_UPDATES:-$((N_ARRIVALS - 1))}"
MAX_ARRIVALS="${MAX_ARRIVALS:-0}"

echo ""
echo "================ stage 1: joint fine-tune ================"
python3 examples/matey/train_joint_oracle.py \
  --config examples/matey/matey_stream.toml \
  --out "${ORACLE_DIR}" \
  --epochs "${EPOCHS}" \
  --steps-per-arrival "${STEPS_PER_ARRIVAL}" \
  --max-arrivals "${MAX_ARRIVALS}" \
  --set "data.path=${STREAM}" \
  --set "model.pretrained_path=${CKPT}" \
  --set "continual_learning.update_mode=none" \
  2>&1 | tee "${OUTDIR}/run_oracle_train.log"

echo ""
echo "================ stage 2: stream the frozen oracle ================"
python3 -m src.main \
  --config examples/matey/matey_stream.toml \
  --set "data.path=${STREAM}" \
  --set "model.pretrained_path=${ORACLE_DIR}/best_ckpt.tar" \
  --set "continual_learning.update_mode=none" \
  --set "drift_detection.max_stream_updates=${MAX_UPDATES}" \
  --set "visualization.input=${OUTDIR}/stream_oracle.csv" \
  2>&1 | tee "${OUTDIR}/run_oracle.log" || echo "oracle arm exited non-zero"

echo ""
echo "================ stage 3: replot ================"
python3 examples/matey/drift_showcase/plot_cl_gain.py --run "${OUTDIR}" || true
python3 examples/matey/drift_showcase/plot_stream_cl.py --run "${OUTDIR}" || true

echo ""
echo "done: ${OUTDIR}"
