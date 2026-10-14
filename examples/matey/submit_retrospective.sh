#!/usr/bin/env bash
# Score one arm's saved adaptation checkpoints against earlier arrivals.
#
#   sbatch --export=ALL,MATEY_ENV=..,STREAM=..,CKPT=..,OUTDIR=..,ARM=base \
#          examples/matey/submit_retrospective.sh
#
# ARRIVALS defaults to the stream's baseline block plus the first shifted one,
# which is what the forgetting curve is read from; pass "all" for the full
# continual-learning R-matrix at roughly three times the cost.
#
#SBATCH --account=lrn097
#SBATCH --job-name=matey-retro
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-matey-retro-%j.out
#SBATCH --error=slurm-matey-retro-%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATEY_ENV="${MATEY_ENV:?set MATEY_ENV to the MATEY environment setup script}"
STREAM="${STREAM:?set STREAM to the stream root}"
CKPT="${CKPT:?set CKPT to the pretrained MATEY checkpoint}"
OUTDIR="${OUTDIR:?set OUTDIR to the run directory holding stream_<arm>.csv}"
ARM="${ARM:?set ARM to the arm being scored}"
ARRIVALS="${ARRIVALS:-0-11}"

cd "${ROOT}"

MIOPEN_CACHE="${MIOPEN_CACHE:-${SCRATCH:-/tmp}/miopen_cache}"
mkdir -p "${MIOPEN_CACHE}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE}"

unset PYTHONPATH
# shellcheck disable=SC1090
source "${MATEY_ENV}"
USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
export PYTHONPATH="${USER_SITE}:${ROOT}/src:${ROOT}:${MATEY_SRC:-}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

python3 examples/matey/eval_retrospective.py \
  --config examples/matey/matey_stream.toml \
  --arm "${ARM}" \
  --ckpts "${OUTDIR}/ckpts_${ARM}" \
  --run-log "${OUTDIR}/run_${ARM}.log" \
  --arrivals "${ARRIVALS}" \
  --out "${OUTDIR}/retro_${ARM}.csv" \
  --set "data.path=${STREAM}" \
  --set "model.pretrained_path=${CKPT}" \
  2>&1 | tee "${OUTDIR}/retro_${ARM}.log"
