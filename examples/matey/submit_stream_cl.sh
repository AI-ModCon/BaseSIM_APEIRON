#!/usr/bin/env bash
# Stream the staged SOLPS arrivals past a pretrained MATEY checkpoint, with drift
# detection dispatching continual learning.
#
# Two arms over the identical stream:
#   cl    update_mode=base  -- detection dispatches adaptation
#   nocl  update_mode=none  -- same stream, no adaptation (the control)
#
# The control is what makes this a result rather than a demo: without it, a
# falling error curve could just be the later arrivals being easier. Run both
# into the same OUTDIR, then compare:
#
#   OUTDIR=output/stream_$(date +%Y%m%d_%H%M%S)
#   sbatch --export=ALL,OUTDIR="$OUTDIR" examples/matey/submit_stream_cl.sh nocl
#   sbatch --export=ALL,OUTDIR="$OUTDIR" examples/matey/submit_stream_cl.sh cl
#   python examples/matey/plot_stream_arms.py "$OUTDIR" --events-at 10 14 19 23
#
# Wall time on one MI250X over 24 arrivals: ~2.5 min for nocl, ~17 min for cl.
# They are separate jobs because Frontier caps one-node batch jobs at two hours.
#
#SBATCH --account=lrn097
#SBATCH --job-name=matey-stream-cl
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-matey-stream-%j.out
#SBATCH --error=slurm-matey-stream-%j.err

set -euo pipefail

# Two levels up from this script, so the job does not depend on the submitting
# directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Site-specific; override in the environment rather than editing this file.
MATEY_ENV="${MATEY_ENV:?set MATEY_ENV to the MATEY environment setup script}"
STREAM="${STREAM:?set STREAM to the stream root holding stream_manifest.json}"
CKPT="${CKPT:?set CKPT to the pretrained MATEY checkpoint}"
OUTDIR="${OUTDIR:?set OUTDIR; both arms must share one so they can be compared}"

cd "${ROOT}"
mkdir -p "${OUTDIR}"

# MIOpen caches compiled kernels per user; point it somewhere writable and
# persistent or every run pays the compile cost again.
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

N_ARRIVALS="$(python -c "
import json
print(json.load(open('${STREAM}/stream_manifest.json'))['n_arrivals'])
")"
# ContinuousMonitor calls update_data_stream() once before its loop and once per
# extension, so it consumes max_stream_updates + 1 arrivals; asking for
# n_arrivals extensions would request one past the end.
MAX_UPDATES="${MAX_UPDATES:-$((N_ARRIVALS - 1))}"
echo "stream has ${N_ARRIVALS} arrivals -> max_stream_updates=${MAX_UPDATES}"

arm="${1:-${ARM:-cl}}"
case "${arm}" in
  cl)   MODE="base" ;;
  nocl) MODE="none" ;;
  *) echo "usage: $0 [cl|nocl]" >&2; exit 2 ;;
esac

echo "================ arm=${arm} (update_mode=${MODE}) ================"
python3 -m src.main \
  --config examples/matey/matey_stream.toml \
  --set "data.path=${STREAM}" \
  --set "model.pretrained_path=${CKPT}" \
  --set "continual_learning.update_mode=${MODE}" \
  --set "drift_detection.max_stream_updates=${MAX_UPDATES}" \
  --set "visualization.input=${OUTDIR}/stream_${arm}.csv" \
  2>&1 | tee "${OUTDIR}/run_${arm}.log"

cp "${STREAM}/stream_manifest.json" "${OUTDIR}/" 2>/dev/null || true
echo "done: ${OUTDIR}/stream_${arm}.csv"
