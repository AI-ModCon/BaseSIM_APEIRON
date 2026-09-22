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
#   python examples/matey/plot_adaptation_sequence.py "$OUTDIR" --stream "$STREAM"
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
# %j logs land in the submitting directory; keep them out of the tree.
#SBATCH --output=slurm-matey-stream-%j.out
#SBATCH --error=slurm-matey-stream-%j.err

set -euo pipefail

# sbatch copies the script to a spool directory, so BASH_SOURCE does not point
# into the repo. Use the submitting directory, which sbatch preserves.
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

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

# DETECTOR selects the drift detector; with EnsembleDetector, ENSEMBLE lists the
# sub-detectors and VOTING is any|majority|unanimous.
DETECTOR="${DETECTOR:-KSWINDetector}"
ENSEMBLE="${ENSEMBLE:-[\"ADWINDetector\", \"KSWINDetector\", \"PageHinkleyDetector\"]}"
VOTING="${VOTING:-any}"

# One arm per job. Beyond cl/nocl these are the continual-learning strategies
# compared in the catastrophic-forgetting study; "_mix" replays historical data
# alongside the arriving simulation.
arm="${1:-${ARM:-cl}}"
MIX=false
EXTRA=()
case "${arm}" in
  cl|base)    MODE="base" ;;
  nocl)       MODE="none" ;;
  # Same configuration as `base`, different seed. Without a replicate, a small
  # difference between two strategies cannot be told from run-to-run noise.
  base2)      MODE="base"; EXTRA+=(--set "seed=4242") ;;
  base_mix)   MODE="base";        MIX=true ;;
  ewc)        MODE="ewc_online" ;;
  ewc_mix)    MODE="ewc_online";  MIX=true ;;
  # Anchors the penalty on the pre-trained weights instead of re-anchoring on
  # each round's result, which is what the default "rolling" mode does.
  ewc_anchor) MODE="ewc_online"
              EXTRA+=(--set "continual_learning.ewc_anchor_mode=pretrained") ;;
  kfac)       MODE="kfac_online" ;;
  kfac_mix)   MODE="kfac_online"; MIX=true ;;
  # rho_x builds a perturbation direction from the difference of the two
  # batches, which needs them on a common grid; across machines they are not.
  # JVP_RHO_THETA is the SAM radius. The shipped 0.05 was chosen for models
  # trained from scratch; a converged surrogate fine-tuned at 3e-6 needs far
  # less, so it is exposed here rather than buried in the config.
  jvp)        MODE="jvp_reg"
              EXTRA+=(--set "continual_learning.jvp_rho_x=0.0"
                      --set "continual_learning.jvp_rho_theta=${JVP_RHO_THETA:-0.05}") ;;
  oracle)     MODE="none"
              CKPT="${ORACLE_CKPT:?set ORACLE_CKPT for the oracle arm}" ;;
  *) echo "usage: $0 [nocl|base|base2|base_mix|ewc|ewc_mix|ewc_anchor|kfac|kfac_mix|jvp|oracle]" >&2; exit 2 ;;
esac

# Every adapting arm must write checkpoints, or its retrospective evaluation has
# nothing to load. Per arm, not per OUTDIR: a shared directory would interleave
# snapshots from different strategies and silently corrupt every one of them.
if [[ "${MODE}" == "none" ]]; then
  MAX_CKPTS="${MAX_CKPTS:-0}"
else
  MAX_CKPTS="${MAX_CKPTS:-64}"
  if [[ "${MAX_CKPTS}" -eq 0 ]]; then
    echo "arm ${arm} adapts but MAX_CKPTS=0; the retrospective needs snapshots" >&2
    exit 2
  fi
fi
CKPTS_PATH="${CKPTS_PATH:-${OUTDIR}/ckpts_${arm}${TAG:-}}"

echo "================ arm=${arm} (update_mode=${MODE}, mix=${MIX}) ================"
python3 -m src.main \
  --config examples/matey/matey_stream.toml \
  --set "data.path=${STREAM}" \
  --set "model.pretrained_path=${CKPT}" \
  --set "continual_learning.update_mode=${MODE}" \
  --set "continual_learning.mix_historic_data=${MIX}" \
  --set "train.batch_size=${BATCH_SIZE:-1}" \
  --set "train.max_iter=${MAX_ITER:-500}" \
  --set "model.max_ckpts=${MAX_CKPTS}" \
  --set "model.ckpts_path=${CKPTS_PATH}" \
  --set "drift_detection.detector_name=${DETECTOR}" \
  --set "drift_detection.ensemble_detectors=${ENSEMBLE}" \
  --set "drift_detection.ensemble_voting=${VOTING}" \
  --set "drift_detection.max_stream_updates=${MAX_UPDATES}" \
  --set "drift_detection.kswin_window_size=${KSWIN_WINDOW:-60}" \
  --set "drift_detection.kswin_stat_size=${KSWIN_STAT:-20}" \
  --set "visualization.input=${OUTDIR}/stream_${arm}${TAG:-}.csv" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  2>&1 | tee "${OUTDIR}/run_${arm}${TAG:-}.log"

cp "${STREAM}/stream_manifest.json" "${OUTDIR}/" 2>/dev/null || true
echo "done: ${OUTDIR}/stream_${arm}${TAG:-}.csv"
