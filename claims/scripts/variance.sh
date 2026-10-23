#!/bin/bash
# Take REPEATED independent draws of each layer and record the spread.
#
#   claims/scripts/variance.sh                      # 3 draws of every layer
#   DRAWS=5 claims/scripts/variance.sh config       # 5 draws of one layer
#
# Why this exists: a kept-ratio is ONE DRAW from a stochastic producer, and
# the campaign reported single draws (47% early, 51% after the tests were
# strengthened). Without a spread those two numbers cannot be compared — the
# difference may be two samples from one distribution, in which case the loop
# of "reconstruct, find the gap, strengthen the criteria, re-measure" has no
# reliable gradient and could be chasing noise. This measures the noise.
#
# Three rules the measurement depends on:
#
#   1. The producer is PINNED (RETICULI_MODEL), so draws differ by sampling
#      alone and not by which model answered. The ledger records the model
#      the producer actually ran, which a reader can check per draw.
#   2. Draws are INDEPENDENT: one fresh room each, no retries folded in, and
#      no --recursive (a chain multiplies per-layer luck, LOG A24/R7).
#   3. A producer REFUSAL IS NOT A SPEC FAILURE. Hitting the Codex usage
#      limit says nothing about the criteria, so those draws are recorded as
#      `quota` and excluded from the ratio statistics rather than counted as
#      zeros, which would fabricate variance that is really rate limiting.
#
# Output: claims/measurements/variance-<stamp>.jsonl, one object per draw,
# plus a summary table on stdout. Safe to interrupt; completed draws persist.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
REPO=$PWD
RETICULI=${RETICULI_SRC:-$REPO/../reticuli/src}
PY=${RETICULI_PYTHON:-python3.13}
ROOMS=${VARIANCE_ROOMS:-${TMPDIR:-/tmp}/apeiron-variance}
DRAWS=${DRAWS:-3}
MODEL=${RETICULI_MODEL:-gpt-6-astra}
PRODUCER=${VARIANCE_PRODUCER:-"env PYTHONPATH=$RETICULI RETICULI_MODEL=$MODEL python3 -m reticuli.producers.codex"}
OUT_DIR=$REPO/claims/measurements
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LEDGER=$OUT_DIR/variance-$STAMP.jsonl

ALL=(config evaluation profilers model_harness updaters logger
     drift_detection trainer monitor entrypoints)
LAYERS=("$@")
[ ${#LAYERS[@]} -eq 0 ] && LAYERS=("${ALL[@]}")

mkdir -p "$ROOMS" "$OUT_DIR"
# The model must be in the env of `ret` ITSELF, not only of the producer
# child: the kernel writes the fallback declaration from its own environment,
# so exporting it to the producer alone leaves a draw unattributed when the
# producer cannot observe what ran (measured: the first three draws here
# recorded no model at all).
export RETICULI_MODEL=$MODEL
ret() { PYTHONPATH=$RETICULI $PY -m reticuli "$@"; }

if ! ret --version >/dev/null 2>&1; then
  echo "cannot run reticuli: set RETICULI_SRC to its src/ and RETICULI_PYTHON" >&2
  exit 1
fi

echo "# variance: ${#LAYERS[@]} layer(s) x $DRAWS draw(s), model=$MODEL"
echo "# ledger: $LEDGER"

for claim in "${LAYERS[@]}"; do
  for draw in $(seq 1 "$DRAWS"); do
    room="$ROOMS/$claim-$draw"
    log="$ROOMS/$claim-$draw.log"
    rm -rf "$room"
    started=$(date -u +%s)
    if ret rebuild "claims/$claim" --producer "$PRODUCER" -o "$room" >"$log" 2>&1; then
      status=rebuilt
    elif grep -qi "usage limit\|rate limit\|quota" "$log"; then
      status=quota          # the producer refused; the criteria were not tested
    else
      status=failed
    fi
    seconds=$(( $(date -u +%s) - started ))

    # what the producer says it actually ran, per draw -- a draw whose model
    # is unknown cannot be pooled with the others
    observed=$($PY - "$room" <<'PYEOF'
import json, os, sys
room = sys.argv[1]
path = os.path.join(room, ".reticuli", "ledger.jsonl")
model = source = None
try:
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "producer":
                model, source = event.get("model"), event.get("model_source")
except OSError:
    pass
print(json.dumps({"model": model, "model_source": source}))
PYEOF
)

    if [ "$status" = rebuilt ]; then
      measured=$($PY claims/scripts/surface_diff.py --json "$room" "$claim" 2>/dev/null)
      [ -z "$measured" ] && measured='{}'
      if ret crosscheck "claims/$claim" "$room" >/dev/null 2>&1; then
        crosschecked=true
      else
        crosschecked=false
      fi
    else
      measured='{}'
      crosschecked=false
    fi

    $PY - "$claim" "$draw" "$status" "$seconds" "$crosschecked" \
          "$measured" "$observed" >>"$LEDGER" <<'PYEOF'
import json, sys
claim, draw, status, seconds, crosschecked, measured, observed = sys.argv[1:8]
row = {"claim": claim, "draw": int(draw), "status": status,
       "seconds": int(seconds), "crosschecked": crosschecked == "true"}
for blob in (measured, observed):
    try:
        extra = json.loads(blob)
    except ValueError:
        extra = {}
    if isinstance(extra, dict):
        row.update({k: v for k, v in extra.items() if k not in ("room",)})
print(json.dumps(row, sort_keys=True))
PYEOF
    kept=$($PY -c "
import json,sys
try: print(f\"{json.loads(sys.argv[1]).get('kept','-'):.0%}\")
except Exception: print('-')" "$measured" 2>/dev/null)
    printf '%-18s draw %s  %-8s kept=%-5s %ss\n' \
      "$claim" "$draw" "$status" "${kept:--}" "$seconds"
  done
done

echo
$PY claims/scripts/variance_report.py "$LEDGER"
