#!/bin/bash
# Draw repeatedly from TWO generations of one claim and compare the results.
#
#   claims/scripts/paired.sh <layer> <claim-before> <claim-after> [draws]
#
# The experiment the whole method rests on, and the one the campaign never
# ran. Everything else here measures a single generation of criteria; this
# measures the DIFFERENCE between two, with the producer pinned and the
# implementation under test held constant. If strengthening a test suite
# really does change what an independent reconstruction is forced to build,
# the kept-ratio moves by more than the noise floor A33 measured. If it does
# not, the central claim is an anecdote and should be withdrawn.
#
# A "generation" is any directory holding the claim at a root: the live claim
# under claims/<layer>, or an old rebuild room, which carries the recipe and
# the pinned inputs it was built from and so still verifies at its own root.
#
#   claims/scripts/paired.sh config /tmp/m3-config claims/config 3
#
# Validity conditions, which the caller must check and this script cannot:
#   * src/ unchanged between the two roots -- otherwise the ratio moves
#     because the target moved, not because the criteria did
#   * the same producer and model for every draw
#   * independent rooms, no retries folded into the count
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
REPO=$PWD
RETICULI=${RETICULI_SRC:-$REPO/../reticuli/src}
PY=${RETICULI_PYTHON:-python3.13}
ROOMS=${PAIRED_ROOMS:-${TMPDIR:-/tmp}/apeiron-paired}
MODEL=${RETICULI_MODEL:-gpt-6-astra}
PRODUCER=${PAIRED_PRODUCER:-"env PYTHONPATH=$RETICULI RETICULI_MODEL=$MODEL python3 -m reticuli.producers.codex"}
OUT_DIR=$REPO/claims/measurements

LAYER=${1:-}
BEFORE=${2:-}
AFTER=${3:-}
DRAWS=${4:-3}
if [ -z "$LAYER" ] || [ -z "$BEFORE" ] || [ -z "$AFTER" ]; then
  sed -n '2,12p' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LEDGER=$OUT_DIR/paired-$LAYER-$STAMP.jsonl
mkdir -p "$ROOMS" "$OUT_DIR"
export RETICULI_MODEL=$MODEL          # the kernel reads its own environment
ret() { PYTHONPATH=$RETICULI $PY -m reticuli "$@"; }

root_of() {
  $PY - "$1" <<'PYEOF'
import json, os, sys
try:
    print(json.load(open(os.path.join(sys.argv[1], ".reticuli", "manifest.json")))["root"])
except Exception:
    print("")
PYEOF
}

echo "# paired: $LAYER, $DRAWS draw(s) per generation, model=$MODEL"
echo "# before: $BEFORE  root $(root_of "$BEFORE" | cut -c1-12)"
echo "# after:  $AFTER  root $(root_of "$AFTER" | cut -c1-12)"
echo "# ledger: $LEDGER"

for side in before after; do
  claim=$BEFORE
  [ "$side" = after ] && claim=$AFTER
  root=$(root_of "$claim")
  for draw in $(seq 1 "$DRAWS"); do
    room="$ROOMS/$LAYER-$side-$draw"
    log="$room.log"
    rm -rf "$room"
    started=$(date -u +%s)
    if ret rebuild "$claim" --producer "$PRODUCER" -o "$room" >"$log" 2>&1; then
      status=rebuilt
    elif grep -qi "usage limit\|rate limit\|quota" "$log"; then
      status=quota        # the producer refused; the criteria were not tested
    else
      status=failed
    fi
    seconds=$(( $(date -u +%s) - started ))

    measured='{}'
    if [ "$status" = rebuilt ]; then
      measured=$($PY claims/scripts/surface_diff.py --json "$room" "$LAYER" 2>/dev/null)
      [ -z "$measured" ] && measured='{}'
    fi
    observed=$($PY - "$room" <<'PYEOF'
import json, os, sys
path = os.path.join(sys.argv[1], ".reticuli", "ledger.jsonl")
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
    $PY - "$LAYER" "$side" "$root" "$draw" "$status" "$seconds" \
          "$measured" "$observed" >>"$LEDGER" <<'PYEOF'
import json, sys
layer, side, root, draw, status, seconds, measured, observed = sys.argv[1:9]
row = {"claim": layer, "side": side, "claim_root": root, "draw": int(draw),
       "status": status, "seconds": int(seconds)}
for blob in (measured, observed):
    try:
        extra = json.loads(blob)
    except ValueError:
        extra = {}
    if isinstance(extra, dict):
        row.update({k: v for k, v in extra.items() if k != "room"})
print(json.dumps(row, sort_keys=True))
PYEOF
    kept=$($PY -c "
import json,sys
try: print(f\"{json.loads(sys.argv[1])['kept']:.0%}\")
except Exception: print('-')" "$measured" 2>/dev/null)
    printf '%-8s draw %s  %-8s kept=%-5s %ss\n' "$side" "$draw" "$status" "${kept:--}" "$seconds"
  done
done

echo
$PY claims/scripts/paired_report.py "$LEDGER"
