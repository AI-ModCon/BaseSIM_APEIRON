# Shared helpers for the A6 sweep scripts. Sourced, not run.
set -euo pipefail
PY=${PYTHON:-python}
A6=${A6_DIR:-tutorials/a6}
OUT=${A6_OUT:-output/a6}
BASE="$A6/configs/a6_mnist.toml"

if [[ ! -d src || ! -d examples ]]; then
  echo "Run this from the Apeiron repo root (the folder with src/ and examples/)." >&2
  exit 1
fi
mkdir -p "$OUT"
FAILED=()

# run NAME MODULE [args...]  -> writes $OUT/NAME.csv and $OUT/NAME.seconds, log in $OUT/NAME.log
run() {
  local name=$1 module=$2; shift 2
  echo ">>> $name"
  local start=$SECONDS
  if ! "$PY" -m "$module" --config "$BASE" \
        --set logging.metrics_output_path="$OUT/$name.csv" "$@" > "$OUT/$name.log" 2>&1; then
    echo "    FAILED -- see $OUT/$name.log (last lines below)"; tail -n 15 "$OUT/$name.log"; FAILED+=("$name"); return 0
  fi
  echo $((SECONDS - start)) > "$OUT/$name.seconds"
  echo "    done in $((SECONDS - start)) s"
}
