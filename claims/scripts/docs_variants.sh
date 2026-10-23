#!/bin/bash
# Build a doc-pinned variant of each claim: the same criteria, plus the
# layer's own documentation as a pinned input, sealed at its own root.
#
#   claims/scripts/docs_variants.sh              # every layer
#   claims/scripts/docs_variants.sh logger       # just one
#
# WHY A SEPARATE ROOT, never an added input on the production claim (A36):
# gates decide, documentation is only read, and nothing verifies that a
# reconstruction obeyed a document. Pinning docs into the real claims would
# raise the kept-ratio while leaving verification strength untouched --
# silently changing the measurand from "what the tests determine" to "what
# tests plus docs let a producer guess" -- and would move all ten roots,
# voiding the baseline A33 calibrated. Kept separate, the DIFFERENCE between
# the two is the measurement, and that difference is what decides whether
# documenting a behaviour can stand in for asserting it.
#
# The mapping below is each layer's OWN documentation, not everything a
# contributor might read. architecture.md is the global overview and is
# deliberately excluded except for monitor, whose loop it is the only doc to
# describe. Narrow keeps each result attributable to one document.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
REPO=$PWD
RETICULI=${RETICULI_SRC:-$REPO/../reticuli/src}
PY=${RETICULI_PYTHON:-python3.13}
DEST=${DOCS_VARIANTS:-${TMPDIR:-/tmp}/apeiron-docs-variants}

docs_for() {
  case "$1" in
    config)          echo "docs/configurations.md" ;;
    evaluation)      echo "docs/api/evaluation.md" ;;
    profilers)       echo "docs/profiler.md" ;;
    model_harness)   echo "docs/model_harness.md" ;;
    updaters)        echo "docs/continuous_learning.md" ;;
    logger)          echo "docs/tracking.md" ;;
    drift_detection) echo "docs/drift_detectors.md" ;;
    trainer)         echo "docs/continuous_learning.md" ;;
    monitor)         echo "docs/architecture.md" ;;
    entrypoints)     echo "docs/quickstart.md" ;;
    *)               echo "" ;;
  esac
}

ALL=(config evaluation profilers model_harness updaters logger
     drift_detection trainer monitor entrypoints)
LAYERS=("$@")
[ ${#LAYERS[@]} -eq 0 ] && LAYERS=("${ALL[@]}")

mkdir -p "$DEST"
ret() { PYTHONPATH=$RETICULI $PY -m reticuli "$@"; }

for layer in "${LAYERS[@]}"; do
  docs=$(docs_for "$layer")
  if [ -z "$docs" ]; then
    printf '%-18s %s\n' "$layer" "no documentation mapped -- skipped"
    continue
  fi
  claim="claims/$layer"
  room="$DEST/$layer-docs"
  rm -rf "$room"
  cp -R "$claim" "$room" || { echo "$layer: copy failed" >&2; continue; }

  names=""
  for doc in $docs; do
    cp "$REPO/$doc" "$room/$(basename "$doc")" || continue
    names="$names $(basename "$doc")"
  done

  # declare the docs as pinned inputs, and say in the recipe why this root
  # exists so the variant explains itself to whoever finds it
  $PY - "$room" "$layer" $names <<'PYEOF'
import pathlib, sys
room, layer, docs = sys.argv[1], sys.argv[2], sys.argv[3:]
p = pathlib.Path(room) / "reticuli.toml"
s = p.read_text()
# the recipes write `inputs` both ways -- a multi-line list and a single
# line -- so handle both rather than only the shape the first layer used
if "inputs = [\n" in s:
    added = "".join(f'    "{d}",\n' for d in docs)
    s = s.replace("inputs = [\n", "inputs = [\n" + added, 1)
elif "inputs = [" in s:
    added = "".join(f'"{d}", ' for d in docs)
    s = s.replace("inputs = [", "inputs = [" + added, 1)
else:
    sys.exit(f"{layer}: no inputs list to extend")
banner = (
    f"# EXPERIMENT (A36): the {layer} layer with its own documentation pinned\n"
    f"# as an input ({', '.join(docs)}), so the producer reads it alongside\n"
    "# the tests. The difference against the tests-only claim is what the\n"
    "# DOCUMENTATION contributes. A separate root on purpose: pinning docs\n"
    "# into the production claim would raise the kept-ratio without\n"
    "# strengthening any gate, since nothing verifies a reconstruction\n"
    "# obeyed a document.\n#\n"
)
p.write_text(banner + s)
PYEOF
  [ $? -ne 0 ] && continue

  out=$(ret pack "$room" 2>&1 | tail -1)
  printf '%-18s %-26s %s\n' "$layer" "$(echo $names)" "$out"
done
