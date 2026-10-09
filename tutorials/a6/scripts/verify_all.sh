#!/usr/bin/env bash
# Instructor check: run every lab end to end, time it, and write a report.
# Run from the Apeiron repo root:   bash tutorials/a6/scripts/verify_all.sh
# Takes roughly the length of all labs combined; results land in output/a6/verify/.
export A6_OUT=${A6_OUT:-output/a6/verify}
source "$(dirname "$0")/_common.sh"
REPORT="$OUT/report.md"
T0=$SECONDS

{
  echo "# A6 verification report"
  echo
  echo "- date: $(date)"
  echo "- host: $(uname -srm)"
  echo "- python: $("$PY" -c 'import sys; print(sys.version.split()[0])')"
  echo "- torch: $("$PY" -c 'import torch; print(torch.__version__, "| cuda:", torch.cuda.is_available(), "| mps:", torch.backends.mps.is_available())')"
  echo "- apeiron commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$REPORT"

# Lab 1
run lab1 src.main
# Lab 2 and 3 (full grid)
A6_OUT="$OUT" bash "$A6/scripts/run_lab2.sh" | tee "$OUT/lab2_summary.txt"
A6_OUT="$OUT" bash "$A6/scripts/run_lab3.sh" --all | tee "$OUT/lab3_summary.txt"
# Lab 2 stretch: custom detector through the plug-in driver
echo ">>> lab2_threshold (custom detector)"
start=$SECONDS
"$PY" "$A6/scripts/a6_run.py" --config "$BASE" \
  --detector "$A6/scripts/custom_detector.py:ThresholdDetector" \
  --set logging.metrics_output_path="$OUT/lab2_threshold.csv" > "$OUT/lab2_threshold.log" 2>&1 \
  && echo $((SECONDS - start)) > "$OUT/lab2_threshold.seconds" || FAILED+=(lab2_threshold)
# Lab 4: harness template
echo ">>> lab4_template"
start=$SECONDS
"$PY" "$A6/scripts/a6_run.py" --config "$A6/configs/a6_custom.toml" \
  --harness "$A6/scripts/harness_template.py:DriftingRegressionHarness" \
  --set logging.metrics_output_path="$OUT/lab4_template.csv" > "$OUT/lab4_template.log" 2>&1 \
  && echo $((SECONDS - start)) > "$OUT/lab4_template.seconds" || FAILED+=(lab4_template)

"$PY" "$A6/scripts/plot_run.py" "$OUT/lab1.csv" --out "$OUT/lab1.png" || true
"$PY" "$A6/scripts/plot_run.py" "$OUT/lab4_template.csv" --out "$OUT/lab4_template.png" || true

{
  echo
  echo "## All runs"
  echo
  echo '```'
  "$PY" "$A6/scripts/compare_runs.py" "$OUT"/*.csv
  echo '```'
  echo
  echo "Total wall time: $((SECONDS - T0)) s"
  echo
  echo "Failed runs: ${FAILED[*]:-none}"
} >> "$REPORT"
echo
echo "Report: $REPORT   (send this file plus $OUT/*.png back for the docs update)"
