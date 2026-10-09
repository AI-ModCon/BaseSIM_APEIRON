#!/usr/bin/env bash
# Lab 2: compare drift detectors on the same stream, with the model frozen (detect-only).
# Run from the Apeiron repo root:   bash tutorials/a6/scripts/run_lab2.sh
source "$(dirname "$0")/_common.sh"

D=drift_detection
run lab2_adwin        src.drift_only --set $D.detector_name=ADWINDetector
run lab2_adwin_touchy src.drift_only --set $D.detector_name=ADWINDetector --set $D.adwin_delta=0.1
run lab2_kswin        src.drift_only --set $D.detector_name=KSWINDetector \
                        --set $D.kswin_window_size=60 --set $D.kswin_stat_size=20 --set $D.kswin_seed=1
run lab2_pagehinkley  src.drift_only --set $D.detector_name=PageHinkleyDetector
run lab2_ensemble_any src.drift_only --set $D.detector_name=EnsembleDetector \
                        --set "$D.ensemble_detectors=[\"ADWINDetector\",\"KSWINDetector\",\"PageHinkleyDetector\"]" \
                        --set $D.ensemble_voting=any \
                        --set $D.kswin_window_size=60 --set $D.kswin_stat_size=20 --set $D.kswin_seed=1
run lab2_ensemble_majority src.drift_only --set $D.detector_name=EnsembleDetector \
                        --set "$D.ensemble_detectors=[\"ADWINDetector\",\"KSWINDetector\",\"PageHinkleyDetector\"]" \
                        --set $D.ensemble_voting=majority \
                        --set $D.kswin_window_size=60 --set $D.kswin_stat_size=20 --set $D.kswin_seed=1

"$PY" "$A6/scripts/compare_runs.py" "$OUT"/lab2_*.csv
"$PY" "$A6/scripts/plot_run.py" "$OUT"/lab2_*.csv --out "$OUT/lab2.png"
