#!/usr/bin/env bash
# Lab 3: compare adaptation strategies. Core set by default; add --all for the full grid.
# Run from the Apeiron repo root:   bash tutorials/a6/scripts/run_lab3.sh [--all]
source "$(dirname "$0")/_common.sh"

CL=continual_learning
# Core: no adaptation vs. naive fine-tuning vs. EWC with replay
run lab3_never          src.cl_only --schedule never
run lab3_base_noreplay  src.main    --set $CL.update_mode=base       --set $CL.mix_historic_data=false
run lab3_ewc_replay     src.main    --set $CL.update_mode=ewc_online --set $CL.mix_historic_data=true

if [[ "${1:-}" == "--all" ]]; then
  run lab3_base_replay    src.main    --set $CL.update_mode=base        --set $CL.mix_historic_data=true
  run lab3_ewc_noreplay   src.main    --set $CL.update_mode=ewc_online  --set $CL.mix_historic_data=false
  run lab3_kfac_replay    src.main    --set $CL.update_mode=kfac_online --set $CL.mix_historic_data=true
  run lab3_jvp            src.main    --set $CL.update_mode=jvp_reg
  # Is the detector worth it? Same updater, triggered on a clock instead of by drift.
  run lab3_every_window   src.cl_only --schedule periodic --period 1 \
                            --set $CL.update_mode=ewc_online --set $CL.mix_historic_data=true
  run lab3_every_3rd      src.cl_only --schedule periodic --period 3 \
                            --set $CL.update_mode=ewc_online --set $CL.mix_historic_data=true
fi

"$PY" "$A6/scripts/compare_runs.py" "$OUT"/lab3_*.csv
"$PY" "$A6/scripts/plot_run.py" "$OUT"/lab3_*.csv --out "$OUT/lab3.png"
