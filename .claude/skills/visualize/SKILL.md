---
name: visualize
description: |
  Visualize experiment results from a BaseSim run. Use when the user wants to
  see metrics dashboards, accuracy curves, drift detection events, or loss
  plots from a completed experiment. Generates PNG dashboards from CSV metrics.
argument-hint: "<config_path>"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

Visualize results from a BaseSim continual learning experiment.

## Arguments
- `$0`: Path to a TOML config file that contains a `[visualization]` section specifying the input CSV and output PNG paths.

## What Gets Generated
The visualization module creates a 6-panel dashboard PNG showing:
1. **Test Accuracy** over time with baseline threshold
2. **Historical Test Accuracy** (forgetting measure)
3. **Loss Metrics** (generation loss, forgetting loss)
4. **Computational Performance** (FLOPs per operation)
5. **Throughput** (TFLOP/s)
6. **Execution Time** analysis

Drift events are marked as red vertical lines on all panels.

## Procedure

1. **Validate config.** Read the TOML file and extract the `[visualization]` section:
   - `input`: path to CSV metrics file (generated during experiment)
   - `output`: path for the output dashboard PNG
   - `baseline`: accuracy threshold for the dashboard

2. **Check input CSV exists:**
   ```bash
   ls -la <input_csv_path>
   ```
   If not found, inform the user they need to run an experiment first. Suggest:
   ```
   /run-experiment <config_path>
   ```

3. **Preview the CSV data** to confirm it has valid content:
   ```bash
   head -20 <input_csv_path>
   wc -l <input_csv_path>
   ```

4. **Run the visualizer:**
   ```bash
   poetry run python -m src.visualize --config $0
   ```

5. **Report results:**
   - Confirm the dashboard PNG was generated at the configured output path
   - Summarize key metrics from the CSV:
     - Number of drift events
     - Accuracy range (min, max, final)
     - Number of CL training events triggered
     - Total evaluation steps
