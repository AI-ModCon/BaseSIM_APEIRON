# Lab 2: Choosing a drift detector

**18 minutes.** You compare detectors on exactly the same stream, then tune one.

## Theory in two minutes

A drift detector turns a stream of numbers into "has something changed?" In Apeiron that number is a **model-performance signal**: the aggregated accuracy, error or loss that the monitor feeds it every `detection_interval` batches. Detectors differ in what kind of change they look for:

| Detector | Looks for | Knobs | Good when |
|---|---|---|---|
| **ADWIN** (adaptive windowing) | a change in the **mean**: it keeps a variable-length window and splits it when the two halves differ by more than chance allows | `adwin_delta`: smaller = stricter, fewer false alarms | gradual or step changes in average performance |
| **KSWIN** (Kolmogorov–Smirnov windowing) | a change in the **distribution** of recent values vs. a reference window | `kswin_alpha`, `kswin_window_size`, `kswin_stat_size` | spikier, heavy-tailed errors where the mean alone misleads |
| **Page-Hinkley** | a sustained **cumulative deviation** from the running mean | `ph_threshold` (in the metric's own units), `ph_delta`, `ph_min_instances` | abrupt shifts you want caught fast |
| **EnsembleDetector** | runs several of the above and votes | `ensemble_detectors`, `ensemble_voting` = `any` / `majority` / `unanimous` | combining sensitivity and robustness |

Every detector trades **missed drift** (the model is wrong and nobody knows) against **false alarms** (you pay for adaptations you did not need). The right balance depends on what a wrong prediction costs you compared with an unneeded update.

Three practical details that trip people up:

- **Warm-up.** KSWIN needs `kswin_window_size` observations before it can fire at all, and Page-Hinkley needs `ph_min_instances`. With 10 000 stream images, 32 per batch and a check every 10 batches, one window gives about 31 checks. That is why this lab lowers KSWIN's window from 100 to 60.
- **Units.** Page-Hinkley's `ph_threshold` is in the units of your metric. The default of 50 suits accuracy in percent; an MAE around 0.1 needs something closer to 1.
- **Cadence.** ADWIN (the `river` implementation) only tests for a split every 32 observations. Very few checks per window means slow reactions.

## 1. Run the sweep, detect-only (≈ [[MEASURE: run_lab2.sh total time]])

To compare detectors fairly, keep the model frozen so every detector sees the identical stream. `src.drift_only` does exactly that: it monitors and records detections, but never adapts.

```bash
bash tutorials/a6/scripts/run_lab2.sh
```

It runs six configurations, prints a summary table, and saves `output/a6/lab2.png`:

| Run | Setting |
|---|---|
| `lab2_adwin` | ADWIN, default `delta = 0.002` |
| `lab2_adwin_touchy` | ADWIN, `delta = 0.1` (much more sensitive) |
| `lab2_kswin` | KSWIN, window 60 |
| `lab2_pagehinkley` | Page-Hinkley, defaults |
| `lab2_ensemble_any` | ADWIN + KSWIN + Page-Hinkley, fire if **any** fires |
| `lab2_ensemble_majority` | the same three, fire if **most** fire |

Each single run is an ordinary command you can copy, for example:

```bash
python -m src.drift_only --config tutorials/a6/configs/a6_mnist.toml \
    --set drift_detection.detector_name=PageHinkleyDetector \
    --set logging.metrics_output_path=output/a6/lab2_pagehinkley.csv
```

Reference result: [[MEASURE: lab2 summary table from verify report]].

## 2. Read the results (5 min)

- Which detector fires **first** after the stream starts to degrade? Which fires **most often**?
- `touchy` vs. default ADWIN: how many extra detections did the more sensitive setting buy, and do they line up with real window changes?
- `any` vs. `majority`: how do the voting rules change the count?
- For your application, which is worse: a late detection or an unneeded adaptation?

## 3. Tune one (5 min)

Pick a detector and change one knob at a time. Some useful ones:

```bash
--set drift_detection.adwin_delta=0.01          # sensitivity
--set drift_detection.detection_interval=5      # check twice as often (noisier inputs)
--set drift_detection.aggregation=median        # robust to a few terrible batches
--set drift_detection.metric_index=1            # watch the loss instead of accuracy
```

Remember to give each run its own `--set logging.metrics_output_path=output/a6/lab2_<name>.csv`, then compare:

```bash
python tutorials/a6/scripts/compare_runs.py output/a6/lab2_*.csv
```

## Stretch: write your own detector

`tutorials/a6/scripts/custom_detector.py` is a 60-line detector: learn a baseline, fire after a few consecutive checks that are more than *k* standard deviations worse. A detector only needs `update(value) -> DriftSignal` and `reset()`. Run it inside the full adapt loop with the plug-in driver:

```bash
python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml \
    --detector tutorials/a6/scripts/custom_detector.py:ThresholdDetector \
    --set logging.metrics_output_path=output/a6/lab2_threshold.csv
```

Then change the rule to something that fits your domain: a fixed tolerance from your requirements, a physics-based residual check, a rate-of-change limit.

To pick a detector for your own signal with help, the `choose-detector` agent skill asks about your metric and expected drift, then writes a validated `[drift_detection]` block.
