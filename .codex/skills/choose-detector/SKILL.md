---
name: choose-detector
description: Help the user pick a drift detector and tune its settings for an apeiron run. Use when the user asks which detector to use, how to configure drift detection, what ADWIN/KSWIN/PageHinkley/threshold values to set, or wants a ready-to-use [drift_detection] config block. Asks a few questions about the monitored metric and drift shape, recommends a detector, writes a filled-in TOML block, and validates that it loads. Does not run a full experiment; for that use explore-examples or custom-experiment.
metadata:
  short-description: Recommend and configure a drift detector
---

# Choose Detector

Help the user choose a drift detector and produce a validated `[drift_detection]`
config block. The authoritative reference is `docs/drift_detectors.md`; read it
first and stay consistent with it rather than restating values that may change.

## Inputs

- Optional config path: if provided, patch that file's `[drift_detection]`
  section in place. If omitted, emit a standalone block the user can paste in.

## Ground Truth

Only three detectors are drop-in for the current `ContinuousMonitor` flow, because
`_check_drift()` calls `detector.update(agg_metric)` with a single aggregated
scalar:

- `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector`.

The others are not drop-in and must not be offered as defaults:

- `ModelPerformanceDetector` needs reference data and batch DataFrames.
- `EvalDetector` (`ModelEvalDetector`) needs extra `update(...)` kwargs the monitor does not send.
- `EnsembleDetector` raises `NotImplementedError` in the loader.

Confirm before relying on this:

```bash
sed -n '1,80p' src/apeiron/drift_detection/load_drift_detector.py
```

## Procedure

### 1. Understand the Signal

Ask the user:

- Which metric feeds the detector, and its scale (bounded like accuracy/error in `[0, 1]`, or unbounded like a loss)?
- Expected drift shape: abrupt mean jumps, gradual drift, or distribution/variance change with little mean movement.
- Sensitivity preference: react early and tolerate false alarms, or fire only on clear sustained drift.
- Cadence: how many checks before a first detection is wanted, and how many batches per check.

Note that the scalar detectors fire on change in either direction; they do not
distinguish good from bad. If the user only cares about degradation, say so
plainly (that is the non-wired `EvalDetector`'s job).

### 2. Recommend a Detector

- distribution/variance/shape change without mean movement: KSWIN
- abrupt mean shift, fast and cheap: PageHinkley
- gradual, mixed, or general default: ADWIN

State the choice and a one-line reason; name the runner-up if it is close.

### 3. Recommend Settings

Pull defaults and semantics from `docs/drift_detectors.md` and scale to the metric:

- ADWIN: `adwin_delta` is the main sensitivity knob; keep the two thresholds at defaults unless steering the regime split.
- KSWIN: `kswin_alpha` for sensitivity; size the windows to retained samples (`stat_size < window_size`).
- PageHinkley: `ph_threshold` scale depends on the metric. For a bounded metric in `[0,1]` the default `50` is very large and rarely fires, so start around `1-10` and tune; larger losses need larger thresholds. `ph_delta` is the slack, `ph_min_instances` is warm-up.

Set `detection_interval`, `aggregation`, `metric_index`, and `max_stream_updates`
from the cadence answers, and explain any non-default value.

### 4. Produce the Config Block

Emit a complete, paste-ready section:

```toml
[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

adwin_delta = 0.002
adwin_minor_threshold = 0.3
adwin_moderate_threshold = 0.6
```

If a config path was given, patch its `[drift_detection]` section, matching the
existing file style.

### 5. Validate (no full run)

Build the config and instantiate the detector to confirm the TOML parses and the
detector accepts the params:

```bash
PYTHONPATH=src poetry run python -c "
from apeiron.config.configuration import build_config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
cfg = build_config(['--config', '<config_path>'])
d = load_drift_detector(cfg)
print('OK:', type(d).__name__)
print(cfg.drift_detection)
"
```
(`PYTHONPATH=src` is required so `import apeiron` resolves; the package lives
under `src/apeiron`.)

For a standalone block, validate against an example TOML with `--set` overrides
(a full `Config` still needs `[model]`/`[data]`/`[train]`). Report the recommended
detector, the reason, the non-default settings, and that the config loaded.

## Useful Commands

A/B a detector on a shipped example without editing files:

```bash
poetry run python -m src.main --config examples/mnist/mnist.toml \
  --set drift_detection.detector_name=PageHinkleyDetector \
  --set drift_detection.ph_threshold=5
```

Keep `docs/drift_detectors.md` as the single source of truth; if the doc and this
skill disagree, fix the doc and follow it.
