---
name: choose-detector
description: Help the user pick a drift detector and tune its settings for an apeiron run. Use when the user asks which detector to use, how to configure drift detection, what ADWIN/KSWIN/PageHinkley/threshold values to set, whether to combine detectors into an ensemble and which voting rule to use, or wants a ready-to-use [drift_detection] config block. Asks a few questions about the monitored metric and drift shape, recommends a detector, writes a filled-in TOML block, and validates that it loads. Does not run a full experiment; for that use explore-examples or custom-experiment.
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

`ContinuousMonitor._check_drift()` calls `detector.update(agg_metric)` with a
single aggregated scalar and no kwargs. Anything needing more than that scalar is
not drop-in.

Drop-in:

- `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` take the scalar directly.
- `EnsembleDetector` forwards the scalar to each sub-detector via
  `update(value, **kwargs)`, so it is drop-in provided every name in
  `ensemble_detectors` is one of the three scalar detectors above. Naming a
  non-drop-in detector as a sub-detector just moves the failure into the ensemble.

Not drop-in, and must not be offered as defaults:

- `ModelPerformanceDetector` needs reference data and batch DataFrames; `update()`
  raises `ValueError` unless `set_reference()` ran first.
- `EvalDetector` (`ModelEvalDetector`) needs extra `update(...)` kwargs the monitor
  does not send (`modelHarness`, `reference_validation_metrics`, `higher_is_better`).

Confirm before relying on this:

```bash
sed -n '1,100p' src/apeiron/drift_detection/load_drift_detector.py
```

## Procedure

### 1. Understand the Signal

Ask the user:

- Which metric feeds the detector, and its scale (bounded like accuracy/error in `[0, 1]`, or unbounded like a loss)?
- Expected drift shape: abrupt mean jumps, gradual drift, or distribution/variance change with little mean movement.
- Sensitivity preference: react early and tolerate false alarms, or fire only on clear sustained drift. A strong preference either way, or "I expect more than one kind of drift", is the cue to consider an ensemble.
- Cadence: how many checks before a first detection is wanted, and how many batches per check.

Note that the scalar detectors fire on change in either direction; they do not
distinguish good from bad. If the user only cares about degradation, say so
plainly (that is the non-wired `EvalDetector`'s job).

### 2. Recommend a Detector

- distribution/variance/shape change without mean movement: KSWIN
- abrupt mean shift, fast and cheap: PageHinkley
- gradual, mixed, or general default: ADWIN
- more than one drift shape expected, or a sensitivity preference a single detector cannot express: Ensemble over two or three of the above

Prefer a single detector when one clearly fits. The ensemble costs an update on
every sub-detector per check and is harder to tune, since each sub-detector is
still driven by its own hyperparameters in the same config block. Reach for it
when the shapes genuinely differ (PageHinkley for abrupt jumps plus KSWIN for
variance changes) or when the user wants a deliberate bias they can state as a
voting rule.

State the choice and a one-line reason; name the runner-up if it is close.

### 3. Recommend Settings

Pull defaults and semantics from `docs/drift_detectors.md` and scale to the metric:

- ADWIN: `adwin_delta` is the main sensitivity knob; keep the two thresholds at defaults unless steering the regime split.
- KSWIN: `kswin_alpha` for sensitivity; size the windows to retained samples (`stat_size < window_size`).
- PageHinkley: `ph_threshold` scale depends on the metric. For a bounded metric in `[0,1]` the default `50` is very large and rarely fires, so start around `1-10` and tune; larger losses need larger thresholds. `ph_delta` is the slack, `ph_min_instances` is warm-up.
- Ensemble: `ensemble_detectors` lists the sub-detector names. Each is built from this same `[drift_detection]` block, so a detector type can appear at most once and still needs its own hyperparameters set here. An empty list, a nested `"EnsembleDetector"`, or an unknown voting name raises `ValueError` at load. `ensemble_voting` sets the bias:
  - `any` (alias `or`) fires when any sub-detector fires. Most sensitive; use when a missed drift costs more than a needless CL dispatch.
  - `majority` (default) needs strictly more than half. Balanced, but needs 3+ detectors to differ from `unanimous`.
  - `unanimous` (aliases `all`, `and`) needs every detector to fire. Most conservative; suppresses small or noisy changes at the cost of latency.

  Note that `drift_score` is the mean of the sub-detector scores and the regime is
  a plurality vote, both independent of the voting rule, so the
  `adwin_minor_threshold` / `adwin_moderate_threshold` regime split gets diluted by
  sub-detectors reporting a score of 0.

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

For an ensemble, list the sub-detectors and keep each one's hyperparameters in the
same block:

```toml
[drift_detection]
detector_name = "EnsembleDetector"
ensemble_detectors = ["ADWINDetector", "PageHinkleyDetector"]
ensemble_voting = "unanimous"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

# Sub-detectors read their usual hyperparameters from this same block
adwin_delta = 0.002
ph_threshold = 30
ph_delta = 0.5
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
print(getattr(d, 'voting', ''), [type(s).__name__ for s in getattr(d, 'detectors', [])])
print(cfg.drift_detection)
"
```
(`PYTHONPATH=src` is required so `import apeiron` resolves; the package lives
under `src/apeiron`.)

For an ensemble this is worth more than a syntax check: it is where an empty
`ensemble_detectors`, a nested `EnsembleDetector`, an unknown voting name, or an
unknown sub-detector name surfaces as a `ValueError` instead of at run time. The
second print confirms the resolved voting rule and that every sub-detector built.

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

`--set` values go through `json.loads`, so a list needs JSON syntax and shell
quoting:

```bash
--set 'drift_detection.ensemble_detectors=["ADWINDetector","KSWINDetector"]'
```

Precedence when sources disagree: the code in `src/apeiron/drift_detection/` wins,
then `docs/drift_detectors.md`, then this skill. Fix whichever is stale rather than
working around it; this file has been wrong about detector wiring before.
