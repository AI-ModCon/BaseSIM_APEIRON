# Choosing a Drift Detector

{doc}`drift_detectors` is the authoritative reference for what each detector
does and every option it takes. This page is the decision guide: which one to
pick, and how to scale its knobs to your metric.

## Start here: what is actually plug-and-play

`ContinuousMonitor._check_drift()` calls `detector.update(agg_metric)` with a
**single aggregated scalar**. Only three detectors work with that signature out
of the box:

- `ADWINDetector`
- `KSWINDetector`
- `PageHinkleyDetector`

The others need extra wiring and should not be treated as defaults:

| Detector | Why it is not drop-in |
| --- | --- |
| `ModelPerformanceDetector` | Needs reference data plus batch `DataFrame`s, which the monitor does not pass. |
| `EvalDetector` (`ModelEvalDetector`) | Needs `modelHarness`, `reference_validation_metrics`, and `higher_is_better` kwargs the monitor does not send. |
| `EnsembleDetector` | `load_drift_detector` raises `NotImplementedError`; sub-detector config is not wired up. |

```{important}
The three scalar detectors fire on **change in either direction** — they do not
know "better" from "worse". If you only care about degradation, that is what
`EvalDetector` is for, and it requires the extra wiring described in
{doc}`drift_detectors`.
```

## Pick a detector

| Your situation | Use | Why |
| --- | --- | --- |
| Distribution / variance / shape changes with little mean movement | **KSWIN** | A two-sample KS test compares full distributions, not just means. |
| Abrupt mean shifts; you want fast and cheap detection | **PageHinkley** | A cumulative-sum test, low memory, quick to react. |
| Gradual drift, mixed drift, or "not sure — give me a sane default" | **ADWIN** | Adaptive windowing handles gradual *and* abrupt shifts with no fixed window size. |

## Scale the knobs to your metric

### ADWIN

`adwin_delta` is the main sensitivity knob.

- Lower (`~0.001`) — stricter test, fewer and later detections, fewer false alarms.
- Higher (`~0.01`) — more sensitive.
- Default `0.002`; typical range `0.001 – 0.01`.

Leave `adwin_minor_threshold` / `adwin_moderate_threshold` at `0.3` / `0.6`
unless you want to steer the regime split (`continual_learning` →
`fine_tuning` → `retrain`).

### KSWIN

- `kswin_alpha` — KS significance level; lower is stricter. Default `0.005`.
- `kswin_window_size` — total retained samples; the reference window is
  `window_size - stat_size`. Larger is a more stable baseline but slower to
  adapt. Default `100`.
- `kswin_stat_size` — most-recent samples tested against the reference; must be
  `< window_size`. Default `30`.

### Page-Hinkley

```{warning}
`ph_threshold` scale **depends on your metric**. For a bounded metric in
`[0, 1]` (accuracy, error rate) the default of `50` is enormous and will
essentially never fire — start around `1 – 10` and tune. For larger-magnitude
losses, larger thresholds are appropriate.
```

- `ph_min_instances` — warm-up samples before detection can fire. Default `30`.
- `ph_delta` — slack per deviation, i.e. the smallest change treated as real.
  Higher ignores small fluctuations. Default `0.005`.
- `ph_alpha` — forgetting factor for the running mean; closer to `1` weights
  history more. Default `0.9999`.

## Set the cadence

These keys are detector-independent and control *how often* the detector sees a
value:

- `detection_interval` — check drift every N monitored batches. `<= 0` disables
  checks entirely (and therefore disables CL dispatch).
- `aggregation` — `mean`, `median`, or `last` over the buffered values.
- `metric_index` — index into the harness's `eval_metrics` ordering.
- `max_stream_updates` — stop monitoring after this many stream extensions.

Warm-up matters: Page-Hinkley needs `ph_min_instances` updates and KSWIN needs
`kswin_window_size` samples before detection is meaningful. With
`detection_interval = 10`, KSWIN's default window of `100` means 1000 monitored
batches before the reference window is full.

## Paste-ready blocks

::::{tab-set}

:::{tab-item} ADWIN (general default)
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
:::

:::{tab-item} KSWIN (distribution shift)
```toml
[drift_detection]
detector_name = "KSWINDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

kswin_alpha = 0.005
kswin_window_size = 100
kswin_stat_size = 30
```
:::

:::{tab-item} Page-Hinkley (abrupt mean shift)
```toml
[drift_detection]
detector_name = "PageHinkleyDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

ph_min_instances = 30
ph_delta = 0.005
# Bounded metric in [0, 1]: start small, not at the default 50.
ph_threshold = 5
ph_alpha = 0.9999
```
:::

::::

## Validate before a full run

Build the config and instantiate the detector to confirm the TOML parses and the
detector name is accepted — no training required:

```bash
poetry run python -c "
from apeiron import build_config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
cfg = build_config(['--config', 'examples/mnist/mnist.toml'])
d = load_drift_detector(cfg)
print(type(d).__name__, d.update(0.5))
"
```

```{seealso}
The repository ships a `choose-detector` agent skill that runs this decision
process interactively and patches your config file. See {doc}`agent_skills`.
```
