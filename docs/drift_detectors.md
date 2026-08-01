# Drift Detectors

This document describes detector classes under `src/apeiron/drift_detection/`, the `drift_detection` config section, and how detector outputs drive continual learning.

## Core Types

Defined in `src/apeiron/drift_detection/detectors/base.py`:

- `LearningRegime`:
  - `stable`
  - `continual_learning`
  - `fine_tuning`
  - `retrain`
- `DriftSignal` fields:
  - `regime`
  - `drift_detected`
  - `drift_score`
  - `confidence` (optional)
  - `metadata` (optional dictionary)
- `BaseDriftDetector` interface:
  - `update(value: float, **kwargs) -> DriftSignal`
  - `reset() -> None`

## Global Drift Config

`src/apeiron/config/configuration.py` defines `DriftDetectionCfg`:

| Key | Default | Meaning |
| --- | --- | --- |
| `detector_name` | `"ADWINDetector"` | Detector class selected by `load_drift_detector(...)`. |
| `detection_interval` | `10` | Check drift every N monitored batches. If `<= 0`, checks are disabled. |
| `aggregation` | `"mean"` | How buffered metric values are aggregated before detector update. Supported by monitor: `mean`, `median`, `last`. |
| `metric_index` | `0` | Index into `modelHarness.eval_metrics` order. |
| `reset_after_learning` | `False` | If true, detector state resets after each CL event. |
| `max_stream_updates` | `20` | Monitoring stops after this many stream extensions. |
| `adwin_delta` | `0.002` | ADWIN confidence/sensitivity parameter. |
| `adwin_minor_threshold` | `0.3` | ADWIN regime threshold (CL boundary). |
| `adwin_moderate_threshold` | `0.6` | ADWIN regime threshold (fine-tuning boundary). |
| `kswin_alpha` | `0.005` | KSWIN significance level. |
| `kswin_window_size` | `100` | KSWIN reference window size. |
| `kswin_stat_size` | `30` | KSWIN recent sample window size. |
| `ph_min_instances` | `30` | Page-Hinkley warm-up count before detection is meaningful. |
| `ph_delta` | `0.005` | Page-Hinkley change magnitude parameter. |
| `ph_threshold` | `50` | Page-Hinkley trigger threshold. |
| `ph_alpha` | `0.9999` | Page-Hinkley forgetting factor. |

## Detector Selection (`detector_name`)

`src/apeiron/drift_detection/load_drift_detector.py` currently supports:

- `ADWINDetector`
- `KSWINDetector`
- `PageHinkleyDetector`
- `ModelPerformanceDetector`
- `EvalDetector` (maps to `ModelEvalDetector`)

`EnsembleDetector` is present as a class but intentionally not wired in the loader and raises `NotImplementedError`.

## Detector Classes And Options

### `ADWINDetector` (`src/apeiron/drift_detection/detectors/statistical_detectors.py`)

Brief Explanation:

ADaptive WINdowing (ADWIN) keeps a variable-length sliding window of the monitored metric and cuts it in two whenever the means of the two sub-windows differ significantly; a cut signals drift and the window shrinks to the recent regime. **It handles both gradual and abrupt shifts, needs no fixed window size, and works well as the general-purpose default for a scalar such as loss or error rate.**

Constructor options:

- `adwin_delta`: confidence bound for a window cut. Lower = stricter test, so fewer/later detections and fewer false alarms; Higher = more sensitive. Typical range `0.001 - 0.01` (default `0.002`).
- `adwin_minor_threshold`: recent-drift-rate boundary below which drift maps to the `continual_learning` regime. Higher pushes more events into CL rather than fine-tuning. Range `0 - 1` (default `0.3`).
- `adwin_moderate_threshold`: drift-rate boundary above which drift escalates to `retrain`; between the two thresholds is `fine_tuning`. Must be `>= minor_threshold`; range `0 - 1` (default `0.6`).
- `name` (constructor-only; not exposed in config)

Behavior:

- Updates on one scalar value at each check.
- Computes `drift_score` as recent drift frequency over up to 100 updates.
- Emits regime based on `drift_detected` and score thresholds.

### `KSWINDetector`

Brief Explanation: 

Kolmogorov-Smirnov WINdowing (KSWIN) runs a two-sample KS test comparing the most recent `kswin_stat_size` samples against a reference window of `kswin_window_size` samples, firing when the two distributions differ at significance level `kswin_alpha`. As it compares full distributions rather than just means, it catches shape/variance changes an averaging detector would miss, **making it a fit for cases where the distribution shifts without the mean moving much.**

Constructor options:

- `kswin_alpha`: KS-test significance level. Lower = stricter, fewer false alarms but slower to react; Higher = more sensitive. Typical range `0.001 -0.01` (default `0.005`).
- `kswin_window_size`: total samples retained; the reference window is `window_size - stat_size`. Larger = more stable baseline but more memory and slower adaptation. Typically `50 - 300` (default `100`).
- `kswin_stat_size`: number of most-recent samples KS-tested against the reference. Larger = smoother but laggier detection. Must be `< window_size`; typically `20 - 50` (default `30`).
- `minor_threshold` (constructor default only; not in config loader call): drift-rate boundary for the `continual_learning` regime, `0 - 1` (default `0.3`).
- `moderate_threshold` (constructor default only; not in config loader call): drift-rate boundary for the `retrain` regime, `0 - 1` (default `0.6`).
- `name` (constructor-only)

Behavior:

- Uses Kolmogorov-Smirnov windowing from `river`.
- Score is recent drift frequency over up to 50 updates.

### `PageHinkleyDetector`

Brief Explanation:
 
The Page-Hinkley test accumulates the running deviation of the metric from its historical mean and fires once that cumulative sum exceeds `ph_threshold`. It is cheap, low-memory, and quick to react, **making it a good fit for real-time monitoring of abrupt mean shifts.** 

Constructor options:

- `ph_min_instances`: warm-up samples observed before detection can fire. Higher = more stable baseline but slower first detection. Typically `20 - 100` (default `30`).
- `ph_delta`: slack subtracted from each deviation, i.e. the smallest change treated as meaningful. Higher = ignores small fluctuations (fewer false alarms); Lower = more sensitive. Typically `0.001 - 0.05` (default `0.005`).
- `ph_threshold`: cumulative-sum value that triggers drift. Higher = needs a larger/longer shift, so fewer/later detections; Lower = more sensitive. Scale depends on the metric; typically `10- 100` (default `50`).
- `ph_alpha`: forgetting factor for the running mean; closer to `1` weights history more (slower to adapt), lower forgets faster. Range just below `1`, e.g. `0.99 - 0.9999` (default `0.9999`).
- `minor_threshold` (constructor default only; not in config loader call): drift-rate boundary for the `continual_learning` regime, `0 - 1` (default `0.3`).
- `moderate_threshold` (constructor default only; not in config loader call): drift-rate boundary for the `retrain` regime, `0 - 1` (default `0.6`).
- `name` (constructor-only)

Behavior:

- Online mean-shift detector from `river`.
- Score is recent drift frequency over up to 50 updates.

### `ModelPerformanceDetector` (`src/apeiron/drift_detection/detectors/model_performance_detector.py`)

Brief Explanation:

A batch-level detector built on Evidently's `DataDriftPreset` that, rather than watching a single scalar, compares a reference dataset (data, predictions, targets captured when the model was healthy) against an incoming batch and flags drift when the share of drifted columns exceeds `drift_share_threshold`. This gives a richer, feature-aware signal but requires reference data up front, so use it for distribution-level analysis across many features (see the integration note below before wiring it into the monitor).

Constructor options:

- `reference_data`: baseline feature table (`pandas.DataFrame`) captured when the model was healthy; incoming batches are compared against it.
- `reference_predictions`: baseline model predictions aligned with `reference_data`; optional, enables prediction-drift analysis.
- `reference_targets`: baseline ground-truth labels; optional, enables target/performance-drift analysis.
- `drift_share_threshold`: fraction of columns that must be flagged as drifted before overall drift fires. Higher = more tolerant (fewer alarms), lower = more sensitive. Range `0 - 1` (Evidently default `0.5`).
- `minor_threshold`: drift-score boundary for the `continual_learning` regime, `0- 1` (default `0.3`).
- `moderate_threshold`: drift-score boundary for the `retrain` regime, `0- 1` (default `0.6`).
- `name`

Behavior modes:

- Batch mode: accepts `data` (`pandas.DataFrame`) and optional `predictions`/`targets`, runs Evidently `DataDriftPreset`.
- Scalar fallback mode: if given only `value`, uses internal simple score logic.

Integration note:

- The class requires reference initialization (`set_reference(...)` or constructor reference data) before `update(...)`.
- Current loader path instantiates it without reference data, so it is not plug-and-play in `ContinuousMonitor` without additional initialization code.

### `ModelEvalDetector` (`detector_name = "EvalDetector"`)

Brief Explanation:

The most direct detector: on each update it re-evaluates the model via `modelHarness.eval()` and compares each metric against a reference value, respecting a per-metric `higher_is_better` flag, declaring drift when current performance falls short of the reference. It is a straightforward "has the model gotten worse?" check. **It is best to use when you have trustworthy reference validation metrics and want an interpretable signal.**

Constructor options:

- `name` only.

Expected `update(...)` kwargs:

- `modelHarness`: the harness whose `.eval()` is called to compute current validation metrics.
- `reference_validation_metrics`: baseline metric values to compare against; must be the same length/order as `modelHarness.eval()` output.
- `higher_is_better`: per-metric dict/flags stating direction of "good"; `True` means a drop below reference counts as drift, `False` means a rise above reference does.

Integration note:

- `ContinuousMonitor._check_drift()` currently passes only one scalar metric to `update(...)`.
- Using `EvalDetector` in the current monitor flow requires extra wiring to pass the expected kwargs.

### `EnsembleDetector`

Brief Explanation:

Wraps several sub-detectors and combines their signals under a `voting` rule (`majority`, `unanimous`, `any`, or weighted) so you can trade sensitivity against false-alarm rate: e.g. `any` reacts to the first detector that fires while `unanimous` requires full agreement. It is conceptually useful for robustness, but note it is not currently loadable from config (see integration note).

Constructor options:

- `detectors: list[BaseDriftDetector]`: sub-detectors whose signals are combined; more detectors = more robust but more compute.
- `voting`: `majority`, `unanimous`, `any`, or weighted fallback: how signals combine. `any` = most sensitive (first firing wins), `majority` = balanced, `unanimous` = most conservative (all must agree).
- `name`

Integration note:

- Class implementation exists, but dynamic config loading for sub-detectors is not implemented.

## How Monitor Uses Detectors

`src/apeiron/driver/continuous_monitor.py` does:

1. Evaluate validation batches and buffer metric vectors.
2. At every `detection_interval`, extract column `metric_index`.
3. Aggregate buffer by `aggregation` (`mean`/`median`/`last`).
4. Call `detector.update(agg_metric)`.
5. If `drift_detected`, run continual learning and optionally `detector.reset()`.

## Practical Config Snippet

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

Code for the workflow in the Monitor implementation:

```python
detector = load_drift_detector(cfg)

data = modelHarness.get_stream_dataloader()

for batch_idx, batch in tqdm(
    enumerate(val_loader),
    desc="Inference on batches",
    leave=False,
):
    # Inference on batch and compute all metrics
    metrics = self._evaluate_batch(batch)
    metric_buffer.append(metrics)


metric_idx = cfg.drift_detection.metric_index
metric_values = [m[metric_idx] for m in metric_buffer]

# aggregate metrics
aggregation = cfg.drift_detection.aggregation
if aggregation == "mean":
    agg_metric = float(np.mean(metric_values))
elif aggregation == "median":
    agg_metric = float(np.median(metric_values))
elif aggregation == "last":
    agg_metric = float(metric_values[-1])

drift_signal = detector.update(agg_metric)
if drift_signal.drift_detected:
    handle_drift(drift_signal)

self.detector.reset()
modelHarness.update_data_stream()
```
