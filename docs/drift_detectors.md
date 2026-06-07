# Drift Detectors

This document describes detector classes under `src/drift_detection/`, the `drift_detection` config section, and how detector outputs drive continual learning.

## Core Types

Defined in `src/drift_detection/detectors/base.py`:

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

`src/config/configuration.py` defines `DriftDetectionCfg`:

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

`src/drift_detection/load_drift_detector.py` currently supports:

- `ADWINDetector`
- `KSWINDetector`
- `PageHinkleyDetector`
- `ModelPerformanceDetector`
- `EvalDetector` (maps to `ModelEvalDetector`)

`EnsembleDetector` is present as a class but intentionally not wired in the loader and raises `NotImplementedError`.

## Detector Classes And Options

### `ADWINDetector` (`src/drift_detection/detectors/statistical_detectors.py`)

Constructor options:

- `delta` (config: `adwin_delta`)
- `minor_threshold` (config: `adwin_minor_threshold`)
- `moderate_threshold` (config: `adwin_moderate_threshold`)
- `name` (constructor-only; not exposed in config)

Behavior:

- Updates on one scalar value at each check.
- Computes `drift_score` as recent drift frequency over up to 100 updates.
- Emits regime based on `drift_detected` and score thresholds.

### `KSWINDetector`

Constructor options:

- `alpha` (config: `kswin_alpha`)
- `window_size` (config: `kswin_window_size`)
- `stat_size` (config: `kswin_stat_size`)
- `minor_threshold` (constructor default only; not in config loader call)
- `moderate_threshold` (constructor default only; not in config loader call)
- `name` (constructor-only)

Behavior:

- Uses Kolmogorov-Smirnov windowing from `river`.
- Score is recent drift frequency over up to 50 updates.

### `PageHinkleyDetector`

Constructor options:

- `min_instances` (config: `ph_min_instances`)
- `delta` (config: `ph_delta`)
- `threshold` (config: `ph_threshold`)
- `alpha` (config: `ph_alpha`)
- `minor_threshold` (constructor default only; not in config loader call)
- `moderate_threshold` (constructor default only; not in config loader call)
- `name` (constructor-only)

Behavior:

- Online mean-shift detector from `river`.
- Score is recent drift frequency over up to 50 updates.

### `ModelPerformanceDetector` (`src/drift_detection/detectors/model_performance_detector.py`)

Constructor options:

- `reference_data`
- `reference_predictions`
- `reference_targets`
- `drift_share_threshold`
- `minor_threshold`
- `moderate_threshold`
- `name`

Behavior modes:

- Batch mode: accepts `data` (`pandas.DataFrame`) and optional `predictions`/`targets`, runs Evidently `DataDriftPreset`.
- Scalar fallback mode: if given only `value`, uses internal simple score logic.

Integration note:

- The class requires reference initialization (`set_reference(...)` or constructor reference data) before `update(...)`.
- Current loader path instantiates it without reference data, so it is not plug-and-play in `ContinuousMonitor` without additional initialization code.

### `ModelEvalDetector` (`detector_name = "EvalDetector"`)

Constructor options:

- `name` only.

Expected `update(...)` kwargs:

- `modelHarness`
- `reference_validation_metrics`
- `higher_is_better`

Integration note:

- `ContinuousMonitor._check_drift()` currently passes only one scalar metric to `update(...)`.
- Using `EvalDetector` in the current monitor flow requires extra wiring to pass the expected kwargs.

### `EnsembleDetector`

Constructor options:

- `detectors: list[BaseDriftDetector]`
- `voting`: `majority`, `unanimous`, `any`, or weighted fallback
- `name`

Integration note:

- Class implementation exists, but dynamic config loading for sub-detectors is not implemented.

## How Monitor Uses Detectors

`src/driver/continuous_monitor.py` does:

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
