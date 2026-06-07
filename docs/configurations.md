
# Apeiron Configuration Reference

This document describes all currently supported TOML configuration options available in Apeiron.
The code to parse the configuration file can be found in `src/apeiron/config/configuration.py`.

## Root-Level Configuration

```toml
seed = 1337
device = "auto"
multi_gpu = false
verbosity = "INFO"
```

| Option | Type | Description |
|----------|------|-------------|
| `seed` | int | Required, random seed used throughout the experiment. |
| `device` | str | Required, execution device (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). |
| `multi_gpu` | bool | Enables multi-GPU selection logic when CUDA is available. Default: false. |
| `verbosity` | str | Logging verbosity level (DEBUG, INFO, INFO:n, WARNING, ERROR, CRITICAL). Default: INFO. |

## [model]

```toml
[model]
name = "myModel"
pretrained_path = "/path/to/model.pt"
max_ckpts = 1
ckpts_path = "/path/to/checkpoint"
```

| Option | Type | Description |
|----------|------|-------------|
| `name` | str | Required, model identifier used by the example factory. |
| `pretrained_path` | str | Path to checkpoint weights. Default: none. |
| `max_ckpts` | int | Number of retained model checkpoints after CL. Default: 0 (no checkpoint is saved). |
| `ckpts_path` | str | Checkpoint output directory. |


## [data]

```toml
[data]
name = "dataset"
path = "data.csv"
batch_size = 32
```

| Option | Type | Description |
|----------|------|-------------|
| `name` | str | Required, dataset/example identifier. |
| `path` | str | Reguired, dataset location. |
| `batch_size` | int | Streaming batch size. Default 1 (inference/drift detection is done on each individual sample). |


## [train]

```toml
[train]
batch_size = 64
num_workers = 4
init_lr = 0.001
grad_accumulation_steps = 1
max_iter = 600
```

| Option | Type | Description |
|----------|------|-------------|
| `batch_size` | int | Required, taining batch size. |
| `num_workers` | int | Required, dataLoader workers. |
| `init_lr` | float | Required, initial learning rate. |
| `grad_accumulation_steps` | int | Gradient accumulation count. Default: 1. |
| `max_iter` | int | Maximum CL iterations. Default: 600. |


## [continual_learning]

```toml
[continual_learning]
update_mode = "base"

jvp_lambda = 0.001
jvp_deltax_norm = 1

ewc_lambda = 1000.0
ewc_ema_decay = 0.95

kfac_lambda = 0.01
kfac_ema_decay = 0.95
```

| Option | Type | Description |
|----------|------|-------------|
| `update_mode` | str | Required, CL strategy to use (base, jvp_reg, ewc_online, kfac_online, none). |
| `jvp_lambda` | float | Weight for JVP regularization term (`jvp_reg` mode). Default: 0.001. |
| `jvp_deltax_norm` | float | Scale factor for JVP input perturbation direction. Default: 1. |
| `ewc_lambda` | float | EWC regularization strength (`ewc_online` mode). Default: 1000. |
| `ewc_ema_decay` | float | EMA decay for online Fisher prior in EWC. Default: 0.95. |
| `kfac_lambda` | float | KFAC penalty strength (`kfac_online` mode). Default: 0.01. |
| `kfac_ema_decay` | float | EMA decay for running Kronecker factors in KFAC mode. Default: 0.95. |

Details about the CL algorithms available can be found in [docs/continuous_learning.md](continuous_learning.md)


## [drift_detection]

All the parameters are optional. Default values are provided in the examples below.

Core settings:

```toml
[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20
```

| Option | Type | Description |
|----------|------|-------------|
| `detector_name` | str | Drift detection algorithm (ADWINDetector, KSWINDetector, PageHinkleyDetector, ModelPerformanceDetector, ModelEvalDetector, EnsembleDetector). |
| `detection_interval` | int | Check drift every N monitored batches. If `<= 0`, checks are disabled. |
| `aggregation` | str | How buffered metric values are aggregated before detector update. Supported by monitor: `mean`, `median`, `last`. |
| `metric_index` | int | Index into `modelHarness.eval_metrics` order. |
| `reset_after_learning` | bool | If true, detector state resets after each CL event. |
| `max_stream_updates` | int | Monitoring stops after this many stream extensions. |

ADWIN:

```toml
adwin_delta = 0.002
adwin_minor_threshold = 0.3
adwin_moderate_threshold = 0.6
```

| Option | Type | Description |
|----------|------|-------------|
| `adwin_delta` | float | ADWIN confidence/sensitivity parameter. |
| `adwin_minor_threshold` | float | ADWIN regime threshold (CL boundary). |
| `adwin_moderate_threshold` | float | ADWIN regime threshold (fine-tuning boundary). |

KSWIN:

```toml
kswin_alpha = 0.005
kswin_window_size = 100
kswin_stat_size = 30
```

| Option | Type | Description |
|----------|------|-------------|
| `kswin_alpha` | float | KSWIN significance level. |
| `kswin_window_size` | int | KSWIN reference window size. |
| `kswin_stat_size` | int | KSWIN recent sample window size. |

Page-Hinkley:

```toml
ph_min_instances = 30
ph_delta = 0.005
ph_threshold = 50
ph_alpha = 0.9999
```

| Option | Type | Description |
|----------|------|-------------|
| `ph_min_instances` | int | Page-Hinkley warm-up count before detection is meaningful. |
| `ph_delta` | float | Page-Hinkley change magnitude parameter. |
| `ph_threshold` | float | Page-Hinkley trigger threshold. |
| `ph_alpha` | float | Page-Hinkley forgetting factor. |


Details about the drift detection algorithms available can be found in [docs/drift_detectors.md](drift_detectors.md)


## [visualization]

The visualization configuration options are used to store the results of the metrics captured during the run.

```toml
[visualization]
input = "output/results.csv"
```

| Option | Type | Description |
|----------|------|-------------|
| `input` | str | Required, path to the CSV output file storing the metrics. |

Metrics used:
```
cl/cperf_detector_flop
cl/cperf_detector_flops
cl/cperf_detector_time
cl/cperf_infer_flop
cl/cperf_infer_flops
cl/cperf_infer_time
cl/cperf_optimizer_flop
cl/cperf_optimizer_flops
cl/cperf_optimizer_time
cl/cperf_update_fwd_bwd_flop
cl/cperf_update_fwd_bwd_flops
cl/cperf_update_fwd_bwd_time
cl/drift_event_id
cl/jvp_reg_forgetting_loss
cl/jvp_reg_generation_loss
cl/jvp_reg_total_loss
cl/step
drift/confidence
drift/cperf_detector_flop
drift/cperf_detector_flops
drift/cperf_detector_time
drift/cperf_infer_flop
drift/cperf_infer_flops
drift/cperf_infer_time
drift/cperf_optimizer_flop
drift/cperf_optimizer_flops
drift/cperf_optimizer_time
drift/cperf_update_fwd_bwd_flop
drift/cperf_update_fwd_bwd_flops
drift/cperf_update_fwd_bwd_time
drift/detected
drift/metric_0
drift/regime
drift/score
drift/step
eval/accuracy
eval/loss
eval/step
eval/test_curr_acc
eval/test_hist_acc
```

Example output file:
```csv
step,metric,value
10,eval/accuracy,62.5
10,eval/loss,2.0406203269958496
10,eval/step,10
10,drift/score,0.0
10,drift/regime,stable
10,drift/confidence,0.998
10,drift/metric_0,44.6875
10,drift/cperf_infer_flop,1624834048.0
10,drift/cperf_infer_time,0.00556622925796546
10,drift/cperf_infer_flops,291909293113.4319
10,drift/cperf_detector_flop,0.0
10,drift/cperf_detector_time,8.56249826028943e-05
10,drift/cperf_detector_flops,0.0
10,cl/jvp_reg_total_loss,3.4211268424987793
10,cl/jvp_reg_forgetting_loss,0.0
10,cl/jvp_reg_generation_loss,3.4211268424987793
```

## [logging]

```toml
[logging]
backend = "wandb"
experiment_name = "experiment"
mlflow_tracking_uri = "http://localhost:5000"
```

| Option | Type | Description |
|----------|------|-------------|
| `backend` | str | Logging backend (wandb, mlflow, or none). Default: wandb. |
| `experiment_name` | str | W&B project name or MLflow experiment name. Default: None. |
| `mlflow_tracking_uri` | str | MLflow tracking server URI. Default: None. |


## Complete Valid Example

```toml
seed = 1337
device = "auto"
multi_gpu = false
verbosity = "INFO"

[model]
name = "mnist"
pretrained_path = "examples/mnist/mnist.pth"
max_ckpts = 0
ckpts_path = "output/mnist"

[data]
name = "mnist"
path = ""
batch_size = 32

[train]
batch_size = 64
num_workers = 4
init_lr = 0.001

[continual_learning]
update_mode = "base"

[drift_detection]
detector_name = "ADWINDetector"

[logging]
backend = "none"

[visualization]
input = "output/results.csv"
```
