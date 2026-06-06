
# Apeiron Configuration Reference

This document describes all currently supported TOML configuration options available in Apeiron, based on `src/apeiron/config/configuration.py`.

## Root-Level Configuration

```toml
seed = 1337
device = "auto"
multi_gpu = false
verbosity = "INFO"
```

| Option | Type | Description |
|----------|------|-------------|
| `seed` | int | Random seed used throughout the experiment. |
| `device` | str | Execution device (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`). |
| `multi_gpu` | bool | Enables multi-GPU selection logic when CUDA is available. |
| `verbosity` | str | Logging verbosity level. |

## [model]

```toml
[model]
name = "myModel"
pretrained_path = ""
max_ckpts = 0
ckpts_path = ""
```

| Option | Type | Description |
|----------|------|-------------|
| `name` | str | Model identifier used by the example factory. |
| `pretrained_path` | str | Path to checkpoint weights. |
| `max_ckpts` | int | Number of retained CL checkpoints. |
| `ckpts_path` | str | Checkpoint output directory. |

Unsupported unless `ModelCfg` is extended:

```toml
architecture_path = "..."
class_name = "..."
input_dim = 512
num_classes = 10
```

## [data]

```toml
[data]
name = "dataset"
path = "data.csv"
batch_size = 32
```

| Option | Type | Description |
|----------|------|-------------|
| `name` | str | Dataset/example identifier. |
| `path` | str | Dataset location. |
| `batch_size` | int | Streaming batch size. |

Unsupported unless `DataCfg` is extended:

```toml
target_col = "label"
num_features = 512
feature_columns = ["a", "b"]
```

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
| `batch_size` | int | Training batch size. |
| `num_workers` | int | DataLoader workers. |
| `init_lr` | float | Initial learning rate. |
| `grad_accumulation_steps` | int | Gradient accumulation count. |
| `max_iter` | int | Maximum CL iterations. |

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

## [drift_detection]

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

ADWIN:

```toml
adwin_delta = 0.002
adwin_minor_threshold = 0.3
adwin_moderate_threshold = 0.6
```

KSWIN:

```toml
kswin_alpha = 0.005
kswin_window_size = 100
kswin_stat_size = 30
```

Page-Hinkley:

```toml
ph_min_instances = 30
ph_delta = 0.005
ph_threshold = 50
ph_alpha = 0.9999
```

## [visualization]

```toml
[visualization]
input = "output/results.csv"
```

## [logging]

```toml
[logging]
backend = "wandb"
experiment_name = "experiment"
mlflow_tracking_uri = "http://localhost:5000"
```

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
