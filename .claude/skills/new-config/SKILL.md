---
name: new-config
description: |
  Generate a new TOML config file for a BaseSim experiment. Use when the user
  wants to create a configuration for a specific dataset, detector, updater,
  and training parameter combination. Can also modify an existing config.
argument-hint: "<output_path> [base_config_to_copy_from]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
---

Generate a new TOML configuration file for a BaseSim experiment.

## Arguments
- `$0`: Output path for the new config file (e.g., `examples/mnist/mnist_ewc.toml`)
- `$1`: (Optional) Existing config to use as a starting template

## Reference: Configuration Dataclasses
!`grep -A 5 "class.*Cfg" src/config/configuration.py`

## Available Options

### Datasets (data.name)
- `mnist` -- MNIST handwritten digits (auto-downloads). Model: name="dummy", pretrained_path="examples/mnist/mnist.pth"
- `cifar10` -- CIFAR-10 images (auto-downloads). Model: name="vit16b" or "vgg11", pretrained_path="examples/cifar/cifar10_vit.pth" or "examples/cifar/cifar10_vgg11.pth"
- `imagenet` -- ImageNet (requires local data). Model: name="vit16b", pretrained_path="examples/imagenet/imagenet_vit.pth"

### Drift Detectors (drift_detection.detector_name)
| Detector | Best For | Key Params |
|---|---|---|
| `ADWINDetector` | Gradual + abrupt changes | adwin_delta=0.002, adwin_minor_threshold=0.3, adwin_moderate_threshold=0.6 |
| `KSWINDetector` | Distribution changes | kswin_alpha=0.005, kswin_window_size=100, kswin_stat_size=30 |
| `PageHinkleyDetector` | Abrupt mean changes | ph_min_instances=30, ph_delta=0.005, ph_threshold=50, ph_alpha=0.9999 |
| `ModelPerformanceDetector` | Batch-level analysis | (evidently defaults) |
| `ModelEvalDetector` | Direct eval comparison | metric_index |

### CL Updaters (continual_learning.update_mode)
| Mode | Strategy | Key Params |
|---|---|---|
| `base` | Vanilla SGD | (none) |
| `jvp_reg` | JVP regularization | jvp_lambda (default 0.001), jvp_deltax_norm (default 1) |
| `ewc_online` | Elastic Weight Consolidation | ewc_lambda (default 1000.0), ewc_ema_decay (default 0.95) |
| `kfac_online` | KFAC approximation | kfac_lambda (default 0.01), kfac_ema_decay (default 0.95) |
| `none` | No CL updates | (none) |

## Minimal Template
```toml
seed = 1337
device = "auto"
multi_gpu = false

[model]
name = ""
pretrained_path = ""

[data]
name = ""
path = ""

[train]
batch_size = 64
num_workers = 4
init_lr = 0.001
max_iter = 600
grad_accumulation_steps = 1

[continual_learning]
update_mode = "base"

[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

[visualization]
baseline = 90.0
input = "output/experiment.csv"
output = "output/experiment_dashboard.png"
```

## Procedure

1. If `$1` is provided, read it as the base template. Otherwise use the minimal template above.
2. Ask the user what they want to configure (or apply values they already specified):
   - Dataset and model
   - Drift detector and its hyperparameters
   - CL update strategy and its hyperparameters
   - Training parameters (batch size, learning rate, max iterations)
3. Fill in model.name and model.pretrained_path based on dataset choice.
4. Add the appropriate detector-specific hyperparameters for the chosen detector.
5. Set visualization input/output paths based on the experiment name.
6. Write the final TOML to `$0`.
7. Validate the config is parseable:
   ```bash
   python -c "import tomllib; tomllib.load(open('$0', 'rb')); print('Config is valid TOML')"
   ```
