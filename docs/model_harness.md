# Model Harness

This document describes the model harness contract and the concrete harness classes currently shipped in the repository.

## Base Class Contract

All harnesses inherit from `BaseModelHarness` in `src/model/torch_model_harness.py`.

Required methods:

| Method | Return | Purpose |
| --- | --- | --- |
| `get_optmizer()` | `torch.optim.Optimizer` | Returns optimizer used by continual learning loops. |
| `update_data_stream()` | `None` | Advances stream state and rebuilds current loaders. |
| `get_stream_dataloader()` | `data_loader` | Returns the loader for the current stream of data. |
| `get_train_dataloaders()` | `(train_loader, val_loader)` | Returns loaders for the current drift state. |
| `get_hist_dataloaders()` | `(hist_train_loader, hist_val_loader)` or `(None, None)` | Returns historical replay loaders used by CL methods. |
| `get_criterion()` | callable loss fn | Returns criterion compatible with model output/labels. |

Important inherited behavior:

- `eval()` loops over `get_train_dataloaders()[1]` and aggregates all metrics in `self.eval_metrics`.
- `history_eval()` loops over historical validation data and returns `None` if there is no history.
- `_unpack(batch)` assumes `(x, y)` tuples. Override it if your loader format differs.

Required/expected attributes:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `self.model` | `nn.Module` | Model moved to `cfg.device` in base constructor. |
| `self.cfg` | `Config` | Frozen configuration object. |
| `self.eval_metrics` | `dict[str, callable]` | Ordered metric map. Metric order controls `metric_index` behavior in drift detection. |
| `self.higher_is_better` | `dict[str, bool]` | Used by `ModelEvalDetector` style logic. Optional for monitor-only operation. |

## Runtime Lifecycle

1. `examples/utils.py:get_example` picks a harness from `cfg.data.name`.
2. `ContinuousMonitor.run()` calls `modelHarness.update_data_stream()` once before monitoring starts.
3. Monitoring evaluates per-batch metrics from `eval_metrics`.
4. On drift, `ContinuousTrainer` pulls both current and historical loaders from the harness.

## Config Keys Used By Harnesses

| Config key | Used by | Meaning |
| --- | --- | --- |
| `model.name` | CIFAR/ImageNet harnesses | Backbone architecture name used by `load_model(...)`. |
| `model.pretrained_path` | All provided harnesses | Optional checkpoint to load before streaming starts. |
| `data.name` | example selector + CIFAR internals | Chooses harness (`mnist`, `cifar10`, `imagenet`) and dataset family logic. |
| `data.path` | ImageNet harness | Root directory containing `train/` and `val/` class folders. |
| `train.batch_size` | all harnesses | Loader batch size. |
| `train.num_workers` | all harnesses | DataLoader worker count. |
| `train.init_lr` | all harnesses | Initial optimizer learning rate. |
| `seed` | all harnesses | Drives deterministic drift augmentation sampling (`seed + task_counter`). |
| `device` | base class + checkpoint load | Device placement for model/tensors. |

## Concrete Harnesses

### `MNIST_CNN` (`examples/mnist/model.py`)

- Default model: internal `Cnn`.
- Criterion: `torch.nn.NLLLoss`.
- Optimizer: Adam over all model params with `cfg.train.init_lr`.
- Data source: torchvision MNIST from `./data` (downloads automatically if missing).
- Drift mechanism: affine transform sampled each stream update and applied to full train/val views.
- Historical loaders: available from the second stream update onward (`task_counter > 1`).

### `CIFAR_VISION` (`examples/cifar/model.py`)

- Model wrapper: `VisionModelCifar` with either CNN-family or ViT-family backend.
- Criterion: `torch.nn.CrossEntropyLoss`.
- Optimizer: Adam with `cfg.train.init_lr`.
- Data source: torchvision CIFAR from `./data`.
- Supports `cfg.data.name` of `cifar10` or `cifar100` internally.
- `get_example(...)` currently routes only `cifar10` to this harness.

Supported `model.name` values for CIFAR/ImageNet loaders:

- `alexnet`
- `vgg11`
- `vgg16`
- `vgg19`
- `inception`
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnext50_32x4d`
- `resnext101_32x8d`
- `densenet121`
- `densenet169`
- `densenet201`
- `regnet_x_400mf`
- `regnet_x_8gf`
- `regnet_x_16gf`
- `vit16b`
- `vit16l`
- `vit32l`
- `vit14h`
- `vit14g`

### `IMAGENET_VISION` (`examples/imagenet/model.py`)

- Model wrapper: `VisionModelImageNet`.
- Criterion: `torch.nn.CrossEntropyLoss`.
- Optimizer: Adam with `cfg.train.init_lr`.
- Data source: `torchvision.datasets.ImageFolder`.
- Requires `cfg.data.path` with this layout:
  - `<path>/train/<class_id>/*.JPEG`
  - `<path>/val/<class_id>/*.JPEG`

## Common Pitfalls

- `metric_index` in drift detection is position-based, not name-based. Keep metric order stable in `eval_metrics`.
- `get_stream_dataloader()` must return a non-`None` loader after `update_data_stream()`.
- `get_train_dataloaders()` must return non-`None` loaders after `update_data_stream()`.
- If your batch is not exactly `(x, y)`, override `_unpack` in your harness.

# Creating a Custom  Model Harness

A custom model harness is the integration point between a user model and Apeiron. The harness allows Apeiron to monitor model performance, detect drift, trigger continual learning, and manage retraining without requiring modifications to the underlying scientific model.

## Required Files

A custom example typically consists of:

```text
examples/<example_name>/
├── __init__.py
├── model.py
├── utils.py
└── <example_name>.toml
```

In addition, the example must be registered in:

```text
examples/utils.py
```

through the `get_example()` factory.

## Step 1: Create the Harness Class

All custom harnesses must inherit from `BaseModelHarness`.

```python
class MyHarness(BaseModelHarness):
    def __init__(self, cfg):
        model = MyModel()
        super().__init__(cfg=cfg, model=model)
```

The base class handles:

- Device placement
- Checkpoint support
- Evaluation loops
- Continual-learning integration
- Drift-monitor integration

## Step 2: Define Evaluation Metrics

Apeiron's drift detectors operate on metric streams.

At least one metric should be exposed through:

```python
self.eval_metrics = {
    "accuracy": accuracy,
}
```

or for regression:

```python
self.eval_metrics = {
    "mse": regression_mse,
}
```

The order of metrics is important because drift detection uses:

```toml
metric_index = 0
```

which refers to the first metric in `eval_metrics`.

## Step 3: Implement Required Methods

### Optimizer

```python
def get_optmizer(self):
    return torch.optim.Adam(
        self.model.parameters(),
        lr=self.cfg.train.init_lr,
    )
```

### Criterion

```python
def get_criterion(self):
    return torch.nn.CrossEntropyLoss()
```

or for regression:

```python
def get_criterion(self):
    return torch.nn.MSELoss()
```

### Current Training Loaders

```python
def get_train_dataloaders(self):
    return self.train_loader, self.val_loader
```

These loaders are used during continual-learning updates.

### Stream Loader

```python
def get_stream_dataloader(self):
    return self.stream_loader
```

This loader is used during monitoring and drift detection.

### Historical Loaders

```python
def get_hist_dataloaders(self):
    if self.task_counter == 0:
        return None, None

    return self.hist_train_loader, self.hist_val_loader
```

Historical loaders provide replay data for continual-learning algorithms such as EWC and KFAC.

### Stream Updates

```python
def update_data_stream(self):
    self.task_counter += 1

    self.train_loader = build_train_loader()
    self.val_loader = build_val_loader()
    self.stream_loader = build_stream_loader()
```

This function is called by the monitor whenever a new stream segment becomes active.

## Step 4: Build Dataset Utilities

Dataset loading should be implemented in `utils.py`.

Typical responsibilities include:

- Reading datasets
- Feature preprocessing
- Label extraction
- Train/validation splitting
- Drift simulation
- DataLoader construction

Example:

```python
def make_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
```

## Step 5: Register the Example

Add a branch to `examples/utils.py`:

```python
elif cfg.data.name == "mydataset":
    from examples.mydataset.model import MyHarness

    return MyHarness(cfg=cfg)
```

The value must match:

```toml
[data]
name = "mydataset"
```

## Step 6: Create the Configuration File

Create:

```text
examples/<example_name>/<example_name>.toml
```

Requirested and optional parameters are described in [configurations.md](configurations.md).

Only keys defined in `configuration.py` are allowed. Custom dataset-specific parameters should be implemented inside the harness or utility code unless the configuration dataclasses are extended.

## Step 7: Validate the Integration

Validate TOML syntax:

```bash
python -c "import tomllib; tomllib.load(open('examples/mydataset/mydataset.toml', 'rb')); print('TOML OK')"
```

Validate factory registration:

```bash
poetry run python -c "from examples.utils import get_example; print('factory OK')"
```

## Step 8: Run a Smoke Test

Before running a full experiment, execute a small CPU-only test:

```bash
poetry run python -m src.main \
  --config examples/mydataset/mydataset.toml \
  --set train.max_iter=2 \
  --set drift_detection.max_stream_updates=2 \
  --set drift_detection.detection_interval=1 \
  --set device=cpu \
  --set logging.backend=none
```

A successful smoke test confirms:

- Configuration loading
- Harness construction
- Dataset loading
- Drift-monitor integration
- Continual-learning integration
- Metric reporting

## Common Errors

### `KeyError: 'model'`

The TOML file is missing the `[model]` section.

### `ImportError` when loading the harness

The factory registration in `examples/utils.py` is missing or incorrect.

### `TypeError: DataCfg.__init__() got an unexpected keyword argument ...`

A TOML key is not defined in the configuration dataclasses.

### `IndexError: list index out of range` during drift detection

No evaluation metrics are being emitted. Ensure `self.eval_metrics` contains at least one metric.

### `get_stream_dataloader()` returns `None`

The monitor cannot evaluate data until `update_data_stream()` constructs and returns a valid stream loader.

## Summary

A custom harness only needs to provide six core behaviors:

1. Construct the model.
2. Construct the optimizer.
3. Construct the criterion.
4. Provide current train/validation loaders.
5. Provide historical replay loaders.
6. Update the active data stream.

Once those interfaces are implemented, Apeiron can automatically provide monitoring, drift detection, continual learning, checkpointing, logging, and experiment management for the underlying scientific model.
