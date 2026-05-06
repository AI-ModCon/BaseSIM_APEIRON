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
