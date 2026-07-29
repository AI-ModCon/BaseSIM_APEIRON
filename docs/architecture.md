# Architecture

Apeiron is built around four extension points — configuration, a **model
harness**, a **drift detector**, and an **updater** — wired together by a
monitoring driver. Everything else is plumbing.

## Runtime flow

```{mermaid}
flowchart TD
    A[src/main.py<br/>build_config] --> B[examples/utils.py<br/>get_example -> BaseModelHarness]
    B --> C[ContinuousMonitor.run]
    C --> D[evaluate stream batch<br/>buffer eval_metrics]
    D --> E{every<br/>detection_interval?}
    E -- no --> D
    E -- yes --> F[aggregate metric<br/>mean / median / last]
    F --> G[detector.update value]
    G --> H{drift_detected?}
    H -- no --> D
    H -- yes --> I[ContinuousTrainer<br/>outer_cl_training_loop]
    I --> J[updater hooks<br/>base / jvp_reg / ewc / kfac]
    J --> K[optional checkpoint<br/>optional detector.reset]
    K --> D
```

Step by step:

1. `src/main.py` builds a `Config` from TOML, `APP_` environment variables, and
   `--set` CLI overrides.
2. `examples/utils.py:get_example` selects a concrete `BaseModelHarness` from
   `cfg.data.name`.
3. `apeiron/driver/continuous_monitor.py` evaluates streaming batches and calls
   the detector every `detection_interval` batches.
4. On drift, `apeiron/training/continuous_trainer.py` runs a CL loop using an
   updater from `apeiron/training/updater/create_updater.py`.
5. Logging is stage-aware (`eval`, `drift`, `cl`) via `apeiron/logger/`.

## Modules

| Module | Role |
| --- | --- |
| `apeiron/config/configuration.py` | TOML/env/CLI config assembly into frozen dataclasses. |
| `apeiron/model/torch_model_harness.py` | `BaseModelHarness` — the model + data-stream contract. |
| `apeiron/driver/continuous_monitor.py` | `ContinuousMonitor` — the monitoring and drift loop. |
| `apeiron/drift_detection/` | Detector classes and the `load_drift_detector` factory. |
| `apeiron/training/continuous_trainer.py` | `ContinuousTrainer` — outer/inner CL loops with gradient accumulation. |
| `apeiron/training/updater/` | CL update strategies behind the `BaseUpdater` hooks. |
| `apeiron/evaluation/metrics.py` | `accuracy()` and `accuracy_topk()`. |
| `apeiron/logger/` | Console output plus W&B / MLflow metrics backends. |
| `apeiron/profilers/` | `FLOPSProfiler` built on PyTorch `FlopCounterMode`. |

The installable package lives under `src/apeiron/` and is imported as `apeiron`
(see `packages = [{ include = "apeiron", from = "src" }]` in `pyproject.toml`).

## Extension points

Each extension point is an ABC with a factory in front of it, so adding a new
implementation means subclassing and registering — not editing the driver.

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} `BaseModelHarness`
:link: model_harness
:link-type: doc

Exposes your model, optimizer, criterion, stream loader, train/val loaders, and
historical replay loaders. Selected by `data.name`.
:::

:::{grid-item-card} `BaseDriftDetector`
:link: drift_detectors
:link-type: doc

`update(value) -> DriftSignal`. Selected by `drift_detection.detector_name` via
`load_drift_detector`.
:::

:::{grid-item-card} `BaseUpdater`
:link: continuous_learning
:link-type: doc

Hooks around forward/backward and the optimizer step. Selected by
`continual_learning.update_mode` via `create_updater`.
:::

::::

## Example harnesses

The bundled examples double as reference implementations of the harness
contract:

| Harness | File | Notes |
| --- | --- | --- |
| `MNIST_CNN` | `examples/mnist/model.py` | CNN on MNIST with affine drift simulation. |
| `CIFAR_VISION` | `examples/cifar/model.py` | ViT/VGG on CIFAR-10 with affine drift. |
| `IMAGENET_VISION` | `examples/imagenet/model.py` | ViT on ImageNet with affine drift. |

`examples/utils.py:get_example(cfg)` is the factory that dispatches on
`cfg.data.name`.
