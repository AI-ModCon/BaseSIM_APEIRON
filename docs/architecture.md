# Architecture

Apeiron is built around four extension points — configuration, a **model
harness**, a **drift detector**, and an **updater** — wired together by a
monitoring driver. Everything else is plumbing.

The driver is the default way to run the workflow, not the only one: each
component keeps a standalone API, so detection and adaptation can be used
separately or embedded in someone else's loop. See
{ref}`running-without-the-driver`.

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

(running-without-the-driver)=

## Running without the driver

`ContinuousMonitor` is a convenience: it wires detection and adaptation together
into the loop above and is what `src/main.py` runs. Nothing else depends on it.
The detector, the trainer, the updater, and the harness are each constructed
independently and can be driven on their own — half the pipeline, or none of it,
embedded in a training loop you already have.

Two half-pipeline entry points ship with the repo. Both take the same flags as
`main.py` (`--config`, `--set key=val`, `--device`, `--multi-gpu`) and emit the
same CSV schema, so their runs are directly comparable to a full run.

| Entry point | Runs | Bypasses |
| --- | --- | --- |
| `python -m src.main` | `ContinuousMonitor` — detect, then adapt | — |
| `python -m src.drift_only` | `DriftOnlyMonitor` — detection trace over the stream, weights frozen | `ContinuousTrainer` |
| `python -m src.cl_only` | `ScheduledCLRunner` — CL fired by a `TriggerSchedule` | the drift detector |

**Detection only** (`src/drift_only.py`): streams data past a frozen model and
records every detector firing without adapting. Use it to tune a detector
offline — every check lands in the metrics CSV under the `drift/` stage with its
score, regime and confidence.

```{code-block} bash
:caption: Detection trace, no adaptation

poetry run python -m src.drift_only --config examples/mnist/mnist.toml
```

**Adaptation only** (`src/cl_only.py`): triggers the CL loop on a fixed schedule
instead of on detected drift — `periodic`, `random` (rate- or budget-matched),
`fixed` (explicit window indices), or `never` (the frozen-model lower bound).
This is the control arm for judging whether a detector's firing points were
actually worth their cost.

```{code-block} bash
:caption: Budget-matched control — 3 triggers over the run

poetry run python -m src.cl_only --config examples/mnist/mnist.toml \
    --schedule periodic --period 14
```

Both expose a function form (`run_drift_only(cfg, modelHarness)` and
`run_manual_cl(cfg, modelHarness, schedule)`) that returns a summary dict, so
they can be called from a sweep script rather than the shell.

### Component APIs

Below the entry points, each piece stands alone.

```{code-block} python
:caption: A detector on any scalar stream — no harness, no config

from apeiron.drift_detection import ADWINDetector

detector = ADWINDetector(delta=0.002)
for value in my_metric_stream:          # any float you already compute
    signal = detector.update(value)
    if signal.drift_detected:
        print(signal.regime, signal.drift_score)
```

`load_drift_detector(cfg)` is the config-driven equivalent when you already have
a `Config`; detectors themselves take plain constructor arguments.

```{code-block} python
:caption: The CL loop on its own, triggered by whatever you like

from apeiron.logger import get_logger
from apeiron.training import ContinuousTrainer

trainer = ContinuousTrainer(
    cfg=cfg, modelHarness=harness, logger=get_logger(), profiler=None
)
trainer.outer_cl_training_loop(drift_event_id=1)
```

And a single updater can be built with `create_updater(cfg, modelHarness)` and
its hooks called from your own training step, without `ContinuousTrainer` at
all. Adding Apeiron to an existing training loop this way is what the
`integrate-apeiron` skill automates — see {doc}`agent_skills`.

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
