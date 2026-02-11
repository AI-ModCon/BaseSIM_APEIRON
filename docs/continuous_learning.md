# Continuous Learning

This document describes the continual-learning path triggered after drift detection.

## Main Components

- `ContinuousTrainer` in `src/training/continuous_trainer.py`
- Updater factory `create_updater(...)` in `src/training/updater/create_updater.py`
- Updater implementations in `src/training/updater/`

## Training Loop Flow

When `ContinuousMonitor` detects drift:

1. `ContinuousTrainer.outer_cl_training_loop(...)` is called.
2. Current and historical loaders are pulled from the model harness.
3. Selected updater runs `cl_preprocessing()`.
4. For each step up to `cfg.train.max_iter`, trainer runs `inner_cl_training_loop(...)`.
5. Each inner step does:
   - `optimizer.zero_grad()`
   - `updater.update_pre_fwd_bwd()`
   - repeated `updater.fwd_bwd(...)` for gradient accumulation
   - `updater.update_post_fwd_bwd()`
   - `optimizer.step()`
   - `updater.update_post_optimizer_call()`
6. Updater runs `cl_postprocessing()` once at end.

## `train` Config Keys Used By CL

Defined in `TrainCfg` (`src/config/configuration.py`):

| Key | Default | Meaning |
| --- | --- | --- |
| `batch_size` | required | Per-loader batch size; also used as minimum accepted batch in `_safe_next`. |
| `num_workers` | required | DataLoader worker count used by harness loaders. |
| `init_lr` | required | Optimizer learning rate used by provided harnesses. |
| `grad_accumulation_steps` | `1` | Number of forward/backward micro-steps before one optimizer step. |
| `max_iter` | `600` | Max CL optimizer updates per drift event. |

## `continual_learning` Config Keys

Defined in `ContinualLearningCfg`:

| Key | Default | Meaning |
| --- | --- | --- |
| `update_mode` | `"base"` | Updater selection key. |
| `jvp_lambda` | `0.001` | Weight for JVP regularization term (`jvp_reg` mode). |
| `jvp_deltax_norm` | `1` | Scale factor for JVP input perturbation direction. |
| `ewc_lambda` | `1000.0` | EWC regularization strength (`ewc_online` mode). |
| `ewc_ema_decay` | `0.95` | EMA decay for online Fisher prior in EWC. |
| `kfac_lambda` | `0.01` | KFAC penalty strength (`kfac_online` mode). |
| `kfac_ema_decay` | `0.95` | EMA decay for running Kronecker factors in KFAC mode. |

## Updater Modes (`update_mode`)

### `base` -> `BaseUpdater`

- File: `src/training/updater/base.py`
- Behavior: plain supervised forward/backward on current batch only.
- Extra config: none.

### `jvp_reg` -> `JVPRegUpdater`

- File: `src/training/updater/jvp_reg.py`
- Uses:
  - current batch gradient
  - historical batch gradient
  - JVP-based regularization term
- Relevant config:
  - `continual_learning.jvp_lambda`
  - `continual_learning.jvp_deltax_norm`
- If no historical batch is available, it falls back to base update behavior.

### `ewc_online` -> `OnlineEWCUpdater`

- File: `src/training/updater/ewc.py`
- Keeps running parameter anchor (`theta_star`) and diagonal Fisher estimate.
- Adds EWC gradient penalty before optimizer step.
- Updates Fisher/anchor once per CL event in `cl_postprocessing()`.
- Relevant config:
  - `continual_learning.ewc_lambda`
  - `continual_learning.ewc_ema_decay`

### `kfac_online` -> `OnlineKFACUpdater`

- File: `src/training/updater/kfac.py`
- Tracks layer-wise activation/gradient statistics via hooks.
- Applies KFAC-structured EWC-like penalty.
- Supports modules:
  - `nn.Linear`
  - `nn.Conv2d`
- Relevant config:
  - `continual_learning.kfac_lambda`
  - `continual_learning.kfac_ema_decay`

### `none` -> `NoUpdater`

- File: `src/training/updater/no_updater.py`
- `fwd_bwd(...)` is a no-op and returns `-1.0`.
- Useful for disabling CL gradient updates while keeping monitoring flow intact.

## CL Trigger Conditions

CL is only dispatched when detector output has `drift_detected = True`. Drift checks depend on `drift_detection` settings:

- `detection_interval > 0` enables periodic checks.
- `detection_interval <= 0` disables checks and therefore disables CL dispatch in current monitor logic.

## Minimal CL Config Example

```toml
[train]
batch_size = 64
num_workers = 4
init_lr = 0.001
grad_accumulation_steps = 2
max_iter = 600

[continual_learning]
update_mode = "jvp_reg"
jvp_lambda = 10.0
jvp_deltax_norm = 1.0

ewc_lambda = 1000.0
ewc_ema_decay = 0.95
kfac_lambda = 1e-2
kfac_ema_decay = 0.95
```
