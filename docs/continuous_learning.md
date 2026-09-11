# Continuous Learning

This document describes the continual-learning path triggered after drift detection.

## Main Components

- `ContinuousTrainer` in `src/apeiron/training/continuous_trainer.py`
- Updater factory `create_updater(...)` in `src/apeiron/training/updater/create_updater.py`
- Updater implementations in `src/apeiron/training/updater/`

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

Defined in `TrainCfg` (`src/apeiron/config/configuration.py`):

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
| `mix_historic_data` | `false` | Mix historical replay data into `fwd_bwd()` (ignored by `jvp_reg`, which mixes the streams itself, and by `none`). |
| `jvp_rho_theta` | `0.05` | SAM parameter-perturbation radius (`jvp_reg` mode). |
| `jvp_rho_x` | `1.0` | Perturbation radius on the historical inputs along the drift direction; `0` gives parameter-only SAM. |
| `jvp_data_sign` | `1.0` | `+1` perturbs old inputs toward the current distribution, `-1` toward the old one. |
| `ewc_lambda` | `1000.0` | EWC regularization strength (`ewc_online` mode). |
| `ewc_ema_decay` | `0.95` | EMA decay for online Fisher prior in EWC. |
| `kfac_lambda` | `0.01` | KFAC penalty strength (`kfac_online` mode). |
| `kfac_ema_decay` | `0.95` | EMA decay for running Kronecker factors in KFAC mode. |
| `importance_weighting` | `false` | Enable prioritized sampling based on per-sample loss deltas. |
| `importance_alpha` | `1.0` | Priority exponent: `1.0` = linear, `<1.0` = flatter, `>1.0` = sharper prioritization. |

## Prioritized Sampling

When `importance_weighting = true`, the trainer rebuilds DataLoaders at the start of each CL round to use `WeightedRandomSampler` with per-sample priorities:

```
priority_i = (L(w_current, x_i) - L(theta_star, x_i))^importance_alpha
```

where `theta_star` is a snapshot of the model weights from the previous CL round.

- Samples the model has "forgotten" (higher loss vs. anchor) are sampled more frequently
- Training loss remains unchanged (no gradient distortion)
- Default `importance_alpha = 1.0` gives linear prioritization; increase for sharper, decrease for flatter

### Mutual Exclusivity with `mix_historic_data`

**Prioritized sampling and `mix_historic_data` are mutually exclusive.** When `importance_weighting = true`, the framework automatically disables `mix_historic_data` (with a warning) because:

- Prioritized sampling uses weighted sampling to focus on forgotten samples
- Mixing concatenates historical data directly into batches
- These are two different replay strategies that should not be combined

### Which Loaders Get Prioritized?

- **Current task loader**: Always prioritized when `importance_weighting = true`
- **Historical loader**: Only prioritized for `jvp_reg` (which always reads `hist_batch` directly)
- **For `base`, `ewc_online`, `kfac_online`**: Only current loader is prioritized; historical data is accessed via prioritized sampling on the current loader

Reference: Raghavan & Papadimitriou, FGCS 2025.

## Updater Modes (`update_mode`)

### `base` -> `BaseUpdater`

- File: `src/apeiron/training/updater/base.py`
- Behavior: plain supervised forward/backward on current batch only.
- Extra config: none.

### `jvp_reg` -> `JVPRegUpdater`

- File: `src/apeiron/training/updater/jvp_reg.py`
- The JVP updater now implements a first-order Sharpness-Aware / Bertsimas
  **robust** update rather than the original Jacobian-Vector-Product
  regularization term.
- Each step:
  1. Computes the current-batch gradient and uses its normalized negation `u_new`
     as the SAM parameter-perturbation direction.
  2. Builds one combined batch, half current and half historical, and shifts the
     historical inputs by `jvp_rho_x` along the unit batch-mean drift direction
     `unit(mean(X_cur) - mean(X_hist))`, signed by `jvp_data_sign`.
  3. Back-propagates the combined loss evaluated at `theta + jvp_rho_theta * u_new`,
     then restores the parameters. Only first-order information is used -- no
     Hessian and no third-order terms.
- Keeping the current batch inside the loss every step anchors online accuracy, so
  adaptation does not trade it away for retention.
- Relevant config:
  - `continual_learning.jvp_rho_theta`
  - `continual_learning.jvp_rho_x` (set to `0` for parameter-only SAM)
  - `continual_learning.jvp_data_sign`
- Manages the historical batch itself, so it ignores `mix_historic_data`.
- If no historical batch is available, it falls back to base update behavior.

### `ewc_online` -> `OnlineEWCUpdater`

- File: `src/apeiron/training/updater/ewc.py`
- Keeps running parameter anchor (`theta_star`) and diagonal Fisher estimate.
- Adds EWC gradient penalty before optimizer step.
- Updates Fisher/anchor once per CL event in `cl_postprocessing()`.
- Relevant config:
  - `continual_learning.ewc_lambda`
  - `continual_learning.ewc_ema_decay`

### `kfac_online` -> `OnlineKFACUpdater`

- File: `src/apeiron/training/updater/kfac.py`
- Tracks layer-wise activation/gradient statistics via hooks.
- Applies KFAC-structured EWC-like penalty.
- Supports modules:
  - `nn.Linear`
  - `nn.Conv2d`
- Relevant config:
  - `continual_learning.kfac_lambda`
  - `continual_learning.kfac_ema_decay`

### `none` -> `NoUpdater`

- File: `src/apeiron/training/updater/no_updater.py`
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
jvp_rho_theta = 0.05
jvp_rho_x = 1.0
jvp_data_sign = 1.0

ewc_lambda = 1000.0
ewc_ema_decay = 0.95
kfac_lambda = 1e-2
kfac_ema_decay = 0.95

# Optional: enable prioritized sampling
importance_weighting = true
importance_alpha = 1.0
```

Codes wanting to do continual learning should use the `ContinuousTrainer` class that takes the configuration parameters, model harness, logger and the profiler.

```python
cfg = build_config ( argv )

xtrainer = ContinuousTrainer(
    cfg=cfg,
    modelHarness=modelHarness,
    logger=logger,
    profiler=flops_profiler,
)

trainer.outer_cl_training_loop(drift_event_id=drift_count)
if modelHarness.ckpts_enabled:
    ckptpath = modelHarness.save_ckpt(event=drift_count)
```
