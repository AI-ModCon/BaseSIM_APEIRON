---
name: new-updater
description: |
  Create a new continual learning update strategy for the BaseSim framework.
  Use when the user wants to add a new regularization method beyond the
  built-in base, JVP, EWC, and KFAC updaters.
argument-hint: "<UpdaterClassName> [update_mode_name]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

Scaffold a new continual learning updater for the BaseSim framework.

## Arguments
- `$0`: Class name for the new updater (e.g., "SIUpdater", "MASUpdater", "PackNetUpdater", "AGEMUpdater")
- `$1`: (Optional) update_mode string for config lookup. Defaults to lowercase `$0` without "Updater" suffix.

## Reference: Base Class and Existing Implementations

### Base updater interface with all hook methods
!`cat src/training/updater/base.py`

### EWC implementation (most complete example of regularization-based updater)
!`cat src/training/updater/ewc.py`

### JVP regularization implementation
!`cat src/training/updater/jvp_reg.py`

### Updater factory
!`cat src/training/updater/create_updater.py`

### CL config parameters
!`grep -A 20 "class ContinualLearningCfg" src/config/configuration.py`

### How the trainer calls updater hooks
!`cat src/training/continuous_trainer.py`

## Required Interface

Every updater subclasses `BaseUpdater`. The available hooks are called in this order per training iteration:

```
cl_preprocessing()              # Once, before CL loop starts
  for each optimizer step:
    update_pre_fwd_bwd()        # Before gradient computation
    for each accumulation step:
      fwd_bwd(batch, hist_batch) -> loss  # Forward + backward
    update_post_fwd_bwd() -> reg_loss     # After backward, before optimizer.step()
    optimizer.step()
    update_post_optimizer_call()           # After optimizer step
cl_postprocessing()             # Once, after CL loop ends
```

```python
from training.updater.base import BaseUpdater
from config.configuration import Config
from model.torch_model_harness import BaseModelHarness

class $0(BaseUpdater):
    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        super().__init__(cfg, modelHarness)
        # self.criterion and self.model are set by BaseUpdater
        # Initialize regularization-specific state here

    def cl_preprocessing(self) -> None:
        """Save reference parameters, compute importance weights, etc."""

    def fwd_bwd(self, batch, hist_batch=None) -> float:
        """Forward + backward. Override for custom loss computation.
        Return the loss scalar. Must call loss.backward()."""

    def update_post_fwd_bwd(self) -> float:
        """Apply gradient penalties/modifications after backward.
        Return regularization loss value (for logging)."""

    def update_post_optimizer_call(self) -> None:
        """Update running statistics after parameter update."""

    def cl_postprocessing(self) -> None:
        """Commit estimates, update EMA buffers, etc."""
```

## Common CL Strategy Patterns

### Regularization-based (EWC, SI, MAS)
- `cl_preprocessing()`: Snapshot reference parameters
- `update_post_fwd_bwd()`: Add penalty term based on parameter importance
- `cl_postprocessing()`: Update importance estimates with EMA

### Gradient-based (A-GEM, OGD)
- `fwd_bwd()`: Custom forward/backward with gradient projection
- `update_post_fwd_bwd()`: Project gradients onto feasible region

### Replay-based (with updater hooks)
- `fwd_bwd()`: Use `hist_batch` for experience replay
- Already partially supported by `jvp_reg` pattern

## Files to Create/Modify

### 1. Create `src/training/updater/<mode>.py`
The new updater implementation file.

### 2. Update `src/training/updater/create_updater.py`
Add a new branch:
```python
if cfg.continual_learning.update_mode == "<mode>":
    from training.updater.<module> import $0
    return $0(cfg, modelHarness)
```

### 3. Update `src/config/configuration.py`
Add new hyperparameters to `ContinualLearningCfg` with sensible defaults. Follow the naming pattern: `<prefix>_<param>` (e.g., `si_lambda`, `si_epsilon`).

## Procedure

1. Discuss the CL algorithm with the user:
   - What type of regularization or constraint does it apply?
   - What state needs to be maintained between tasks?
   - What are its hyperparameters?
2. Read the existing updater implementations for structural reference.
3. Create the updater file following the EWC pattern structure.
4. Wire into the factory in `create_updater.py`.
5. Add config parameters to `ContinualLearningCfg`.
6. Verify imports and factory work:
   ```bash
   cd /home/user/BaseSim_Framework && poetry run python -c "from training.updater.create_updater import create_updater; print('Factory imports OK')"
   ```

## Design Guidelines
- All hooks decorated with `@torch.no_grad()` except `fwd_bwd()` (which needs gradients)
- Regularization losses from `update_post_fwd_bwd()` are logged separately from generation loss
- Use `self.model.parameters()` or `self.model.named_parameters()` for parameter access
- Divide loss by `self.cfg.train.grad_accumulation_steps` in `fwd_bwd()` (see base implementation)
- Keep device management consistent -- use `self.cfg.device` for tensor placement
