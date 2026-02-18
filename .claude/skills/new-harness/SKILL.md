---
name: new-harness
description: |
  Create a new model harness for integrating a custom model and dataset into
  the BaseSim framework. Use when the user wants to add support for a new
  dataset (beyond MNIST, CIFAR-10, ImageNet) or a new model architecture.
  Generates the harness class, data utilities, TOML config, and registers
  it in the example factory.
argument-hint: "<name> [dataset_name]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

Scaffold a new model harness for a custom dataset/model in the BaseSim framework.

## Arguments
- `$0`: Name for the new example directory (e.g., "fashionmnist", "svhn", "custom_tabular")
- `$1`: (Optional) Dataset identifier for `data.name` in config, defaults to `$0`

## Reference: Existing Implementations

Read these files before generating code to ensure consistency with current patterns:

### Base class interface
!`cat src/model/torch_model_harness.py`

### Canonical example (MNIST harness)
!`cat examples/mnist/model.py`

### Data utilities pattern
!`cat examples/mnist/utils.py`

### Factory that registers harnesses
!`cat examples/utils.py`

## Files to Create

### 1. `examples/$0/__init__.py`
Empty init file.

### 2. `examples/$0/model.py`
Subclass of `BaseModelHarness` implementing all abstract methods:

```python
class <Name>Harness(BaseModelHarness):
    def __init__(self, cfg: Config):
        model = <build_nn_module>()
        super().__init__(cfg=cfg, model=model)
        self.eval_metrics = {"accuracy": accuracy}  # from evaluation.metrics
        # Load pretrained weights if available
        # Initialize task_counter, data state, aug_history

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def update_data_stream(self) -> None:
        # Increment task_counter
        # Apply drift simulation (e.g., affine transforms for images)
        # Rebuild data loaders with new transforms
        # Track augmentation history for replay

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        # Return (train_loader, val_loader) for current data
        # Call _dispose_current_loaders() first if loaders exist

    def get_hist_data_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        # Return historical data loaders for CL replay
        # Return (None, None) when task_counter == 1

    def get_criterion(self) -> CriterionFn:
        return nn.CrossEntropyLoss()
```

Key patterns from existing harnesses:
- `eval_metrics` must be `Dict[str, Callable[[Tensor, Tensor], scalar]]`
- Use `_dispose_current_loaders()` helper for memory cleanup before rebuilding loaders
- `update_data_stream()` increments `task_counter` and applies cumulative augmentation drift
- `get_hist_data_loaders()` returns `(None, None)` when `task_counter == 1` (no history yet)

### 3. `examples/$0/utils.py`
Dataset loading utilities following the MNIST pattern:
- `get_<name>_train()` / `get_<name>_val()` -- load raw dataset
- `FixedAffine` -- Custom transform for deterministic drift simulation
- `TransformedView` -- Dataset wrapper applying transforms
- `sample_aug(seed)` -- Sample random augmentation parameters deterministically
- `make_loader(dataset, batch_size, num_workers, shuffle)` -- DataLoader factory

### 4. `examples/$0/<name>.toml`
TOML config following the project convention. Set appropriate:
- `[model]` name and pretrained_path
- `[data]` name matching the factory branch
- `[train]` reasonable defaults for the dataset
- `[drift_detection]` with ADWINDetector defaults
- `[visualization]` with output paths

### 5. Update `examples/utils.py`
Add an `elif cfg.data.name == "<dataset_name>":` branch that imports and returns the new harness.

## Procedure

1. Create the `examples/$0/` directory.
2. Generate `__init__.py` (empty).
3. Generate `model.py` with the harness class. Ask the user about:
   - Model architecture (CNN, ViT, MLP, etc.)
   - Dataset source (torchvision, custom, HuggingFace, etc.)
   - Number of classes, input dimensions
   - Type of drift simulation appropriate for the domain
4. Generate `utils.py` with data loading utilities.
5. Generate the TOML config file.
6. Update `examples/utils.py` factory with the new branch.
7. Verify the import chain works:
   ```bash
   cd /home/user/BaseSim_Framework && poetry run python -c "from examples.utils import get_example; print('Factory imports OK')"
   ```

## Important Notes
- The model's `__init__` receives `cfg: Config` and must call `super().__init__(cfg=cfg, model=<nn.Module>)`
- `eval_metrics` must include at minimum an `accuracy` metric from `evaluation.metrics`
- For non-image datasets, adapt the `FixedAffine` pattern to domain-appropriate transforms (e.g., feature noise for tabular, token perturbation for text)
- Pretrained weights are optional -- the harness should handle missing weight files gracefully
