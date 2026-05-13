# Creating an Application Model Harness

This document describes how to create a custom model harness for Apeiron using `BaseModelHarness`.

A model harness is the integration layer between your application and the Apeiron continual-learning framework. It is responsible for:

- Managing data streams
- Providing dataloaders
- Configuring optimizers and loss functions
- Defining evaluation metrics
- Supporting checkpointing and drift evaluation

---

# Overview

To integrate a model into Apeiron, create a subclass of:

```python
from apeiron.model.torch_model_harness import BaseModelHarness
```

Your subclass adapts your application's:

- datasets
- models
- training streams
- evaluation logic

to Apeiron's runtime lifecycle.

---

# Required Lifecycle Methods

Your harness must implement the following methods.

| Method | Purpose |
|---|---|
| `get_optmizer()` | Return the optimizer |
| `update_data_stream()` | Refresh or replace active stream data |
| `get_stream_dataloader()` | Return continual-learning stream loader |
| `get_hist_dataloaders()` | Return historical train/validation loaders |
| `get_train_dataloaders()` | Return active train/validation loaders |
| `get_criterion()` | Return the training loss function |

---

# Minimal Harness Example

```python
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness


class ApplicationHarness(BaseModelHarness):

    def __init__(self, cfg: Config, model: nn.Module):
        super().__init__(cfg, model)

        self._train_loader = self._make_loader(split="train")
        self._val_loader = self._make_loader(split="val")

        self._hist_train_loader = None
        self._hist_val_loader = None

        self.eval_metrics["accuracy"] = self._accuracy

    def get_optmizer(self):
        return Adam(
            self.model.parameters(),
            lr=self.cfg.model.lr
        )

    def update_data_stream(self) -> None:
        """
        Refresh stream data.

        Replace this with application-specific logic.
        """

        self._train_loader = self._make_loader(split="train")
        self._val_loader = self._make_loader(split="val")

    def get_stream_dataloader(self) -> DataLoader:
        return self._train_loader

    def get_hist_dataloaders(self):
        return (
            self._hist_train_loader,
            self._hist_val_loader
        )

    def get_train_dataloaders(self):
        return (
            self._train_loader,
            self._val_loader
        )

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def _make_loader(self, split: str) -> DataLoader:

        x = torch.randn(128, 10)
        y = torch.randint(0, 2, (128,))

        dataset = TensorDataset(x, y)

        return DataLoader(
            dataset,
            batch_size=self.cfg.model.batch_size
        )

    @staticmethod
    def _accuracy(
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:

        preds = y_hat.argmax(dim=1)

        return (preds == y).float().mean()
```

---

# Constructor Requirements

Every harness constructor should:

1. Accept:
   - `Config`
   - `torch.nn.Module`

2. Call:

```python
super().__init__(cfg, model)
```

This initializes:

- device placement
- configuration access
- metric registry

Example:

```python
def __init__(self, cfg: Config, model: nn.Module):
    super().__init__(cfg, model)
```

---

# Optimizer Configuration

Implement:

```python
get_optmizer()
```

This method should return a PyTorch optimizer.

Example:

```python
def get_optmizer(self):
    return Adam(
        self.model.parameters(),
        lr=self.cfg.model.lr
    )
```

Parameter groups are supported:

```python
return Adam([
    {"params": backbone.parameters(), "lr": 1e-5},
    {"params": head.parameters(), "lr": 1e-3},
])
```

---

# Data Stream Management

Implement:

```python
update_data_stream()
```

This method is responsible for refreshing or replacing the current data stream.

Typical use cases:

- sliding windows
- simulated drift
- streaming inference
- periodic dataset refresh
- online learning

Example:

```python
def update_data_stream(self):
    self._train_loader = load_new_stream()
```

---

# Stream Dataloader

Implement:

```python
get_stream_dataloader()
```

This dataloader is used for continual learning.

Example:

```python
def get_stream_dataloader(self):
    return self._train_loader
```

---

# Historical Dataloaders

Implement:

```python
get_hist_dataloaders()
```

Used for:

- retention testing
- drift measurement
- historical evaluation

Expected return type:

```python
(train_loader, val_loader)
```

If no historical data exists:

```python
return (None, None)
```

---

# Train and Validation Dataloaders

Implement:

```python
get_train_dataloaders()
```

Expected return:

```python
(train_loader, val_loader)
```

The validation loader (`index 1`) is used internally by:

```python
eval()
history_eval()
```

Example:

```python
def get_train_dataloaders(self):
    return self._train_loader, self._val_loader
```

---

# Loss Functions

Implement:

```python
get_criterion()
```

Example:

```python
def get_criterion(self):
    return nn.CrossEntropyLoss()
```

Any PyTorch-compatible loss function is supported.

---

# Evaluation Metrics

Metrics are stored in:

```python
self.eval_metrics
```

Each metric must accept:

```python
(y_hat, y)
```

and return:

- tensor
- float
- scalar numeric value

Example:

```python
self.eval_metrics["accuracy"] = self._accuracy
```

Metric implementation:

```python
@staticmethod
def _accuracy(y_hat, y):
    preds = y_hat.argmax(dim=1)
    return (preds == y).float().mean()
```

---

# Batch Format

By default, batches are expected to be:

```python
(x, y)
```

If your dataloader returns:

- dictionaries
- metadata
- custom objects
- multimodal batches

override:

```python
_unpack()
```

Example:

```python
def _unpack(self, batch):

    x = batch["features"]
    y = batch["labels"]

    return x, y
```

---

# Evaluation Lifecycle

The framework provides:

```python
eval()
```

and:

```python
history_eval()
```

These methods:

- switch model to evaluation mode
- iterate over validation loaders
- compute registered metrics
- aggregate metric averages

No implementation is required unless custom behavior is needed.

---

# Checkpointing

Checkpointing is automatically enabled when:

```python
cfg.model.max_ckpts > 0
```

and:

```python
cfg.model.ckpts_path
```

are configured.

Save checkpoints with:

```python
save_ckpt(event)
```

Example:

```python
self.save_ckpt(event=4)
```

Generated checkpoint files:

```text
drift_adaptation_4.pt
```

A `latest` pointer file is also maintained automatically.

Older checkpoints are removed once the retention limit is exceeded.

---

# Recommended Project Structure

Example application layout:

```text
application/
├── data/
├── models/
├── harness/
│   └── application_harness.py
├── training/
└── configs/
```

---

# Best Practices

## Keep stream logic isolated

Avoid embedding stream refresh logic throughout the application.

Prefer:

```python
update_data_stream()
```

as the single source of truth.

---

## Use `_unpack()` for compatibility

Avoid hardcoding batch assumptions in evaluation or training loops.

Override `_unpack()` instead.

---

## Register metrics once

Register metrics during initialization:

```python
self.eval_metrics["f1"] = self._f1
```

Avoid re-registering metrics dynamically.

---

## Keep loaders persistent

Avoid reconstructing datasets unnecessarily unless drift or stream updates require it.
