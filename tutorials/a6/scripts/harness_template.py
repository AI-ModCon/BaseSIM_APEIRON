"""A complete model harness for a drifting regression stream -- copy it, then swap in your problem.

It runs as-is on synthetic data so you can see a working harness end to end:

    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_custom.toml \
        --harness tutorials/a6/scripts/harness_template.py:DriftingRegressionHarness

To adapt it to your own work, change the three places marked  >>> YOUR PROBLEM <<< :
  1. build_model()      -- your PyTorch model (or load your trained checkpoint)
  2. load_window(t)     -- return (inputs, targets) for stream window t
  3. eval_metrics       -- the scores Apeiron should monitor (first one is metric_index 0)

What Apeiron needs from a harness (see docs/model_harness.md in the repo):
  update_data_stream()     move to the next window of data
  get_stream_dataloader()  batches the model is scored on while monitoring
  get_train_dataloaders()  (train, val) loaders for the current window, used when adapting
  get_hist_dataloaders()   (train, val) loaders over past windows, used for replay; (None, None) if none
  get_optimizer()          optimizer for adaptation
  get_criterion()          loss function
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness


def mae(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y_hat - y).abs().mean()


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((y_hat - y) ** 2).mean()


class DriftingRegressionHarness(BaseModelHarness):
    N_FEATURES = 8
    N_TRAIN, N_VAL, N_STREAM = 2048, 512, 2048  # samples per window
    MAX_HISTORY = 4  # past windows kept for replay

    def __init__(self, cfg: Config):
        torch.manual_seed(cfg.seed)
        self._gen = torch.Generator().manual_seed(cfg.seed)
        self._w0 = torch.randn(self.N_FEATURES, generator=self._gen)
        self._w_dir = torch.randn(self.N_FEATURES, generator=self._gen)

        self.cfg = cfg  # build_model() may read the config (e.g. a checkpoint path)
        super().__init__(cfg=cfg, model=self.build_model())

        # >>> YOUR PROBLEM <<< (3): what to monitor. The first entry is metric_index = 0.
        self.eval_metrics = {"mae": mae, "mse": mse}
        self.higher_is_better = {"mae": False, "mse": False}

        self.t = -1  # current window index; update_data_stream() moves to 0 first
        self._history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._current_train_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
        # Set on every update_data_stream() call; Apeiron calls it before using them.
        self._train: DataLoader
        self._val: DataLoader
        self._stream: DataLoader

        self._pretrain()

    # ------------------------------------------------------------------ your problem
    def build_model(self) -> nn.Module:
        # >>> YOUR PROBLEM <<< (1): return your model. To start from a trained checkpoint:
        #     model = MyNet(); model.load_state_dict(torch.load(self.cfg.model.pretrained_path)); return model
        return nn.Sequential(
            nn.Linear(self.N_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def load_window(self, t: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # >>> YOUR PROBLEM <<< (2): return n samples (x, y) from stream window t.
        # For real data this is where you read the next file, time slice, shot range,
        # simulation batch, ... The synthetic stream below drifts in two ways at once:
        #   covariate shift -- the inputs move away from the training range, and
        #   concept drift   -- the input -> output relationship itself rotates.
        shift = 0.35 * t
        w = self._w0 + 0.25 * t * self._w_dir
        x = torch.randn(n, self.N_FEATURES, generator=self._gen) + shift
        y = torch.tanh(x @ w / math.sqrt(self.N_FEATURES)) + 0.3 * torch.sin(
            x[:, :1]
        ).squeeze(1)
        y = y + 0.05 * torch.randn(n, generator=self._gen)
        return x, y.unsqueeze(1)

    # ------------------------------------------------------------------ Apeiron interface
    def update_data_stream(self) -> None:
        if self._current_train_tensors is not None:  # current window becomes history
            self._history.append(self._current_train_tensors)
            self._history = self._history[-self.MAX_HISTORY :]
        self.t += 1
        x_tr, y_tr = self.load_window(self.t, self.N_TRAIN)
        x_va, y_va = self.load_window(self.t, self.N_VAL)
        x_st, y_st = self.load_window(self.t, self.N_STREAM)
        self._current_train_tensors = (x_tr, y_tr)

        # Adaptation skips training batches smaller than [train] batch_size, so keep only
        # full batches (drop_last) and make sure a window holds at least one batch.
        bs = self.cfg.train.batch_size
        assert self.N_TRAIN >= bs, (
            "each window needs at least [train] batch_size samples"
        )
        self._train = DataLoader(
            TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True, drop_last=True
        )
        self._val = DataLoader(TensorDataset(x_va, y_va), batch_size=bs)
        self._stream = DataLoader(
            TensorDataset(x_st, y_st), batch_size=self.cfg.data.batch_size
        )

    def get_stream_dataloader(self) -> DataLoader:
        return self._stream

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return self._train, self._val

    def get_hist_dataloaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if not self._history:
            return None, None
        ds = ConcatDataset([TensorDataset(x, y) for x, y in self._history])
        n_val = min(len(ds), self.N_VAL)
        bs = self.cfg.train.batch_size
        train = DataLoader(
            ds, batch_size=bs, shuffle=True, drop_last=True
        )  # full batches only, as above
        val = DataLoader(torch.utils.data.Subset(ds, range(n_val)), batch_size=bs)
        return train, val

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self):
        return nn.MSELoss()

    # ------------------------------------------------------------------ helpers
    def _pretrain(self, steps: int = 400) -> None:
        """Stand-in for 'the model you already trained': fit window 0 once, then deploy.

        Delete this if build_model() loads a trained checkpoint.
        """
        x, y = self.load_window(0, self.N_TRAIN)
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(x), (128,), generator=self._gen)
            opt.zero_grad()
            loss_fn(
                self.model(x[idx].to(self.cfg.device)), y[idx].to(self.cfg.device)
            ).backward()
            opt.step()
        self.model.eval()
