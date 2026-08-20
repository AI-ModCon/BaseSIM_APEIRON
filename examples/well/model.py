"""Streaming neural PDE surrogate over a drifting Well stream.

``PDESurrogate`` is a compact residual 2D-CNN that maps the fields at time ``t``
to the fields at ``t+1`` (next-step prediction). It normalizes internally using
per-channel statistics baked in at conversion time and predicts a residual, so it
operates in raw physical units externally -- training and evaluation are
consistent whether or not a batch passes through ``_unpack``.

``WellHarness`` wires that model to a committed ``WindowStore`` via
:class:`~apeiron.data.windowed_harness.WindowedHarness`, so the whole
drift-detection + continual-learning machinery (and the Phase-3 sharding) applies
unchanged. The monitored metric is VRMSE (lower is better); crossing from one
``tcool`` regime to the next is the drift the detector must catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, cast

import torch
import torch.nn as nn

from apeiron.config.configuration import Config
from apeiron.data.window_store import WindowStore
from apeiron.data.windowed_harness import WindowedHarness
from apeiron.evaluation.metrics import mae, vrmse
from examples.well.convert import WELL_META

_DEFAULT_WIDTH = 32
_DEFAULT_DEPTH = 4


class PDESurrogate(nn.Module):
    """Residual next-step field predictor: ``field_{t+1} = field_t + f(field_t)``."""

    def __init__(
        self,
        n_channels: int,
        mean: Sequence[float],
        std: Sequence[float],
        width: int = _DEFAULT_WIDTH,
        depth: int = _DEFAULT_DEPTH,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        )

        layers: list[nn.Module] = [
            nn.Conv2d(n_channels, width, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(1, depth) - 1):
            layers += [
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(1, width),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, n_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        x_norm = (x - mean) / std
        delta_norm = self.head(self.body(x_norm))
        return x + delta_norm * std  # residual, back in raw units


class WellHarness(WindowedHarness):
    """WindowedHarness for the Well next-step surrogate task."""

    def __init__(self, cfg: Config) -> None:
        store_path = cfg.data.window_store_path or cfg.data.path
        meta = json.loads((Path(store_path) / WELL_META).read_text())
        store = WindowStore(store_path, catalog=False)

        width = cfg.model.width or _DEFAULT_WIDTH
        depth = cfg.model.depth or _DEFAULT_DEPTH
        model = PDESurrogate(
            n_channels=meta["n_channels"],
            mean=meta["norm_mean"],
            std=meta["norm_std"],
            width=width,
            depth=depth,
        )

        super().__init__(
            cfg,
            model,
            store,
            criterion=nn.MSELoss(),
            optimizer_factory=lambda m: torch.optim.Adam(
                m.parameters(), lr=cfg.train.init_lr
            ),
            eval_metrics={"vrmse": vrmse, "mae": mae},
        )
        # VRMSE and MAE are error metrics: lower is better (drives BWT/FWT signs).
        self.higher_is_better = {"vrmse": False, "mae": False}
        self.meta = meta
