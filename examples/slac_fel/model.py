# examples/slac_fel/model.py
"""SLAC FEL model harness for the BaseSim continuous-learning framework.

This harness wraps a 7-layer ELU regression network trained to predict
HXR pulse intensity from accelerator settings.  This example uses real temporal drift:
the pre-processed accelerator data is chronologically sorted and sliced into time windows,
each of which is served in order by ``update_data_stream()``.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness

from examples.slac_fel.utils import (
    FELDataset,
    load_fel_data,
    load_scalers,
    make_loader,
    split_into_windows,
)


# ---------------------------------------------------------------------------
# Neural-network architecture  (matches training script at 62c1a89)
# ---------------------------------------------------------------------------
class FELNet(nn.Module):
    """7-layer ELU regression network for FEL pulse-intensity prediction.

    Architecture (from ``train_fel_model.py``):
        Linear → ELU  ×3  →  Linear → ELU → Dropout  ×2
        → Linear → ELU → Dropout → Linear → ELU → Linear(out)
    """

    def __init__(self, input_size: int, output_size: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Dropout(p=0.05),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Regression metrics (match MetricFn  =  Callable[[Tensor, Tensor], Any])
# ---------------------------------------------------------------------------
@torch.no_grad()
def mse_metric(y_hat: Tensor, y: Tensor) -> Tensor:
    """Mean-squared error computed on the batch."""
    return F.mse_loss(y_hat, y)


@torch.no_grad()
def r2_metric(y_hat: Tensor, y: Tensor) -> Tensor:
    """Batch R² (coefficient of determination).

    R² = 1 − SS_res / SS_tot.  Returns a scalar tensor.
    """
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ---------------------------------------------------------------------------
# Model harness
# ---------------------------------------------------------------------------

# Fraction of each time window reserved for validation
_VAL_FRACTION: float = 0.2


class SLAC_FEL(BaseModelHarness):
    """Continuous-learning harness for the LCLS FEL regression model.

    The data path (``cfg.data.path``) must point to a directory containing:

    * ``data.pkl``            – pre-filtered, chronologically-sorted DataFrame
    * ``input_scaler.pt``     – BoTorch ``AffineInputTransform`` for inputs
    * ``output_scaler.pt``    – BoTorch ``AffineInputTransform`` for the target
    * ``feature_config.yml``  – YAML listing input / output variable names

    Each call to :meth:`update_data_stream` advances to the next time window.
    """

    def __init__(self, cfg: Config) -> None:
        # ----- load data & split into windows --------------------------------
        X, y, self.timestamps = load_fel_data(cfg.data.path, device=cfg.device)
        self.input_scaler, self.output_scaler = load_scalers(
            cfg.data.path, device=cfg.device
        )

        input_size = X.shape[1]
        output_size = y.shape[1]

        self.windows = split_into_windows(X, y)

        # ----- build model ---------------------------------------------------
        model = FELNet(input_size=input_size, output_size=output_size)

        super().__init__(cfg=cfg, model=model)

        # ----- load pretrained weights (optional) ----------------------------
        pretrained_path = cfg.model.pretrained_path
        if pretrained_path:
            try:
                state = torch.load(
                    pretrained_path, map_location=cfg.device, weights_only=False
                )
                # Handle state_dict vs full model save
                if isinstance(state, dict):
                    # Strip '_orig_mod.' prefix if present (from torch.compile)
                    cleaned: Dict[str, Any] = {}
                    for k, v in state.items():
                        key = (
                            k.replace("_orig_mod.", "")
                            if k.startswith("_orig_mod.")
                            else k
                        )
                        # Handle 'net.' prefix mismatch
                        cleaned[key] = v
                    self.model.load_state_dict(cleaned, strict=False)
                else:
                    # Full model was saved with torch.save(model, ...)
                    self.model.load_state_dict(state.state_dict(), strict=False)
                print(f"Loaded pretrained FEL model from {pretrained_path}")
            except FileNotFoundError:
                print(
                    f"Warning: Pretrained model not found at {pretrained_path}, "
                    "using randomly initialized weights"
                )
            except Exception as e:
                print(f"Warning: Failed to load pretrained FEL model: {e}")

        # ----- eval metrics (regression) -------------------------------------
        self.eval_metrics = {"mse": mse_metric, "r2": r2_metric}
        self.higher_is_better = {"mse": False, "r2": True}

        # ----- streaming state -----------------------------------------------
        self.window_idx: int = 0
        self.history_windows: List[Tuple[Tensor, Tensor]] = []

        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None

    # --------------------------------------------------------------------- #
    # Required overrides
    # --------------------------------------------------------------------- #

    def get_optmizer(self) -> Optimizer:  # noqa: D102  (spelling kept for ABC)
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self):  # noqa: D102
        return nn.MSELoss()

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:  # noqa: D102
        assert self._cur_train_loader is not None and self._cur_val_loader is not None
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Return loaders over all previously-seen time windows.

        Returns ``(None, None)`` until at least two windows have been served.
        """
        if self.window_idx <= 1:
            return None, None

        # Concatenate all history windows
        hist_train_views: List[FELDataset] = []
        hist_val_views: List[FELDataset] = []

        for X_w, y_w in self.history_windows:
            n = X_w.shape[0]
            n_val = max(1, int(n * _VAL_FRACTION))
            n_train = n - n_val
            hist_train_views.append(FELDataset(X_w[:n_train], y_w[:n_train]))
            hist_val_views.append(FELDataset(X_w[n_train:], y_w[n_train:]))

        ds_hist_train: ConcatDataset[Any] = ConcatDataset(hist_train_views)
        ds_hist_val: ConcatDataset[Any] = ConcatDataset(hist_val_views)

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        return (
            make_loader(
                ds_hist_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
            ),
            make_loader(ds_hist_val, bs, shuffle=False, num_workers=nw, pin_memory=pin),
        )

    def update_data_stream(self) -> None:
        """Advance to the next chronological time window.

        The current window is added to the history, and new train/val loaders
        are built from the upcoming window.
        """
        self._dispose_current_loaders()

        if self.window_idx >= len(self.windows):
            print(
                f"Warning: All {len(self.windows)} time windows exhausted; "
                "wrapping around to the first window."
            )
            self.window_idx = 0

        X_w, y_w = self.windows[self.window_idx]

        # Archive previous window in history (skip the very first call)
        if self.window_idx > 0:
            prev_X, prev_y = self.windows[self.window_idx - 1]
            # Only add if not already stored (idempotency guard)
            if len(self.history_windows) < self.window_idx:
                self.history_windows.append((prev_X, prev_y))

        # Train / val split (last _VAL_FRACTION chronologically)
        n = X_w.shape[0]
        n_val = max(1, int(n * _VAL_FRACTION))
        n_train = n - n_val

        ds_train = FELDataset(X_w[:n_train], y_w[:n_train])
        ds_val = FELDataset(X_w[n_train:], y_w[n_train:])

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        self._cur_train_loader = make_loader(
            ds_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        self._cur_val_loader = make_loader(
            ds_val, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )

        print(
            f"[SLAC-FEL] Window {self.window_idx + 1}/{len(self.windows)}: "
            f"{n_train} train / {n_val} val samples"
        )
        self.window_idx += 1

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _dispose_current_loaders(self) -> None:
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()
