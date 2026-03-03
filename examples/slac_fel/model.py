# examples/slac_fel/model.py
"""SLAC FEL model harness for the BaseSim continuous-learning framework.

This harness wraps a 7-layer ELU regression network with dropout regularisation,
trained to predict HXR pulse intensity from accelerator settings. The pre-processed
accelerator data is chronologically sorted and sliced into time windows,
each of which is served in order by ``update_data_stream()``.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness

from examples.slac_fel.utils import (
    FELDataset,
    discover_window_files,
    load_feature_config,
    load_fel_data,
    load_scalers,
    load_window_file,
    make_loader,
    split_into_windows,
)


_log = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Neural-network architecture  (matches FELNeuralNetwork in train_fel_model.py from commit  4e1f676
# -------------------------------------------------------------------------------------------------
class FELNet(nn.Module):
    """7-layer fully-connected ELU regression network.
    Predicts FEL pulse intensity from scaled accelerator inputs."""

    def __init__(self, input_size=None, output_size=1):
        super(FELNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
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
        if not torch.isfinite(x).all():
            n_bad = (~torch.isfinite(x)).sum().item()
            _log.warning(
                "FELNet.forward: %d non-finite values in input. Replacing with 0",
                n_bad,
            )
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        out = self.net(x)

        # Detect NaN outputs
        if not torch.isfinite(out).all():
            if torch.isnan(out).any():
                _log.warning("Model produced NaN predictions. Replacing with 0.")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

        return out


# ---------------------------------------------------------------------------
# Regression metrics (match MetricFn  =  Callable[[Tensor, Tensor], Any])
# ---------------------------------------------------------------------------
@torch.no_grad()
def mse_metric(y_hat: Tensor, y: Tensor) -> Tensor:
    """Mean-squared error computed on the batch."""
    return F.mse_loss(y_hat, y)


# ---------------------------------------------------------------------------
# Model harness
# ---------------------------------------------------------------------------

# Fraction of each time window reserved for validation
_VAL_FRACTION: float = 0.2


class SLAC_FEL(BaseModelHarness):
    """Continuous-learning harness for the LCLS FEL regression model.

    The data path (``cfg.data.path``) must point to a directory containing:

    * ``data_1.pkl``, ``data_2.pkl``, … – pre-filtered, chronologically-sorted
      DataFrames, one per time window.  Each file is loaded on demand by
      :meth:`update_data_stream` (preferred, memory-efficient layout).
    * **OR** ``data.pkl`` – a single monolithic DataFrame that will be split
      into fixed-size windows via ``cfg.data.window_size`` (legacy fallback).

     The data path (``cfg.model``) must point to a directory containing:
    * ``input_scaler.pt``             – BoTorch ``AffineInputTransform`` for inputs
    * ``output_scaler.pt``            – BoTorch ``AffineInputTransform`` for the target
    * ``feature_config.yml``          – YAML listing input / output variable names
    * ``model_pretrained.pt`` (optional) – pretrained checkpoint for the FELNet model

    Each call to :meth:`update_data_stream` advances to the next time window.
    """

    def __init__(self, cfg: Config) -> None:
        # ----- scalers & feature config (always needed) ----------------------
        self.input_scaler, self.output_scaler = load_scalers(
            cfg.model.config_path, device=cfg.device
        )
        self.input_cols, self.output_cols = load_feature_config(cfg.model.config_path)

        # ----- discover per-file windows or fall back to monolithic ----------
        self._window_files = discover_window_files(cfg.data.path)
        self._lazy = len(self._window_files) > 0

        if self._lazy:
            # Per-file mode: each pickle is one window, loaded on demand.
            # Load the first file just to determine tensor dimensions.
            X0, y0 = load_window_file(
                self._window_files[0],
                self.input_cols,
                self.output_cols,
                self.input_scaler,
                self.output_scaler,
            )
            input_size = X0.shape[1]
            output_size = y0.shape[1]
            self.num_windows = len(self._window_files)
            # Pre-loaded windows list is not used in lazy mode;
            # we keep a reference only for the first window so that the
            # very first update_data_stream call is free.
            self._prefetched_window: Optional[Tuple[Tensor, Tensor]] = (X0, y0)
            self.windows: List[Tuple[Tensor, Tensor]] = []  # unused in lazy mode
            print(
                f"[SLAC-FEL] Lazy mode: {self.num_windows} window files found "
                f"(input_dim={input_size}, output_dim={output_size})"
            )
        else:
            # Legacy mode: single data.pkl split into fixed-size windows.
            X, y, self.timestamps = load_fel_data(cfg.data.path, device=cfg.device)
            input_size = X.shape[1]
            output_size = y.shape[1]
            self.windows = split_into_windows(X, y, window_size=cfg.data.window_size)
            self.num_windows = len(self.windows)
            self._prefetched_window = None
            print(
                f"[SLAC-FEL] Legacy mode: {self.num_windows} windows "
                f"(window_size={cfg.data.window_size}, "
                f"input_dim={input_size}, output_dim={output_size})"
            )

        # ----- build model ---------------------------------------------------
        pretrained_path = cfg.model.pretrained_path
        if pretrained_path:
            model = self._load_pretrained_direct(
                pretrained_path, input_size, output_size, cfg.device
            )
        else:
            model = FELNet(input_size=input_size, output_size=output_size)

        super().__init__(cfg=cfg, model=model)

        # ----- eval metrics (regression) -------------------------------------
        self.eval_metrics = {"mse": mse_metric}
        self.higher_is_better = {"mse": False}

        # ----- streaming state -----------------------------------------------
        self.window_idx: int = 0
        self.history_windows: List[Tuple[Tensor, Tensor]] = []
        self._current_window: Optional[Tuple[Tensor, Tensor]] = None

        # Cap history to prevent unbounded memory growth
        self.max_history_windows: int = 200

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

        In lazy mode each call loads the next ``data_<N>.pkl`` file from disk.
        In legacy mode the pre-split in-memory windows are used.
        """
        self._dispose_current_loaders()

        # ── Archive the *previous* window into history ────────────────────
        # We saved a reference in _current_window during the last call so
        # we never have to re-load it from disk.
        if self._current_window is not None:
            self.history_windows.append(self._current_window)
            self._current_window = None
            # Evict oldest windows when cap is reached
            while len(self.history_windows) > self.max_history_windows:
                self.history_windows.pop(0)

        if self.window_idx >= self.num_windows:
            print(
                f"Warning: All {self.num_windows} time windows exhausted; "
                "wrapping around to the first window."
            )
            self.window_idx = 0

        # ── Load the window tensors ──────────────────────────────────────
        if self._lazy:
            # Use the prefetched first window if available
            if self._prefetched_window is not None and self.window_idx == 0:
                X_w, y_w = self._prefetched_window
                self._prefetched_window = None  # free after first use
            else:
                X_w, y_w = load_window_file(
                    self._window_files[self.window_idx],
                    self.input_cols,
                    self.output_cols,
                    self.input_scaler,
                    self.output_scaler,
                )
        else:
            X_w, y_w = self.windows[self.window_idx]

        # Keep a reference so the next call can archive it without reloading
        self._current_window = (X_w, y_w)

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
            f"[SLAC-FEL] Window {self.window_idx + 1}/{self.num_windows}: "
            f"{n_train} train / {n_val} val samples"
        )
        self.window_idx += 1

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _load_pretrained_direct(
        path: str, input_size: int, output_size: int, device: str
    ) -> FELNet:
        """Load a pretrained checkpoint directly with no weight modifications.

        Supports two save formats produced by ``train_fel_model.py``:

        1. A raw ``nn.Sequential`` (saved via ``torch.save(model.net, ...)``).
           The Sequential is wrapped inside a new :class:`FELNet` whose
           architecture is defined entirely by the checkpoint.
        2. A ``state_dict`` (plain ``dict``).  The input dimension is inferred
           from the first Linear layer's weight shape so that :class:`FELNet`
           is constructed to match exactly, then ``load_state_dict`` is called
           with ``strict=True``.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        RuntimeError
            If the checkpoint shapes are incompatible with the data (e.g. the
            data has a different number of input features than the model
            expects).
        """
        state = torch.load(path, map_location=device, weights_only=False)

        if isinstance(state, nn.Sequential):
            # Format 1: checkpoint is the raw nn.Sequential
            # Infer input/output dims from the first and last Linear layers
            first_linear = next(m for m in state.modules() if isinstance(m, nn.Linear))
            last_linear = list(m for m in state.modules() if isinstance(m, nn.Linear))[
                -1
            ]
            ckpt_in = first_linear.in_features
            ckpt_out = last_linear.out_features

            if ckpt_in != input_size:
                raise RuntimeError(
                    f"Pretrained model expects {ckpt_in} input features but "
                    f"the data has {input_size}.  Ensure the feature_config.yml "
                    f"and scalers match the checkpoint."
                )

            model = FELNet(input_size=ckpt_in, output_size=ckpt_out)
            model.net.load_state_dict(state.state_dict(), strict=True)
            print(f"Loaded pretrained FEL model (nn.Sequential) from {path}")

        elif isinstance(state, dict):
            # Format 2: checkpoint is a state_dict
            # Strip torch.compile artefact from keys
            sd = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

            # Infer input dim from the first weight tensor
            first_weight_key = next(
                k for k in sd if k.endswith(".weight") and sd[k].dim() == 2
            )
            ckpt_in = sd[first_weight_key].shape[1]

            if ckpt_in != input_size:
                raise RuntimeError(
                    f"Pretrained model expects {ckpt_in} input features but "
                    f"the data has {input_size}.  Ensure the feature_config.yml "
                    f"and scalers match the checkpoint."
                )

            model = FELNet(input_size=ckpt_in, output_size=output_size)
            model.load_state_dict(sd, strict=True)
            print(f"Loaded pretrained FEL model (state_dict) from {path}")

        else:
            raise TypeError(
                f"Unexpected checkpoint type {type(state).__name__} from {path}. "
                f"Expected nn.Sequential or state_dict."
            )

        return model

    def _dispose_current_loaders(self) -> None:
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()
