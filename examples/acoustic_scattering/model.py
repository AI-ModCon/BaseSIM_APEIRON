"""Acoustic scattering maze harness — next-frame prediction with DenseViT."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from apeiron.config.configuration import Config
from apeiron.evaluation.metrics import vrmse
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from examples.acoustic_scattering.src.scoring import build_scorer
from examples.acoustic_scattering.src.utils import (
    FramePairDataset,
    SelectiveFramePairDataset,
    compute_trajectory_complexity,
    load_prepared,
    make_loader,
    sort_by_complexity,
    split_into_brackets,
    split_test_set,
)
from examples.acoustic_scattering.src.vit_dense import vit_dense_base, vit_dense_small


def _build_model(cfg: Config) -> nn.Module:
    name = cfg.model.name.lower()
    if name == "vit_dense_small":
        return vit_dense_small()
    return vit_dense_base()


class ACOUSTIC_SCATTERING(BaseModelHarness):
    """Complexity-ordered acoustic-scattering next-frame prediction.

    The dataset is a single .pt tensor (seq, len, x, y).  Trajectories are
    sorted by spatial variance (simple -> complex) and divided into contiguous
    brackets.  Each call to ``update_data_stream`` advances to the next
    bracket, simulating temporal drift from easy to hard mazes.

    When ``cfg.data.scoring_strategy`` is not ``"none"``, samples within each
    bracket are scored by prediction error or uncertainty, and only the
    top ``selection_ratio`` fraction is used for training and replay.
    """

    def __init__(self, cfg: Config):
        model = _build_model(cfg)
        super().__init__(cfg=cfg, model=model)

        # Load pretrained weights if available
        try:
            state_dict = torch.load(
                cfg.model.pretrained_path,
                map_location=cfg.device,
                weights_only=False,
            )
            self.model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {cfg.model.pretrained_path}")
        except FileNotFoundError:
            print(
                f"Warning: Pretrained model not found at "
                f"{cfg.model.pretrained_path}, using random init"
            )
        except Exception as e:
            print(f"Warning: Failed to load pretrained model: {e}")

        # --- dataset & complexity ordering ---
        self.tensor_data, precomputed_complexity, _meta = load_prepared(cfg.data.path)
        complexities = compute_trajectory_complexity(
            self.tensor_data, precomputed=precomputed_complexity
        )
        sorted_ids = sort_by_complexity(complexities)

        # Hold out test trajectories (uniform across complexity) if requested
        self.test_traj_ids: List[int] = []
        if cfg.data.test_fraction > 0:
            train_ids_all, self.test_traj_ids = split_test_set(
                sorted_ids, cfg.data.test_fraction, seed=cfg.seed
            )
        else:
            train_ids_all = sorted_ids

        self.brackets = split_into_brackets(train_ids_all, cfg.data.n_brackets)

        self.bracket_ptr = 0
        self.task_counter = 0

        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None
        self._cur_stream_loader: Optional[DataLoader] = None

        # Metrics: lower is better for both
        self.eval_metrics: Dict[str, Any] = {
            "vrmse": vrmse,
            "loss": self.get_criterion(),
        }
        self.higher_is_better: Dict[str, bool] = {
            "vrmse": False,
            "loss": False,
        }

        # --- active selection ---
        self.scorer = build_scorer(
            cfg.data.scoring_strategy, mc_samples=cfg.data.mc_samples
        )
        self.selection_ratio: float = cfg.data.selection_ratio
        self.scoring_profiler = FLOPSProfiler()

        # Memory pool: trajectory indices kept from previous brackets for replay
        self.memory_pool: List[int] = []

        # Cumulative CL FLOP counter across all drift events
        self.cumulative_cl_flops: float = 0.0

    # ------------------------------------------------------------------
    # Scoring & selection
    # ------------------------------------------------------------------

    def _score_and_select(self, dataset: FramePairDataset) -> List[int]:
        """Score every sample in *dataset* and return indices of the top fraction.

        The FLOP cost of scoring is tracked under the ``"scoring"`` tag via
        the harness's dedicated ``scoring_profiler``.
        """
        device = self.cfg.device
        with self.scoring_profiler.measure_flops(tag="scoring"):
            scores = self.scorer.score(self.model, dataset, device)  # type: ignore[union-attr]

        k = max(1, int(len(dataset) * self.selection_ratio))
        _, top_indices = scores.topk(k)
        return sorted(top_indices.tolist())

    # ------------------------------------------------------------------
    # Loader lifecycle
    # ------------------------------------------------------------------

    def _dispose_current_loaders(self) -> None:
        for attr in ("_cur_train_loader", "_cur_val_loader", "_cur_stream_loader"):
            if getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()

    def update_data_stream(self) -> None:
        self._dispose_current_loaders()

        bracket_indices = self.brackets[self.bracket_ptr]
        self.bracket_ptr = min(self.bracket_ptr + 1, len(self.brackets) - 1)

        # 80/20 train/val split within bracket
        n = len(bracket_indices)
        split = max(1, int(0.8 * n))
        train_ids = bracket_indices[:split]
        val_ids = bracket_indices[split:] if split < n else bracket_indices[-1:]

        bs = self.cfg.train.batch_size
        stream_bs = self.cfg.data.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        ds_full_train = FramePairDataset(self.tensor_data, train_ids)
        ds_val = FramePairDataset(self.tensor_data, val_ids)

        # Active selection: score and filter training data
        if self.scorer is not None:
            selected = self._score_and_select(ds_full_train)
            ds_train = SelectiveFramePairDataset(ds_full_train, selected)
        else:
            ds_train = ds_full_train  # type: ignore[assignment]

        self._cur_train_loader = make_loader(
            ds_train, bs, shuffle=True, num_workers=nw, pin_memory=pin
        )
        # Val and stream loaders stay on the full bracket (unbiased evaluation)
        self._cur_val_loader = make_loader(
            ds_val, bs, shuffle=False, num_workers=nw, pin_memory=pin
        )
        self._cur_stream_loader = make_loader(
            ds_val, stream_bs, shuffle=True, num_workers=nw, pin_memory=pin
        )

        # Accumulate memory pool with current bracket's training indices
        self.memory_pool.extend(train_ids)
        self.task_counter += 1

    # ------------------------------------------------------------------
    # BaseModelHarness interface
    # ------------------------------------------------------------------

    def get_stream_dataloader(self) -> DataLoader:
        return self._cur_stream_loader  # type: ignore[return-value]

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return self._cur_train_loader, self._cur_val_loader  # type: ignore[return-value]

    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if self.task_counter <= 1:
            return None, None

        if not self.memory_pool:
            return None, None

        bs = self.cfg.train.batch_size
        nw = self.cfg.train.num_workers
        pin = torch.cuda.is_available()

        ds_full = FramePairDataset(self.tensor_data, self.memory_pool)

        if self.scorer is not None:
            # Re-score memory with current model and prune low-utility samples
            selected = self._score_and_select(ds_full)

            # Prune memory_pool: keep only trajectory indices whose
            # frame-pairs survived scoring.
            surviving_traj_ids: set[int] = set()
            for si in selected:
                pair_traj_idx, _ = ds_full.pairs[si]
                surviving_traj_ids.add(pair_traj_idx)
            self.memory_pool = [t for t in self.memory_pool if t in surviving_traj_ids]

            # Train/val split over the selected indices
            n_sel = len(selected)
            split = max(1, int(0.8 * n_sel))
            ds_train: Dataset = SelectiveFramePairDataset(ds_full, selected[:split])
            ds_val: Dataset = SelectiveFramePairDataset(
                ds_full, selected[split:] if split < n_sel else selected[-1:]
            )
        else:
            # No scoring: split trajectory-level, same as original behaviour
            all_ids = self.memory_pool
            n_ids = len(all_ids)
            id_split = max(1, int(0.8 * n_ids))
            train_traj = all_ids[:id_split]
            val_traj = all_ids[id_split:] if id_split < n_ids else all_ids[-1:]
            ds_train = FramePairDataset(self.tensor_data, train_traj)
            ds_val = FramePairDataset(self.tensor_data, val_traj)

        return (
            make_loader(ds_train, bs, shuffle=True, num_workers=nw, pin_memory=pin),
            make_loader(ds_val, bs, shuffle=False, num_workers=nw, pin_memory=pin),
        )

    # ------------------------------------------------------------------
    # Test-set evaluation & data budget
    # ------------------------------------------------------------------

    def get_test_dataloader(self) -> Optional[DataLoader]:
        """Build a DataLoader over the held-out test trajectories."""
        if not self.test_traj_ids:
            return None
        ds = FramePairDataset(self.tensor_data, self.test_traj_ids)
        return make_loader(
            ds,
            self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @torch.no_grad()
    def final_evaluation(self) -> Dict[str, float]:
        """Evaluate the model on the held-out test set.

        Returns a dict of metric names → scalar values.  Returns ``{}``
        if no test set was configured.
        """
        loader = self.get_test_dataloader()
        if loader is None:
            return {}

        self.model.eval()
        device = self.cfg.device

        # Accumulate per-batch metric sums
        metric_sums: Dict[str, float] = {k: 0.0 for k in self.eval_metrics}
        n_batches = 0

        for batch in loader:
            x, y = self._unpack(batch)
            x, y = x.to(device), y.to(device)
            y_hat = self.model(x)
            for key, metric_fn in self.eval_metrics.items():
                metric_sums[key] += self._to_scalar(metric_fn(y_hat, y))
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metric_sums.items()}

    def get_data_budget(self) -> int:
        """Total unique frame-pairs across all training brackets.

        This is the number of samples Track 1 (offline) must match so
        that both tracks see the same amount of data.
        """
        total = 0
        for bracket in self.brackets:
            total += len(FramePairDataset(self.tensor_data, bracket))
        return total

    # ------------------------------------------------------------------
    # Scoring FLOP bookkeeping
    # ------------------------------------------------------------------

    def accumulate_scoring_flops(self) -> None:
        """Add scoring FLOPs from the profiler to the cumulative counter."""
        perf = self.scoring_profiler.get_performance()
        scoring_flops = perf.get("scoring_flop", 0.0)
        if scoring_flops > 0:
            # Sum all recorded scoring flops (profiler stores averages,
            # so multiply by number of measurements)
            n_measurements = len(
                self.scoring_profiler.profiles.get("scoring", {}).get("flop", [])
            )
            self.cumulative_cl_flops += scoring_flops * n_measurements

    def get_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)
