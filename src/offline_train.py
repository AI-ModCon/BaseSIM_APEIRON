"""Offline supervised training baseline (Track 1).

Trains a model on randomly sampled frame-pairs drawn from all training
brackets — no curriculum, no drift detection, no continual learning.
Produces a ``results.json`` comparable to the CL pipeline's output.

Usage::

    poetry run python -m src.offline_train \
        --config examples/acoustic_scattering/acoustic_scattering_offline.toml \
        --set data.path=<data.pt>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from apeiron.config.configuration import build_config, Config
from apeiron.profilers import FLOPSProfiler

from examples.acoustic_scattering.model import ACOUSTIC_SCATTERING
from examples.acoustic_scattering.src.utils import (
    FramePairDataset,
    SelectiveFramePairDataset,
    make_loader,
)


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    profiler: FLOPSProfiler,
) -> float:
    """One epoch of supervised training. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with profiler.measure_flops(tag="forward"):
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    metrics: Dict[str, Any],
    device: str,
) -> Dict[str, float]:
    """No-grad evaluation. Returns dict of metric_name → mean value."""
    model.eval()
    metric_sums: Dict[str, float] = {k: 0.0 for k in metrics}
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        for key, metric_fn in metrics.items():
            val = metric_fn(y_hat, y)
            metric_sums[key] += val.item() if hasattr(val, "item") else float(val)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metric_sums.items()}


def save_results(
    cfg: Config,
    test_metrics: Dict[str, float],
    profiler: FLOPSProfiler,
    epochs: int,
    data_budget: int,
) -> Path:
    """Write results.json to the checkpoint directory."""
    fwd_flops = sum(profiler.profiles.get("forward", {}).get("flop", []))

    results = {
        "track": "offline",
        "test_metrics": test_metrics,
        "flops": {
            "train": fwd_flops,
            "scoring": 0.0,
            "inference": 0.0,
            "total": fwd_flops,
        },
        "epochs": epochs,
        "data_budget": data_budget,
    }

    out_dir = Path(cfg.model.ckpts_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")
    return out_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    harness = ACOUSTIC_SCATTERING(cfg)

    # 1. Data budget: total frame-pairs the CL pipeline would see
    data_budget = harness.get_data_budget()
    print(f"Data budget (from brackets): {data_budget} frame-pairs")

    # 2. Build flat dataset over ALL train trajectories
    all_train_ids: list[int] = []
    for bracket in harness.brackets:
        all_train_ids.extend(bracket)
    full_ds = FramePairDataset(harness.tensor_data, all_train_ids)

    # 3. Random sample: data_budget for training + 20% extra for validation
    rng = torch.Generator().manual_seed(cfg.seed)
    n_available = len(full_ds)
    n_val = max(1, int(data_budget * 0.25))  # 20% of total (= 25% of train)
    n_total = min(data_budget + n_val, n_available)
    perm = torch.randperm(n_available, generator=rng)[:n_total]

    train_ds = SelectiveFramePairDataset(full_ds, perm[:data_budget].tolist())
    val_ds = SelectiveFramePairDataset(full_ds, perm[data_budget:].tolist())

    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    pin = torch.cuda.is_available()
    train_loader = make_loader(
        train_ds, bs, shuffle=True, num_workers=nw, pin_memory=pin
    )
    val_loader = make_loader(val_ds, bs, shuffle=False, num_workers=nw, pin_memory=pin)

    # 4. Standard training loop with early stopping
    model = harness.model
    optimizer = harness.get_optmizer()
    criterion = harness.get_criterion()
    profiler = FLOPSProfiler()

    best_val_loss = float("inf")
    patience = 10
    stale = 0
    final_epoch = 0

    for epoch in range(cfg.train.max_iter):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, profiler
        )
        val_metrics = evaluate(model, val_loader, harness.eval_metrics, cfg.device)
        val_loss = val_metrics.get("loss", train_loss)
        final_epoch = epoch + 1

        print(
            f"Epoch {epoch + 1}/{cfg.train.max_iter}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"val_vrmse={val_metrics.get('vrmse', float('nan')):.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            # Save best model weights
            out_dir = Path(cfg.model.ckpts_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    # Reload best weights for final evaluation
    best_path = Path(cfg.model.ckpts_path) / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=cfg.device, weights_only=True)
        )

    # 5. Final test evaluation
    test_metrics = harness.final_evaluation()
    print(f"Test metrics: {test_metrics}")

    # 6. Save results
    save_results(cfg, test_metrics, profiler, final_epoch, data_budget)

    return 0


if __name__ == "__main__":
    sys.exit(main())
