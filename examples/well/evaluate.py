"""Score finished models against The Well's held-out test split.

The monitoring run never touches ``test``. It streams and adapts on ``train``
and evaluates on ``valid``, so the test split stays clean for exactly this: a
separate pass, after the fact, over whatever a run produced.

A run leaves one checkpoint per drift event in ``[model] ckpts_path``. Scoring
every one of them against every regime gives a matrix rather than a number, and
the matrix is where the interesting part is -- the diagonal says how well the
model did on the regime it had just adapted to, and the columns to the left of
it say what that adaptation cost the regimes it had already left.

Usage::

    # every checkpoint a run produced, plus the trivial baselines
    poetry run python -m examples.well.evaluate \\
        --config examples/well/well_trl2d.toml --baselines

    # add a reference model to compare against, e.g. The Well's published
    # baseline, which was trained on all regimes jointly
    poetry run python -m examples.well.evaluate \\
        --config examples/well/well_trl2d.toml \\
        --reference model.safetensors --reference-name published

``--cap`` scores only the first N samples of each regime, which is enough to
rank models and much faster than the full 97.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from apeiron.config.configuration import Config, build_config

from examples.well.datasets import RegimeCache, WellDataset, get_dataset
from examples.well.model import (
    N_STEPS_OUTPUT,
    UNET_WIDTHS,
    RegimeDataset,
    RegimeFrames,
    vrmse,
)
from examples.well.unet import UNetClassic

SPLIT = "test"


def build_model(spec: WellDataset, width: int) -> nn.Module:
    """A model shaped for this dataset, at one of the published widths."""
    from examples.well.model import N_STEPS_INPUT

    return UNetClassic(
        dim_in=N_STEPS_INPUT * spec.n_channels,
        dim_out=N_STEPS_OUTPUT * spec.n_channels,
        n_spatial_dims=spec.n_spatial_dims,
        spatial_resolution=spec.spatial_resolution,
        init_features=width,
    )


def load_weights(path: Path) -> Dict[str, Tensor]:
    """Weights from a ``.safetensors`` or a checkpoint written by ``save_ckpt``."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # save_ckpt stores build_checkpoint_payload(), which is a bare state_dict
    # by default; tolerate a wrapper that keeps it under a key.
    if isinstance(payload, dict) and "state_dict" in payload:
        return dict(payload["state_dict"])
    return dict(payload)


def checkpoints(directory: Path) -> List[Tuple[str, Path]]:
    """``drift_adaptation_<n>.pt`` in event order, labelled by event."""
    found = []
    for path in directory.glob("drift_adaptation_*.pt"):
        try:
            event = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        found.append((event, path))
    return [(f"event_{e:03d}", p) for e, p in sorted(found)]


def persistence(x: Tensor, n_channels: int) -> Tensor:
    """Repeat the last input frame. The baseline any surrogate has to beat."""
    return x[:, -N_STEPS_OUTPUT * n_channels :]


def score(
    model: Optional[nn.Module],
    cache: RegimeCache,
    regimes: Sequence[str],
    stats: Tuple,
    batch_size: int,
    device: str,
    cap: Optional[int],
    baseline: Optional[str] = None,
) -> Dict[str, float]:
    """Mean VRMSE per regime for one model, or for a trivial baseline."""
    mean, std = stats
    if model is not None:
        model.to(device).eval()
    out: Dict[str, float] = {}
    for regime in regimes:
        dataset: Dataset = RegimeDataset(
            RegimeFrames(cache.array(SPLIT, regime)), mean, std
        )
        if cap is not None and cap < len(dataset):  # type: ignore[arg-type]
            dataset = Subset(dataset, range(cap))
        total = count = 0.0
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=batch_size):
                x, y = x.to(device), y.to(device)
                if baseline == "persistence":
                    y_hat = persistence(x, mean.shape[0])
                elif baseline == "zero":
                    y_hat = torch.zeros_like(y)
                else:
                    assert model is not None
                    y_hat = model(x)
                total += vrmse(y_hat, y).item() * len(x)
                count += len(x)
        out[regime] = total / count
    return out


def report(rows: List[Tuple[str, Dict[str, float]]], regimes: Sequence[str]) -> None:
    """One line per model, one column per regime, plus the mean."""
    labels = [r.rsplit("_", 1)[-1] for r in regimes]
    width = max(len(name) for name, _ in rows) + 1
    header = f"{'model':<{width}}" + "".join(f"{lab:>8}" for lab in labels)
    print(f"\n{header}{'MEAN':>9}")
    print("-" * len(header + "    MEAN "))
    for name, scores in rows:
        line = f"{name:<{width}}" + "".join(f"{scores[r]:8.4f}" for r in regimes)
        mean = sum(scores.values()) / len(scores)
        print(f"{line}{mean:9.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score models on The Well's test split"
    )
    parser.add_argument("--config", required=True, help="the run's TOML")
    parser.add_argument(
        "--checkpoints", help="directory of checkpoints (default: [model] ckpts_path)"
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="a .safetensors or .pt to score alongside; repeatable",
    )
    parser.add_argument(
        "--reference-name",
        action="append",
        default=[],
        help="label for each --reference, in order",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="also score persistence and zero prediction",
    )
    parser.add_argument(
        "--random", action="store_true", help="also score an untrained model"
    )
    parser.add_argument("--cap", type=int, help="samples per regime (default: all)")
    parser.add_argument("--out", help="write the matrix to this CSV")
    args = parser.parse_args(argv)

    cfg: Config = build_config(["--config", args.config])
    spec = get_dataset(cfg.data.name.split(":", 1)[1])
    cache = RegimeCache(spec, cfg.data.path)
    regimes = cache.regimes()
    stats = cache.statistics()
    width = UNET_WIDTHS[cfg.model.name]
    device = cfg.device
    common = dict(
        cache=cache,
        regimes=regimes,
        stats=stats,
        batch_size=cfg.train.batch_size,
        device=device,
        cap=args.cap,
    )

    print(f"dataset   {spec.name}")
    print(
        f"split     {SPLIT} ({len(regimes)} regimes"
        f"{f', first {args.cap} samples each' if args.cap else ''})"
    )
    print(f"model     UNetClassic init_features={width} ({cfg.model.name})")
    print(f"device    {device}")

    rows: List[Tuple[str, Dict[str, float]]] = []

    if args.baselines:
        for label in ("persistence", "zero"):
            rows.append((label, score(None, baseline=label, **common)))  # type: ignore[arg-type]

    if args.random:
        # Seed first, exactly as the harness does before it draws its weights,
        # so this row is the model the run actually started from rather than
        # some other random draw. Without it the baseline moves between calls
        # and "did adapting help?" has no fixed reference.
        torch.manual_seed(cfg.seed)
        rows.append(("initial", score(build_model(spec, width), **common)))  # type: ignore[arg-type]

    names = list(args.reference_name)
    for index, source in enumerate(args.reference):
        path = Path(source)
        label = names[index] if index < len(names) else path.stem
        model = build_model(spec, width)
        model.load_state_dict(load_weights(path), strict=True)
        rows.append((label, score(model, **common)))  # type: ignore[arg-type]

    directory = Path(args.checkpoints or cfg.model.ckpts_path or "")
    if directory and directory.is_dir():
        found = checkpoints(directory)
        print(f"checkpoints {len(found)} in {directory}")
        for label, path in found:
            model = build_model(spec, width)
            model.load_state_dict(load_weights(path), strict=True)
            rows.append((label, score(model, **common)))  # type: ignore[arg-type]
    elif not rows:
        parser.error(f"no checkpoints at {directory!r} and nothing else to score")

    if not rows:
        parser.error("nothing to score")

    report(rows, regimes)

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", *regimes, "mean"])
            for name, scores in rows:
                values = [scores[r] for r in regimes]
                writer.writerow([name, *values, sum(values) / len(values)])
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
