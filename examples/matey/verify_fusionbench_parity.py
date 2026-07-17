#!/usr/bin/env python3
"""Verify APEIRON MATEY inference matches FusionBench NRMSE on sin4 slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FusionBench parity gate: baseline sin4 NRMSE should be ~0.01."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="SOLPS bundle root (train/ + valid/ with slice .nc files).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="MATEY checkpoint file (best_ckpt.tar).",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "examples/matey/matey_inference_drift.toml"),
        help="Base TOML config.",
    )
    parser.add_argument("--min-nrmse", type=float, default=0.005)
    parser.add_argument("--max-nrmse", type=float, default=0.025)
    parser.add_argument("--max-batches", type=int, default=15)
    return parser.parse_args()


def _run_parity_eval(args: argparse.Namespace) -> tuple[float, list[float], dict]:
    from config.configuration import build_config
    from examples.matey.model import MATEYHarness
    from logger import configure_backend, get_logger, reset_logger

    cfg = build_config(
        [
            "verify_fusionbench_parity",
            "--config",
            args.config,
            "--set",
            f"data.path={args.baseline}",
            "--set",
            "data.alt_path=",
            "--set",
            f"model.pretrained_path={args.checkpoint}",
            "--set",
            "data.dset_type=SOLPS2DwION",
            "--set",
            f"eval.max_val_batches={args.max_batches}",
            "--set",
            "eval.leadtime=1",
            "--set",
            "eval.use_step_inference=true",
            "--set",
            "logging.backend=none",
            "--set",
            "verbosity=INFO:1",
        ]
    )

    configure_backend(cfg)
    reset_logger()
    get_logger(verbosity=cfg.verbosity, backend="none")

    harness = MATEYHarness(cfg)
    harness.model.to(cfg.device)
    harness.model.eval()
    harness.update_data_stream()
    _, val_loader = harness.get_cur_data_loaders()

    nrmse_values: list[float] = []
    first_batch_diag: dict = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            x, y = harness._unpack(batch)
            x, y = x.to(cfg.device), y.to(cfg.device)
            y_hat = harness.model(x)
            value = harness._to_scalar(harness.eval_metrics["nrmse_mean"](y_hat, y))
            nrmse_values.append(float(value))

            if batch_idx == 0:
                target = harness._select_target_tensor(
                    y, harness._adapter_model.last_rollout_steps
                )
                first_batch_diag = {
                    "leadtime": (
                        int(x.leadtime.flatten()[0].item())
                        if x.leadtime is not None
                        else None
                    ),
                    "rollout_steps": harness._adapter_model.last_rollout_steps,
                    "use_step_inference": harness._adapter_model.use_step_inference,
                    "pred_shape": tuple(y_hat.shape),
                    "target_shape": tuple(target.shape),
                    "first_batch_nrmse_mean": float(value),
                    "autoregressive_param": bool(
                        getattr(harness._params, "autoregressive", False)
                    ),
                }

    if not nrmse_values:
        raise RuntimeError("Parity eval produced zero batches.")

    return float(sum(nrmse_values) / len(nrmse_values)), nrmse_values, first_batch_diag


def main() -> int:
    args = _parse_args()
    baseline = Path(args.baseline).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    if not baseline.is_dir():
        print(f"ERROR: baseline path missing: {baseline}", file=sys.stderr)
        return 2
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint missing: {checkpoint}", file=sys.stderr)
        return 2

    print(f"FusionBench parity check")
    print(f"  baseline:   {baseline}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  pass band:  [{args.min_nrmse}, {args.max_nrmse}]")

    try:
        mean_nrmse, per_batch, diag = _run_parity_eval(args)
    except Exception as exc:
        print(f"FAIL: parity eval error: {exc}", file=sys.stderr)
        return 1

    print(f"  batches:    {len(per_batch)}")
    print(f"  nrmse_mean: {mean_nrmse:.6f} (per-batch: {[f'{v:.6f}' for v in per_batch]})")

    passed = args.min_nrmse <= mean_nrmse <= args.max_nrmse
    if passed:
        print(f"PASS: sin4 NRMSE {mean_nrmse:.6f} within FusionBench band.")
        return 0

    print(f"FAIL: sin4 NRMSE {mean_nrmse:.6f} outside [{args.min_nrmse}, {args.max_nrmse}].")
    print("Diagnostics (batch 0):")
    for key, value in diag.items():
        print(f"  {key}: {value}")
    if mean_nrmse > 0.1:
        print(
            "Hint: high NRMSE usually means autoregressive rollout, wrong leadtime, "
            "or full-trajectory eval instead of FusionBench slice bundles."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
