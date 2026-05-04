"""Deterministic MNIST first-drift validation against EWC CL loss references."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from apeiron.config.configuration import build_config
from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.logger import get_logger
import apeiron.logger.logger as logger_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


REFERENCE_LOSS_CSV = (
    Path(__file__).parent / "references" / "mnist_first_drift_ewc_losses.csv"
)
_MNIST_RAW_REQUIRED = (
    Path("data/MNIST/raw/train-images-idx3-ubyte"),
    Path("data/MNIST/raw/train-labels-idx1-ubyte"),
    Path("data/MNIST/raw/t10k-images-idx3-ubyte"),
    Path("data/MNIST/raw/t10k-labels-idx1-ubyte"),
)
_LOSS_METRIC_MAP = {
    "cl/jvp_reg_generation_loss": "generation_loss",
    "cl/jvp_reg_forgetting_loss": "forgetting_loss",
    "cl/jvp_reg_total_loss": "total_loss",
}


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _extract_cl_loss_table(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    filtered = df[df["metric"].isin(_LOSS_METRIC_MAP)].copy()
    if filtered.empty:
        raise AssertionError("No CL loss metrics found in generated CSV.")

    filtered["value"] = pd.to_numeric(filtered["value"], errors="raise")
    filtered["metric"] = filtered["metric"].map(_LOSS_METRIC_MAP)
    table = (
        filtered.pivot(index="step", columns="metric", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    table = table[["step", "generation_loss", "forgetting_loss", "total_loss"]]
    for col in ("generation_loss", "forgetting_loss", "total_loss"):
        table[col] = pd.to_numeric(table[col], errors="raise")
    return table.sort_values("step").reset_index(drop=True)


def _run_mnist_until_first_drift(output_csv: Path) -> None:
    from examples.utils import get_example

    cfg = build_config(
        [
            "--config",
            "examples/mnist/mnist.toml",
            "--set",
            "seed=1337",
            "--set",
            "device=cpu",
            "--set",
            "train.num_workers=0",
            "--set",
            "train.max_iter=2",
            "--set",
            "continual_learning.update_mode=ewc_online",
            "--set",
            "drift_detection.detection_interval=1",
            "--set",
            "drift_detection.max_stream_updates=1",
        ]
    )
    _set_deterministic_seed(cfg.seed)

    logger_module._default_logger = None
    logger = get_logger(verbosity="WARNING", wandb_enabled=False, csv_path=output_csv)
    logger.init(cfg, project="mnist-validation")

    harness = get_example(cfg)
    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    harness.update_data_stream()

    forced_signal = DriftSignal(
        drift_detected=True,
        drift_score=1.0,
        regime=LearningRegime.CONTINUAL_LEARNING,
        confidence=1.0,
        metadata={"forced": True},
    )

    with patch.object(monitor.detector, "update", return_value=forced_signal):
        _, val_loader = harness.get_cur_data_loaders()
        for batch in val_loader:
            metrics = monitor._evaluate_batch(batch)
            monitor.metric_buffer.append(metrics)
            monitor.batch_count += 1

            if monitor.batch_count % monitor.detection_interval == 0:
                signal = monitor._check_drift()
                if signal.drift_detected:
                    monitor._handle_drift(signal)
                    break

    logger.finish()
    logger_module._default_logger = None


@pytest.mark.slow
def test_mnist_first_drift_losses_match_reference(tmp_path: Path) -> None:
    missing = [p for p in _MNIST_RAW_REQUIRED if not p.exists()]
    if missing:
        pytest.skip("MNIST raw files are missing locally; skipping validation test.")

    output_csv = tmp_path / "mnist_first_drift_metrics.csv"
    _run_mnist_until_first_drift(output_csv=output_csv)

    actual = _extract_cl_loss_table(output_csv)
    expected = pd.read_csv(REFERENCE_LOSS_CSV)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
