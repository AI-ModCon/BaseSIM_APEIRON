"""End-to-end tests for the three entry points.

main.py, drift_only.py, and cl_only.py (~1,000 lines) had no tests at all.
These run each main() against a real TOML on a tiny in-memory harness — only
the example factory is substituted — and assert the exit code, the monitoring
loop having actually run, and the metrics CSV landing where the config says.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import Config
from apeiron.evaluation.metrics import accuracy
from apeiron.logger import reset_logger
from apeiron.model.torch_model_harness import BaseModelHarness

# The entry points import `examples.*`, which lives at the repository root
# outside the installable package, so they resolve only the way they are
# documented to run: from the root (`python -m src.main`). Mirror that.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _write_toml(tmp_path: Path, csv_path: Path) -> Path:
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        textwrap.dedent(f"""\
        seed = 42
        device = "cpu"
        multi_gpu = false

        [model]
        name = "tiny"
        pretrained_path = ""

        [data]
        name = "test"
        path = "/tmp"
        batch_size = 4

        [train]
        batch_size = 8
        num_workers = 0
        init_lr = 0.01
        max_iter = 2

        [drift_detection]
        detector_name = "PageHinkleyDetector"
        detection_interval = 2
        metric_index = 0
        max_stream_updates = 3
        ph_min_instances = 2
        ph_delta = 0.0
        ph_threshold = 0.1

        [logging]
        backend = "none"
        metrics_output_path = "{csv_path}"
        """)
    )
    return toml


class _ShiftingHarness(BaseModelHarness):
    """A tiny harness whose stream distribution moves every window.

    Self-contained rather than built on the shared DummyHarness: these tests
    drive whole entry points, so the harness contract they exercise should be
    visible here and not move when a shared fixture changes.
    """

    N, IN_F, NC = 32, 4, 3

    def __init__(self, cfg: Config):
        super().__init__(cfg, nn.Linear(self.IN_F, self.NC))
        self.eval_metrics = {"accuracy": accuracy, "loss": self.get_criterion()}
        self.higher_is_better = {"accuracy": True, "loss": False}
        self._window = 0
        self._train_ds = self._make(seed=0, shift=0.0)
        self._val_ds = self._make(seed=1, shift=0.0)

    def _make(self, seed: int, shift: float) -> TensorDataset:
        g = torch.Generator().manual_seed(seed)
        return TensorDataset(
            torch.randn(self.N, self.IN_F, generator=g) + shift,
            torch.randint(0, self.NC, (self.N,), generator=g),
        )

    def update_data_stream(self) -> None:
        self._val_ds = self._make(seed=100 + self._window, shift=2.0 * self._window)
        self._window += 1

    def get_stream_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.cfg.data.batch_size)

    def get_train_dataloaders(self):
        bs = self.cfg.train.batch_size
        return (
            DataLoader(self._train_ds, batch_size=bs),
            DataLoader(self._val_ds, batch_size=bs),
        )

    def get_hist_dataloaders(self):
        return (None, None)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        return SGD(self.model.parameters(), lr=self.cfg.train.init_lr)


@pytest.fixture(autouse=True)
def _fresh_logger():
    # get_logger() is a singleton that ignores its arguments once an
    # instance exists; every entry-point run must configure its own
    reset_logger()
    yield
    reset_logger()


def _csv_lines(csv_path: Path) -> list[str]:
    assert csv_path.is_file(), "the entry point must write the metrics CSV"
    return [ln for ln in csv_path.read_text().splitlines() if ln.strip()]


class TestMainEntryPoint:
    def test_runs_to_completion_and_writes_metrics(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        toml = _write_toml(tmp_path, csv_path)
        with patch("main.get_example", side_effect=lambda cfg: _ShiftingHarness(cfg)):
            import main

            assert main.main(["--config", str(toml)]) == 0
        lines = _csv_lines(csv_path)
        assert any("eval/" in ln for ln in lines), "the monitor never evaluated"
        assert any("drift/" in ln for ln in lines), "no drift check was logged"


class TestDriftOnlyEntryPoint:
    def test_runs_to_completion_and_prints_summary(self, tmp_path, capsys):
        csv_path = tmp_path / "metrics.csv"
        toml = _write_toml(tmp_path, csv_path)
        with patch(
            "drift_only.get_example", side_effect=lambda cfg: _ShiftingHarness(cfg)
        ):
            import drift_only

            assert drift_only.main(["--config", str(toml)]) == 0
        out = capsys.readouterr().out
        assert "Drift-only summary" in out
        assert "stream_updates: 3" in out
        _csv_lines(csv_path)

    def test_the_model_is_never_trained(self, tmp_path):
        # drift-only means detection without adaptation: weights are frozen
        csv_path = tmp_path / "metrics.csv"
        toml = _write_toml(tmp_path, csv_path)
        seen = {}

        def make(cfg):
            h = _ShiftingHarness(cfg)
            seen["weights"] = [p.detach().clone() for p in h.model.parameters()]
            seen["harness"] = h
            return h

        with patch("drift_only.get_example", side_effect=make):
            import drift_only

            assert drift_only.main(["--config", str(toml)]) == 0
        for p, before in zip(seen["harness"].model.parameters(), seen["weights"]):
            assert torch.equal(p.detach(), before)


class TestClOnlyEntryPoint:
    @pytest.mark.parametrize(
        "flags",
        [
            pytest.param(["--schedule", "fixed", "--trigger-at", "1"], id="fixed"),
            pytest.param(["--schedule", "periodic", "--period", "2"], id="periodic"),
            pytest.param(["--schedule", "never"], id="never"),
        ],
    )
    def test_schedules_run_to_completion(self, tmp_path, capsys, flags):
        csv_path = tmp_path / "metrics.csv"
        toml = _write_toml(tmp_path, csv_path)
        with patch(
            "cl_only.get_example", side_effect=lambda cfg: _ShiftingHarness(cfg)
        ):
            import cl_only

            assert cl_only.main([*flags, "--config", str(toml)]) == 0
        out = capsys.readouterr().out
        assert "Scheduled CL summary" in out
        _csv_lines(csv_path)

    def test_periodic_without_period_or_budget_refuses(self, tmp_path):
        # the default schedule is periodic and needs one of the two, so a
        # bare invocation (or one that only passes --trigger-at, forgetting
        # --schedule fixed) fails this way
        csv_path = tmp_path / "metrics.csv"
        toml = _write_toml(tmp_path, csv_path)
        with patch(
            "cl_only.get_example", side_effect=lambda cfg: _ShiftingHarness(cfg)
        ):
            import cl_only

            with pytest.raises(ValueError, match="--period or --budget"):
                cl_only.main(["--trigger-at", "1", "--config", str(toml)])
