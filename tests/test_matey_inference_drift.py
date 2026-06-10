from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_cfg(
    *,
    data_path: str,
    alt_path: str = "",
    pretrained_path: str = "",
) -> Config:
    return Config(
        model=ModelCfg(name="matey_vit", pretrained_path=pretrained_path),
        data=DataCfg(
            name="matey_inference_drift",
            path=data_path,
            alt_path=alt_path,
        ),
        train=TrainCfg(
            batch_size=1, num_workers=0, init_lr=0.001, max_iter=2
        ),
        continual_learning=ContinualLearningCfg(update_mode="none"),
        drift_detection=DriftDetectionCfg(
            detection_interval=1, max_stream_updates=2
        ),
        seed=7,
        device="cpu",
        multi_gpu=False,
    )


def _make_solps_tree(root: Path) -> None:
    for split in ("train", "valid"):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "sample.nc").write_bytes(b"")


def _minimal_harness(baseline: Path, shift: Path | None):
    from examples.matey.model_inference_drift import MATEYInferenceDriftHarness

    harness = MATEYInferenceDriftHarness.__new__(MATEYInferenceDriftHarness)
    harness._baseline_root = baseline.resolve()
    harness._alt_root = shift.resolve() if shift is not None else None
    harness.task_counter = 0
    harness._data_root = baseline.resolve()
    harness._params = MagicMock()
    harness._solps_split = None
    harness._configure_user_data_paths = MagicMock()
    harness._configure_solps_staged_pool = MagicMock()
    return harness


class TestMATEYInferenceDriftHarness:
    def test_domain_alternates_when_alt_path_set(self, tmp_path: Path):
        root = _project_root()
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        baseline = tmp_path / "baseline"
        shift = tmp_path / "shift"
        _make_solps_tree(baseline)
        _make_solps_tree(shift)

        harness = _minimal_harness(baseline, shift)
        domains_seen: list[str] = []
        roots_seen: list[Path] = []

        for expected_domain, expected_root in [
            ("baseline", baseline),
            ("shift", shift),
            ("baseline", baseline),
            ("shift", shift),
        ]:
            domains_seen.append(harness._active_domain_label())
            roots_seen.append(harness._active_data_root())
            assert harness._active_domain_label() == expected_domain
            assert harness._active_data_root() == expected_root.resolve()
            harness.task_counter += 1

        assert domains_seen == ["baseline", "shift", "baseline", "shift"]
        assert roots_seen == [
            baseline.resolve(),
            shift.resolve(),
            baseline.resolve(),
            shift.resolve(),
        ]

    def test_alt_path_missing_raises(self, tmp_path: Path):
        root = _project_root()
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        baseline = tmp_path / "baseline"
        _make_solps_tree(baseline)
        cfg = _make_cfg(data_path=str(baseline), alt_path=str(tmp_path / "missing"))

        from examples.matey.model_inference_drift import MATEYInferenceDriftHarness

        with pytest.raises(FileNotFoundError, match="alt_path"):
            MATEYInferenceDriftHarness(cfg)

    def test_utils_dispatches_inference_drift_name(self):
        root = _project_root()
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from examples.utils import get_example

        cfg = _make_cfg(data_path="/tmp/baseline")
        with patch(
            "examples.matey.model_inference_drift.MATEYInferenceDriftHarness"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            get_example(cfg)
            mock_cls.assert_called_once_with(cfg=cfg)
