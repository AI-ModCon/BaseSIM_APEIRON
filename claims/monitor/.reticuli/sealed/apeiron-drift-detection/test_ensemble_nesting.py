"""An ensemble detector must not be built out of itself.\n\nThe factory refuses the recursion rather than building a detector that\nwould recurse forever on its first update."""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.drift_detection.load_drift_detector import load_drift_detector


class TestEnsembleNesting:
    def _cfg(self, default_cfg, **dd):
        return replace(default_cfg, drift_detection=DriftDetectionCfg(**dd))

    def test_an_ensemble_cannot_contain_itself(self, default_cfg):
        cfg = self._cfg(
            default_cfg,
            detector_name="EnsembleDetector",
            ensemble_detectors=["ADWINDetector", "EnsembleDetector"],
        )
        with pytest.raises(ValueError, match="cannot be nested inside itself"):
            load_drift_detector(cfg)
