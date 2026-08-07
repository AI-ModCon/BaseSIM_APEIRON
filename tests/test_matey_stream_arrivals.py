"""Stream-arrival guards for the MATEY example.

These cover two failure modes that are silent rather than loud, so nothing
downstream can tell them from a real result.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # repo root, so `examples.matey` imports
    sys.path.append(str(_ROOT))

from examples.matey.model_stream import MATEYStreamHarness  # noqa: E402

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "matey"


def _stub_harness(tmp_path, arrivals):
    """A MATEYStreamHarness with only what update_data_stream's guards touch."""
    harness = MATEYStreamHarness.__new__(MATEYStreamHarness)
    harness._arrivals = arrivals
    harness._stream_root = tmp_path
    harness.task_counter = 0
    return harness


class TestMissingArrivalBundle:
    """A bundle that is not on disk must stop the run, not repeat the last one.

    ``_configure_user_data_paths`` returns quietly when neither ``train/`` nor
    ``valid/`` exists, so without a guard the loaders keep serving the PREVIOUS
    arrival while the log announces the new one -- a repeated bundle is then
    indistinguishable from a genuinely flat regime, and those stale loaders get
    latched as the forgetting baseline.
    """

    def test_absent_bundle_raises_and_names_it(self, tmp_path):
        harness = _stub_harness(tmp_path, [{"dir": "seg_007_gone", "case": "a"}])
        with pytest.raises(
            FileNotFoundError, match=r"seg_007_gone.*no train/ or valid/"
        ):
            harness.update_data_stream()


class TestConfiguredMetricIndex:
    """``metric_index`` is a positional index into an ordered dict of metrics.

    Nothing validates it at runtime, so an off-by-one silently drives drift
    detection off a single field. ``matey.toml`` shipped ``0`` while its comment
    claimed that was the aggregate; index 0 is ``nrmse_ne2d``.
    """

    EXPECTED_ORDER = [
        "nrmse_ne2d",
        "nrmse_te2d",
        "nrmse_ti2d",
        "nrmse_mean",
        "nrmse",
        "rmse",
        "loss",
    ]

    @pytest.mark.parametrize("config_name", ["matey.toml", "matey_stream.toml"])
    def test_configs_monitor_the_aggregate(self, config_name):
        # Parsed by hand rather than with tomllib, which needs Python 3.11 and so
        # would skip on the 3.10 MATEY runtime this example actually runs under.
        text = (EXAMPLE_DIR / config_name).read_text()
        match = re.search(r"^metric_index\s*=\s*(\d+)", text, re.MULTILINE)
        assert match, f"{config_name} has no metric_index"
        index = int(match.group(1))
        assert self.EXPECTED_ORDER[index] == "nrmse_mean", (
            f"{config_name} monitors {self.EXPECTED_ORDER[index]!r}, "
            f"not the across-field aggregate"
        )
