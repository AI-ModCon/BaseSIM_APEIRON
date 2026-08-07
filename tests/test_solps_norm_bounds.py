"""The per-device SOLPS normalisation envelopes must bracket their own data.

A wrong envelope does not raise -- it rescales a field into a range the model
never saw, and in the limiting case flattens it to a constant. That failure is
invisible in the loss curve of the run that causes it and shows up much later as
"the surrogate cannot predict this device". The KSTAR entry carried eV bounds
against Joule data for exactly this reason, which silently zeroed te2d and ti2d.

The data-backed test is skipped when the shared trees are not mounted, so this
file is still useful off Frontier: the unit-consistency and coverage checks below
run anywhere.
"""

from __future__ import annotations

import os

import pytest

# SOLPS2DwIONDataset subclasses a MATEY reader, and MATEY is an optional
# dependency. Skip rather than fail collection without it.
pytest.importorskip("matey", reason="the MATEY package is not installed")

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # repo root, so `examples.matey` imports
    sys.path.append(str(_ROOT))

from examples.matey.solps.solps2dwion_dataset import (  # noqa: E402
    _CASE_MINMAX,
    SOLPS2DwIONDataset,
)

# Elementary charge -- the eV <-> Joule factor that the KSTAR bug turned on.
EV_IN_JOULES = 1.602176634e-19

# b2time.nc sources, one per device, relative to the pre-training tree. Set
# SOLPS_PRETRAIN_ROOT to point these at your own copy; the data-backed tests skip
# themselves for any source that is not present.
_PRETRAIN_ROOT = os.environ.get("SOLPS_PRETRAIN_ROOT", "").rstrip("/")
# The held-out D3D scenario lives outside the pre-training tree. It matters here
# because the D3D envelope is numerically its min/max.
_HELDOUT_ROOT = os.environ.get("SOLPS_HELDOUT_ROOT", "").rstrip("/")

_RELATIVE_SOURCES = {
    "SOLPS-D3D": [
        (
            _PRETRAIN_ROOT,
            "SOLPS2DwION/D3D/174310_D/"
            "puff2.5e21_ss_Sequence_sin4_308_2d_output/b2time.nc",
        ),
        (
            _HELDOUT_ROOT,
            "D3D/174310_D/puff2.5e21_ss_noLat_dribble_308_2d_output/b2time.nc",
        ),
    ],
    "SOLPS-KSTAR": [
        (_PRETRAIN_ROOT, "SOLPS2DwION/KSTAR/19077_D/puff5e20_td_linear_ramp/b2time.nc"),
    ],
}

_SOURCES = {
    case: [f"{root}/{rel}" for root, rel in entries if root]
    for case, entries in _RELATIVE_SOURCES.items()
}

_FIELD_VARS = {"ne": "ne2d", "te": "te2d", "ti": "ti2d"}


def _available(paths):
    return [p for p in paths if os.path.exists(p)]


class TestEnvelopeStructure:
    """Checks that need no data access."""

    def test_every_inferable_case_has_an_envelope(self):
        # _infer_case can only ever return these; each must be registered, or
        # _bounds_for raises at runtime deep inside a training job.
        ds = SOLPS2DwIONDataset.__new__(SOLPS2DwIONDataset)
        for token in ("D3D", "KSTAR"):
            case = ds._infer_case(f"/some/path/{token}/run/b2time.nc")
            assert case in _CASE_MINMAX, f"{case} has no normalisation envelope"

    def test_unknown_case_raises_instead_of_borrowing(self):
        ds = SOLPS2DwIONDataset.__new__(SOLPS2DwIONDataset)
        with pytest.raises(KeyError, match="No normalisation envelope"):
            ds._bounds_for({"SOLPS-D3D": (0.0, 1.0)}, "SOLPS-ELSEWHERE")

    @pytest.mark.parametrize("case", sorted(_CASE_MINMAX))
    def test_bounds_are_ordered_and_positive_width(self, case):
        for field, (lo, hi) in _CASE_MINMAX[case].items():
            assert hi > lo, f"{case}/{field}: bounds not increasing"

    @pytest.mark.parametrize("case", sorted(_CASE_MINMAX))
    def test_temperatures_are_in_joules_not_ev(self, case):
        """Guard against the exact regression that motivated this file.

        SOLPS edge temperatures are O(1-1000) eV, i.e. O(1e-19 - 1e-16) J. An
        upper bound above 1e-10 means somebody stored eV.
        """
        for field in ("te", "ti"):
            _, hi = _CASE_MINMAX[case][field]
            assert hi < 1e-10, (
                f"{case}/{field} upper bound {hi:.4g} looks like eV, not Joules "
                f"({hi / EV_IN_JOULES:.4g} J would be the eV reading)"
            )


class TestEnvelopeMatchesData:
    """Checks that read the real b2time.nc files."""

    @pytest.mark.parametrize("case", sorted(_SOURCES))
    def test_bounds_bracket_the_data(self, case):
        netCDF4 = pytest.importorskip("netCDF4")
        import numpy as np

        paths = _available(_SOURCES[case])
        if not paths:
            pytest.skip(f"no b2time.nc available for {case}")

        envelope = _CASE_MINMAX[case]
        for path in paths:
            ds = netCDF4.Dataset(path)
            try:
                for field, var in _FIELD_VARS.items():
                    arr = np.asarray(ds[var][:])
                    lo, hi = envelope[field]
                    normalised = (arr - lo) / (hi - lo)
                    # A tolerance is needed because the D3D envelope comes from
                    # the dribble run, so sin4 sits strictly inside it, while
                    # single-run envelopes are exactly the data range.
                    assert normalised.min() >= -0.05, (
                        f"{case}/{field} in {os.path.basename(os.path.dirname(path))}: "
                        f"normalised minimum {normalised.min():.4g} is far below 0"
                    )
                    assert normalised.max() <= 1.05, (
                        f"{case}/{field} in {os.path.basename(os.path.dirname(path))}: "
                        f"normalised maximum {normalised.max():.4g} is far above 1"
                    )
            finally:
                ds.close()

    @pytest.mark.parametrize("case", sorted(_SOURCES))
    def test_no_field_collapses_to_a_constant(self, case):
        """The KSTAR failure mode, stated directly.

        With eV bounds against Joule data the whole field mapped to a single
        value. Require every field to retain real dynamic range once normalised.
        """
        netCDF4 = pytest.importorskip("netCDF4")
        import numpy as np

        paths = _available(_SOURCES[case])
        if not paths:
            pytest.skip(f"no b2time.nc available for {case}")

        envelope = _CASE_MINMAX[case]
        for path in paths:
            ds = netCDF4.Dataset(path)
            try:
                for field, var in _FIELD_VARS.items():
                    arr = np.asarray(ds[var][:])
                    lo, hi = envelope[field]
                    normalised = (arr - lo) / (hi - lo)
                    spread = float(normalised.max() - normalised.min())
                    assert spread > 0.01, (
                        f"{case}/{field}: normalised spread is {spread:.3e} -- the "
                        f"channel is effectively constant, which is what wrong "
                        f"units look like"
                    )
            finally:
                ds.close()

    def test_kstar_old_ev_bounds_would_have_been_caught(self):
        """Regression witness: the previous bounds must fail the constant check.

        Without this, the checks above could be satisfied by a tolerance that is
        simply too loose to notice the original bug.
        """
        netCDF4 = pytest.importorskip("netCDF4")
        import numpy as np

        paths = _available(_SOURCES["SOLPS-KSTAR"])
        if not paths:
            pytest.skip("no KSTAR b2time.nc available")

        old_ev_bounds = (1e-05, 220.66874753097187)
        ds = netCDF4.Dataset(paths[0])
        try:
            arr = np.asarray(ds["te2d"][:])
        finally:
            ds.close()
        lo, hi = old_ev_bounds
        normalised = (arr - lo) / (hi - lo)
        assert float(normalised.max() - normalised.min()) < 1e-6
