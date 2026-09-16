"""SOLPS2DwION NetCDF loader for MATEY (b2time.nc with ne2d/te2d/ti2d)."""

from __future__ import annotations

import os

import numpy as np
import torch

from matey.data_utils.netcdf_datasets import BasenetCDFDirectoryDataset

SOLPS_SCALE2EV = 6.241509074460763e18


def _env_cubsizes(default: list[int]) -> list[int]:
    """Spatial (H, W) for the block split, overridable for A/B testing.

    ``_specifics`` is a staticmethod called before any config exists, so this
    is an env var rather than a config field.
    """
    raw = os.environ.get("SOLPS_CUBSIZES", "").strip()
    if not raw:
        return list(default)
    parts = [p for p in raw.replace("[", "").replace("]", "").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"SOLPS_CUBSIZES must be 'H,W'; got {raw!r}")
    return [int(p) for p in parts]


# Per-device min/max normalisation envelopes, in the units b2time.nc actually
# stores: ne in m^-3, te/ti in JOULES, tflux as the sum over nstrat.
#
# Getting the units wrong here does not raise -- it silently flattens a field to
# a constant, which reads downstream as "the model cannot predict this device".
# The KSTAR entry used to carry te/ti bounds of (1e-05, 220.66874753097187),
# which are that run's own min/max expressed in **eV**. Normalising the real
# Joule-valued data against them mapped the entire field, minimum and maximum
# alike, to -4.53e-08: te2d and ti2d became constant channels, and the KSTAR
# NRMSE of ~0.216 that followed was an artefact of this, not a statement about
# cross-device generalisation. Multiplying those eV bounds by the elementary
# charge reproduces the observed Joule range to 12 significant figures, which is
# how the mismatch was identified.
#
# Bounds below are each run's own min/max over its full time series, except D3D,
# which is left exactly as it was because the FusionBench parity result
# (NRMSE 0.0100 vs the ~0.011 reference) is calibrated against these numbers.
# Note the D3D values are numerically the min/max of the *held-out* dribble run
# rather than of the pre-training pool; that is pre-existing and deliberate,
# since it is what Demo/verify_solps_units.py uses.
_CASE_MINMAX = {
    "SOLPS-D3D": {
        "ne": (3.993869294415e16, 1.539813668964e21),
        "te": (4.854264896419e-20, 1.245952398209e-16),
        "ti": (1.102931835423e-19, 1.363727869099e-16),
        # Sum of tflux over nstrat (input actuator for TurBT AR checkpoints).
        "tflux": (3.0e23, 7.6e23),
    },
    "SOLPS-KSTAR": {
        "ne": (7.563805294615e14, 2.190466584132e20),
        # Was (1e-05, 220.66874753097187) -- the same numbers in eV.
        "te": (1.602176634000e-24, 3.535503111482e-17),
        "ti": (2.651905813672e-20, 3.819545622800e-17),
        # Was the D3D envelope, which put KSTAR's summed tflux entirely
        # negative, at [-0.587, -0.509].
        "tflux": (2.976367041786e22, 6.603777825904e22),
    },
}


def register_envelopes(extra: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Add per-device envelopes supplied by the data root.

    ``matey_settings.json`` carries these so that a device whose name may not
    appear in this repository can still be normalised correctly. The key is the
    token that appears in its file paths.
    """
    for token, bounds in extra.items():
        _CASE_MINMAX[f"SOLPS-{token.upper()}"] = {
            field: (float(lo), float(hi)) for field, (lo, hi) in bounds.items()
        }


class SOLPS2DwIONDataset(BasenetCDFDirectoryDataset):
    """Loader for SOLPS-ITER b2time.nc exports (time, ny, nx)."""

    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ["ne2d", "te2d", "ti2d"]
        type_name = "SOLPS2DwION"
        # cubsizes is consumed as (H, W) and reaches the model through
        # blockdict["Ind_dim"] = [D, H, W]. The b2time.nc arrays are
        # (time, ny=38, nx=98) and _reconstruct_sample yields (t, C, 38, 98),
        # so the tensor's trailing dim is 98. Declaring [98, 38] therefore sets
        # W=38 and the prediction comes back cropped to 38x38 instead of 38x98.
        # MATEY's own SOLPSDataset declares [98, 38]; whether that is correct
        # for its KSTAR arrays is a separate question, but it is demonstrably
        # wrong for this one. Override with SOLPS_CUBSIZES="98,38" to reproduce
        # the old behaviour.
        cubsizes = _env_cubsizes(default=[38, 98])
        split_level = None
        return time_index, sample_index, field_names, type_name, split_level, cubsizes

    field_names = _specifics()[2]

    def _infer_case(self, filepath: str) -> str:
        """Resolve the device from the path.

        This used to default to D3D, which made _bounds_for()'s KeyError
        unreachable: an unrecognised path did not fail, it borrowed the D3D
        envelope. That is how a device whose ne peaks an order of magnitude
        above the D3D ceiling was normalised for a long time without anything
        saying so. Longest token first, so a token that contains another still
        resolves to itself.
        """
        upper = filepath.upper()
        tokens = sorted(
            (c.split("-", 1)[1] for c in _CASE_MINMAX), key=len, reverse=True
        )
        for token in tokens:
            if token in upper:
                return f"SOLPS-{token}"
        raise KeyError(
            f"No device token in {filepath!r}; known tokens are {sorted(tokens)}. "
            f"Keep the device name in the path, or register its envelope under "
            f"'norm_envelopes' in the data root's matey_settings.json."
        )

    def _dataset_source_path(self, dat) -> str:
        fp = getattr(dat, "filepath", None)
        if callable(fp):
            return str(fp())
        if fp:
            return str(fp)
        return str(self.path)

    def get_min_max(self, filepath: str | None = None):
        # Dict-by-case layout matches Demo/FusionBench _denorm_solps_tensor.
        self.neminmax = {k: v["ne"] for k, v in _CASE_MINMAX.items()}
        self.teminmax = {k: v["te"] for k, v in _CASE_MINMAX.items()}
        self.timinmax = {k: v["ti"] for k, v in _CASE_MINMAX.items()}
        self.tfluxminmax = {k: v["tflux"] for k, v in _CASE_MINMAX.items()}
        return self._infer_case(filepath or str(self.path))

    def _bounds_for(self, mapping: dict, case: str):
        if case in mapping:
            return mapping[case]
        # This used to fall through to the first entry (D3D). Silently borrowing
        # another device's envelope does not fail, it just mis-scales the field:
        # a device whose ne peaks an order of magnitude above the D3D ceiling
        # gets normalised inputs well outside the intended range, and nothing
        # says so. Fail loudly instead: a missing entry is a data-registration
        # bug, and every case _infer_case can return is covered by _CASE_MINMAX.
        raise KeyError(
            f"No normalisation envelope for case {case!r}. Known cases: "
            f"{sorted(mapping)}. Add the device's min/max to _CASE_MINMAX in "
            f"{__file__} rather than letting it borrow another device's bounds."
        )

    def _get_norm_data(self, data, filepath: str | None = None):
        case = self.get_min_max(filepath)
        ne_min, ne_max = self._bounds_for(self.neminmax, case)
        te_min, te_max = self._bounds_for(self.teminmax, case)
        ti_min, ti_max = self._bounds_for(self.timinmax, case)
        data[:, :, :, 0] = (data[:, :, :, 0] - ne_min) / (ne_max - ne_min)
        data[:, :, :, 1] = (data[:, :, :, 1] - te_min) / (te_max - te_min)
        data[:, :, :, 2] = (data[:, :, :, 2] - ti_min) / (ti_max - ti_min)
        return data

    def _get_specific_stats(self, dat):
        time_dim = "nt" if "nt" in dat.dimensions else "time"
        steps = dat.dimensions[time_dim].size
        return 1, steps

    def _get_specific_bcs(self, dat):
        return [0, 0]

    def _read_input_control(self, dat, time_idx: int, n_steps: int, leadtime: int):
        if "tflux" not in dat.variables:
            raise KeyError(
                "SOLPS2DwION checkpoint expects input_control_act but "
                f"'tflux' is missing in {self._dataset_source_path(dat)}"
            )
        raw = np.ma.getdata(
            dat.variables["tflux"][time_idx - n_steps : time_idx + leadtime]
        )
        if raw.ndim == 2:
            raw = raw.sum(axis=-1)
        filepath = self._dataset_source_path(dat)
        case = self.get_min_max(filepath)
        lo, hi = self._bounds_for(self.tfluxminmax, case)
        return ((raw - lo) / (hi - lo)).astype(np.float32)

    def _reconstruct_sample(self, dat, leadtime, time_idx, n_steps):
        filepath = self._dataset_source_path(dat)
        lt = int(leadtime.item()) if hasattr(leadtime, "item") else int(leadtime)
        ne = np.ma.getdata(dat.variables["ne2d"][time_idx - n_steps : time_idx, :, :])
        te = np.ma.getdata(dat.variables["te2d"][time_idx - n_steps : time_idx, :, :])
        ti = np.ma.getdata(dat.variables["ti2d"][time_idx - n_steps : time_idx, :, :])

        ne_y = np.ma.getdata(dat.variables["ne2d"][time_idx : time_idx + lt, :, :])
        te_y = np.ma.getdata(dat.variables["te2d"][time_idx : time_idx + lt, :, :])
        ti_y = np.ma.getdata(dat.variables["ti2d"][time_idx : time_idx + lt, :, :])

        comb_x = np.stack([ne, te, ti], axis=-1).astype(np.float32)
        comb_y = np.stack([ne_y, te_y, ti_y], axis=-1).astype(np.float32)
        comb = np.concatenate((comb_x, comb_y), axis=0)
        comb_norm = self._get_norm_data(comb, filepath)
        input_control = (
            self._read_input_control(dat, time_idx, n_steps, lt)
            if self.input_control_act
            else None
        )
        return (
            comb_norm.transpose(0, 3, 1, 2),
            leadtime.to(torch.float32),
            input_control,
        )
