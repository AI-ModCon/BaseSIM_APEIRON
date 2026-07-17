"""SOLPS2DwION NetCDF loader for MATEY (b2time.nc with ne2d/te2d/ti2d)."""

from __future__ import annotations

import numpy as np
import torch

from matey.data_utils.netcdf_datasets import BasenetCDFDirectoryDataset

SOLPS_SCALE2EV = 6.241509074460763e18

# D3D training envelope (from MATEY / Demo verify_solps_units.py)
_CASE_MINMAX = {
    "SOLPS-D3D": {
        "ne": (3.993869294415e16, 1.539813668964e21),
        "te": (4.854264896419e-20, 1.245952398209e-16),
        "ti": (1.102931835423e-19, 1.363727869099e-16),
        # Sum of tflux over nstrat (input actuator for TurBT AR checkpoints).
        "tflux": (3.0e23, 7.6e23),
    },
    "SOLPS-KSTAR": {
        "ne": (756380529461540.2, 2.1904665841323272e20),
        "te": (1e-05, 220.66874753097187),
        "ti": (1e-05, 220.66874753097187),
        "tflux": (3.0e23, 7.6e23),
    },
}


class SOLPS2DwIONDataset(BasenetCDFDirectoryDataset):
    """Loader for SOLPS-ITER b2time.nc exports (time, ny, nx)."""

    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ["ne2d", "te2d", "ti2d"]
        type_name = "SOLPS2DwION"
        cubsizes = [98, 38]
        split_level = None
        return time_index, sample_index, field_names, type_name, split_level, cubsizes

    field_names = _specifics()[2]

    def _infer_case(self, filepath: str) -> str:
        upper = filepath.upper()
        for token in ("D3D", "KSTAR", "SPARC"):
            if token in upper:
                return f"SOLPS-{token}"
        return "SOLPS-D3D"

    def _dataset_source_path(self, dat) -> str:
        fp = getattr(dat, "filepath", None)
        if callable(fp):
            return str(fp())
        if fp:
            return str(fp)
        return str(self.path)

    def get_min_max(self, filepath: str | None = None):
        case = self._infer_case(filepath or str(self.path))
        bounds = _CASE_MINMAX.get(case, _CASE_MINMAX["SOLPS-D3D"])
        self.neminmax = bounds["ne"]
        self.teminmax = bounds["te"]
        self.timinmax = bounds["ti"]
        self.tfluxminmax = bounds["tflux"]

    def _get_norm_data(self, data, filepath: str | None = None):
        self.get_min_max(filepath)
        ne_min, ne_max = self.neminmax
        te_min, te_max = self.teminmax
        ti_min, ti_max = self.timinmax
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
        self.get_min_max(filepath)
        lo, hi = self.tfluxminmax
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
        return comb_norm.transpose(0, 3, 1, 2), leadtime.to(torch.float32), input_control
