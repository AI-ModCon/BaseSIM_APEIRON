#!/usr/bin/env bash
# Stage FusionBench-compatible SOLPS slice bundles under mateydata/solps_drift_fb/.
# Does NOT modify mateydata/solps_drift/ (full b2time symlinks).
set -euo pipefail

MATEYDATA="${1:-/lustre/orion/lrn097/scratch/asvillar/mateydata}"
OUT_ROOT="${2:-${MATEYDATA}/solps_drift_fb}"
SIN4="${3:-/lustre/orion/lrn097/proj-shared/fusionMT-data/SOLPS2DwION/D3D/174310_D/puff2.5e21_ss_Sequence_sin4_308_2d_output/b2time.nc}"
DRIBBLE="${4:-${MATEYDATA}/Datasets_notusedinpretraining/D3D/174310_D/puff2.5e21_ss_noLat_dribble_308_2d_output/b2time.nc}"
N_SLICES="${5:-3}"
WINDOW="${6:-60}"

for f in "$SIN4" "$DRIBBLE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing SOLPS file: $f" >&2
    exit 1
  fi
done

MATEY_ENV="${MATEY_ENV:-/lustre/orion/world-shared/stf218/junqi/forge/matey-env-rocm631.sh}"
if ! python3 -c "import netCDF4" 2>/dev/null; then
  if [[ -f "${MATEY_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${MATEY_ENV}"
  else
    echo "ERROR: netCDF4 not found and MATEY_ENV missing: ${MATEY_ENV}" >&2
    exit 1
  fi
fi

python3 - "$OUT_ROOT" "$SIN4" "$DRIBBLE" "$N_SLICES" "$WINDOW" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml

out_root = Path(sys.argv[1])
sin4 = Path(sys.argv[2])
dribble = Path(sys.argv[3])
n_slices = int(sys.argv[4])
window = int(sys.argv[5])


def default_slices(n_time: int, n_slices: int, window: int, margin: int = 15) -> List[Tuple[int, int]]:
    if n_time <= window + margin + 1:
        return [(margin, min(n_time, margin + window))]
    usable = n_time - window - margin
    starts = [margin + int(i * usable / max(n_slices - 1, 1)) for i in range(n_slices)]
    return [(s, s + window) for s in starts]


def write_time_subset_nc(src_path: Path, dst_path: Path, start: int, stop: int) -> None:
    import netCDF4 as nc

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(str(src_path)) as src, nc.Dataset(str(dst_path), "w", format="NETCDF4") as dst:
        for dim_name, dim in src.dimensions.items():
            size = len(dim)
            if dim_name == "time":
                size = max(0, stop - start)
            dst.createDimension(dim_name, size if not dim.isunlimited() else None)
        for var_name, var in src.variables.items():
            dims = var.dimensions
            out_var = dst.createVariable(var_name, var.dtype, dims)
            for attr in var.ncattrs():
                out_var.setncattr(attr, var.getncattr(attr))
            if "time" in dims:
                t_axis = dims.index("time")
                sl = [slice(None)] * len(dims)
                sl[t_axis] = slice(start, stop)
                out_var[:] = var[tuple(sl)]
            else:
                out_var[:] = var[:]
        for attr in src.ncattrs():
            dst.setncattr(attr, src.getncattr(attr))


def build_bundle(bundle_dir: Path, bundle_id: str, src_nc: Path, used_in_pretraining: bool) -> None:
    import netCDF4 as nc

    rel_shot = "D3D/174310_D"
    valid_dir = bundle_dir / "valid" / rel_shot
    train_dir = bundle_dir / "train" / rel_shot
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    valid_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(str(src_nc)) as d:
        n_time = d.dimensions["time"].size
    slices = default_slices(n_time, n_slices, window)

    slice_records = []
    for i, (start, stop) in enumerate(slices):
        fname = f"slice_{i:02d}_t{start:04d}_{stop:04d}.nc"
        out_valid = valid_dir / fname
        write_time_subset_nc(src_nc, out_valid, start, stop)
        out_train = train_dir / fname
        if out_train.exists() or out_train.is_symlink():
            out_train.unlink()
        out_train.symlink_to(out_valid.resolve())
        slice_records.append(
            {
                "slice_id": f"slice_{i:02d}",
                "time_start": start,
                "time_stop": stop,
                "n_steps": stop - start,
                "source_b2time_nc": str(src_nc),
                "source_time_indices": list(range(start, stop)),
                "output_file": str(out_valid),
            }
        )

    subset_meta = {
        "source_b2time": str(src_nc),
        "slices": [{"start": s, "stop": e, "n_steps": e - s} for s, e in slices],
        "n_files": len(slices),
    }
    with (bundle_dir / "subset_meta.json").open("w") as f:
        json.dump(subset_meta, f, indent=2)

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_id": bundle_id,
        "fusionbench_task": "fore_conf_solps_fld2d",
        "used_in_pretraining": used_in_pretraining,
        "matey_dataset_key": "SOLPS2DwION",
        "source": {
            "dataset_family": "SOLPS2DwION",
            "shot": rel_shot,
            "b2time_nc": str(src_nc),
        },
        "slices": slice_records,
        "build": {
            "script": "examples/matey/stage_solps_fusionbench_bundles.sh",
            "n_slices": n_slices,
            "window_steps": window,
            "subset_meta": subset_meta,
        },
    }
    with (bundle_dir / "manifest.yaml").open("w") as f:
        yaml.dump(manifest, f, sort_keys=False, default_flow_style=False)


build_bundle(out_root / "baseline", "train_Sequence_sin4", sin4, True)
build_bundle(out_root / "ood", "heldout_noLat_dribble", dribble, False)

print(f"Staged FusionBench SOLPS bundles under: {out_root}")
print(f"  baseline (sin4):   {out_root / 'baseline'}")
print(f"  ood (dribble):     {out_root / 'ood'}")
print(f"  slices: {n_slices} x {window} steps in train/ and valid/")
PY

cat <<EOF
FusionBench bundle staging complete.

Parity check (GPU):
  cd /lustre/orion/lrn097/scratch/asvillar/src/BaseSIM_APEIRON
  ./examples/matey/verify_fusionbench_parity.py \\
    --baseline ${OUT_ROOT}/baseline \\
    --checkpoint /lustre/orion/lrn097/scratch/asvillar/mateydata/models/leadtime_1/best_ckpt.tar

Drift run:
  ./examples/matey/submit_inference_drift_fb.sh
EOF
