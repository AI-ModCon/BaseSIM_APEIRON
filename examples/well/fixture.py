"""Generate Well-shaped HDF5 files with no download.

The files match the on-disk schema of
``polymathic-ai/turbulent_radiative_layer_2D`` exactly (root attrs;
``dimensions/{time,x,y}``; ``scalars/tcool``; ``t0_fields/{density,pressure}``;
``t1_fields/velocity`` with a trailing component dim; boundary-condition masks),
so :mod:`examples.well.wellio` and the converter exercise the *same* code path on
a fixture as on real data.

Crucially, the field statistics depend on the ``tcool`` parameter -- lower
cooling time yields higher-frequency, higher-amplitude turbulence -- so ordering
the stream by ``tcool`` produces genuine, physically-flavored concept drift
rather than synthetic noise. Everything is deterministic given ``seed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Real turbulent_radiative_layer_2D tcool values (a subset); each is a regime.
DEFAULT_TCOOLS = (0.03, 0.06, 0.10, 0.18, 0.32, 0.56, 1.00)


def _regime_fields(
    tcool: float, n_time: int, height: int, width: int, seed: int
) -> dict[str, np.ndarray]:
    """Synthesize density/pressure/velocity whose statistics depend on tcool."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Faster cooling (small tcool) -> finer, stronger structure.
    freq = 1.0 + 3.0 / (tcool + 0.1)
    amp = 1.0 / (tcool + 0.1)

    density = np.empty((n_time, height, width), dtype=np.float32)
    pressure = np.empty((n_time, height, width), dtype=np.float32)
    velocity = np.empty((n_time, height, width, 2), dtype=np.float32)

    for t in range(n_time):
        phase = 2.0 * np.pi * (t / max(1, n_time))
        # Drifting cellular structure + parameter-scaled turbulent noise.
        base = np.sin(freq * np.pi * (xs + 0.2 * np.cos(phase))) * np.cos(
            freq * np.pi * (ys - 0.2 * np.sin(phase))
        )
        turb = amp * rng.standard_normal((height, width)).astype(np.float32)
        density[t] = 1.0 + 0.5 * base + 0.15 * turb
        pressure[t] = 0.6 + 0.3 * base * base + 0.1 * turb
        # Rotational/shear velocity, magnitude grows as cooling speeds up.
        velocity[t, ..., 0] = amp * np.cos(freq * np.pi * ys + phase)
        velocity[t, ..., 1] = amp * np.sin(freq * np.pi * xs - phase)

    return {"density": density, "pressure": pressure, "velocity": velocity}


def write_well_file(
    path: str | Path,
    tcool: float,
    *,
    n_trajectories: int = 1,
    n_time: int = 12,
    height: int = 16,
    width: int = 24,
    seed: int = 0,
) -> Path:
    """Write one Well-schema HDF5 file for a single ``tcool`` regime."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = "well_fixture"
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = 2
        f.attrs["n_trajectories"] = n_trajectories
        f.attrs["simulation_parameters"] = ["tcool"]
        f.attrs["tcool"] = float(tcool)

        dims = f.create_group("dimensions")
        dims.create_dataset("time", data=np.arange(n_time, dtype=np.float32))
        dims.create_dataset("x", data=np.linspace(0, 1, height, dtype=np.float32))
        dims.create_dataset("y", data=np.linspace(0, 1, width, dtype=np.float32))

        f.create_group("scalars").create_dataset("tcool", data=np.float32(tcool))

        t0 = f.create_group("t0_fields")
        t1 = f.create_group("t1_fields")
        f.create_group("t2_fields")

        density = np.empty((n_trajectories, n_time, height, width), dtype=np.float32)
        pressure = np.empty_like(density)
        velocity = np.empty(
            (n_trajectories, n_time, height, width, 2), dtype=np.float32
        )
        for traj in range(n_trajectories):
            fields = _regime_fields(tcool, n_time, height, width, seed + traj)
            density[traj] = fields["density"]
            pressure[traj] = fields["pressure"]
            velocity[traj] = fields["velocity"]

        t0.create_dataset("density", data=density)
        t0.create_dataset("pressure", data=pressure)
        t1.create_dataset("velocity", data=velocity)

        bc = f.create_group("boundary_conditions")
        xg = bc.create_group("x_periodic")
        xg.create_dataset("mask", data=np.ones(height, dtype=bool))
        yg = bc.create_group("y_open")
        yg.create_dataset("mask", data=np.zeros(width, dtype=bool))

    return path


def generate_fixture(
    out_dir: str | Path,
    tcools: tuple[float, ...] = DEFAULT_TCOOLS,
    *,
    n_trajectories: int = 1,
    n_time: int = 12,
    height: int = 16,
    width: int = 24,
    seed: int = 0,
) -> list[Path]:
    """Write one file per ``tcool`` regime; returns the paths (regime order)."""
    out_dir = Path(out_dir)
    paths = []
    for i, tcool in enumerate(tcools):
        name = f"well_fixture_tcool_{tcool:.2f}.hdf5"
        paths.append(
            write_well_file(
                out_dir / name,
                tcool,
                n_trajectories=n_trajectories,
                n_time=n_time,
                height=height,
                width=width,
                seed=seed + 100 * i,
            )
        )
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write Well-shaped fixture HDF5 files.")
    ap.add_argument("out_dir")
    ap.add_argument("--n-time", type=int, default=12)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--n-trajectories", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = generate_fixture(
        args.out_dir,
        n_trajectories=args.n_trajectories,
        n_time=args.n_time,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    print(f"wrote {len(paths)} fixture files to {args.out_dir}")
