#!/usr/bin/env python3
"""Data-based drift detection for MATEY on SOLPS-ITER fusion data.

This showcase runs APEIRON's drift detectors over a SOLPS-ITER edge-plasma
stream that transitions from a case used in MATEY pre-training
(``Sequence_sin4``) to a held-out case that was never seen in pre-training
(``noLat_dribble``).

The monitored signal is *data-based*: a per-window distance between the
incoming plasma state and the operating envelope represented in the
pre-training case.  This is deliberately independent of the MATEY checkpoint,
so it is unaffected by the SOLPS2DwION loader-parity issue documented in
``notes/BaseSIM_APEIRON/AGENT_CONTEXT.md``.

Outputs an NPZ + JSON bundle consumed by ``plot_solps_drift_showcase.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import ks_2samp

# APEIRON detectors live under <repo>/src
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from apeiron.drift_detection.detectors.statistical_detectors import (  # noqa: E402
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)

# --- SOLPS field descriptors -------------------------------------------------
# Physical state descriptors extracted per simulation frame.  These are the
# quantities an operator (or a reasoning agent proposing plasma states) would
# have access to without running the surrogate.
DESCRIPTORS = [
    "tflux",  # total gas-puff throughput (actuator)
    "ne_mean",
    "ne_p50",
    "ne_p99",
    "te_mean",
    "te_p99",
    "ti_mean",
    "ti_p99",
    "nesepm",  # separatrix density, outer midplane
    "tesepm",  # separatrix electron temperature
    "tisepm",
]

SCALE2EV = 6.241509074460763e18  # SOLPS stores te/ti in Joule


@dataclass
class CaseData:
    name: str
    path: str
    time: np.ndarray
    desc: np.ndarray  # (nt, n_desc)
    ne: np.ndarray  # (nt, ny, nx) kept for field plots / KS
    te: np.ndarray
    ti: np.ndarray


def load_case(name: str, path: str, keep_fields: bool = True) -> CaseData:
    nf = sio.netcdf_file(path, "r", mmap=False)
    try:
        t = nf.variables["timesa"][:].astype(np.float64).copy()
        tflux = nf.variables["tflux"][:].astype(np.float64).sum(-1).copy()
        ne = nf.variables["ne2d"][:].astype(np.float64).copy()
        te = nf.variables["te2d"][:].astype(np.float64).copy() * SCALE2EV
        ti = nf.variables["ti2d"][:].astype(np.float64).copy() * SCALE2EV
        nesepm = nf.variables["nesepm"][:, 0].astype(np.float64).copy()
        tesepm = nf.variables["tesepm"][:, 0].astype(np.float64).copy()
        tisepm = nf.variables["tisepm"][:, 0].astype(np.float64).copy()
    finally:
        nf.close()

    ax = (1, 2)
    desc = np.stack(
        [
            tflux,
            ne.mean(ax),
            np.percentile(ne, 50, axis=ax),
            np.percentile(ne, 99, axis=ax),
            te.mean(ax),
            np.percentile(te, 99, axis=ax),
            ti.mean(ax),
            np.percentile(ti, 99, axis=ax),
            nesepm,
            tesepm,
            tisepm,
        ],
        axis=1,
    )
    return CaseData(
        name=name,
        path=path,
        time=t,
        desc=desc,
        ne=ne if keep_fields else np.empty(0),
        te=te if keep_fields else np.empty(0),
        ti=ti if keep_fields else np.empty(0),
    )


# --- drift scoring -----------------------------------------------------------


def ks_drift_score(
    fields: dict[str, np.ndarray],
    ref_pool: dict[str, np.ndarray],
    w: int,
    n_sample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Data-based drift score: per-field two-sample Kolmogorov-Smirnov statistic
    of each monitoring window against a pooled pre-training reference.

    This is the distribution-comparison test named in the paper and the same
    statistic APEIRON's KSWIN detector applies to a metric stream -- used here on
    the *incoming plasma state* rather than on the model error.  Scoring the raw
    field values (thousands of cells per window) rather than a handful of summary
    descriptors keeps the statistic away from its saturation limit.

    Returns (mean KS over fields, per-field KS [nwin, n_field], field order).
    """
    names = sorted(fields)
    rng = np.random.default_rng(seed)
    nwin = min(len(fields[n]) // w for n in names)
    out = np.zeros((nwin, len(names)))
    for j, n in enumerate(names):
        ref = ref_pool[n]
        for i in range(nwin):
            chunk = fields[n][i * w : (i + 1) * w].ravel()
            cur = rng.choice(chunk, size=min(n_sample, chunk.size), replace=False)
            out[i, j] = ks_2samp(ref, cur).statistic
    return out.mean(1), out, names


def window_reduce(x: np.ndarray, w: int, how: str = "mean") -> np.ndarray:
    n = (len(x) // w) * w
    x = x[:n].reshape(-1, w)
    return x.mean(1) if how == "mean" else np.median(x, axis=1)


def field_ks_per_window(
    field: np.ndarray, ref_pool: np.ndarray, w: int, n_sample: int, seed: int
) -> np.ndarray:
    """Two-sample KS statistic between each monitoring window's field values and
    a pooled reference drawn from the pre-training case."""
    rng = np.random.default_rng(seed)
    nwin = len(field) // w
    out = np.empty(nwin)
    ref = rng.choice(ref_pool, size=min(n_sample, ref_pool.size), replace=False)
    for i in range(nwin):
        chunk = field[i * w : (i + 1) * w].ravel()
        cur = rng.choice(chunk, size=min(n_sample, chunk.size), replace=False)
        out[i] = ks_2samp(ref, cur).statistic
    return out


# --- detector harness --------------------------------------------------------


def run_detector(detector, values: np.ndarray) -> dict:
    """Feed a scalar stream through an APEIRON detector, returning fire indices."""
    fired, scores, regimes = [], [], []
    for i, v in enumerate(values):
        sig = detector.update(float(v))
        scores.append(sig.drift_score)
        regimes.append(sig.regime.value if sig.regime else "n/a")
        if sig.drift_detected:
            fired.append(i)
    return {"fired": fired, "score": scores, "regime": regimes}


def detector_suite(short_stream: bool, seed: int = 0) -> dict:
    """Detector configurations, with identical names across cadences.

    "library default" is river's own defaults, as shipped in APEIRON's config.
    They assume streams of millions of samples. "tuned to stream" rescales the
    window and threshold parameters to the few hundred monitoring windows a
    scientific stream actually produces -- two orders of magnitude fewer. A
    Page-Hinkley threshold of 50 simply never accumulates that far in 291
    windows, so it never fires, which is a property of the stream length rather
    than of the drift.
    """
    ks = (20, 8) if short_stream else (60, 20)
    ph_min, ph_thr = (10, 1.0) if short_stream else (30, 5.0)
    return {
        "ADWIN library default": lambda: ADWINDetector(delta=0.002),
        "ADWIN tuned to stream": lambda: ADWINDetector(delta=0.05),
        "KSWIN library default": lambda: KSWINDetector(
            alpha=0.005, window_size=100, stat_size=30, seed=seed
        ),
        "KSWIN tuned to stream": lambda: KSWINDetector(
            alpha=0.005, window_size=ks[0], stat_size=ks[1], seed=seed
        ),
        "Page-Hinkley library default": lambda: PageHinkleyDetector(
            min_instances=30, delta=0.005, threshold=50.0
        ),
        "Page-Hinkley tuned to stream": lambda: PageHinkleyDetector(
            min_instances=ph_min, delta=0.005, threshold=ph_thr
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = os.environ.get("SOLPS_DRIFT_ROOT", "")
    if not root:
        raise SystemExit("set SOLPS_DRIFT_ROOT to the two-root drift showcase tree")
    ap.add_argument("--baseline", default=f"{root}/baseline/valid/d3d_sequence_sin4.nc")
    ap.add_argument("--ood", default=f"{root}/ood/valid/d3d_noLat_dribble.nc")
    ap.add_argument("--baseline-name", default="Sequence_sin4 (in pre-training)")
    ap.add_argument("--ood-name", default="noLat_dribble (held out)")
    ap.add_argument(
        "--n-ref", type=int, default=150, help="baseline frames defining the envelope"
    )
    ap.add_argument(
        "--window", type=int, default=5, help="frames per monitoring window"
    )
    ap.add_argument(
        "--short-window",
        type=int,
        default=40,
        help="frames per window for the short-stream (FusionBench-slice-like) run",
    )
    ap.add_argument("--ks-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] baseline {args.baseline}")
    base = load_case(args.baseline_name, args.baseline)
    print(f"[load] ood      {args.ood}")
    ood = load_case(args.ood_name, args.ood)
    print(f"       baseline nt={len(base.time)}  ood nt={len(ood.time)}")

    # Envelope from the first n_ref frames of the in-pre-training case.
    # Pooled per-field reference drawn from the in-pre-training case.
    _rng = np.random.default_rng(args.seed)
    FIELDS = ("ne", "te", "ti")
    ref_pool = {
        n: _rng.choice(
            getattr(base, n)[: args.n_ref].ravel(),
            size=args.ks_samples * 3,
            replace=False,
        )
        for n in FIELDS
    }
    stream_fields = {
        n: np.concatenate([getattr(base, n)[args.n_ref :], getattr(ood, n)], axis=0)
        for n in FIELDS
    }

    # Held-out portion of the baseline is monitored like any other data, so the
    # in-distribution false-alarm rate is measurable.
    base_mon = base.desc[args.n_ref :]
    stream_desc = np.concatenate([base_mon, ood.desc], axis=0)
    boundary_frame = len(base_mon)

    # In-distribution control: score the reference against itself at the same
    # cadence, so the in-distribution KS level (and hence the alarm threshold)
    # is measured rather than assumed.
    ref_score, _, _ = ks_drift_score(
        {n: getattr(base, n)[: args.n_ref] for n in FIELDS},
        ref_pool,
        args.window,
        args.ks_samples,
        args.seed,
    )

    results = {
        "meta": {
            "baseline": args.baseline,
            "ood": args.ood,
            "baseline_name": args.baseline_name,
            "ood_name": args.ood_name,
            "n_ref_frames": args.n_ref,
            "n_baseline_monitored": int(len(base_mon)),
            "n_ood": int(len(ood.desc)),
            "descriptors": DESCRIPTORS,
            "ref_score_mean": float(ref_score.mean()),
            "ref_score_p99": float(np.percentile(ref_score, 99)),
        },
        "runs": {},
    }

    for tag, w in (("dense", args.window), ("short", args.short_window)):
        s, per_field, field_order = ks_drift_score(
            stream_fields, ref_pool, w, args.ks_samples, args.seed
        )
        boundary_win = boundary_frame // w
        suite = detector_suite(short_stream=(tag == "short"), seed=args.seed)
        det_out = {}
        for name, factory in suite.items():
            r = run_detector(factory(), s)
            fired = r["fired"]
            after = [f for f in fired if f >= boundary_win]
            before = [f for f in fired if f < boundary_win]
            det_out[name] = {
                "fired": fired,
                "first_after_boundary": after[0] if after else None,
                "detection_delay_windows": (after[0] - boundary_win) if after else None,
                "detection_delay_frames": (
                    (after[0] - boundary_win) * w if after else None
                ),
                "false_alarms_before_boundary": len(before),
            }
            d = det_out[name]["detection_delay_windows"]
            print(
                f"[{tag} w={w}] {name:<32} "
                f"detect={'yes' if after else 'NO ':<3} "
                f"delay={d if d is not None else '-':>4} win  "
                f"false_alarms={len(before)}"
            )
        results["runs"][tag] = {
            "per_field_ks": per_field.tolist(),
            "field_order": field_order,
            "window_frames": w,
            "n_windows": int(len(s)),
            "boundary_window": int(boundary_win),
            "score": s.tolist(),
            "detectors": det_out,
        }

    # Field-level KS drift (second, complementary data-based signal).
    rng_pool = np.random.default_rng(args.seed)
    ref_pool = rng_pool.choice(
        base.ne[: args.n_ref].ravel(), size=args.ks_samples * 4, replace=False
    )
    ks_stream = np.concatenate(
        [
            field_ks_per_window(
                base.ne[args.n_ref :], ref_pool, args.window, args.ks_samples, args.seed
            ),
            field_ks_per_window(
                ood.ne, ref_pool, args.window, args.ks_samples, args.seed + 1
            ),
        ]
    )
    results["ks_ne"] = {
        "window_frames": args.window,
        "values": ks_stream.tolist(),
        "boundary_window": int(boundary_frame // args.window),
    }

    np.savez_compressed(
        outdir / "drift_showcase.npz",
        ref_score=ref_score,
        stream_desc=stream_desc,
        base_desc=base.desc,
        ood_desc=ood.desc,
        base_time=base.time,
        ood_time=ood.time,
        boundary_frame=boundary_frame,
        ks_stream=ks_stream,
        base_ne_frames=base.ne[:: max(1, len(base.ne) // 8)],
        ood_ne_frames=ood.ne[:: max(1, len(ood.ne) // 8)],
        base_te_frames=base.te[:: max(1, len(base.te) // 8)],
        ood_te_frames=ood.te[:: max(1, len(ood.te) // 8)],
        base_ne_mean=base.ne.mean(0),
        ood_ne_mean=ood.ne.mean(0),
        base_te_mean=base.te.mean(0),
        ood_te_mean=ood.te.mean(0),
        descriptors=np.array(DESCRIPTORS, dtype=object),
    )
    with open(outdir / "drift_showcase.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[done] wrote {outdir}/drift_showcase.npz and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
