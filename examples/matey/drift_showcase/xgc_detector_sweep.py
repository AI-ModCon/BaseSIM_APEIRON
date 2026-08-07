#!/usr/bin/env python3
"""Run APEIRON's detectors over the corrected XGC drift streams.

Two streams, and the contrast between them is the point:

**device stream** -- pre-trained devices first, then held-out ones.  The
monitored scalar is the coverage score: the KS of the current monitoring
window against the *closest* pre-trained device.  A detector should fire once
the stream crosses into held-out devices.

**same-machine control** -- one DIII-D scenario followed by the other, scored
against the first.  The machine never changes, only the operating scenario.
A detector that fires here is reporting a device change that did not happen,
so this measures the false-alarm side that a delay-only figure cannot show.

Reads the JSON written by ``xgc_mesh_drift.py``; no re-reading of graph data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
_here = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(_here)]
for p in (str(REPO), str(REPO / "src")):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from apeiron.drift_detection.detectors.statistical_detectors import (  # noqa: E402
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)


def windowed_matrix(res: dict) -> dict:
    """W[a][b] = per-frame KS of a's frames against b's pooled reference."""
    w: dict[str, dict[str, list[float]]] = {}
    for lab, vals in res["levels"]["L0_temporal_windowed"].items():
        w.setdefault(lab, {})[lab] = list(vals)
    for rec in res["same_machine_pairs"] + res["cross_machine_pairs"]:
        a, b = rec["a"], rec["b"]
        w.setdefault(a, {})[b] = list(rec["ks_windowed_ab"])
        w.setdefault(b, {})[a] = list(rec["ks_windowed_ba"])
    return w


def detectors(seed: int) -> dict:
    return {
        "ADWIN as-shipped": lambda: ADWINDetector(delta=0.002),
        "ADWIN resized": lambda: ADWINDetector(delta=0.05),
        "KSWIN as-shipped": lambda: KSWINDetector(
            alpha=0.005, window_size=100, stat_size=30, seed=seed
        ),
        "KSWIN resized": lambda: KSWINDetector(
            alpha=0.005, window_size=20, stat_size=8, seed=seed
        ),
        "Page-Hinkley as-shipped": lambda: PageHinkleyDetector(
            min_instances=30, delta=0.005, threshold=50.0
        ),
        "Page-Hinkley resized": lambda: PageHinkleyDetector(
            min_instances=10, delta=0.005, threshold=1.0
        ),
    }


def sweep(stream: np.ndarray, boundary: int, seed: int) -> dict:
    out = {}
    for name, factory in detectors(seed).items():
        d = factory()
        fired = [i for i, v in enumerate(stream) if d.update(float(v)).drift_detected]
        after = [f for f in fired if f >= boundary]
        out[name] = {
            "fired": fired,
            "first_after": after[0] if after else None,
            "delay": (after[0] - boundary) if after else None,
            "false_alarms": len([f for f in fired if f < boundary]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--showcase", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sdir = Path(args.showcase)
    res = json.load(open(sdir / "xgc_mesh_drift.json"))
    labs = res["labels"]
    refs = res["reference_cases"]
    dev = {lab: res["cases"][lab]["device"] for lab in labs}
    w = windowed_matrix(res)

    # --- device stream ----------------------------------------------------
    # Coverage against the closest pre-trained device, self included: a
    # pre-trained case IS covered, and saying so is what makes the baseline
    # meaningful rather than trivially large.
    order = [lab for lab in labs if res["cases"][lab]["in_pretraining"]] + [
        lab for lab in labs if not res["cases"][lab]["in_pretraining"]
    ]
    stream, bounds, cur = [], [], 0
    for lab in order:
        cov = np.min([w[lab][r] for r in refs if r in w[lab]], axis=0)
        stream.append(cov)
        cur += len(cov)
        bounds.append(
            {
                "label": lab,
                "end": cur,
                "in_pretraining": res["cases"][lab]["in_pretraining"],
            }
        )
    stream = np.concatenate(stream)
    boundary = next(
        b["end"] - (b["end"] - (bounds[i - 1]["end"] if i else 0))
        for i, b in enumerate(bounds)
        if not b["in_pretraining"]
    )
    dev_out = sweep(stream, boundary, args.seed)
    print(f"[device] {len(stream)} windows, change point at {boundary}")
    for n, v in dev_out.items():
        print(f"  {n:26} delay={v['delay']}  false_alarms={v['false_alarms']}")

    # --- same-machine control --------------------------------------------
    ctrl = None
    pair = next(
        (r for r in res["same_machine_pairs"] if dev[r["a"]] == dev[r["b"]]), None
    )
    if pair is not None:
        a, b = pair["a"], pair["b"]
        cstream = np.concatenate([w[a][a], w[b][a]])
        cbound = len(w[a][a])
        cout = sweep(cstream, cbound, args.seed)
        ctrl = {
            "reference": a,
            "second_case": b,
            "boundary": int(cbound),
            "stream": [float(x) for x in cstream],
            "detectors": cout,
        }
        print(f"\n[control] {a} -> {b} (same machine), change point {cbound}")
        for n, v in cout.items():
            print(f"  {n:26} fired={len(v['fired'])}  delay={v['delay']}")

    payload = {
        "device_stream": {
            "order": order,
            "bounds": bounds,
            "boundary": int(boundary),
            "stream": [float(x) for x in stream],
            "detectors": dev_out,
        },
        "same_machine_control": ctrl,
    }
    with open(sdir / "xgc_detectors.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[done] {sdir}/xgc_detectors.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
