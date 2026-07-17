#!/usr/bin/env python3
"""Compare saved MATEY pred/target artifacts across baseline vs shift streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.matey.solps_field_maps import field_names_for_maps, squeeze_solps_field_maps


def _load_manifest(manifest_path: Path) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _nrmse_per_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-7
    values = []
    for idx in range(pred.shape[0]):
        diff = pred[idx] - target[idx]
        num = np.mean(diff**2)
        den = np.mean(target[idx] ** 2) + eps
        values.append(np.sqrt(num / den))
    return np.asarray(values, dtype=np.float64)


def _summarize_domain(rows: list[dict], root: Path) -> dict[str, float]:
    if not rows:
        return {}
    field_means: dict[str, list[float]] = {}
    mean_vals: list[float] = []
    for row in rows:
        npz = np.load(root / row["path"], allow_pickle=True)
        pred = squeeze_solps_field_maps(npz["pred"])
        target = squeeze_solps_field_maps(npz["target"])
        per_field = _nrmse_per_field(pred, target)
        names = field_names_for_maps(pred)
        for name, val in zip(names, per_field):
            field_means.setdefault(name, []).append(float(val))
        mean_vals.append(float(np.mean(per_field)))
    out = {f"nrmse_{name}": float(np.mean(vals)) for name, vals in field_means.items()}
    out["nrmse_mean"] = float(np.mean(mean_vals))
    out["n_batches"] = float(len(rows))
    return out


def _pick_rows(rows: list[dict], domain: str, batch_idx: int) -> dict | None:
    matches = [
        r
        for r in rows
        if r.get("domain") == domain and int(r.get("stream_batch_idx", -1)) == batch_idx
    ]
    if not matches:
        return None
    return matches[0]


def plot_field_comparison(
    root: Path,
    baseline_row: dict,
    shift_row: dict,
    output_path: Path,
    *,
    field_idx: int = 0,
) -> None:
    base_npz = np.load(root / baseline_row["path"], allow_pickle=True)
    shift_npz = np.load(root / shift_row["path"], allow_pickle=True)
    pred_b = squeeze_solps_field_maps(base_npz["pred"])
    pred_s = squeeze_solps_field_maps(shift_npz["pred"])
    gt_b = squeeze_solps_field_maps(base_npz["target"])
    gt_s = squeeze_solps_field_maps(shift_npz["target"])
    names = field_names_for_maps(pred_b)
    field_name = names[field_idx]

    gt_base = gt_b[field_idx]
    gt_shift = gt_s[field_idx]
    pred_base = pred_b[field_idx]
    pred_shift = pred_s[field_idx]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (axes[0, 0], gt_base, f"GT baseline ({field_name})"),
        (axes[0, 1], gt_shift, f"GT shift ({field_name})"),
        (axes[0, 2], gt_shift - gt_base, "GT shift − baseline"),
        (axes[1, 0], pred_base, f"Pred baseline ({field_name})"),
        (axes[1, 1], pred_shift, f"Pred shift ({field_name})"),
        (axes[1, 2], pred_shift - pred_base, "Pred shift − baseline"),
    ]
    for ax, arr, title in panels:
        im = ax.imshow(arr, origin="lower", aspect="auto")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("output/matey_inference_drift_artifacts"),
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="Stream-local batch index to compare (same index in baseline/shift).",
    )
    parser.add_argument(
        "--field-idx",
        type=int,
        default=0,
        help="Field index: 0=ne2d, 1=te2d, 2=ti2d.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/matey_inference_drift_field_compare.png"),
    )
    args = parser.parse_args()

    root = args.artifacts_dir.resolve()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Re-run inference with [eval_outputs].enabled=true."
        )

    rows = _load_manifest(manifest_path)
    baseline_rows = [r for r in rows if r.get("domain") == "baseline"]
    shift_rows = [r for r in rows if r.get("domain") == "shift"]

    print(f"Artifacts root: {root}")
    print(f"Saved batches: {len(rows)} (baseline={len(baseline_rows)}, shift={len(shift_rows)})")

    base_summary = _summarize_domain(baseline_rows, root)
    shift_summary = _summarize_domain(shift_rows, root)
    print("\nRecomputed NRMSE from saved pred/target:")
    for key in sorted(set(base_summary) | set(shift_summary)):
        if key == "n_batches":
            continue
        b = base_summary.get(key, float("nan"))
        s = shift_summary.get(key, float("nan"))
        print(f"  {key:12s}  baseline={b:.6f}  shift={s:.6f}  delta={s - b:+.6f}")

    baseline_row = _pick_rows(rows, "baseline", args.batch_idx)
    shift_row = _pick_rows(rows, "shift", args.batch_idx)
    if baseline_row and shift_row:
        plot_field_comparison(
            root,
            baseline_row,
            shift_row,
            args.output,
            field_idx=args.field_idx,
        )
        print(f"\nSaved field comparison plot: {args.output}")
    else:
        print(
            f"\nNo baseline/shift pair for batch_idx={args.batch_idx}; "
            "skipping field plot."
        )


if __name__ == "__main__":
    main()
