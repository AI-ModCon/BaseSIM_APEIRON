#!/usr/bin/env python3
"""2D SOLPS field maps: baseline vs shift (GT, pred, and domain diffs).

Reads saved inference artifacts (manifest.jsonl + NPZ). Fields are shown on the
MATEY 2D patch grid (normalized ne2d/te2d/ti2d), typically 38×38 for SOLPS2DwION.
"""

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


def _pick_row(rows: list[dict], domain: str, batch_idx: int) -> dict | None:
    matches = [
        r
        for r in rows
        if r.get("domain") == domain and int(r.get("stream_batch_idx", -1)) == batch_idx
    ]
    return matches[0] if matches else None


def _load_maps(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz = np.load(root / row["path"], allow_pickle=True)
    pred = squeeze_solps_field_maps(npz["pred"])
    target = squeeze_solps_field_maps(npz["target"])
    names = field_names_for_maps(pred)
    return pred, target, names


def _robust_limits(*arrays: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = [a[np.isfinite(a)] for a in arrays if a.size]
    if not finite:
        return 0.0, 1.0
    stacked = np.concatenate([a.ravel() for a in finite])
    vmin, vmax = float(np.percentile(stacked, 2)), float(np.percentile(stacked, 98))
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        return -bound, bound
    if vmin >= vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_baseline_vs_shift(
    artifacts_dir: Path,
    output_path: Path,
    *,
    batch_idx: int = 0,
    fields: list[str] | None = None,
) -> None:
    root = artifacts_dir.resolve()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    rows = _load_manifest(manifest_path)
    base_row = _pick_row(rows, "baseline", batch_idx)
    shift_row = _pick_row(rows, "shift", batch_idx)
    if base_row is None or shift_row is None:
        raise ValueError(
            f"No baseline/shift pair for batch_idx={batch_idx}. "
            f"Have domains: {sorted({r.get('domain') for r in rows})}"
        )

    pred_b, gt_b, names = _load_maps(root, base_row)
    pred_s, gt_s, _ = _load_maps(root, shift_row)
    if fields is None:
        fields = names
    field_indices = [names.index(f) for f in fields if f in names]
    if not field_indices:
        raise ValueError(f"No matching fields in {names}")

    n_rows = len(field_indices)
    col_titles = [
        "GT baseline",
        "GT shift",
        "GT shift − baseline",
        "Pred baseline",
        "Pred shift",
        "Pred shift − baseline",
        "|Pred−GT| baseline",
        "|Pred−GT| shift",
    ]
    fig, axes = plt.subplots(
        n_rows,
        len(col_titles),
        figsize=(2.2 * len(col_titles), 2.4 * n_rows),
        squeeze=False,
    )

    h, w = gt_b.shape[-2], gt_b.shape[-1]
    extent = [0, w, 0, h]

    for row_i, fidx in enumerate(field_indices):
        name = names[fidx]
        panels = [
            (gt_b[fidx], "field", f"{name}: GT baseline"),
            (gt_s[fidx], "field", f"{name}: GT shift"),
            (gt_s[fidx] - gt_b[fidx], "diff", f"{name}: ΔGT"),
            (pred_b[fidx], "field", f"{name}: pred baseline"),
            (pred_s[fidx], "field", f"{name}: pred shift"),
            (pred_s[fidx] - pred_b[fidx], "diff", f"{name}: Δpred"),
            (np.abs(pred_b[fidx] - gt_b[fidx]), "error", f"{name}: |err| base"),
            (np.abs(pred_s[fidx] - gt_s[fidx]), "error", f"{name}: |err| shift"),
        ]
        for col_i, (arr, kind, _) in enumerate(panels):
            ax = axes[row_i, col_i]
            if kind == "diff":
                vmin, vmax = _robust_limits(arr, symmetric=True)
                cmap = "RdBu_r"
            elif kind == "error":
                vmin, vmax = 0.0, float(np.percentile(arr[np.isfinite(arr)], 98))
                cmap = "magma"
            else:
                vmin, vmax = _robust_limits(gt_b[fidx], gt_s[fidx], pred_b[fidx], pred_s[fidx])
                cmap = "viridis"
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="equal",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=8)
            if col_i == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.set_xlabel("nx")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    metrics_b = base_row.get("metrics", {})
    metrics_s = shift_row.get("metrics", {})
    fig.suptitle(
        "Baseline vs shift — 2D SOLPS fields (normalized patch grid)\n"
        f"batch_idx={batch_idx}  "
        f"nrmse_mean: baseline={metrics_b.get('nrmse_mean', float('nan')):.4f}  "
        f"shift={metrics_s.get('nrmse_mean', float('nan')):.4f}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gt_domain_diff_summary(
    artifacts_dir: Path,
    output_path: Path,
) -> None:
    """Mean |GT_shift − GT_baseline| per field across all saved batch pairs."""
    root = artifacts_dir.resolve()
    rows = _load_manifest(root / "manifest.jsonl")
    base_by_idx = {int(r["stream_batch_idx"]): r for r in rows if r.get("domain") == "baseline"}
    shift_by_idx = {int(r["stream_batch_idx"]): r for r in rows if r.get("domain") == "shift"}
    common = sorted(set(base_by_idx) & set(shift_by_idx))
    if not common:
        raise ValueError("No overlapping batch indices between baseline and shift")

    diffs: dict[str, list[float]] = {}
    for idx in common:
        _, gt_b, names = _load_maps(root, base_by_idx[idx])
        _, gt_s, _ = _load_maps(root, shift_by_idx[idx])
        for fidx, name in enumerate(names):
            val = float(np.mean(np.abs(gt_s[fidx] - gt_b[fidx])))
            diffs.setdefault(name, []).append(val)

    names = list(diffs.keys())
    means = [float(np.mean(diffs[n])) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(names)])
    ax.set_ylabel("mean |GT_shift − GT_baseline|")
    ax.set_title("Domain shift in ground truth (saved batches)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/matey_inference_drift_latest"),
        help="Timestamped run dir or artifacts/ path.",
    )
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument(
        "--fields",
        type=str,
        default="ne2d,te2d,ti2d",
        help="Comma-separated field names to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Main 2D comparison PNG (default: <run-dir>/baseline_vs_shift_2d.png).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if (run_dir / "manifest.jsonl").exists():
        artifacts_dir = run_dir
    elif (run_dir / "artifacts" / "manifest.jsonl").exists():
        artifacts_dir = run_dir / "artifacts"
    else:
        raise FileNotFoundError(f"No manifest under {run_dir} or {run_dir}/artifacts")

    run_root = artifacts_dir.parent if artifacts_dir.name == "artifacts" else artifacts_dir
    out_main = args.output or (run_root / "baseline_vs_shift_2d.png")
    out_summary = run_root / "baseline_vs_shift_gt_diff_summary.png"

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    plot_baseline_vs_shift(artifacts_dir, out_main, batch_idx=args.batch_idx, fields=fields)
    plot_gt_domain_diff_summary(artifacts_dir, out_summary)

    print(f"Saved {out_main}")
    print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()
