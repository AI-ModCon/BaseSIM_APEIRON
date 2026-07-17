"""Save MATEY inference predictions and ground truth for offline comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from config.configuration import EvalOutputsCfg
from examples.matey.solps_field_maps import (
    SOLPS_ION_FIELD_NAMES,
    field_names_for_maps,
    squeeze_solps_field_maps,
)


@dataclass
class EvalArtifactContext:
    stream_id: int
    domain: str
    data_root: str
    stream_batch_idx: int
    global_batch_idx: int


class MateyEvalArtifactWriter:
    """Write per-batch pred/target NPZ files plus a JSONL manifest."""

    def __init__(self, cfg: EvalOutputsCfg):
        self.cfg = cfg
        self.root = Path(cfg.dir)
        if not self.root.is_absolute():
            self.root = Path.cwd() / self.root
        self.root = self.root.resolve()
        self._manifest_path = self.root / "manifest.jsonl"
        self._saved_this_stream = 0

    def reset_stream(self) -> None:
        self._saved_this_stream = 0

    def maybe_save(
        self,
        *,
        pred: Tensor,
        target: Tensor,
        metrics: dict[str, float],
        ctx: EvalArtifactContext,
    ) -> Path | None:
        if not self.cfg.enabled:
            return None
        if self._saved_this_stream >= self.cfg.max_batches_per_stream:
            return None
        if ctx.stream_batch_idx % self.cfg.save_stride != 0:
            return None

        stream_dir = self.root / f"stream_{ctx.stream_id:02d}_{ctx.domain}"
        stream_dir.mkdir(parents=True, exist_ok=True)
        out_path = stream_dir / f"batch_{ctx.stream_batch_idx:05d}.npz"

        pred_np = squeeze_solps_field_maps(pred.detach().cpu().float().numpy())
        target_np = squeeze_solps_field_maps(target.detach().cpu().float().numpy())
        field_names = field_names_for_maps(pred_np)

        np.savez_compressed(
            out_path,
            pred=pred_np,
            target=target_np,
            field_names=np.array(field_names, dtype=object),
        )

        record: dict[str, Any] = {
            "path": str(out_path.relative_to(self.root)),
            "stream_id": ctx.stream_id,
            "domain": ctx.domain,
            "data_root": ctx.data_root,
            "stream_batch_idx": ctx.stream_batch_idx,
            "global_batch_idx": ctx.global_batch_idx,
            "pred_shape": list(pred_np.shape),
            "target_shape": list(target_np.shape),
            "field_names": field_names,
            "metrics": metrics,
        }
        with self._manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        self._saved_this_stream += 1
        return out_path
