from __future__ import annotations

from pathlib import Path

import torch

from config.configuration import EvalOutputsCfg
from examples.matey.src.eval_artifacts import EvalArtifactContext, MateyEvalArtifactWriter


def test_matey_eval_artifact_writer_saves_npz_and_manifest(tmp_path: Path):
    cfg = EvalOutputsCfg(
        enabled=True,
        dir=str(tmp_path / "artifacts"),
        max_batches_per_stream=2,
        save_stride=1,
    )
    writer = MateyEvalArtifactWriter(cfg)
    writer.reset_stream()

    pred = torch.randn(1, 3, 4, 5)
    target = torch.randn(1, 3, 4, 5)
    metrics = {"nrmse_mean": 0.42, "nrmse_ne2d": 0.5, "nrmse_te2d": 0.3, "nrmse_ti2d": 0.45}
    ctx = EvalArtifactContext(
        stream_id=1,
        domain="baseline",
        data_root="/tmp/baseline",
        stream_batch_idx=0,
        global_batch_idx=2,
    )

    out = writer.maybe_save(pred=pred, target=target, metrics=metrics, ctx=ctx)
    assert out is not None
    assert out.exists()

    skipped = writer.maybe_save(
        pred=pred,
        target=target,
        metrics=metrics,
        ctx=EvalArtifactContext(
            stream_id=1,
            domain="baseline",
            data_root="/tmp/baseline",
            stream_batch_idx=2,
            global_batch_idx=4,
        ),
    )
    assert skipped is None

    manifest = (tmp_path / "artifacts" / "manifest.jsonl").read_text(encoding="utf-8")
    assert "baseline" in manifest
    assert "nrmse_mean" in manifest

    loaded = __import__("numpy").load(out, allow_pickle=True)
    assert loaded["pred"].shape == (3, 4, 5)
    assert loaded["target"].shape == (3, 4, 5)
    assert list(loaded["field_names"]) == ["ne2d", "te2d", "ti2d"]
