"""Tests for the Well scaling benchmark (examples/well/benchmark.py).

Gated on h5py; runs the single-process modes against the fixture store. The
distributed throughput mode is exercised by the distributed smoke, not here.
"""

from __future__ import annotations

import glob

import pytest

pytest.importorskip("h5py")

from examples.well.benchmark import (  # noqa: E402
    bench_frontier,
    bench_memory,
    bench_resume,
    bench_throughput,
)
from examples.well.convert import convert_files  # noqa: E402
from examples.well.fixture import generate_fixture  # noqa: E402


@pytest.fixture()
def well_store(tmp_path):
    generate_fixture(
        tmp_path / "src", tcools=(0.03, 0.3, 1.0), n_time=16, height=8, width=12
    )
    convert_files(
        sorted(glob.glob(str(tmp_path / "src" / "*.hdf5"))),
        tmp_path / "store",
        window_steps=8,
        val_fraction=0.25,
    )
    return tmp_path / "store"


class TestBenchmark:
    def test_memory_pointer_flat_vs_copy_grows(self, well_store):
        rows = bench_memory(str(well_store), str(well_store.parent / "m.csv"), [1, 20])
        by = {(r["mode"], r["tasks"]): r for r in rows}
        # predicted copy cost scales with task count; pointer adds ~nothing.
        assert (
            by[("copy", 20)]["predicted_copy_mb"] > by[("copy", 1)]["predicted_copy_mb"]
        )
        assert (
            by[("pointer", 20)]["rss_added_mb"] <= by[("copy", 20)]["rss_added_mb"] + 1
        )

    def test_frontier_more_adaptation_lowers_error(self, well_store):
        rows = bench_frontier(str(well_store), str(well_store.parent / "f.csv"), [0, 6])
        never = next(r for r in rows if r["budget"] == 0)
        always = next(r for r in rows if r["budget"] == 6)
        assert always["final_vrmse"] < never["final_vrmse"]

    def test_resume_reloads_task_history(self, well_store):
        rows = bench_resume(
            str(well_store),
            str(well_store.parent / "r.csv"),
            str(well_store.parent / "ck"),
        )
        row = rows[0]
        assert row["tasks_reloaded"] == row["tasks_before"] > 0
        assert row["diagonals_match"] is True

    def test_throughput_pure_counts_real_samples(self, well_store):
        rows = bench_throughput(
            str(well_store),
            str(well_store.parent / "t.csv"),
            width=8,
            batch_size=6,
            max_iter=3,
            pure=True,
        )
        row = rows[0]  # single process -> rank 0 writes the row
        assert row["world_size"] == 1
        assert row["pure_throughput"] == 1
        assert row["train_samples"] > 0  # real samples, not batches * batch_size
        assert row["stream_samples"] > 0
        assert "final_vrmse" in row  # still reported even with transfer eval off
