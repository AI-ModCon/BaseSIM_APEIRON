"""Measure the memory-scaling branch on the Well surrogate task.

Four modes, one CLI. Each writes a CSV row (or rows) to ``--out``.

* ``memory``    -- peak/added RSS as the retained-task count grows, **pointer**
                   (new) vs **copy** (pre-Phase-1) task eval sets. The headline
                   memory capability, as a controlled A/B on the same run.
* ``throughput``-- wall-clock + samples/s + per-phase and collective timings for
                   one run. Launch under ``torchrun``/``srun`` and vary the world
                   size for strong/weak scaling; the Amdahl breakdown falls out
                   of ``comm.timings()``.
* ``frontier``  -- final error vs adaptation budget (the accuracy-cost frontier
                   ``cl_only`` exists to trace). Run at several world sizes to
                   check the frontier is preserved.
* ``resume``    -- kill/resume continuity: the forgetting history (task records)
                   and model survive a restart.

Everything runs on the fixture (no download) for a quick local smoke; point
``--store`` at a real converted Well store and scale ``--width``/data on GPU.

    poetry run python -m examples.well.benchmark memory --store /tmp/wellstore \
        --out out/mem.csv --task-counts 1,5,20,50
    torchrun --nproc_per_node=4 -m examples.well.benchmark throughput \
        --store /data/wellstore --out out/scale.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import psutil

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)
from apeiron.distributed import comm
from apeiron.driver.schedules import NeverSchedule, PeriodicSchedule
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction
from apeiron.driver.trigger_policy import SchedulePolicy
from apeiron.logger.logger import Logger
from examples.well.model import WellHarness


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _max_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS/BSD reports bytes.
    return ru / 1e3 if sys.platform.startswith("linux") else ru / 1e6


def _cfg(
    store: str,
    *,
    batch_size: int = 8,
    max_iter: int = 20,
    width: int = 16,
    depth: int = 3,
    max_stream_updates: int = 12,
    ckpts_path: str = "",
    max_ckpts: int = 0,
    device: str = "",
) -> Config:
    # Match the device to the collective backend (cuda for nccl, cpu for gloo);
    # a GPU node with an nccl group and cpu tensors would fail the collectives.
    resolved_device = device or str(comm.device())
    return Config(
        model=ModelCfg(
            name="well_surrogate",
            width=width,
            depth=depth,
            ckpts_path=ckpts_path,
            max_ckpts=max_ckpts,
        ),
        data=DataCfg(
            name="well", path=store, window_store_path=store, batch_size=batch_size
        ),
        train=TrainCfg(
            batch_size=batch_size, num_workers=0, init_lr=1e-3, max_iter=max_iter
        ),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(
            detector_name="PageHinkleyDetector",
            detection_interval=3,
            metric_index=0,
            aggregation="mean",
            max_stream_updates=max_stream_updates,
        ),
        seed=0,
        device=resolved_device,
    )


def _silent() -> Logger:
    return Logger(verbosity="ERROR", backend="none", csv_path=None)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# memory A/B: pointer vs copy task eval sets
# ---------------------------------------------------------------------------


def bench_memory(store: str, out: str, task_counts: list[int]) -> list[dict]:
    rows = []
    for copy in (False, True):
        for k in task_counts:
            h = WellHarness(_cfg(store))
            h._copy_task_evalsets = copy
            h.max_task_records = k + 1  # do not evict during the sweep
            h.update_data_stream()
            assert h.current_window_id is not None
            val_x, _ = h.store.window(h.current_window_id).load_full(h.val_split)
            val_mb = val_x.numpy().nbytes / 1e6

            gc.collect()
            rss0 = _rss_mb()
            for _ in range(k):
                # placeholder diagonal -- we are measuring the eval-set retention,
                # not the metric; register freezes a pointer or an in-RAM copy.
                h.register_task([0.0], window_id=h.current_window_id)
            gc.collect()
            added = _rss_mb() - rss0

            rows.append(
                {
                    "mode": "copy" if copy else "pointer",
                    "tasks": k,
                    "val_split_mb": round(val_mb, 3),
                    "rss_added_mb": round(added, 2),
                    "max_rss_mb": round(_max_rss_mb(), 1),
                    "predicted_copy_mb": round(val_mb * k, 2),
                }
            )
            del h
    _write_csv(out, rows)
    return rows


# ---------------------------------------------------------------------------
# throughput / scaling (launch under torchrun for world_size > 1)
# ---------------------------------------------------------------------------


def bench_throughput(
    store: str, out: str, *, width: int, batch_size: int, max_iter: int
) -> list[dict]:
    comm.init_from_env()
    comm.time_collectives = True
    comm.reset_timings()

    cfg = _cfg(
        store,
        batch_size=batch_size,
        width=width,
        max_iter=max_iter,
        max_stream_updates=_n_windows(store),
    )
    h = WellHarness(cfg)
    engine = StreamEngine(
        cfg, h, SchedulePolicy(PeriodicSchedule(1)), AdaptAction(), logger=_silent()
    )

    comm.barrier()
    t0 = time.perf_counter()
    summary = engine.run()
    comm.barrier()
    dt = time.perf_counter() - t0

    rows: list[dict] = []
    if comm.is_main:
        samples = engine.batch_count * batch_size
        timings = comm.timings()
        rows.append(
            {
                "world_size": comm.world_size,
                "wall_s": round(dt, 3),
                "windows": summary["stream_updates"],
                "global_batches": engine.batch_count,
                "samples": samples,
                "samples_per_s": round(samples / dt, 1) if dt else 0.0,
                "fires": summary["fires"],
                "comm_all_gather_s": round(timings.get("all_gather", 0.0), 4),
                "comm_broadcast_s": round(timings.get("broadcast", 0.0), 4),
                "comm_all_reduce_s": round(timings.get("all_reduce", 0.0), 4),
                "final_vrmse": round(float(summary["final_metrics"][0]), 5),
            }
        )
        _write_csv(out, rows)
    comm.shutdown()
    return rows


# ---------------------------------------------------------------------------
# accuracy-cost frontier
# ---------------------------------------------------------------------------


def bench_frontier(store: str, out: str, budgets: list[int]) -> list[dict]:
    n = _n_windows(store)
    rows = []
    for budget in budgets:
        cfg = _cfg(store, max_stream_updates=n)
        h = WellHarness(cfg)
        if budget <= 0:
            policy = SchedulePolicy(NeverSchedule())
        else:
            period = max(1, round(n / budget))
            policy = SchedulePolicy(PeriodicSchedule(period))
        engine = StreamEngine(cfg, h, policy, AdaptAction(), logger=_silent())
        summary = engine.run()
        rows.append(
            {
                "world_size": comm.world_size,
                "budget": budget,
                "triggers": summary["fires"],
                "final_vrmse": round(float(summary["final_metrics"][0]), 5),
                "final_mae": round(float(summary["final_metrics"][1]), 5),
            }
        )
    _write_csv(out, rows)
    return rows


# ---------------------------------------------------------------------------
# resumability
# ---------------------------------------------------------------------------


def bench_resume(store: str, out: str, ckpt_dir: str) -> list[dict]:
    n = _n_windows(store)
    half = max(1, n // 2)
    cfg = _cfg(store, ckpts_path=ckpt_dir, max_ckpts=3, max_stream_updates=n)

    # Run the first half with checkpointing + task-record persistence.
    h1 = WellHarness(cfg)
    e1 = StreamEngine(
        replace(
            cfg, drift_detection=replace(cfg.drift_detection, max_stream_updates=half)
        ),
        h1,
        SchedulePolicy(PeriodicSchedule(1)),
        AdaptAction(),
        logger=_silent(),
    )
    s1 = e1.run()
    tasks_before = len(h1._task_records)

    # "Restart": a fresh harness reloads the persisted forgetting history.
    h2 = WellHarness(cfg)
    reloaded = h2.load_task_records()
    diag_match = [r.diagonal for r in h2._task_records] == [
        r.diagonal for r in h1._task_records
    ]

    rows = [
        {
            "windows_before_kill": s1["stream_updates"],
            "tasks_before": tasks_before,
            "tasks_reloaded": reloaded,
            "diagonals_match": diag_match,
            "bwt_recomputable": h2.eval_past_tasks() is not None,
        }
    ]
    _write_csv(out, rows)
    return rows


def _n_windows(store: str) -> int:
    from apeiron.data.window_store import WindowStore

    return len(WindowStore(store, catalog=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["memory", "throughput", "frontier", "resume"])
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task-counts", default="1,5,20,50")
    p.add_argument("--budgets", default="0,1,3,6")
    p.add_argument("--ckpt-dir", default="output/well_resume_ckpts")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=20)
    args = p.parse_args(argv)

    if args.mode == "memory":
        bench_memory(
            args.store, args.out, [int(x) for x in args.task_counts.split(",")]
        )
    elif args.mode == "throughput":
        bench_throughput(
            args.store,
            args.out,
            width=args.width,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
        )
    elif args.mode == "frontier":
        bench_frontier(args.store, args.out, [int(x) for x in args.budgets.split(",")])
    elif args.mode == "resume":
        bench_resume(args.store, args.out, args.ckpt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
