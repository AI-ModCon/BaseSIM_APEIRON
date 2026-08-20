"""Real multi-process smoke for the distributed path (CPU / gloo).

Launch with torchrun (or srun). Verifies, with >1 rank:
  * the harness shards the window so shards exactly cover it,
  * the monitoring loop + data-parallel adaptation complete without deadlock,
  * model parameters stay identical across ranks (rank-0 broadcast at the start
    of each adapt + gradient all-reduce each step).

    DIST_SMOKE_STORE=/tmp/store poetry run torchrun \
        --nproc_per_node=2 --nnodes=1 tests/dist_smoke.py

Exits non-zero on any failed assertion (rank 0 prints the verdict).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)
from apeiron.data.window_store import WindowStore
from apeiron.data.windowed_harness import WindowedHarness
from apeiron.distributed import comm
from apeiron.driver.schedules import PeriodicSchedule
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction
from apeiron.driver.trigger_policy import SchedulePolicy
from apeiron.evaluation.metrics import accuracy
from apeiron.logger.logger import Logger


def main() -> None:
    comm.init_from_env()
    assert comm.is_distributed, "run under torchrun --nproc_per_node>=2"

    store_path = os.environ["DIST_SMOKE_STORE"]

    # Rank 0 writes the immutable windows; other ranks wait for them.
    if comm.is_main:
        store = WindowStore(store_path, catalog=False)
        rng = np.random.default_rng(0)
        for i in range(3):
            x = rng.standard_normal((40, 4)).astype(np.float32)
            y = rng.integers(0, 3, 40).astype(np.int64)
            store.commit(x, y, val_fraction=0.5, t_start=f"w{i}", t_end=f"w{i}")
    comm.barrier()

    cfg = Config(
        model=ModelCfg(name="tiny"),
        data=DataCfg(name="win", path=store_path, batch_size=5),
        train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.1, max_iter=3),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(max_stream_updates=3),
        seed=0,
        device="cpu",
    )

    store = WindowStore(store_path, catalog=False)

    # Deliberately different init per rank -> the broadcast at the start of
    # adaptation must reconcile them; if it does not, the sync check below fails.
    torch.manual_seed(comm.rank + 1)
    model = nn.Linear(4, 4)
    for p in model.parameters():
        nn.init.normal_(p, std=1.0)

    harness = WindowedHarness(
        cfg,
        model,
        store,
        criterion=nn.CrossEntropyLoss(),
        optimizer_factory=lambda m: SGD(m.parameters(), lr=0.1),
        eval_metrics={"accuracy": accuracy},
    )

    # 1) Shards exactly cover the window.
    handle = store.window(store.window_ids()[0])
    local_n = int(len(handle.indices("all", shard=comm.shard())))
    shard_sizes = comm.all_gather_object(local_n)

    logger = Logger(verbosity="ERROR", backend="none", csv_path=None)
    engine = StreamEngine(
        cfg,
        harness,
        SchedulePolicy(PeriodicSchedule(1)),  # adapt every window
        AdaptAction(),
        logger=logger,
    )
    summary = engine.run()

    # 2) Parameters identical across ranks after data-parallel training.
    flat = torch.cat([p.detach().reshape(-1) for p in harness.model.parameters()])
    all_params = comm.all_gather_object(flat.tolist())

    if comm.is_main:
        assert sum(shard_sizes) == handle.manifest.n_samples, (
            shard_sizes,
            handle.manifest.n_samples,
        )
        assert max(shard_sizes) - min(shard_sizes) <= 1, shard_sizes
        ref = np.array(all_params[0])
        for other in all_params[1:]:
            assert np.allclose(ref, np.array(other), atol=1e-5), "params diverged!"
        assert summary["fires"] == 3
        print(
            f"world_size={comm.world_size} shard_sizes={shard_sizes} "
            f"window_n={handle.manifest.n_samples}"
        )
        print(
            f"fires={summary['fires']} decisions={summary['decision_points']} "
            f"batches={summary['batches']}"
        )
        print("params in sync across ranks: True")
        print("DIST SMOKE OK")

    comm.shutdown()


if __name__ == "__main__":
    main()
