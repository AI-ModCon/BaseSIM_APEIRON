#!/usr/bin/env python3
"""Behaviour probe: apeiron.training.continuous_trainer, off the pinned set.

The trainer regrew at 47% (A33) and is where A30 sits -- the forgetting
metric a continual learning paper reports, which was unverified until tests
were written for it. The campaign also saw `criterion` dropped from the
reconstruction's surface, a seam the layer exposes to callers.

Where this aims:

  compute_bwt     backward transfer, the number a paper publishes. Pure
                  arithmetic over the task registry: a mean of
                  R[T][i] - R[i][i]. Probed over an empty registry (documented
                  to return None), one task, several, a metric index past the
                  end, and diagonals of mismatched width -- none of which the
                  tests reach.
  _safe_next      restarts an exhausted loader and enforces a minimum batch
                  size, with a documented refusal when no batch in the loader
                  can satisfy it. Three branches, two of them error paths, and
                  the iterator it hands back is explicitly not the one it was
                  given.
  the loops       outer and inner, run over fixed data, reduced to the losses
                  and the resulting weights. With SGD and no momentum the
                  weights are an exact function of the data, so two
                  implementations of the same loop land on identical numbers.

Everything the layer prints is silenced; only the probe's JSON reaches
stdout.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _fixtures import (  # noqa: E402
    attempt, emit, fixed_loader, make_cfg, make_harness, params, show, silence,
)

silence()

import torch  # noqa: E402

from apeiron.logger.logger import Logger  # noqa: E402
from apeiron.profilers.count_flops import FLOPSProfiler  # noqa: E402
from apeiron.training.continuous_trainer import ContinuousTrainer  # noqa: E402


def built(extra: str = "", overrides=(), n: int = 8, batch: int = 4,
          profiler: bool = True):
    cfg = make_cfg(extra=extra, overrides=overrides)
    harness = make_harness(cfg, n=n, batch=batch)
    trainer = ContinuousTrainer(
        cfg, harness, Logger(backend="none"),
        FLOPSProfiler(warmup_iters=0) if profiler else None)
    return trainer, harness


def main() -> int:
    out = {}

    # --- 1. compute_bwt, the published number -----------------------------
    out["compute_bwt[no-tasks]"] = attempt(lambda: show(built()[0].compute_bwt()))

    def with_tasks(diagonals, metric_index: int = 0):
        trainer, harness = built()
        for diagonal in diagonals:
            harness.register_task(list(diagonal))
        return show(trainer.compute_bwt(metric_index))

    for name, diagonals in (
        ("one-task", [[50.0, 1.0]]),
        ("two-tasks", [[50.0, 1.0], [60.0, 2.0]]),
        ("three-tasks", [[50.0, 1.0], [60.0, 2.0], [70.0, 3.0]]),
        ("identical-diagonals", [[50.0, 1.0], [50.0, 1.0]]),
        ("zero-diagonals", [[0.0, 0.0], [0.0, 0.0]]),
        ("negative-diagonals", [[-10.0, -1.0], [-20.0, -2.0]]),
        ("huge-diagonals", [[1e9, 1.0], [1e9, 1.0]]),
        ("nan-diagonal", [[float("nan"), 1.0]]),
    ):
        out[f"compute_bwt[{name}]"] = attempt(lambda d=diagonals: with_tasks(d))

    for index in (0, 1, 2, 5, -1):
        out[f"compute_bwt[metric_index={index}]"] = attempt(
            lambda i=index: with_tasks([[50.0, 1.0], [60.0, 2.0]], i))
    out["compute_bwt[narrow-diagonal]"] = attempt(
        lambda: with_tasks([[50.0]], 0))
    out["compute_bwt[empty-diagonal]"] = attempt(lambda: with_tasks([[]], 0))
    # the eviction cap interacts with BWT: it averages over SURVIVORS
    def capped(cap: int):
        trainer, harness = built()
        harness.max_task_records = cap
        for value in (10.0, 20.0, 30.0, 40.0):
            harness.register_task([value, 1.0])
        return show({"bwt": show(trainer.compute_bwt(0)),
                     "retained": show(len(harness.task_diagonals))})
    for cap in (1, 2, 10):
        out[f"compute_bwt[cap={cap}]"] = attempt(lambda c=cap: capped(c))

    # --- 2. _safe_next, three branches and two error paths ----------------
    def safe(n: int, batch: int, min_batch=None, exhaust: bool = False):
        trainer, _ = built()
        loader = fixed_loader(n, batch)
        iterator = iter(loader)
        if exhaust:
            for _ in loader:
                pass
            iterator = iter([])
        got_iter, got_batch = trainer._safe_next(iterator, loader, min_batch)
        return show({"shapes": show([list(t.shape) for t in got_batch]),
                     "same_iterator": show(got_iter is iterator)})

    out["_safe_next[plain]"] = attempt(lambda: safe(8, 4))
    out["_safe_next[exhausted-restarts]"] = attempt(
        lambda: safe(8, 4, exhaust=True))
    out["_safe_next[min-satisfied]"] = attempt(lambda: safe(8, 4, min_batch=4))
    out["_safe_next[min-below-batch]"] = attempt(lambda: safe(8, 4, min_batch=2))
    # the documented refusal: no batch in the loader can satisfy min_batch
    out["_safe_next[min-impossible]"] = attempt(lambda: safe(8, 4, min_batch=16))
    out["_safe_next[ragged-last-batch,min=3]"] = attempt(
        lambda: safe(7, 3, min_batch=3))
    out["_safe_next[empty-loader]"] = attempt(lambda: safe(0, 4))
    out["_safe_next[empty-loader,min=1]"] = attempt(lambda: safe(0, 4, min_batch=1))
    out["_safe_next[min=0]"] = attempt(lambda: safe(8, 4, min_batch=0))
    out["_safe_next[min-negative]"] = attempt(lambda: safe(8, 4, min_batch=-1))

    # a batch whose second element has no shape: documented to be accepted
    def unshaped():
        trainer, _ = built()
        loader = [[torch.zeros(2, 4), "not-a-tensor"]]
        return attempt(lambda: trainer._safe_next(iter(loader), loader, 2))
    out["_safe_next[unshaped-second-element]"] = unshaped()

    # --- 3. the inner loop ------------------------------------------------
    def inner(n: int = 8, batch: int = 4, hist: bool = True, overrides=()):
        trainer, harness = built(overrides=overrides, n=n, batch=batch)
        cur, _ = harness.get_train_dataloaders()
        hist_loader = harness.get_hist_dataloaders()[0] if hist else None
        result = trainer.inner_cl_training_loop(
            0, cur, iter(cur), hist_loader,
            iter(hist_loader) if hist_loader is not None else None)
        losses = [round(float(x), 5) for x in result[:2]]
        return show({"losses": losses, "weights": params(harness.model)})

    out["inner[with-hist]"] = attempt(lambda: inner())
    out["inner[no-hist]"] = attempt(lambda: inner(hist=False))
    out["inner[single-sample]"] = attempt(lambda: inner(n=1, batch=1))
    for mode in ("base", "ewc_online", "kfac_online", "jvp_reg", "none"):
        out[f"inner[update_mode={mode}]"] = attempt(
            lambda m=mode: inner(overrides=["--set",
                                            f"continual_learning.update_mode={m}"]))
    for mix in ("true", "false"):
        out[f"inner[mix_historic_data={mix}]"] = attempt(
            lambda x=mix: inner(overrides=[
                "--set", f"continual_learning.mix_historic_data={x}"]))

    # --- 4. the outer loop ------------------------------------------------
    def outer(event: int = 0, overrides=(), n: int = 8, batch: int = 4):
        trainer, harness = built(overrides=overrides, n=n, batch=batch)
        iters = trainer.outer_cl_training_loop(event)
        return show({"iters": show(iters), "weights": params(harness.model)})

    out["outer[default]"] = attempt(lambda: outer())
    out["outer[event-id=7]"] = attempt(lambda: outer(event=7))
    for max_iter in ("1", "2", "5"):
        out[f"outer[max_iter={max_iter}]"] = attempt(
            lambda m=max_iter: outer(overrides=["--set", f"train.max_iter={m}"]))
    out["outer[max_iter=0]"] = attempt(
        lambda: outer(overrides=["--set", "train.max_iter=0"]))
    for mode in ("base", "ewc_online", "kfac_online", "jvp_reg", "none"):
        out[f"outer[update_mode={mode}]"] = attempt(
            lambda m=mode: outer(overrides=[
                "--set", f"continual_learning.update_mode={m}",
                "--set", "train.max_iter=2"]))
    out["outer[empty-stream]"] = attempt(lambda: outer(n=0))
    out["outer[no-profiler]"] = attempt(
        lambda: show(ContinuousTrainer(
            make_cfg(), make_harness(), Logger(backend="none"), None)
            .outer_cl_training_loop(0)))

    # --- 5. the surface callers hold ---------------------------------------
    trainer, harness = built()
    out["surface[criterion-present]"] = show(
        type(getattr(trainer, "criterion", None)).__name__)
    out["surface[optimizer-present]"] = show(
        type(getattr(trainer, "optimizer", None)).__name__)
    out["surface[cl_updater-present]"] = show(
        type(getattr(trainer, "cl_updater", None)).__name__)
    out["surface[profiler-present]"] = show(
        type(getattr(trainer, "profiler", None)).__name__)
    out["surface[attrs]"] = show(sorted(
        a for a in vars(trainer) if not a.startswith("__")))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
