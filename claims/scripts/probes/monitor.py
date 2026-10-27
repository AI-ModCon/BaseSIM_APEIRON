#!/usr/bin/env python3
"""Behaviour probe: apeiron.driver.continuous_monitor, off the pinned set.

The monitor is the framework's main loop and the campaign's worst-regrowing
layer (A33: 42%, the lowest of the ten). A29 recorded that the loop was not
pinned by the layer that owns it, and tests were written in response; the
campaign also saw `label` and `last_agg_metric` dropped in 3 of 3 draws --
seams a subclass depends on, which no criterion mentions.

Where this aims:

  aggregation       `_aggregate_buffer` reduces the buffered per-batch
                    metrics to the one scalar the detector sees. It
                    recognises "mean", "median" and "last", and silently
                    falls back to the mean for anything else -- which is A20,
                    a misspelled `aggregation` becoming the mean with no
                    warning. Probed across the documented names, misspellings,
                    casings, and an empty buffer (documented to raise).
  stop condition    `_should_stop` compares a count against a cap; the
                    boundary is off-by-one territory and nothing pins it.
  the subclass seams  `label`, `last_agg_metric`, and the counters, which the
                    side runners (`drift_only.py`, `cl_only.py`) read.
  the whole loop    `run()` over a fixed stream with a real detector,
                    reduced to the counters and the final weights. The loop
                    is deterministic here because the stream, the model and
                    the detector are all fixed.

Everything the layer prints is silenced; only the probe's JSON reaches
stdout.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _fixtures import (  # noqa: E402
    attempt, emit, make_cfg, make_harness, params, show, silence,
)

silence()

from apeiron.driver.continuous_monitor import ContinuousMonitor  # noqa: E402


def built(extra: str = "", overrides=(), n: int = 16, batch: int = 4):
    cfg = make_cfg(extra=extra, overrides=overrides)
    return ContinuousMonitor(cfg, make_harness(cfg, n=n, batch=batch))


#: Every name a subclass reads off the monitor. Each is read through
#: `getattr` with a sentinel, and that is the whole point: the first version
#: of this helper guarded `label` and `last_agg_metric` but read the other
#: eight directly, so an implementation missing any ONE of them raised
#: inside the dict comprehension and took every case that calls `state()`
#: down with it -- including the twelve `run[*]` cases, which are the only
#: ones that exercise the loop end to end. Twelve cases reported a
#: divergence about the monitor's loop while measuring nothing about the
#: loop at all. A missing name must cost exactly one case.
SEAMS = ("label", "last_agg_metric", "stream_update_count", "batch_count",
         "check_count", "drift_event_count", "metric_idx",
         "detection_interval", "max_stream_updates", "aggregation")


def state(monitor) -> str:
    """The counters and seams a subclass reads, each independently reported."""
    return show({name: show(getattr(monitor, name, "<missing>"))
                 for name in SEAMS})


def main() -> int:
    out = {}

    # --- 1. the aggregation table, including A20's silent fallback --------
    buffer = [[1.0, 10.0], [5.0, 50.0], [3.0, 30.0], [2.0, 20.0]]
    for name in ("mean", "median", "last", "MEAN", "Median", " mean ",
                 "mien", "avg", "average", "first", "max", "", "none"):
        monitor = built(overrides=["--set", f"drift_detection.aggregation={name}"])
        monitor.metric_buffer = [list(row) for row in buffer]
        out[f"_aggregate_buffer[{name!r}]"] = attempt(
            lambda m=monitor: m._aggregate_buffer())
        # and the buffer must be cleared whatever the method
        out[f"_aggregate_buffer[{name!r},cleared]"] = show(monitor.metric_buffer)

    # even-length median, which is where a mean-of-middle-two shows up
    for rows, label in (([[1.0], [2.0]], "two"),
                        ([[1.0], [2.0], [3.0], [4.0]], "four"),
                        ([[1.0]], "one"),
                        ([[1.0], [1.0], [1.0]], "constant"),
                        ([[-5.0], [5.0]], "straddling-zero"),
                        ([[float("nan")], [1.0]], "nan"),
                        ([[float("inf")], [1.0]], "inf")):
        for method in ("mean", "median", "last"):
            monitor = built(overrides=["--set",
                                       f"drift_detection.aggregation={method}"])
            monitor.metric_buffer = [list(r) for r in rows]
            out[f"_aggregate_buffer[{method},{label}]"] = attempt(
                lambda m=monitor: m._aggregate_buffer())

    # the documented empty-buffer behaviour
    out["_aggregate_buffer[empty]"] = attempt(
        lambda: built()._aggregate_buffer())
    # a metric index past the end of each row
    wide = built(overrides=["--set", "drift_detection.metric_index=5"])
    wide.metric_buffer = [list(row) for row in buffer]
    out["_aggregate_buffer[metric-index-out-of-range]"] = attempt(
        lambda: wide._aggregate_buffer())
    # a negative index, which Python will happily accept as from-the-end
    neg = built(overrides=["--set", "drift_detection.metric_index=-1"])
    neg.metric_buffer = [list(row) for row in buffer]
    out["_aggregate_buffer[metric-index=-1]"] = attempt(
        lambda: neg._aggregate_buffer())
    # ragged rows
    ragged = built()
    ragged.metric_buffer = [[1.0, 2.0], [3.0]]
    out["_aggregate_buffer[ragged-rows]"] = attempt(
        lambda: ragged._aggregate_buffer())

    # --- 2. the stop condition's boundary ---------------------------------
    for cap in (0, 1, 3):
        for count in (0, 1, 2, 3, 4):
            monitor = built(overrides=["--set",
                                       f"drift_detection.max_stream_updates={cap}"])
            monitor.stream_update_count = count
            out[f"_should_stop[cap={cap},count={count}]"] = attempt(
                lambda m=monitor: m._should_stop())

    # --- 3. the seams a subclass reads ------------------------------------
    out["state[fresh]"] = attempt(lambda: state(built()))

    # One case per seam, so a name the implementation does not have costs
    # exactly one divergence and names itself in the report. `src/cl_only.py`
    # and `src/drift_only.py` subclass this monitor and read `label`,
    # `metric_idx`, `last_agg_metric` and `_aggregate_buffer` off it, so all
    # four are contract whatever the leading underscore suggests -- and none
    # of them is required by any test file this layer's claim pins. Both
    # producer families dropped the same four.
    fresh_monitor = built()
    for name in SEAMS:
        out[f"seam[{name}]"] = attempt(
            lambda n=name: show(getattr(fresh_monitor, n, "<missing>")))
    for name in ("_aggregate_buffer", "_evaluate_batch", "_check_drift",
                 "_should_stop", "_adapt", "run"):
        out[f"seam[{name},callable]"] = attempt(
            lambda n=name: show(callable(getattr(fresh_monitor, n, None))))
    out["detector[cached-property-same-object]"] = attempt(
        lambda: (lambda m: show(m.detector is m.detector))(built()))
    out["detector[type]"] = attempt(lambda: type(built().detector).__name__)
    out["trainer[type]"] = attempt(lambda: type(built().trainer).__name__)
    out["trainer[cached]"] = attempt(
        lambda: (lambda m: show(m.trainer is m.trainer))(built()))

    # --- 4. a drift check end to end --------------------------------------
    checked = built()
    checked.metric_buffer = [list(row) for row in buffer]
    out["_check_drift[populated]"] = attempt(
        lambda: show(checked._check_drift().drift_detected))
    out["_check_drift[sets-last-agg]"] = attempt(
        lambda: show(checked.last_agg_metric))
    out["_check_drift[empty-buffer]"] = attempt(
        lambda: built()._check_drift())

    harness = make_harness(n=8, batch=4)
    one = ContinuousMonitor(harness.cfg, harness)
    out["_evaluate_batch[fixed]"] = attempt(
        lambda: one._evaluate_batch(next(iter(harness.get_stream_dataloader()))))
    out["_evaluate_batch[appends-to-buffer]"] = attempt(
        lambda: show(len(one.metric_buffer)))

    # --- 5. the whole loop, reduced to counters and weights ---------------
    # One case per observable rather than one per run. Bundling the whole
    # state dict and the weights into a single compared value meant that an
    # implementation missing any one seam name diverged on every run
    # configuration, and the counters and trained weights -- the loop's
    # actual result, and the only reason to run the loop at all -- were
    # never compared. Measured: the counters agreed on both producer
    # families throughout while all twelve run cases reported a divergence.
    def run_cases(label: str, overrides=(), n: int = 16, batch: int = 4):
        try:
            monitor = built(overrides=overrides, n=n, batch=batch)
            monitor.run()
        except BaseException as exc:                  # noqa: BLE001 — probed
            out[f"run[{label}].raised"] = show(exc)
            return
        for name in SEAMS:
            out[f"run[{label}].{name}"] = attempt(
                lambda m=monitor, k=name: show(getattr(m, k, "<missing>")))
        out[f"run[{label}].weights"] = attempt(
            lambda m=monitor: params(m.modelHarness.model))

    run_cases("default")
    for cap in ("0", "1", "2"):
        run_cases(f"max_stream_updates={cap}",
                  overrides=["--set",
                             f"drift_detection.max_stream_updates={cap}"])
    for interval in ("1", "2", "100"):
        run_cases(f"detection_interval={interval}",
                  overrides=["--set",
                             f"drift_detection.detection_interval={interval}"])
    for detector in ("ADWINDetector", "PageHinkleyDetector", "KSWINDetector"):
        run_cases(f"detector={detector}",
                  overrides=["--set", f"drift_detection.detector_name={detector}",
                             "--set", "drift_detection.kswin_seed=7",
                             "--set", "drift_detection.max_stream_updates=2"])
    run_cases("empty-stream", n=0)
    run_cases("single-batch-stream", n=1, batch=1)
    run_cases("unknown-detector",
              overrides=["--set", "drift_detection.detector_name=NopeDetector"])

    # --- 6. the adaptation path, which nothing above reaches --------------
    # Every configuration above finishes with `drift_event_count == 0`, so
    # `_adapt` never runs, the trainer is never dispatched, and the weights
    # compared are the fixed initialisation on both sides. That comparison
    # looked like twelve agreements about the loop and was worth nothing:
    # an off-by-one planted in `_should_stop` was caught by the counters in
    # 11 of 12 configurations and by `.weights` in NONE of them. Drift has
    # to be forced before the weights mean anything.
    #
    # Page-Hinkley with a near-zero threshold and `min_instances=1` fires on
    # essentially every check, which takes the loop through drift -> adapt
    # -> outer CL training -> checkpoint. Measured against the source: 5
    # drift events, and `fc.bias` moves from the pinned initialisation.
    FIRES = ["--set", "drift_detection.detector_name=PageHinkleyDetector",
             "--set", "drift_detection.ph_threshold=0.0001",
             "--set", "drift_detection.ph_min_instances=1",
             "--set", "drift_detection.ph_delta=0.0",
             "--set", "drift_detection.detection_interval=1",
             "--set", "drift_detection.max_stream_updates=3"]
    run_cases("adapts", overrides=FIRES)
    # The same stream with training switched off. Drift still fires, so the
    # counters must match the case above while the weights must NOT -- which
    # is what makes the pair evidence that `.weights` is driven by the
    # adaptation path rather than by the stream alone.
    run_cases("adapts,update_mode=none",
              overrides=[*FIRES, "--set", "continual_learning.update_mode=none"])
    for mode in ("base", "ewc_online"):
        run_cases(f"adapts,update_mode={mode}",
                  overrides=[*FIRES, "--set",
                             f"continual_learning.update_mode={mode}"])

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
