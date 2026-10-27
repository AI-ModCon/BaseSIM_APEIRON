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


def state(monitor) -> str:
    """The counters and seams a subclass reads."""
    return show({
        "label": show(getattr(monitor, "label", "<missing>")),
        "last_agg_metric": show(getattr(monitor, "last_agg_metric", "<missing>")),
        "stream_update_count": show(monitor.stream_update_count),
        "batch_count": show(monitor.batch_count),
        "check_count": show(monitor.check_count),
        "drift_event_count": show(monitor.drift_event_count),
        "metric_idx": show(monitor.metric_idx),
        "detection_interval": show(monitor.detection_interval),
        "max_stream_updates": show(monitor.max_stream_updates),
        "aggregation": show(monitor.aggregation),
    })


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
    out["state[label-default]"] = attempt(lambda: show(built().label))
    out["state[last_agg_metric-initial]"] = attempt(
        lambda: show(built().last_agg_metric))
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
    def run_once(overrides=(), n: int = 16, batch: int = 4) -> str:
        monitor = built(overrides=overrides, n=n, batch=batch)
        monitor.run()
        return show({"state": state(monitor),
                     "weights": params(monitor.modelHarness.model)})

    out["run[default]"] = attempt(lambda: run_once())
    for cap in ("0", "1", "2"):
        out[f"run[max_stream_updates={cap}]"] = attempt(
            lambda c=cap: run_once(
                overrides=["--set", f"drift_detection.max_stream_updates={c}"]))
    for interval in ("1", "2", "100"):
        out[f"run[detection_interval={interval}]"] = attempt(
            lambda i=interval: run_once(
                overrides=["--set", f"drift_detection.detection_interval={i}"]))
    for detector in ("ADWINDetector", "PageHinkleyDetector", "KSWINDetector"):
        out[f"run[detector={detector}]"] = attempt(
            lambda d=detector: run_once(
                overrides=["--set", f"drift_detection.detector_name={d}",
                           "--set", "drift_detection.kswin_seed=7",
                           "--set", "drift_detection.max_stream_updates=2"]))
    out["run[empty-stream]"] = attempt(lambda: run_once(n=0))
    out["run[single-batch-stream]"] = attempt(lambda: run_once(n=1, batch=1))
    out["run[unknown-detector]"] = attempt(
        lambda: run_once(overrides=["--set",
                                    "drift_detection.detector_name=NopeDetector"]))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
