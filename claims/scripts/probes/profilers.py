#!/usr/bin/env python3
"""Behaviour probe: apeiron.profilers, off the pinned set.

This layer measures wall-clock time, so most of what it returns is
nondeterministic and cannot be compared between two runs. The probe
therefore targets the parts that ARE determinate, and says so rather than
quietly comparing noise:

  the aten FLOP table      A25 recorded that every FLOPs figure the framework
                           reports rests on this table and that nothing
                           verifies it. A reconstruction that invents its own
                           constants produces plausible numbers that are all
                           wrong, and a kept-ratio would not notice. Compared
                           key by key.
  the formatters           _format_flops / _format_time / _format_throughput
                           turn a float into a human string -- unit
                           thresholds, rounding, and what they do with zero,
                           a negative, a NaN or an absurd magnitude. Pure
                           functions, entirely determinate, and the layer's
                           most-used surface.
  FLOP counts              a fixed tiny model's forward pass under
                           measure_flops, which FlopCounterMode makes exact
                           and repeatable. The elapsed time it records
                           alongside is explicitly dropped.
  structure                which tags and keys get created, which is a
                           contract callers depend on.

What is deliberately NOT compared: any duration, and anything derived from
one (throughput). A probe that compared those would fail randomly and teach
us to ignore it.
"""
import json
import sys

import torch
import torch.nn as nn

from apeiron.profilers.aten_flops_map import ATEN_FLOPS_PER_ELEMENT  # noqa: E402
from apeiron.profilers.count_flops import FLOPSProfiler  # noqa: E402

#: Keys whose values are times or derived from times, and so cannot be
#: compared between two runs.
#:
#: `get_performance` documents its key scheme as `{tag}_flop` (a COUNT),
#: `{tag}_time` (a duration) and `{tag}_flops` (a THROUGHPUT, flop/time).
#: Count and rate therefore differ by a single trailing `s`, and the first
#: version of this probe fell straight into it: a substring filter that
#: caught `time` and `throughput` did not catch `fwd_flops`, so the probe
#: reported a divergence that was two wall-clock measurements. Anyone
#: consuming that dict can make the same mistake, which is worth a finding
#: in its own right.
#: Two key namespaces, and both carry a time. `get_performance` returns
#: `{tag}_time` / `{tag}_flops`; the raw `profiles[tag]` dict underneath uses
#: the bare words `time` and `flop`. A suffix rule alone misses the bare
#: `time`, which is how the second version of this probe still reported two
#: wall-clock readings as a divergence.
TIMED_EXACT = ("time", "flops", "times")
TIMED_SUFFIXES = ("_time", "_flops", "_times")
TIMED_SUBSTRINGS = ("throughput", "duration", "_sec", "per_sec", "elapsed")


def timed(key: str) -> bool:
    low = key.lower()
    return (low in TIMED_EXACT
            or low.endswith(TIMED_SUFFIXES)
            or any(s in low for s in TIMED_SUBSTRINGS))


def show(value) -> str:
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, dict):
        return json.dumps({k: show(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return json.dumps([show(v) for v in value])
    return repr(value)


def attempt(fn, *args, **kwargs) -> str:
    try:
        return show(fn(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 — probed
        return show(exc)


class Tiny(nn.Module):
    """Fixed weights, so the FLOP count is a property of the graph alone."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):
                p.copy_(torch.full_like(p, 0.01 * (i + 1)))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> int:
    out = {}

    # --- 1. the aten FLOP table, key by key -------------------------------
    out["aten[count]"] = show(len(ATEN_FLOPS_PER_ELEMENT))
    out["aten[keys]"] = json.dumps(sorted(ATEN_FLOPS_PER_ELEMENT))
    out["aten[table]"] = json.dumps(
        {k: ATEN_FLOPS_PER_ELEMENT[k] for k in sorted(ATEN_FLOPS_PER_ELEMENT)},
        default=repr)

    # --- 2. the formatters ------------------------------------------------
    profiler = FLOPSProfiler()
    # Every decade, and both sides of each one. A formatter's unit
    # thresholds are the thing most likely to be reconstructed slightly
    # differently, and a probe only sees that if a value lands between the
    # old threshold and the new one. The first version of this list jumped
    # from 1e9 to 1.5e12, so a mutant that moved a threshold from 1e12 to
    # 1e11 went undetected -- the probe had no value in the gap.
    magnitudes = [0, 1, -1, -1500, 0.5, 0.001, 1e-9, 1e-3,
                  float("nan"), float("inf"), float("-inf")]
    for exp in range(0, 22):
        decade = 10.0 ** exp
        magnitudes += [decade, decade * 1.5, decade * 9.99]
    magnitudes += [999, 1001, 1023, 1024, 999_999, 1_500_000, 999_999_999]
    for value in magnitudes:
        out[f"_format_flops[{value!r}]"] = attempt(profiler._format_flops, value)
        out[f"_format_time[{value!r}]"] = attempt(profiler._format_time, value)
        out[f"_format_throughput[{value!r}]"] = attempt(
            profiler._format_throughput, value)

    # --- 3. structure the caller depends on -------------------------------
    fresh = FLOPSProfiler()
    out["init[warmup default]"] = show(fresh.warmup_iters)
    out["init[profiles empty]"] = show(fresh.profiles)
    out["init[tag]"] = show(fresh.tag)
    sized = FLOPSProfiler(warmup_iters=5)
    out["init[warmup set]"] = show(sized.warmup_iters)

    shaped = FLOPSProfiler()
    shaped._add_profile("t")
    out["_add_profile[keys]"] = json.dumps(sorted(shaped.profiles["t"]))
    shaped._add_profile("t")
    out["_add_profile[idempotent]"] = show(shaped.profiles["t"])

    # --- 4. an exact FLOP count -------------------------------------------
    model, x = Tiny(), torch.ones(2, 8)
    counted = FLOPSProfiler(warmup_iters=0)
    try:
        with counted.measure_flops("fwd"):
            model(x)
        rec = counted.profiles.get("fwd", {})
        out["measure_flops[tags]"] = json.dumps(sorted(counted.profiles))
        out["measure_flops[keys]"] = json.dumps(sorted(rec))
        out["measure_flops[flops]"] = json.dumps(
            [show(v) for k, v in sorted(rec.items())
             if not timed(k)])
        out["measure_flops[recorded-a-time]"] = show(
            any(timed(k) for k in rec))
    except BaseException as exc:                     # noqa: BLE001 — probed
        out["measure_flops[error]"] = show(exc)

    # two blocks under one tag: appended or overwritten?
    twice = FLOPSProfiler(warmup_iters=0)
    try:
        for _ in range(2):
            with twice.measure_flops("fwd"):
                model(x)
        rec = twice.profiles.get("fwd", {})
        out["measure_flops[repeat-lengths]"] = json.dumps(
            {k: len(v) if isinstance(v, list) else "scalar"
             for k, v in sorted(rec.items())})
        out["measure_flops[repeat-flops]"] = json.dumps(
            [show(v) for k, v in sorted(rec.items())
             if not timed(k)])
    except BaseException as exc:                     # noqa: BLE001 — probed
        out["measure_flops[repeat-error]"] = show(exc)

    # an exception inside the block: is the tag still closed out cleanly?
    leaky = FLOPSProfiler(warmup_iters=0)
    try:
        with leaky.measure_flops("boom"):
            raise RuntimeError("probe")
    except BaseException as exc:                     # noqa: BLE001 — probed
        out["measure_flops[propagates]"] = show(exc)
    out["measure_flops[state-after-raise]"] = json.dumps(
        {"tag": show(leaky.tag),
         "has_boom": show("boom" in leaky.profiles)})

    # --- 5. get_performance, with times excluded --------------------------
    perf = FLOPSProfiler(warmup_iters=0)
    try:
        with perf.measure_flops("fwd"):
            model(x)
        got = perf.get_performance()
        out["get_performance[keys]"] = json.dumps(sorted(got))
        out["get_performance[untimed]"] = json.dumps(
            {k: show(v) for k, v in sorted(got.items())
             if not timed(k)})
    except BaseException as exc:                     # noqa: BLE001 — probed
        out["get_performance[error]"] = show(exc)
    out["get_performance[empty]"] = attempt(FLOPSProfiler().get_performance)

    json.dump(out, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
