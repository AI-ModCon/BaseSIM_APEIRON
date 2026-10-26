#!/usr/bin/env python3
"""Behaviour probe: the runnable entrypoints, off the pinned set.

This layer is the three scripts a user actually invokes -- `main.py`,
`drift_only.py`, `cl_only.py` -- and it regrew at 52% (A33). A16 lives here:
`--trigger-at` without `--schedule fixed` dies obscurely. A28 lives here too,
and was closed by collapsing the duplicated loop onto `ContinuousMonitor`, so
the side runners are now subclasses that override a decision rather than
copies of a loop.

Where this aims:

  the schedules     `NeverSchedule`, `PeriodicSchedule`, `RandomSchedule`,
                    `FixedSchedule` each answer `should_fire(decision_idx)`.
                    Pure logic over an integer, which makes the firing
                    PATTERN an exact observable -- probed over a long index
                    range so an off-by-one in a period or an inclusive bound
                    shows up as a different pattern, not a different count.
  make_schedule     the CLI-to-schedule mapping, including the documented
                    refusals (periodic with neither period nor budget, fixed
                    with no trigger-at) and A16's combination: a flag that
                    belongs to one schedule passed with another.
  build_parser      defaults and choices, which are a user-facing contract.
  the runners       `DriftOnlyMonitor` and `ScheduledCLRunner` driven over the
                    fixed harness, reduced to their returned dict and the
                    resulting weights.

`main.py` itself is NOT driven: it calls `get_example(cfg)`, which builds a
real MNIST or CIFAR harness and needs the dataset on disk. Its import surface
is checked and its argument handling is probed through `build_config`, which
is where its own decisions live; the loop it then runs is the monitor's, and
that is probed in the monitor layer.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _fixtures import (  # noqa: E402
    attempt, emit, make_cfg, make_harness, params, show, silence,
)

silence()

import cl_only  # noqa: E402
import drift_only  # noqa: E402

SCHEDULES = ("never", "periodic", "random", "fixed")
#: Long enough that a period of 3 and a period of 4 cannot look alike, and
#: that an inclusive upper bound shows as a trailing difference.
HORIZON = 24


def pattern(schedule) -> str:
    """The exact firing pattern, as a string of dots and hashes."""
    return "".join("#" if schedule.should_fire(i) else "." for i in range(HORIZON))


def args_for(**over):
    """A parsed namespace with the parser's own defaults, then overrides."""
    parser = cl_only.build_parser()
    namespace = parser.parse_args([])
    for key, value in over.items():
        setattr(namespace, key, value)
    return namespace


def main() -> int:
    out = {}

    # --- 1. the parser's user-facing contract ------------------------------
    parser = cl_only.build_parser()
    defaults = parser.parse_args([])
    out["parser[defaults]"] = show(
        {k: show(v) for k, v in sorted(vars(defaults).items())})
    for action in parser._actions:
        if action.dest != "help":
            out[f"parser[choices:{action.dest}]"] = show(
                sorted(action.choices) if action.choices else None)
    for argv in (["--schedule", "periodic"], ["--schedule", "never"],
                 ["--schedule", "fixed", "--trigger-at", "1,2,3"],
                 ["--schedule", "nope"], ["--period", "notanint"],
                 ["--trigger-at", "7,19"], ["--budget", "-1"],
                 ["--prob", "2.5"], ["--schedule-seed", "0"],
                 ["--unknown-flag"]):
        out[f"parser[{' '.join(argv)}]"] = attempt(
            lambda a=argv: show({k: show(v) for k, v in
                                 sorted(vars(parser.parse_args(a)).items())}))

    # --- 2. make_schedule, including A16 ----------------------------------
    cfg = make_cfg()
    out["make_schedule[never]"] = attempt(
        lambda: type(cl_only.make_schedule(args_for(schedule="never"), cfg)).__name__)
    # periodic: period wins, else budget, else the documented refusal
    for period, budget in ((1, 0), (3, 0), (0, 4), (0, 1), (0, 0), (3, 4),
                           (-1, 0), (0, -1)):
        out[f"make_schedule[periodic,period={period},budget={budget}]"] = attempt(
            lambda p=period, b=budget: pattern(cl_only.make_schedule(
                args_for(schedule="periodic", period=p, budget=b), cfg)))
    # fixed: the documented refusal, and how the list is parsed
    for spec in ("7,19", "0", "1,1,1", "", " ", "3,,5", "-1,2", "2,1",
                 "1, 2 , 3", "abc", "1.5"):
        out[f"make_schedule[fixed,trigger_at={spec!r}]"] = attempt(
            lambda s=spec: pattern(cl_only.make_schedule(
                args_for(schedule="fixed", trigger_at=s), cfg)))
    # A16: a flag that belongs to another schedule, passed with this one
    out["make_schedule[A16: trigger_at with periodic]"] = attempt(
        lambda: pattern(cl_only.make_schedule(
            args_for(schedule="periodic", trigger_at="7,19"), cfg)))
    out["make_schedule[A16: trigger_at with never]"] = attempt(
        lambda: pattern(cl_only.make_schedule(
            args_for(schedule="never", trigger_at="7,19"), cfg)))
    out["make_schedule[A16: trigger_at with random]"] = attempt(
        lambda: pattern(cl_only.make_schedule(
            args_for(schedule="random", trigger_at="7,19", prob=0.5), cfg)))
    out["make_schedule[period with fixed]"] = attempt(
        lambda: pattern(cl_only.make_schedule(
            args_for(schedule="fixed", period=3), cfg)))
    out["make_schedule[unknown]"] = attempt(
        lambda: cl_only.make_schedule(args_for(schedule="weekly"), cfg))

    # horizon: from the flag when positive, else from the config
    for horizon in (0, 1, 10, -5):
        out[f"make_schedule[random,horizon={horizon}]"] = attempt(
            lambda h=horizon: pattern(cl_only.make_schedule(
                args_for(schedule="random", horizon=h, budget=3,
                         schedule_seed=7), cfg)))

    # --- 3. each schedule's pattern, directly ------------------------------
    out["NeverSchedule[pattern]"] = attempt(
        lambda: pattern(cl_only.NeverSchedule()))
    out["NeverSchedule[describe]"] = attempt(
        lambda: cl_only.NeverSchedule().describe())
    for period in (1, 2, 3, 5, 24, 25, 0, -1):
        out[f"PeriodicSchedule[period={period}]"] = attempt(
            lambda p=period: pattern(cl_only.PeriodicSchedule(p)))
        out[f"PeriodicSchedule[describe,period={period}]"] = attempt(
            lambda p=period: cl_only.PeriodicSchedule(p).describe())
    for spec in ([], [0], [1, 2, 3], [23], [24], [100], [-1], [2, 2],
                 [5, 1, 3]):
        out[f"FixedSchedule[{spec}]"] = attempt(
            lambda s=spec: pattern(cl_only.FixedSchedule(list(s))))
        out[f"FixedSchedule[describe,{spec}]"] = attempt(
            lambda s=spec: cl_only.FixedSchedule(list(s)).describe())

    # random: seeded, so the pattern is exact and must be reproducible
    for seed in (0, 1, 7):
        for prob, budget in ((0.0, 0), (0.5, 0), (1.0, 0), (0.0, 3),
                             (0.0, 24), (0.0, 100)):
            out[f"RandomSchedule[seed={seed},prob={prob},budget={budget}]"] = \
                attempt(lambda s=seed, p=prob, b=budget: pattern(
                    cl_only.RandomSchedule(seed=s, prob=p, budget=b,
                                           horizon=HORIZON)))
    out["RandomSchedule[same-seed-twice]"] = attempt(
        lambda: show(pattern(cl_only.RandomSchedule(
            seed=3, prob=0.5, budget=0, horizon=HORIZON))
            == pattern(cl_only.RandomSchedule(
                seed=3, prob=0.5, budget=0, horizon=HORIZON))))
    out["RandomSchedule[different-seeds-differ]"] = attempt(
        lambda: show(pattern(cl_only.RandomSchedule(
            seed=1, prob=0.5, budget=0, horizon=HORIZON))
            != pattern(cl_only.RandomSchedule(
                seed=2, prob=0.5, budget=0, horizon=HORIZON))))
    out["RandomSchedule[describe]"] = attempt(
        lambda: cl_only.RandomSchedule(seed=1, prob=0.5, budget=0,
                                       horizon=HORIZON).describe())

    # --- 4. the drift-only runner ------------------------------------------
    def drift_run(overrides=(), n: int = 16, batch: int = 4) -> str:
        cfg = make_cfg(overrides=overrides)
        harness = make_harness(cfg, n=n, batch=batch)
        runner = drift_only.DriftOnlyMonitor(cfg, harness)
        result = runner.run()
        return show({
            "result": show({k: show(v) for k, v in sorted(result.items())})
            if isinstance(result, dict) else show(result),
            # a METHOD, not a property: rendering it without calling it
            # yields a bound-method repr carrying a memory address
            "windows": show(runner.drift_windows()
                            if callable(runner.drift_windows)
                            else runner.drift_windows),
            "label": show(runner.label),
            "events": show(runner.drift_event_count),
            "weights": params(harness.model),
        })

    out["DriftOnlyMonitor[label]"] = attempt(
        lambda: show(drift_only.DriftOnlyMonitor(
            make_cfg(), make_harness()).label))
    out["DriftOnlyMonitor[run]"] = attempt(lambda: drift_run())
    for cap in ("1", "2"):
        out[f"DriftOnlyMonitor[run,max_stream_updates={cap}]"] = attempt(
            lambda c=cap: drift_run(
                overrides=["--set", f"drift_detection.max_stream_updates={c}"]))
    out["DriftOnlyMonitor[run,no-adaptation]"] = attempt(
        lambda: drift_run(overrides=["--set",
                                     "continual_learning.update_mode=none"]))
    out["DriftOnlyMonitor[empty-stream]"] = attempt(lambda: drift_run(n=0))

    # --- 5. the scheduled CL runner ----------------------------------------
    def cl_run(schedule_args: dict, overrides=(), n: int = 16,
               batch: int = 4) -> str:
        cfg = make_cfg(overrides=overrides)
        harness = make_harness(cfg, n=n, batch=batch)
        schedule = cl_only.make_schedule(args_for(**schedule_args), cfg)
        runner = cl_only.ScheduledCLRunner(cfg, harness, schedule)
        result = runner.run()
        return show({
            "result": show({k: show(v) for k, v in sorted(result.items())})
            if isinstance(result, dict) else show(result),
            "label": show(runner.label),
            "weights": params(harness.model),
        })

    out["ScheduledCLRunner[never]"] = attempt(
        lambda: cl_run({"schedule": "never"}))
    out["ScheduledCLRunner[periodic,period=1]"] = attempt(
        lambda: cl_run({"schedule": "periodic", "period": 1}))
    out["ScheduledCLRunner[periodic,period=2]"] = attempt(
        lambda: cl_run({"schedule": "periodic", "period": 2}))
    out["ScheduledCLRunner[fixed]"] = attempt(
        lambda: cl_run({"schedule": "fixed", "trigger_at": "0,1"}))
    out["ScheduledCLRunner[random,seeded]"] = attempt(
        lambda: cl_run({"schedule": "random", "prob": 0.5,
                        "schedule_seed": 7}))
    out["ScheduledCLRunner[label]"] = attempt(
        lambda: show(cl_only.ScheduledCLRunner(
            make_cfg(), make_harness(),
            cl_only.NeverSchedule()).label))

    # --- 6. main.py's import surface, without running it -------------------
    import importlib
    out["main[importable]"] = attempt(
        lambda: show(bool(importlib.import_module("main").main)))
    out["main[signature]"] = attempt(
        lambda: str(__import__("inspect").signature(
            importlib.import_module("main").main)))
    out["drift_only[has-main]"] = attempt(
        lambda: show(callable(drift_only.main)))
    out["cl_only[has-main]"] = attempt(lambda: show(callable(cl_only.main)))
    out["drift_only[has-runner-fn]"] = attempt(
        lambda: show(callable(drift_only.run_drift_only)))
    out["cl_only[has-runner-fn]"] = attempt(
        lambda: show(callable(cl_only.run_manual_cl)))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
