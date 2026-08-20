"""Schedule-driven continual learning: trigger CL without a drift detector.

This is a control arm for evaluating how good a drift detector actually is.
A detector run answers "adapting k times at *these* moments gives error E".

It is the shared :class:`~apeiron.driver.stream_engine.StreamEngine` with a
:class:`~apeiron.driver.trigger_policy.SchedulePolicy` (fire on a fixed rule,
ignoring model metrics) in place of the detector policy, and the same
:class:`~apeiron.driver.trigger_action.AdaptAction` a full run uses. Because it is
the same engine, the per-batch evaluation, metric aggregation, FLOPs profiling,
and logged field names all match a detector run -- so the resulting CSV is
directly comparable.

One decision point per stream window
------------------------------------
A schedule uses the engine's ``"window"`` cadence: one decision point at the end
of every stream window. A run over ``drift_detection.max_stream_updates`` windows
has exactly that many decision points, and ``--period 1`` triggers CL on every
window.

Schedules
---------
periodic  Fire every ``--period`` windows.
random    Fire at a matched *rate* rather than a matched cadence. With
          ``--budget`` it fires at exactly that many uniformly sampled
          windows; with ``--prob`` it fires Bernoulli(p) per window. Run
          several ``--schedule-seed`` values to get a null band.
fixed     Fire at explicit window indices. Use this to replay a detector's own
          firing points with one removed (leave-one-out attribution) or
          shifted by a few windows (timing sensitivity).
never     Never adapt -- the frozen-model lower bound.

Examples
--------
    # Lower bound: no adaptation at all.
    python -m src.cl_only --config examples/aeris/aeris.toml \
        --schedule never

    # Upper bound: adapt after every window (42 triggers on aeris.toml).
    python -m src.cl_only --config examples/aeris/aeris.toml \
        --schedule periodic --period 1

    # Budget-matched periodic control: 3 triggers over 42 windows.
    python -m src.cl_only --config examples/aeris/aeris.toml \
        --schedule periodic --period 14

    # Rate-matched random null, exactly 3 triggers, repeated over seeds.
    python -m src.cl_only --config examples/aeris/aeris.toml \
        --schedule random --budget 3 --schedule-seed 0

    # Replay ADWIN's firing points with the second one dropped.
    python -m src.cl_only --config examples/aeris/aeris.toml \
        --schedule fixed --trigger-at 7,29

Any ``--set key=val`` / ``--device`` flags not consumed here are forwarded to
the normal apeiron config builder.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from apeiron.config.configuration import build_config, Config
from apeiron.driver.schedules import (
    FixedSchedule,
    NeverSchedule,
    PeriodicSchedule,
    RandomSchedule,
    TriggerSchedule,
)
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import AdaptAction
from apeiron.driver.trigger_policy import SchedulePolicy
from apeiron.logger import Logger, configure_backend, get_logger
from apeiron.model.torch_model_harness import BaseModelHarness

from examples.utils import get_example


# ---------------------------------------------------------------------------
# Schedule construction from CLI
# ---------------------------------------------------------------------------


def make_schedule(args: argparse.Namespace, cfg: Config) -> TriggerSchedule:
    """Build the schedule described by the parsed CLI arguments."""
    # One decision point per stream window, so the horizon is just the window
    # count -- no need to read it off a prior run.
    horizon = (
        args.horizon if args.horizon > 0 else cfg.drift_detection.max_stream_updates
    )

    if args.schedule == "never":
        return NeverSchedule()
    if args.schedule == "periodic":
        if args.period > 0:
            period = args.period
        elif args.budget > 0:
            period = max(1, round(horizon / args.budget))
        else:
            raise ValueError("--schedule periodic needs --period or --budget")
        return PeriodicSchedule(period)
    if args.schedule == "random":
        seed = args.schedule_seed if args.schedule_seed is not None else cfg.seed
        return RandomSchedule(
            seed=seed, prob=args.prob, budget=args.budget, horizon=horizon
        )
    if args.schedule == "fixed":
        if not args.trigger_at:
            raise ValueError("--schedule fixed needs --trigger-at")
        return FixedSchedule([int(v) for v in args.trigger_at.split(",") if v.strip()])
    raise ValueError(f"Unknown schedule: {args.schedule}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_manual_cl(
    cfg: Config,
    modelHarness: BaseModelHarness,
    schedule: TriggerSchedule,
    logger: Logger | None = None,
) -> dict[str, Any]:
    """Run continual learning on a fixed schedule, bypassing drift detection.

    Returns:
        Summary dict with the realized trigger count and trigger points -- the
        numbers needed to place this run on the accuracy-cost frontier.
    """
    engine = StreamEngine(
        cfg=cfg,
        modelHarness=modelHarness,
        policy=SchedulePolicy(schedule),
        action=AdaptAction(),
        logger=logger,
        detected_sample_rate=1.0,  # keep the full decision trace
    )
    return engine.run()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trigger continual learning on a fixed schedule, bypassing "
        "drift detection (budget-matched control for detector evaluation).",
    )
    p.add_argument(
        "--schedule",
        choices=["periodic", "random", "fixed", "never"],
        default="periodic",
        help="Trigger rule (default: periodic)",
    )
    p.add_argument(
        "--period",
        type=int,
        default=0,
        help="periodic: fire every N stream windows (1 = every window)",
    )
    p.add_argument(
        "--prob",
        type=float,
        default=0.0,
        help="random: per-window firing probability",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Target number of triggers. Set to the trigger count of the "
        "detector run being controlled for.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Windows the budget is spread over (default: "
        "drift_detection.max_stream_updates)",
    )
    p.add_argument(
        "--trigger-at",
        type=str,
        default="",
        help="fixed: comma-separated window indices, e.g. 7,19,29",
    )
    p.add_argument(
        "--schedule-seed",
        type=int,
        default=None,
        help="Seed for the random schedule only; leaves the run seed alone so "
        "placements vary while training stays reproducible (default: cfg.seed)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args, remaining = build_parser().parse_known_args(argv)

    cfg: Config = build_config(remaining)
    modelHarness = get_example(cfg=cfg)

    schedule = make_schedule(args, cfg)

    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.visualization.input if cfg.visualization else None,
    )

    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    summary = run_manual_cl(
        cfg=cfg, modelHarness=modelHarness, schedule=schedule, logger=logger
    )

    logger.finish()

    print("\n==== Scheduled CL summary ====")
    print(f"  schedule: {summary['policy']}")
    print(f"  batches: {summary['batches']}")
    print(f"  stream_updates: {summary['stream_updates']}")
    print(f"  decision_points: {summary['decision_points']}")
    print(f"  triggers: {summary['fires']}")
    print(f"  trigger_points: {summary['fire_points']}")
    print(f"  final_metrics: {summary['final_metrics']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
