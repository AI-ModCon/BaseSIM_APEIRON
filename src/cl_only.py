"""Schedule-driven continual learning: trigger CL without a drift detector.

This is a control arm for evaluating how good a drift detector actually is.
A detector run answers "adapting k times at *these* moments gives error E".

The monitoring loop mirrors ``ContinuousMonitor``: the same per-batch
evaluation, the same metric aggregation, the same FLOPs profiling, and the
same logged field names -- so the resulting CSV is directly comparable to a
detector run. 

One decision point per stream window
------------------------------------
A decision point sits at the end of every stream window, so a run over
``drift_detection.max_stream_updates`` windows has exactly that many decision
points and ``--period 1`` triggers CL on every window.

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
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from apeiron.config.configuration import build_config, Config
from apeiron.logger import get_logger, configure_backend
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from apeiron.training import ContinuousTrainer

from examples.utils import get_example


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


class TriggerSchedule:
    """Decides fire / no-fire at each decision point, ignoring model metrics.

    One decision point occurs per stream window, so ``decision_idx`` is the
    index of the window that just finished streaming and a run has exactly
    ``drift_detection.max_stream_updates`` of them.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def should_fire(self, decision_idx: int) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NeverSchedule(TriggerSchedule):
    """Never adapt. Frozen-model lower bound for the accuracy-cost frontier."""

    def __init__(self) -> None:
        super().__init__("never")

    def should_fire(self, decision_idx: int) -> bool:
        return False


class PeriodicSchedule(TriggerSchedule):
    """Fire every ``period`` windows, starting at window index ``period - 1``.

    ``period = 1`` adapts after every window -- the adaptation upper bound.
    Firing is offset to the *end* of the first period rather than at index 0
    so that ``period = N`` fires exactly ``floor(windows / N)`` times.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError(f"--period must be >= 1, got {period}")
        super().__init__("periodic")
        self.period = period

    def should_fire(self, decision_idx: int) -> bool:
        return (decision_idx + 1) % self.period == 0

    def describe(self) -> str:
        return f"periodic(every {self.period} window(s))"


class RandomSchedule(TriggerSchedule):
    """Fire at random windows -- the rate-matched null.

    Two parametrizations:

    * ``prob``: independent Bernoulli(p) per window. The realized trigger
      count varies run to run, which is the honest null if you want a
      distribution over counts as well as placements.
    * ``budget``: exactly ``budget`` windows sampled uniformly without
      replacement from ``[0, horizon)``, where ``horizon`` defaults to the
      run's window count. Preferred for budget matching, since it holds the
      count fixed and varies only the placement -- the thing being controlled
      for.
    """

    def __init__(
        self,
        seed: int,
        prob: float = 0.0,
        budget: int = 0,
        horizon: int = 0,
    ) -> None:
        super().__init__("random")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.prob = prob
        self.budget = budget
        self.horizon = horizon
        self.fire_at: Optional[set[int]] = None

        if budget > 0:
            if horizon <= 0:
                raise ValueError("--budget requires a positive horizon")
            if budget > horizon:
                raise ValueError(f"--budget {budget} exceeds --horizon {horizon}")
            self.fire_at = set(
                int(i) for i in self.rng.choice(horizon, size=budget, replace=False)
            )
        elif prob <= 0.0:
            raise ValueError("--schedule random needs --prob or --budget")

    def should_fire(self, decision_idx: int) -> bool:
        if self.fire_at is not None:
            return decision_idx in self.fire_at
        return bool(self.rng.random() < self.prob)

    def describe(self) -> str:
        if self.fire_at is not None:
            pts = ",".join(str(i) for i in sorted(self.fire_at))
            return f"random(seed={self.seed}, exactly {self.budget} of {self.horizon}: [{pts}])"
        return f"random(seed={self.seed}, p={self.prob})"


class FixedSchedule(TriggerSchedule):
    """Fire at an explicit list of window indices.

    Drives the per-trigger attribution experiments: replay a detector's own
    firing windows with one dropped (how much was that trigger worth?) or with
    all of them shifted by a few windows (how much does timing precision
    matter?).
    """

    def __init__(self, trigger_at: list[int]) -> None:
        super().__init__("fixed")
        self.fire_at = set(trigger_at)

    def should_fire(self, decision_idx: int) -> bool:
        return decision_idx in self.fire_at

    def describe(self) -> str:
        pts = ",".join(str(i) for i in sorted(self.fire_at))
        return f"fixed([{pts}])"


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


class ScheduledCLRunner:
    """Stream monitor that adapts on a schedule instead of on detected drift.

    Mirrors ``ContinuousMonitor``'s loop -- per-batch evaluation, metric
    aggregation, FLOPs profiling, stream extension, checkpointing -- but
    replaces the detector call with a :class:`TriggerSchedule` consulted once
    per stream window. Metrics are still aggregated and logged at every
    decision point even though nothing consumes them, so the recorded metric
    trace of a schedule run is directly comparable to a detector run.
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        schedule: TriggerSchedule,
    ) -> None:
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.schedule = schedule
        self.logger = get_logger()

        self.flops_profiler = FLOPSProfiler()
        self.trainer = ContinuousTrainer(
            cfg=cfg,
            modelHarness=modelHarness,
            logger=self.logger,
            profiler=self.flops_profiler,
        )

        self.metric_idx = cfg.drift_detection.metric_index
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # State
        self.stream_update_count = 0
        self.batch_count = 0
        self.decision_count = 0
        self.trigger_count = 0
        self.trigger_points: list[int] = []
        self.metric_buffer: list[list[float]] = []

        self.logger.info("==== ScheduledCLRunner initialized ====", level=0)
        self.logger.info("\tDrift detection: BYPASSED", level=1)
        self.logger.info(f"\tSchedule: {schedule.describe()}", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        self.logger.info("\tDecision point: once per stream window", level=1)
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(
            f"\tMax stream updates: {self.max_stream_updates} "
            f"(= decision points available)",
            level=1,
        )

    # -- main loop ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the scheduled-adaptation loop and return a run summary."""
        self.logger.info("==== Starting Scheduled CL Monitoring ====", level=0)
        self.logger.info("\tInitializing first data stream...", level=1)
        self.modelHarness.update_data_stream()

        while self.stream_update_count < self.max_stream_updates:
            try:
                self._process_stream()
            except StopIteration:
                self._extend_stream()

        self.logger.info("==== Scheduled CL Monitoring Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)
        self.logger.info(f"\tDecision points: {self.decision_count}", level=1)
        self.logger.info(f"\tCL triggers fired: {self.trigger_count}", level=1)
        self.logger.info(f"\tTrigger points: {self.trigger_points}", level=1)

        final_metrics = self.modelHarness.eval()
        self.logger.info(f"\tFinal eval metrics: {final_metrics}", level=1)

        return {
            "schedule": self.schedule.describe(),
            "batches": self.batch_count,
            "stream_updates": self.stream_update_count,
            "decision_points": self.decision_count,
            "triggers": self.trigger_count,
            "trigger_points": list(self.trigger_points),
            "final_metrics": final_metrics,
        }

    def _process_stream(self) -> None:
        """Evaluate one stream window, then consult the schedule once.

        The decision point sits at the *end* of the window: the whole window
        has to be evaluated before its aggregate metric exists, and adapting
        mid-window would mean the window's metric was measured under two
        different models.

        Raises:
            StopIteration: Always, once the window is exhausted, to hand
                control back to :meth:`run` for the next stream update.
        """
        stream_loader = self.modelHarness.get_stream_dataloader()

        for _, batch in tqdm(
            enumerate(stream_loader),
            desc="Processing batches",
            leave=False,
        ):
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1

        self._decision_point()

        raise StopIteration()

    def _decision_point(self) -> None:
        """Aggregate the window's metrics, log them, and ask the schedule to fire."""
        agg_metric = self._aggregate_buffer()

        decision_idx = self.decision_count
        self.decision_count += 1

        fire = self.schedule.should_fire(decision_idx)
        self._log_metrics(decision_idx, agg_metric, fire)

        if fire:
            self._trigger_cl(decision_idx)

    def _aggregate_buffer(self) -> float:
        """Reduce the buffered per-batch metrics to the monitored scalar."""
        if not self.metric_buffer:
            raise RuntimeError("Model Harness requires evaluation metrics")

        metric_values = [m[self.metric_idx] for m in self.metric_buffer]
        self.metric_buffer = []

        if self.aggregation == "median":
            return float(np.median(metric_values))
        if self.aggregation == "last":
            return float(metric_values[-1])
        return float(np.mean(metric_values))

    def _trigger_cl(self, decision_idx: int) -> None:
        """Dispatch the continual-learning loop for a scheduled trigger."""
        self.trigger_count += 1
        self.trigger_points.append(decision_idx)

        self.logger.info(
            f"==== SCHEDULED TRIGGER (#{self.trigger_count}) at decision "
            f"point {decision_idx} ====",
            level=0,
        )
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        if timerange is not None:
            self.logger.info(
                f"\tData time range: {timerange[0]} -> {timerange[1]}", level=1
            )

        self.flops_profiler.print_performance(logger=self.logger, level=2)

        self.logger.info("-> Dispatching continual learning module...", level=0)
        self.trainer.outer_cl_training_loop(drift_event_id=self.trigger_count)
        self.logger.info("<- Continual learning complete.", level=0)

        if self.modelHarness.ckpts_enabled:
            ckptpath = self.modelHarness.save_ckpt(event=self.trigger_count)
            self.logger.info(f"* Checkpoint saved to: {ckptpath}", level=0)

        self.logger.info("==== RESUMING MONITORING! ====", level=0)

    def _extend_stream(self) -> None:
        """Load the next data buffer when the current stream is exhausted."""
        self.stream_update_count += 1
        self.logger.info(
            f"\tStream exhausted. Loading next data buffer. "
            f"{self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )
        self.modelHarness.update_data_stream()

    # -- evaluation / logging ---------------------------------------------

    def _evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """Evaluate the model on one streaming batch and return all metrics."""
        self.modelHarness.model.eval()

        profile = self.batch_count > self.flops_profiler.warmup_iters

        with torch.no_grad():
            if profile:
                with self.flops_profiler.measure_flops(tag="infer"):
                    metrics, named = self._forward_metrics(batch)
            else:
                metrics, named = self._forward_metrics(batch)

        if profile:
            self.logger.stage("eval")
            self.logger.log(named)

        return metrics

    def _forward_metrics(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[list[float], dict[str, float]]:
        """Run the forward pass and compute every harness eval metric."""
        x, y = self.modelHarness._unpack(batch)
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)

        y_hat = self.modelHarness.model(x)

        metrics: list[float] = []
        named: dict[str, float] = {}
        for key, metric_fn in self.modelHarness.eval_metrics.items():
            value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
            metrics.append(value)
            named[key] = value
        return metrics, named

    def _log_metrics(self, decision_idx: int, metric_value: float, fired: bool) -> None:
        """Log one decision point.

        Field names match ``ContinuousMonitor._log_metrics`` so schedule runs
        and detector runs land in the same CSV schema, with ``detected``
        carrying the schedule's decision instead of a detector's. ``score`` is
        held at 0.0 -- there is no drift statistic here, and emitting a
        placeholder severity would be indistinguishable from a real one during
        analysis.
        """
        flops_perf = self.flops_profiler.get_performance()

        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        ts_fields = {}
        if timerange is not None:
            ts_fields["data_time_start"] = timerange[0]
            ts_fields["data_time_end"] = timerange[1]

        self.logger.stage("drift")
        self.logger.log(
            {
                "detected": int(fired),
                "score": 0.0,
                "regime": "scheduled",
                "confidence": "N/A",
                f"metric_{self.metric_idx}": metric_value,
                "decision_idx": decision_idx,
                "trigger_count": self.trigger_count + int(fired),
                "stream_idx": self.stream_update_count,
                **ts_fields,
                **{f"cperf_{k}": v for k, v in flops_perf.items()},
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_manual_cl(
    cfg: Config,
    modelHarness: BaseModelHarness,
    schedule: TriggerSchedule,
) -> dict[str, Any]:
    """Run continual learning on a fixed schedule, bypassing drift detection.

    Args:
        cfg: Resolved apeiron configuration.
        modelHarness: Harness supplying the model, stream, and train loaders.
        schedule: Decides which stream windows trigger a CL update.

    Returns:
        Summary dict with the realized trigger count and trigger points --
        the numbers needed to place this run on the accuracy-cost frontier.
    """
    return ScheduledCLRunner(
        cfg=cfg, modelHarness=modelHarness, schedule=schedule
    ).run()


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

    summary = run_manual_cl(cfg=cfg, modelHarness=modelHarness, schedule=schedule)

    logger.finish()

    print("\n==== Scheduled CL summary ====")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
