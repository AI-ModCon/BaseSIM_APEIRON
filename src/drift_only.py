"""Detection-only monitoring: run a drift detector without any adaptation.

Streams data past a frozen model, feeds the monitored metric to the configured
detector, and records every time it fires without triggering continual learning.
The weights that go in are the weights that come out. Drift is checked every
``drift_detection.detection_interval`` batches, so the metric/score trace lines
up batch-for-batch with a full run.

This is the shared :class:`~apeiron.driver.stream_engine.StreamEngine` with a
:class:`~apeiron.driver.trigger_policy.DetectorPolicy` (the same detector a full
run uses) and a :class:`~apeiron.driver.trigger_action.RecordOnlyAction` (note
the detection, leave the model frozen). Because nothing adapts, no trainer or
optimizer is constructed. Unlike the full run, every decision is logged rather
than sampling ``detected=0`` -- the complete score trace is the entire product of
this run.

A ``main.py`` run with ``[continual_learning] update_mode = "none"`` also leaves
the weights untouched, but it still dispatches the whole CL path on every
detection, so use that to measure the pipeline with training made inert, and this
file when the detection trace is the point.

Every check lands in the run's metrics CSV (``[visualization] input``) under the
``drift/`` stage, so the full detection trace is already on disk and this file
writes no output of its own.

Examples
--------
    python -m src.drift_only --config examples/aeris/aeris.toml

Takes the same flags as ``main.py``: ``--config``, ``--set key=val``,
``--device``, ``--multi-gpu``.
"""

from __future__ import annotations

import sys
from typing import Any

from apeiron.config.configuration import build_config, Config
from apeiron.distributed import comm
from apeiron.driver.stream_engine import StreamEngine
from apeiron.driver.trigger_action import RecordOnlyAction
from apeiron.driver.trigger_policy import DetectorPolicy
from apeiron.logger import Logger, configure_backend, get_logger
from apeiron.model.torch_model_harness import BaseModelHarness

from examples.utils import get_example


def run_drift_only(
    cfg: Config,
    modelHarness: BaseModelHarness,
    logger: Logger | None = None,
) -> dict[str, Any]:
    """Run drift detection over the stream without ever adapting the model.

    Returns:
        Summary dict with the detection count, the per-event details, the stream
        windows that fired, and the final eval metrics.
    """
    engine = StreamEngine(
        cfg=cfg,
        modelHarness=modelHarness,
        policy=DetectorPolicy(cfg),
        action=RecordOnlyAction(),
        logger=logger,
        detected_sample_rate=1.0,  # keep the full detection trace
    )
    summary = engine.run()
    summary["drift_windows"] = sorted({int(e["stream_idx"]) for e in summary["events"]})
    return summary


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    comm.init_from_env()
    modelHarness = get_example(cfg=cfg)

    backend = configure_backend(cfg) if comm.is_main else "none"
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=(
            cfg.visualization.input if (cfg.visualization and comm.is_main) else None
        ),
    )

    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    summary = run_drift_only(cfg=cfg, modelHarness=modelHarness, logger=logger)

    logger.finish()
    comm.shutdown()

    if comm.is_main:
        print("\n==== Drift-only summary ====")
        print(f"  detector: {summary['policy']}")
        print(f"  batches: {summary['batches']}")
        print(f"  stream_updates: {summary['stream_updates']}")
        print(f"  drift_checks: {summary['decision_points']}")
        print(f"  drift_events: {summary['fires']}")
        print(f"  drift_windows: {summary['drift_windows']}")
        print(f"  final_metrics: {summary['final_metrics']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
