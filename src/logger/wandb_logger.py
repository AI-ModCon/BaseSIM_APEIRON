"""Weights & Biases logging interface for metrics tracking."""

from __future__ import annotations

import pandas as pd
import wandb
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from config.configuration import Config


StageType = Literal["eval", "drift", "cl"]
VALID_STAGES: tuple[StageType, ...] = ("eval", "drift", "cl")


class WandBLogger:
    """WandB metrics logger with stage-aware step tracking.

    Tracks global `step` and per-stage steps (eval/step, drift/step, cl/step).
    """

    def __init__(self, enabled: bool = True, csv_path: str | Path | None = None):
        """Initialize WandB logger.

        Args:
            enabled: Enable WandB logging
            csv_path: Save metrics to CSV at this path (disabled if None)
        """
        self.enabled = enabled
        self.csv_path = Path(csv_path) if csv_path else None
        self.run: wandb.sdk.wandb_run.Run | None = None
        self.metrics_history: list[dict[str, Any]] = []

        # Step tracking
        self._step = 0
        self._stage_steps: dict[str, int] = {stage: 0 for stage in VALID_STAGES}
        self._current_stage: StageType | None = None

    @property
    def step(self) -> int:
        return self._step

    @property
    def current_stage(self) -> StageType | None:
        return self._current_stage

    def stage(self, name: StageType) -> None:
        """Set current stage ('eval', 'drift', or 'cl')."""
        if name not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{name}'. Must be one of: {VALID_STAGES}")
        self._current_stage = name

    def get_stage_step(self, stage: StageType) -> int:
        return self._stage_steps.get(stage, 0)

    def init(
        self,
        cfg: Config,
        project: str = "basesim-framework",
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        **kwargs,
    ) -> wandb.sdk.wandb_run.Run | None:
        """Initialize wandb run with config."""
        if not self.enabled:
            return None

        self.run = wandb.init(
            project=project,
            name=name,
            config=asdict(cfg),
            tags=tags,
            notes=notes,
            group=group,
            settings=wandb.Settings(silent=True),
            **kwargs,
        )

        if self.run:
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")

        return self.run

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Log metrics with auto-prefixing by current stage.

        Args:
            metrics: Metric name/value pairs
            step: Override global step (None = use internal counter)
            commit: Commit immediately to wandb
            prefix: Auto-prefix metrics with current stage (skips keys with "/")
            increment: Increment step counters
        """
        if increment:
            self._step += 1
            if self._current_stage:
                self._stage_steps[self._current_stage] += 1

        current_step = step if step is not None else self._step

        # Build prefixed metrics
        prefixed_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            if prefix and self._current_stage and "/" not in key:
                prefixed_metrics[f"{self._current_stage}/{key}"] = value
            else:
                prefixed_metrics[key] = value

        # Add step metrics
        prefixed_metrics["step"] = current_step
        if self._current_stage:
            prefixed_metrics[f"{self._current_stage}/step"] = self._stage_steps[
                self._current_stage
            ]

        # Save to history if CSV path provided
        if self.csv_path:
            self.metrics_history.append(prefixed_metrics.copy())

        # Log to wandb
        if self.enabled and self.run:
            wandb.log(prefixed_metrics, commit=commit)

    def save(
        self,
        file_path: str | Path,
        base_path: str | Path | None = None,
        policy: Literal["now", "live", "end"] = "now",
    ) -> None:
        """Save file to wandb."""
        if self.enabled and self.run:
            wandb.save(
                str(file_path),
                base_path=str(base_path) if base_path else None,
                policy=policy,
            )

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert metrics to long-format DataFrame (step, metric, value)."""
        if not self.metrics_history:
            return None

        rows = []
        for entry in self.metrics_history:
            step = entry.get("step")
            for metric_name, value in entry.items():
                if metric_name != "step":
                    rows.append({"step": step, "metric": metric_name, "value": value})

        return pd.DataFrame(rows)

    def to_csv(self, csv_path: str | Path | None = None) -> Path | None:
        """Save metrics to CSV in long format (step, metric, value)."""
        df = self.to_dataframe()
        if df is None:
            return None

        output_path = Path(csv_path) if csv_path else self.csv_path
        if output_path is None:
            return None

        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
        return output_path

    def finish(
        self, exit_code: int | None = None, save_csv: bool = True
    ) -> Path | None:
        """Finish wandb run and optionally save CSV."""
        csv_path = self.to_csv() if save_csv and self.csv_path else None

        if self.enabled and self.run:
            wandb.finish(exit_code=exit_code)
            self.run = None

        return csv_path

    @property
    def url(self) -> str | None:
        return self.run.get_url() if self.run else None
