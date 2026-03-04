"""MLflow logging interface for metrics tracking."""

from __future__ import annotations

import pandas as pd
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from logger.wandb_logger import StageType, VALID_STAGES

if TYPE_CHECKING:
    import mlflow
    from config.configuration import Config


class MLFlowLogger:
    """MLflow metrics logger with stage-aware step tracking.

    Mirrors WandBLogger interface for drop-in replacement.
    Tracks global `step` and per-stage steps (eval/step, drift/step, cl/step).
    """

    def __init__(self, enabled: bool = True, csv_path: str | Path | None = None):
        """Initialize MLflow logger.

        Args:
            enabled: Enable MLflow logging
            csv_path: Save metrics to CSV at this path (disabled if None)
        """
        self.enabled = enabled
        self.csv_path = Path(csv_path) if csv_path else None
        self.run: mlflow.ActiveRun | None = None
        self.metrics_history: list[dict[str, Any]] = []

        # Step tracking
        self._step = 0
        self._stage_steps: dict[str, int] = {stage: 0 for stage in VALID_STAGES}
        self._current_stage: StageType | None = None

        # Lazy import mlflow
        self._mlflow: Any = None

    def _get_mlflow(self) -> Any:
        """Lazy import mlflow to avoid import errors when not installed."""
        if self._mlflow is None:
            import mlflow

            self._mlflow = mlflow
        return self._mlflow

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
    ) -> mlflow.ActiveRun | None:
        """Initialize MLflow run with config.

        Args:
            cfg: Configuration object
            project: MLflow experiment name
            name: Run name
            tags: List of tags
            notes: Run description
            group: Group name (stored as tag)
            **kwargs: Additional arguments (ignored for MLflow compatibility)

        Returns:
            MLflow ActiveRun object or None if disabled
        """
        if not self.enabled:
            return None

        mlflow = self._get_mlflow()

        mlflow.set_experiment(project)

        # Build tags dict
        run_tags = {}
        if group:
            run_tags["group"] = group
        if tags:
            for i, tag in enumerate(tags):
                run_tags[f"user_tag_{i}"] = tag

        self.run = mlflow.start_run(run_name=name, tags=run_tags if run_tags else None)

        # Log config as params (flatten nested dataclass)
        flat_config = self._flatten_config(asdict(cfg))
        # MLflow params have a 500 char limit per value
        truncated_config = {
            k: str(v)[:500] if len(str(v)) > 500 else v for k, v in flat_config.items()
        }
        mlflow.log_params(truncated_config)

        if notes:
            mlflow.set_tag("mlflow.note.content", notes)

        return self.run

    def _flatten_config(self, d: dict, parent_key: str = "") -> dict:
        """Flatten nested dict for MLflow params.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix

        Returns:
            Flattened dictionary with dot-separated keys
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,  # ignored, MLflow commits immediately
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Log metrics with auto-prefixing by current stage.

        Args:
            metrics: Metric name/value pairs
            step: Override global step (None = use internal counter)
            commit: Ignored (MLflow commits immediately)
            prefix: Auto-prefix metrics with current stage (skips keys with "/" or ".")
            increment: Increment step counters
        """
        if increment:
            self._step += 1
            if self._current_stage:
                self._stage_steps[self._current_stage] += 1

        current_step = step if step is not None else self._step

        # Build prefixed metrics (MLflow uses dots, not slashes)
        prefixed_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            if prefix and self._current_stage and "/" not in key and "." not in key:
                prefixed_metrics[f"{self._current_stage}.{key}"] = value
            else:
                # Convert slashes to dots for MLflow compatibility
                prefixed_metrics[key.replace("/", ".")] = value

        # Add step metrics
        prefixed_metrics["step"] = current_step
        if self._current_stage and increment:
            prefixed_metrics[f"{self._current_stage}.step"] = self._stage_steps[
                self._current_stage
            ]

        # Save to history if CSV path provided
        if self.csv_path:
            self.metrics_history.append(prefixed_metrics.copy())

        # Log to MLflow
        if self.enabled and self.run:
            mlflow = self._get_mlflow()
            # Filter to only numeric values (MLflow only accepts numbers for metrics)
            # Exclude bools: bool is a subclass of int in Python, but casting
            # True/False to float(1.0/0.0) corrupts categorical semantics.
            # Instead, booleans are converted to int (0/1) separately.
            numeric_metrics = {
                k: float(v)
                for k, v in prefixed_metrics.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and k != "step"
            }
            # Convert booleans to int (0/1) so they remain discrete metrics
            bool_metrics = {
                k: int(v) for k, v in prefixed_metrics.items() if isinstance(v, bool)
            }
            numeric_metrics.update(bool_metrics)
            if numeric_metrics:
                # WandB uses the global step as x-axis, however, MLflow uses the per-stage step so that
                # sparse global-step values do not cause MLflow to downsample away the rare detected=1 events.
                log_step = (
                    self._stage_steps[self._current_stage]
                    if self._current_stage
                    else current_step
                )
                mlflow.log_metrics(numeric_metrics, step=log_step)

    def save(
        self,
        file_path: str | Path,
        base_path: str | Path | None = None,
        policy: Literal["now", "live", "end"] = "now",
    ) -> None:
        """Save file as MLflow artifact.

        Args:
            file_path: Path to file to save
            base_path: Ignored (MLflow doesn't use base_path)
            policy: Ignored (MLflow saves immediately)
        """
        if self.enabled and self.run:
            mlflow = self._get_mlflow()
            mlflow.log_artifact(str(file_path))

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
        """Finish MLflow run and optionally save CSV.

        Args:
            exit_code: Exit code (non-zero marks run as failed)
            save_csv: Whether to save metrics to CSV

        Returns:
            Path to saved CSV or None
        """
        csv_path = self.to_csv() if save_csv and self.csv_path else None

        if self.enabled and self.run:
            mlflow = self._get_mlflow()
            if exit_code and exit_code != 0:
                mlflow.set_tag("mlflow.runStatus", "FAILED")
            mlflow.end_run()
            self.run = None

        return csv_path

    @property
    def url(self) -> str | None:
        """Get URL to the MLflow run UI."""
        if self.run:
            mlflow = self._get_mlflow()
            tracking_uri = mlflow.get_tracking_uri()
            # Handle file:// URIs (local mlflow)
            if tracking_uri.startswith("file://") or tracking_uri.startswith("/"):
                return None
            return f"{tracking_uri}/#/experiments/{self.run.info.experiment_id}/runs/{self.run.info.run_id}"
        return None
