"""Unified logging interface combining metrics tracking and console output."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Literal

from config.configuration import Config
from logger.console_logger import ConsoleLogger
from logger.wandb_logger import StageType, WandBLogger
from logger.mlflow_logger import MLFlowLogger


# Type alias for metrics backend
MetricsBackend = Literal["wandb", "mlflow", "none"]


class Logger:
    """Unified logger with metrics tracking (WandB/MLflow) and console output."""

    def __init__(
        self,
        verbosity: str = "INFO",
        backend: MetricsBackend = "wandb",
        csv_path: str | Path | None = None,
        # Legacy parameter for backwards compatibility
        wandb_enabled: bool | None = None,
    ):
        """Initialize unified logger.

        Args:
            verbosity: Console level (DEBUG, INFO, INFO:n, WARNING, ERROR, CRITICAL)
            backend: Metrics backend to use ("wandb", "mlflow", or "none")
            csv_path: Save metrics to CSV at this path (disabled if None)
            wandb_enabled: Deprecated. Use backend="wandb" or backend="none" instead.
        """
        # Handle legacy wandb_enabled parameter
        if wandb_enabled is not None:
            backend = "wandb" if wandb_enabled else "none"

        # Initialize metrics backend
        if backend == "mlflow":
            self._metrics: WandBLogger | MLFlowLogger = MLFlowLogger(
                enabled=True, csv_path=csv_path
            )
        elif backend == "wandb":
            self._metrics = WandBLogger(enabled=True, csv_path=csv_path)
        else:  # "none"
            self._metrics = WandBLogger(enabled=False, csv_path=csv_path)

        self._backend = backend
        self._console = ConsoleLogger(
            verbosity=verbosity, get_step=lambda: self._metrics.step
        )

    @property
    def backend(self) -> MetricsBackend:
        """Get the current metrics backend."""
        return self._backend

    # Metrics/WandB methods
    def init(
        self,
        cfg: Config,
        project: str = "basesim-framework",
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        **kwargs,
    ):
        """Initialize wandb run with config."""
        return self._metrics.init(cfg, project, name, tags, notes, group, **kwargs)

    def stage(self, name: StageType) -> None:
        """Set current stage ('eval', 'drift', or 'cl')."""
        self._metrics.stage(name)

    @property
    def step(self) -> int:
        return self._metrics.step

    @property
    def current_stage(self) -> StageType | None:
        return self._metrics.current_stage

    def get_stage_step(self, stage: StageType) -> int:
        return self._metrics.get_stage_step(stage)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Log metrics with stage-aware prefixing."""
        self._metrics.log(metrics, step, commit, prefix, increment)

    def save(
        self,
        file_path: str | Path,
        base_path: str | Path | None = None,
        policy: Literal["now", "live", "end"] = "now",
    ) -> None:
        """Save file to wandb."""
        self._metrics.save(file_path, base_path, policy)

    def to_dataframe(self) -> pd.DataFrame | None:
        """Convert metrics to DataFrame."""
        return self._metrics.to_dataframe()

    def to_csv(self, csv_path: str | Path | None = None) -> Path | None:
        """Save metrics to CSV."""
        return self._metrics.to_csv(csv_path)

    def finish(
        self, exit_code: int | None = None, save_csv: bool = True
    ) -> Path | None:
        """Finish wandb run and save CSV."""
        return self._metrics.finish(exit_code, save_csv)

    @property
    def url(self) -> str | None:
        return self._metrics.url

    # Console logging methods
    @property
    def verbosity(self) -> str:
        return self._console.verbosity

    @verbosity.setter
    def verbosity(self, level: str) -> None:
        self._console.verbosity = level

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._console.debug(msg, *args, **kwargs)

    def info(self, msg: str, level: int = 0, *args, **kwargs) -> None:
        """Log at INFO:level tier (0=standard, 1+=more verbose)."""
        self._console.info(msg, level, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._console.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._console.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._console.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._console.exception(msg, *args, **kwargs)


# Singleton for convenience
_default_logger: Logger | None = None


def get_logger(
    verbosity: str = "INFO",
    backend: MetricsBackend = "wandb",
    csv_path: str | Path | None = None,
    # Legacy parameter for backwards compatibility
    wandb_enabled: bool | None = None,
) -> Logger:
    """Get or create the default Logger instance.

    Args:
        verbosity: Console level (DEBUG, INFO, INFO:n, WARNING, ERROR, CRITICAL)
        backend: Metrics backend to use ("wandb", "mlflow", or "none")
        csv_path: Save metrics to CSV at this path (disabled if None)
        wandb_enabled: Deprecated. Use backend="wandb" or backend="none" instead.

    Returns:
        Logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(
            verbosity=verbosity,
            backend=backend,
            csv_path=csv_path,
            wandb_enabled=wandb_enabled,
        )
    return _default_logger


def reset_logger() -> None:
    """Reset the default logger instance. Useful for testing."""
    global _default_logger
    _default_logger = None


def configure_backend(cfg: Config | None) -> MetricsBackend:
    """Configure and return the logging backend from config."""
    if cfg is None or cfg.logging is None:
        return "wandb"

    backend = cfg.logging.backend
    if backend == "mlflow" and cfg.logging.mlflow_tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)

    if backend not in ("wandb", "mlflow", "none"):
        raise ValueError(f"Invalid logging backend: {backend}")

    return backend
