"""Unified logging interface combining WandB metrics and console output."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Literal

from config.configuration import Config
from logger.console_logger import ConsoleLogger
from logger.wandb_logger import StageType, WandBLogger


class Logger:
    """Unified logger with metrics tracking (WandB) and console output."""

    def __init__(
        self,
        verbosity: str = "INFO",
        wandb_enabled: bool = True,
        csv_path: str | Path | None = None,
    ):
        """Initialize unified logger.

        Args:
            verbosity: Console level (DEBUG, INFO, INFO:n, WARNING, ERROR, CRITICAL)
            wandb_enabled: Enable WandB logging
            csv_path: Save metrics to CSV at this path (disabled if None)
        """
        self._metrics = WandBLogger(enabled=wandb_enabled, csv_path=csv_path)
        self._console = ConsoleLogger(
            verbosity=verbosity, get_step=lambda: self._metrics.step
        )

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
    wandb_enabled: bool = True,
    csv_path: str | Path | None = None,
) -> Logger:
    """Get or create the default Logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(
            verbosity=verbosity, wandb_enabled=wandb_enabled, csv_path=csv_path
        )
    return _default_logger
