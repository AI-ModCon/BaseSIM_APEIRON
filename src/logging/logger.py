"""Weights & Biases logging interface for BaseSim Framework."""

from __future__ import annotations

import wandb
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import asdict

from src.config.configuration import Config


class WandBLogger:
    """Interface for logging experiments with Weights & Biases."""

    def __init__(
        self,
        enabled: bool = True,
        save_to_csv: bool = True,
        csv_path: Optional[str | Path] = None,
    ):
        """
        Initialize the WandB logger.

        Parameters
        ----------
        enabled : bool, optional
            Whether to enable wandb logging. Defaults to True.
        save_to_csv : bool, optional
            Whether to save metrics to CSV file. Defaults to True.
        csv_path : str | Path | None, optional
            Full path to the CSV file where metrics will be saved.
            If None, defaults to "./output/metrics.csv".
        """
        self.enabled = enabled
        self.save_to_csv = save_to_csv
        self.run: Optional[wandb.sdk.wandb_run.Run] = None
        self.metrics_history: List[Dict[str, Any]] = []

        # Set CSV path and create parent directory
        self.csv_path = Path(csv_path) if csv_path else Path("./output/metrics.csv")
        self.csv_path.parent.mkdir(exist_ok=True, parents=True)

    def init(
        self,
        cfg: Config,
        project: str = "basesim-framework",
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs,
    ) -> Optional[wandb.sdk.wandb_run.Run]:
        """
        Initialize a wandb run with configuration.

        Parameters
        ----------
        cfg : Config
            The configuration object to log.
        project : str, optional
            The wandb project name. Defaults to "basesim-framework".
        name : str | None, optional
            Run name. If None, wandb auto-generates one.
        tags : list[str] | None, optional
            Tags for organizing runs.
        notes : str | None, optional
            Description of the run.
        group : str | None, optional
            Group name for organizing related runs.
        **kwargs
            Additional arguments passed to wandb.init().

        Returns
        -------
        wandb.sdk.wandb_run.Run | None
            The initialized wandb run or None if disabled.
        """
        if not self.enabled:
            return None

        # Convert config to dict for logging
        config_dict = asdict(cfg)

        self.run = wandb.init(
            project=project,
            name=name,
            config=config_dict,
            tags=tags,
            notes=notes,
            group=group,
            **kwargs,
        )

        return self.run

    def log(
        self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True
    ) -> None:
        """
        Log metrics to wandb and optionally save to CSV.

        Parameters
        ----------
        metrics : dict[str, Any]
            Dictionary of metric names and values to log.
        step : int | None, optional
            Global step value. If None, wandb auto-increments.
        commit : bool, optional
            Whether to commit this log immediately. Defaults to True.

        Examples
        --------
        >>> logger.log({"train/loss": 0.5, "train/accuracy": 0.85})
        >>> logger.log({"val/loss": 0.3}, step=100)
        """
        # Track metrics locally if CSV saving is enabled
        if self.save_to_csv:
            metrics_with_step = {"step": step, **metrics}
            self.metrics_history.append(metrics_with_step)

        # Log to wandb
        if not self.enabled or self.run is None:
            return

        wandb.log(metrics, step=step, commit=commit)

    def save(
        self,
        file_path: str | Path,
        base_path: Optional[str | Path] = None,
        policy: str = "now",
    ) -> None:
        """
        Save a file to wandb.

        Parameters
        ----------
        file_path : str | Path
            Path to the file to save.
        base_path : str | Path | None, optional
            Base path to determine relative paths in wandb.
        policy : str, optional
            When to upload: "now", "end", or "live". Defaults to "now".

        Examples
        --------
        >>> logger.save("outputs/predictions.csv")
        """
        if not self.enabled or self.run is None:
            return

        wandb.save(
            str(file_path),
            base_path=str(base_path) if base_path else None,
            policy=policy,
        )

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Convert logged metrics to a pandas DataFrame in long format.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with columns ["step", "metric", "value"], or None if no metrics were logged.

        Examples
        --------
        >>> df = logger.to_dataframe()
        >>> print(df.head())
           step         metric  value
        0     1   train/loss    0.5
        1     1   train/accuracy 0.85
        """
        if not self.metrics_history:
            return None

        # Convert to long format: each metric gets its own row
        rows = []
        for entry in self.metrics_history:
            step = entry.get("step")
            for metric_name, value in entry.items():
                if metric_name != "step":
                    rows.append({"step": step, "metric": metric_name, "value": value})

        return pd.DataFrame(rows)

    def to_csv(self, csv_path: Optional[str | Path] = None) -> Optional[Path]:
        """
        Write all logged metrics to a CSV file in long format.

        Parameters
        ----------
        csv_path : str | Path | None, optional
            Full path to the CSV file. If None, uses the path from initialization.

        Returns
        -------
        Path | None
            Path to the saved CSV file, or None if no metrics were logged.

        Examples
        --------
        >>> logger.to_csv()  # Uses path from __init__
        >>> logger.to_csv("./output/custom_metrics.csv")

        Notes
        -----
        CSV format is long format with columns: ["step", "metric", "value"]
        Each metric-value pair gets its own row.
        """
        df = self.to_dataframe()
        if df is None:
            return None

        # Use provided path or default to initialized path
        output_path = Path(csv_path) if csv_path else self.csv_path

        # Ensure parent directory exists
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Write DataFrame to CSV
        df.to_csv(output_path, index=False)

        return output_path

    def finish(
        self, exit_code: Optional[int] = None, save_csv: bool = True
    ) -> Optional[Path]:
        """
        Finish the current wandb run and save metrics to CSV.

        Parameters
        ----------
        exit_code : int | None, optional
            Exit code for the run.
        save_csv : bool, optional
            Whether to save metrics to CSV before finishing. Defaults to True.

        Returns
        -------
        Path | None
            Path to the saved CSV file, or None if not saved.

        Examples
        --------
        >>> logger.finish()
        >>> csv_path = logger.finish(save_csv=True)
        """
        csv_path = None

        # Save metrics to CSV if enabled
        if save_csv and self.save_to_csv:
            csv_path = self.to_csv()

        # Finish wandb run
        if self.enabled and self.run is not None:
            wandb.finish(exit_code=exit_code)
            self.run = None

        return csv_path

    @property
    def url(self) -> Optional[str]:
        """Get the URL of the current run."""
        if self.run is None:
            return None
        return self.run.get_url()


# Singleton instance for convenience
_default_logger: Optional[WandBLogger] = None


def get_logger(
    enabled: bool = True,
    save_to_csv: bool = True,
    csv_path: Optional[str | Path] = None,
) -> WandBLogger:
    """
    Get or create the default WandBLogger instance.

    Parameters
    ----------
    enabled : bool, optional
        Whether to enable wandb logging. Defaults to True.
    save_to_csv : bool, optional
        Whether to save metrics to CSV file. Defaults to True.
    csv_path : str | Path | None, optional
        Full path to the CSV file where metrics will be saved.
        If None, defaults to "./output/metrics.csv".

    Returns
    -------
    WandBLogger
        The default logger instance.

    Examples
    --------
    >>> from src.logging.logger import get_logger
    >>> logger = get_logger(csv_path="./output/my_experiment.csv")
    >>> logger.init(cfg, project="my-experiment")
    >>> logger.log({"loss": 0.5})
    >>> logger.finish()  # Saves to ./output/my_experiment.csv
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = WandBLogger(
            enabled=enabled, save_to_csv=save_to_csv, csv_path=csv_path
        )
    return _default_logger
