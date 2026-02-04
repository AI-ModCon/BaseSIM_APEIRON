from __future__ import annotations

import argparse
import sys
import os
import json
import subprocess
import torch
import dataclasses as _dc

# Handle tomllib for Python 3.10 vs 3.11+
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. Install with: pip install tomli"
        )

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Mapping


def get_available_device(multi_gpu: bool = False) -> torch.device:
    """
    Returns a torch.device with sensible fallbacks:
      - CPU-only hosts: 'cpu'
      - CUDA hosts:
          * multi_gpu=True  -> 'cuda' (let caller handle DDP/DataParallel)
          * multi_gpu=False -> choose GPU with most free memory, then restrict
            CUDA_VISIBLE_DEVICES so only that GPU is visible.
      - Apple Silicon with PyTorch MPS: 'mps' if CUDA is unavailable
    Never raises if nvidia-smi is missing.
    """
    # Single-GPU mode: must set CUDA_VISIBLE_DEVICES *before* CUDA init
    if not multi_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        best = _select_best_gpu()
        if best is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best)

    # Now check CUDA availability (this initializes CUDA)
    if torch.cuda.is_available():
        if multi_gpu:
            return torch.device("cuda")
        # After restricting, there's only cuda:0
        return torch.device("cuda:0")

    # CUDA not available: try MPS (Apple), otherwise CPU
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def _select_best_gpu() -> int | None:
    """Select GPU with most free memory using nvidia-smi (pre-CUDA-init safe)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        )
        rows = [int(x.strip()) for x in out.decode().strip().splitlines() if x.strip()]
        if rows:
            return max(range(len(rows)), key=lambda i: rows[i])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return None


@dataclass(frozen=True)
class ModelCfg:
    name: str
    pretrained_path: str
    # ckpt: str | None = None # perhaps later support checkpointing files


@dataclass(frozen=True)
class TrainCfg:
    batch_size: int
    num_workers: int
    init_lr: float
    grad_accumulation_steps: int = 1
    max_iter: int = 600  # the maximum number of iterations for one cl application


@dataclass(frozen=True)
class DataCfg:
    name: str
    path: str


@dataclass(frozen=True)
class ContinualLearningCfg:
    update_mode: str = "base"

    # For JVP regularization
    jvp_lambda: float = 0.001
    jvp_deltax_norm: float = 1

    # For EWC method
    ewc_lambda: float = 1000.0
    ewc_ema_decay: float = 0.95

    # For KFAC method
    kfac_lambda: float = 0.01
    kfac_ema_decay: float = 0.95


@dataclass(frozen=True)
class DriftDetectionCfg:
    detector_name: str = (
        "ADWINDetector"  # "ADWINDetector", "KSWINDetector", "PageHinkleyDetector", etc.
    )
    detection_interval: int = 10  # Check drift every N batches
    aggregation: str = "mean"  # How to aggregate metrics: "mean", "last", "median"
    metric_index: int = 0  # Which metric to monitor (0=first, 1=second, etc.)
    reset_after_learning: bool = False  # Reset detector after CL loop
    max_stream_updates: int = 20  # Stop after N stream extensions

    # ADWIN hyperparameters
    adwin_delta: float = 0.002
    adwin_minor_threshold: float = 0.3
    adwin_moderate_threshold: float = 0.6

    # KSWIN hyperparameters
    kswin_alpha: float = 0.005
    kswin_window_size: int = 100
    kswin_stat_size: int = 30

    # PageHinkley hyperparameters
    ph_min_instances: int = 30
    ph_delta: float = 0.005
    ph_threshold: float = 50
    ph_alpha: float = 0.9999


@dataclass(frozen=True)
class VisualizationCfg:
    baseline: float = 95.0  # baseline accuracy threshold for drift detection
    input: str = "output/cl_only.csv"  # input CSV file path
    output: str = "output/drift_dashboard.png"  # output dashboard image path


@dataclass(frozen=True)
class LoggingCfg:
    backend: str = "wandb"  # "wandb", "mlflow", or "none"
    mlflow_tracking_uri: str | None = None  # MLflow tracking server URI
    mlflow_experiment_name: str | None = None  # Override experiment name


@dataclass(frozen=True)
class Config:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    continual_learning: ContinualLearningCfg
    drift_detection: DriftDetectionCfg

    seed: int
    device: str
    multi_gpu: bool
    verbosity: str = "INFO"
    visualization: VisualizationCfg | None = None
    logging: LoggingCfg | None = None


def parse_args(argv=None):
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : list[str] | None
        The command line arguments to parse. If None, use sys.argv.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], help="key=val, repeatable")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: auto|cpu|cuda|cuda:N|mps (overrides TOML/env)",
    )
    p.add_argument(
        "--multi-gpu",
        action="store_true",
        help="When --device=auto, prefer multi-GPU CUDA device",
    )

    return p.parse_args(argv)


def load_toml(p: Path) -> dict[str, Any]:
    """
    Load TOML configuration from file.

    Parameters
    ----------
    p : Path
        The file path to load from.

    Returns
    -------
    dict[str, Any]
        The loaded configuration.
    """
    with p.open("rb") as f:
        return tomllib.load(f)


def deep_update(x: dict, y: Mapping) -> dict:
    """
    Recursively update a nested dictionary with values from another mapping.

    Parameters
    ----------
    x : dict
        The dictionary to update.
    y : Mapping
        The mapping containing the values to update with.

    Returns
    -------
    dict
        The updated dictionary.
    """
    for k, v in y.items():
        x[k] = (
            deep_update(dict(x[k]), v)
            if isinstance(v, Mapping) and isinstance(x.get(k), Mapping)
            else v
        )
    return x


def kv_to_nested(items: list[str]) -> dict[str, Any]:
    """
    Recursively build a nested dictionary from a list of key-value strings.

    Parameters
    ----------
    items : list[str]
        List of key-value strings in the format "key=value".

    Returns
    -------
    dict[str, Any]
        The built nested dictionary.
    """
    out: dict[str, Any] = {}
    for s in items:
        k, v = s.split("=", 1)
        try:
            v = json.loads(v)  # parse numbers/bools/lists if given
        except json.JSONDecodeError:
            pass
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def env_overrides(prefix="APP_") -> dict[str, Any]:
    """
    Recursively build a nested dictionary from environment variables that start with the given prefix.

    Parameters
    ----------
    prefix : str, optional
        The prefix to filter environment variables with. Defaults to "APP_".

    Returns
    -------
    dict[str, Any]
        The built nested dictionary.
    """
    items = [
        f"{k[len(prefix) :].lower()}={v}"
        for k, v in os.environ.items()
        if k.startswith(prefix)
    ]
    return kv_to_nested(items)


def build_config(argv=None) -> Config:
    """
    Load configuration from TOML file, environment variables, and command line arguments.

    Parameters
    ----------
    argv : list[str] | None
        The command line arguments to parse. If None, use sys.argv.

    Returns
    -------
    Config
        The final configuration.
    """
    args = parse_args(argv)
    cfg = load_toml(args.config)
    cfg = deep_update(cfg, env_overrides("APP_"))
    cfg = deep_update(cfg, kv_to_nested(args.set))
    # validate/freeze
    model = ModelCfg(**cfg["model"])
    data = DataCfg(**cfg["data"])
    train = TrainCfg(**cfg["train"])
    dd = DriftDetectionCfg(**cfg["drift_detection"])
    cl = ContinualLearningCfg(**cfg.get("continual_learning", {}))
    viz = VisualizationCfg(**cfg["visualization"]) if "visualization" in cfg else None
    log_cfg = LoggingCfg(**cfg["logging"]) if "logging" in cfg else None

    raw_device = str(
        cfg.get(
            "device",
            args.device if getattr(args, "device", None) is not None else "auto",
        )
    )
    multi_gpu_flag = bool(cfg.get("multi_gpu", getattr(args, "multi_gpu", False)))

    resolved_device = (
        str(get_available_device(multi_gpu=multi_gpu_flag))
        if raw_device.lower() == "auto"
        else raw_device
    )

    explicit = {
        "model",
        "data",
        "train",
        "continual_learning",
        "drift_detection",
        "visualization",
        "logging",
        "device",
        "multi_gpu",
    }
    # also exclude any keys not in Config to avoid surprises
    valid = {f.name for f in _dc.fields(Config)}
    extras = {k: v for k, v in cfg.items() if k in valid - explicit}

    final = Config(
        model=model,
        data=data,
        train=train,
        continual_learning=cl,
        drift_detection=dd,
        visualization=viz,
        logging=log_cfg,
        device=resolved_device,
        multi_gpu=multi_gpu_flag,
        **extras,
    )

    Path("resolved_config.json").write_text(json.dumps(asdict(final), indent=2))
    return final
