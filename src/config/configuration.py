from __future__ import annotations

import argparse
import tomllib
import os
import json
import subprocess
import torch
import dataclasses as _dc

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
          * multi_gpu=False -> choose GPU with most free memory (torch.cuda.mem_get_info)
            (falls back to nvidia-smi if needed; then to cuda:0)
      - Apple Silicon with PyTorch MPS: 'mps' if CUDA is unavailable
    Never raises if nvidia-smi is missing.
    """
    # Prefer CUDA if available
    if torch.cuda.is_available():
        if multi_gpu:
            return torch.device("cuda")

        # Single-GPU selection: prefer PyTorch's mem_get_info (no external deps)
        try:
            n = torch.cuda.device_count()
            free_bytes = []
            for i in range(n):
                # Ensure we query device i
                with torch.cuda.device(i):
                    free_i, _total_i = torch.cuda.mem_get_info()
                free_bytes.append(free_i)

            best = max(range(n), key=lambda i: free_bytes[i])
            return torch.device(f"cuda:{best}")

        except Exception:
            # Fallback to nvidia-smi if mem_get_info is unavailable/problematic
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                )
                rows = [
                    int(x.strip())
                    for x in out.decode().strip().splitlines()
                    if x.strip()
                ]
                best = max(range(len(rows)), key=lambda i: rows[i])
                return torch.device(f"cuda:{best}")
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Last resort: first CUDA device
                return torch.device("cuda:0")

    # CUDA not available: try MPS (Apple), otherwise CPU
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


@dataclass(frozen=True)
class ModelCfg:
    name: str
    # ckpt: str | None = None # perhaps later support checkpointing files


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class DataCfg:
    name: str
    path: str


@dataclass(frozen=True)
class ContinuousLearningCfg:
    x_updates: int # Not useful anymore, because of JVP implementation
    theta_updates: int # Not useful anymore, because of JVP implementation
    factor: float # this is now the lambda factor for the JVP regularization.
    x_lr: float # Not useful anymore, because of JVP implementation
    th_lr: float # Not useful anymore, because of JVP implementation
    total_updates: int  # TODO: Make sure that this does not conflict wiht train.epochs
                        # train.epoch is not relevant anymore because, I am doing max_iterations 
                        # inside the continual learning loop. max_iterations is total updates 
                        # now


@dataclass(frozen=True)
class Config:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    continuous_learning: ContinuousLearningCfg

    seed: int
    version: str
    device: str
    multi_gpu: bool


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
        f"{k[len(prefix):].lower()}={v}"
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
    cl = ContinuousLearningCfg(**cfg.get("continuous_learning", {}))

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

    explicit = {"model", "data", "train", "continuous_learning", "device", "multi_gpu"}
    # also exclude any keys not in Config to avoid surprises
    valid = {f.name for f in _dc.fields(Config)}
    extras = {k: v for k, v in cfg.items() if k in valid - explicit}

    final = Config(
        model=model,
        data=data,
        train=train,
        continuous_learning=cl,
        device=resolved_device,
        multi_gpu=multi_gpu_flag,
        **extras,
    )

    Path("resolved_config.json").write_text(json.dumps(asdict(final), indent=2))
    return final
