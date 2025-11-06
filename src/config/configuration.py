# config.py
from __future__ import annotations
import argparse
import tomllib
import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelCfg:
    name: str
    # ckpt: str | None = None # perhaps later support checkpointing files


@dataclass(frozen=True)
class TrainCfg:
    epochs: int


@dataclass(frozen=True)
class DataCfg:
    name: str
    path: str
    batch_size: int = 64
    num_workers: int = 4


@dataclass(frozen=True)
class Config:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    seed: int = 0
    version: str = "1"


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

    final = Config(
        model=model,
        data=data,
        train=train,
        **{k: v for k, v in cfg.items() if k not in {"model", "data", "train"}},
    )
    Path("resolved_config.json").write_text(json.dumps(asdict(final), indent=2))
    return final
