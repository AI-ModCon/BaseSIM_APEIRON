"""Run Apeiron's monitor -> detect -> adapt loop with your own harness and/or detector.

`python -m src.main` only knows the bundled examples (mnist, cifar10, imagenet).
This driver does the same thing as `src/main.py`, but lets you point it at a
harness, a drift detector and/or a continual-learning updater class that live in
your own files, so you do not have to edit anything inside Apeiron.

Run from the Apeiron repo root:

    # Bundled MNIST example, same as `python -m src.main`
    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml

    # Your own detector (or update rule) on the MNIST stream
    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml \
        --detector tutorials/a6/scripts/custom_detector.py:ThresholdDetector
    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml \
        --updater path/to/my_updater.py:MyUpdater

    # Your own model and data stream
    python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_custom.toml \
        --harness tutorials/a6/scripts/harness_template.py:DriftingRegressionHarness

Any other flag (--set key=value, --device) is passed through to Apeiron's config loader.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _repo_root_on_path() -> None:
    """Apeiron's examples are imported as `examples.*`, so the repo root must be importable."""
    root = Path.cwd()
    if not (root / "examples").is_dir() or not (root / "src").is_dir():
        sys.exit(
            "Run this script from the Apeiron repo root (the folder that contains "
            "src/ and examples/)."
        )
    sys.path.insert(0, str(root))


def _load_class(spec: str):
    """Load `path/to/file.py:ClassName`."""
    path, _, name = spec.partition(":")
    if not name:
        sys.exit(f"Expected FILE.py:ClassName, got {spec!r}")
    module_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if module_spec is None or module_spec.loader is None:
        sys.exit(f"Cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError:
        sys.exit(f"{path} has no class named {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--harness", help="FILE.py:ClassName of a BaseModelHarness subclass"
    )
    parser.add_argument(
        "--detector", help="FILE.py:ClassName of a BaseDriftDetector subclass"
    )
    parser.add_argument("--updater", help="FILE.py:ClassName of a BaseUpdater subclass")
    args, apeiron_args = parser.parse_known_args(argv)

    _repo_root_on_path()

    from apeiron.config.configuration import build_config
    from apeiron.driver.continuous_monitor import ContinuousMonitor
    from apeiron.logger import configure_backend, get_logger

    cfg = build_config(apeiron_args)

    # Same order as src/main.py: the logger must exist before the harness is built.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
    )

    if args.harness:
        harness = _load_class(args.harness)(cfg=cfg)
    else:
        from examples.utils import get_example

        harness = get_example(cfg=cfg)

    project = (
        cfg.logging.experiment_name
        if cfg.logging and cfg.logging.experiment_name
        else "apeiron"
    )
    logger.init(cfg, project=project)

    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    if args.detector:
        # Replace the detector built from [drift_detection]; the rest of the loop is unchanged.
        monitor.detector = _load_class(args.detector)()
        logger.info(f"\tUsing custom detector: {monitor.detector}", level=1)
    if args.updater:
        # Replace the update rule chosen by [continual_learning] update_mode.
        monitor.trainer.cl_updater = _load_class(args.updater)(
            cfg=cfg, modelHarness=harness
        )
        logger.info(
            f"\tUsing custom updater: {type(monitor.trainer.cl_updater).__name__}",
            level=1,
        )

    monitor.run()
    csv_path = logger.finish()
    if csv_path:
        print(f"\nMetrics written to {os.fspath(csv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
