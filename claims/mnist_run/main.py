import sys

import numpy as np
import torch

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, Config

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)

    # Make the run reproducible. Two independent RNGs decide what this
    # run does, and neither is seeded by anything in the framework:
    #
    #   torch  -- three of the harness's loaders use shuffle=True with no
    #             generator, so the batch order comes from the global torch
    #             RNG. Unseeded, roughly half the metrics move between runs.
    #   numpy  -- ContinuousMonitor._log_metrics samples `detected=0` rows at
    #             10% via np.random.random() to hold log volume down, so
    #             unseeded the SHAPE of the metrics file varies too: the same
    #             config wrote 1843, 1842 and 1846 rows on three runs.
    #
    # See claims/LOG.md A41.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Must precede get_example(): get_logger() ignores its arguments once an
    # instance exists, so a harness that logs from __init__ would pin the config.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
    )

    modelHarness = get_example(cfg=cfg)

    # Determine project/experiment name
    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    # Create continuous monitor - replaces fixed loop and detector instantiation
    monitor = ContinuousMonitor(
        cfg=cfg,
        modelHarness=modelHarness,
    )

    # Run continuous monitoring
    monitor.run()

    # TODO: Save a model checkpoint

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
