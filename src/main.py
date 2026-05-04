import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, Config

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    # Configure logger
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.visualization.input if cfg.visualization else None,
    )

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
