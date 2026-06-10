import sys

from logger import get_logger, configure_backend, reset_logger
from config.configuration import build_config, Config

from examples.utils import get_example

from driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    # Configure logger (reset singleton: harness init may have created wandb default)
    backend = configure_backend(cfg)
    reset_logger()
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
    get_logger().info("wandb run initialized", level=0)

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
