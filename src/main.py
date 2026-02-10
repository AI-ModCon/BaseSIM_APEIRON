import sys

from logger import get_logger
from config.configuration import build_config, Config

from examples.utils import get_example

from driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    # Configure logger on entry to main
    logger = get_logger(
        verbosity=cfg.verbosity,
        wandb_enabled=True,
        csv_path=cfg.visualization.input if cfg.visualization else None,
    )
    logger.init(cfg, project="main")

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
