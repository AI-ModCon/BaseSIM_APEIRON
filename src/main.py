import sys

from logger import get_logger
from config.configuration import build_config, Config

from examples.utils import get_example

from drift_detection.continuous_monitor import ContinuousMonitor


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    logger = get_logger(
        enabled=True, csv_path=cfg.visualization.input if cfg.visualization else None
    )
    logger.init(cfg, project="main")

    # Create continuous monitor - replaces fixed loop and detector instantiation
    monitor = ContinuousMonitor(
        cfg=cfg,
        modelHarness=modelHarness,
        logger=logger,
    )

    # Run continuous monitoring
    monitor.run()

    print("\nLogged Metrics:\n", logger.to_dataframe())

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
