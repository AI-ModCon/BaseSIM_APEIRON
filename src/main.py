import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, Config

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)

    # Configure the logger before building the harness. get_logger() returns any
    # existing instance without applying its arguments, so a harness that logs
    # from __init__ -- as one carrying its resolved settings does -- would
    # otherwise create the singleton first and pin the default backend and a
    # null CSV path, silently dropping visualization.input for the whole run.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.visualization.input if cfg.visualization else None,
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
