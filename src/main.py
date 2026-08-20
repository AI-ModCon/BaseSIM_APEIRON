import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, Config
from apeiron.distributed import comm

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)

    # Create the process group (no-op single-process) before anything reads it.
    comm.init_from_env()

    # Only rank 0 writes metrics/CSV; other ranks get a silent logger so a
    # multi-rank run does not produce N duplicate wandb runs or CSVs.
    # Must precede get_example(): get_logger() ignores its arguments once an
    # instance exists, so a harness that logs from __init__ would pin the config.
    backend = configure_backend(cfg) if comm.is_main else "none"
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=(
            cfg.visualization.input if (cfg.visualization and comm.is_main) else None
        ),
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
        logger=logger,
    )

    # Run continuous monitoring
    monitor.run()

    # TODO: Save a model checkpoint

    logger.finish()
    comm.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
