import sys
from tqdm import tqdm

from src.logging import get_logger
from src.config.configuration import build_config, Config

from examples.utils import get_example

from src.training.continual_learning import continual_learning_loop
from src.drift_detection.drift_detection_driver import drift_detection_driver


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    logger = get_logger(
        enabled=True, csv_path=cfg.visualization.input if cfg.visualization else None
    )
    logger.init(cfg, project="main")

    # Global step tracked over self-improvement loop.
    #   Managing global step can be implemented into logger in future PR.
    global_step = 0
    progress_bar = tqdm(range(10), desc="CL Tasks", leave=True)
    for i in progress_bar:
        drift_signal = drift_detection_driver(
            cfg, modelHarness, logger, global_step=global_step
        )
        print("Drift Detected:", drift_signal.drift_detected)

        # Self-improvement actuation.
        # NOTE: To test visualization, hardcodes
        # 2 rounds of basic and
        # 3 rounds of jvp_reg
        if drift_signal.drift_detected or i < 5:
            continual_learning_loop(
                cfg=cfg,
                modelHarness=modelHarness,
                logger=logger,
                global_step=global_step,
                basic_only=(i < 2 and not drift_signal.drift_detected),
            )

        # Update steps are tracked and require advancing global step.
        # NOTE: Given drift_signal actuates model update,
        #  when model update is skipped we assume (for now)
        #  global step still proceeds as if model updates.
        global_step += cfg.continuous_learning.max_iter

    print("\nLogged Metrics:\n", logger.to_dataframe())

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
