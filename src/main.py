import sys

from logger import get_logger
from config.configuration import build_config, Config

from examples.utils import get_example

from training.continual_learning import continual_learning_loop
from drift_detection.drift_detection_driver import drift_detection_driver
from drift_detection.load_drift_detector import load_drift_detector


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
    for i in range(20):
        # Create an artificial data_drift
        modelHarness.update_data_stream()

        # drift_signal = drift_detection_driver(
        #     cfg, modelHarness, logger, global_step=global_step
        # )
        detector = load_drift_detector(cfg)

        drift_signal = detector.update(
            modelHarness,
            reference_validation_metrics=[90, 1.0],
            higher_is_better=[True, False],
        )

        print(drift_signal)
        print("Drift Detected:", drift_signal.drift_detected)
        print("------------------")

        # Self-improvement actuation.

        if drift_signal.drift_detected:
            continual_learning_loop(
                cfg=cfg,
                modelHarness=modelHarness,
                logger=logger,
                global_step=global_step,
                basic_only=False,
            )

        # Update steps are tracked and require advancing global step.
        # NOTE: Given drift_signal actuates model update,
        #  when model update is skipped we assume (for now)
        #  global step still proceeds as if model updates.
        global_step += cfg.continuous_learning.max_iter

        # TODO: Save a model checkpoint

    print("\nLogged Metrics:\n", logger.to_dataframe())

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
