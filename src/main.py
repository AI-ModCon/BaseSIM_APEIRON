import sys
from tqdm import tqdm

from logger import get_logger
from config.configuration import build_config, Config

from examples.utils import get_example

from training.continual_learning import continual_learning_loop
from drift_detection.drift_detection_driver import drift_detection_driver


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    logger = get_logger(enabled=True, csv_path="./output/main.csv")
    logger.init(cfg, project="main")

    progress_bar = tqdm(range(10), desc="CL Tasks", leave=True)
    for i in progress_bar:
        drift_signal = drift_detection_driver(cfg, modelHarness, logger)
        logger.log({"drift_signal": drift_signal.drift_detected}, step=i)

        if drift_signal.drift_detected or i == 0:
            continual_learning_loop(
                cfg=cfg, modelHarness=modelHarness, logger=logger, global_iter=i
            )

    print("\nLogged Metrics:\n", logger.to_dataframe())

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
