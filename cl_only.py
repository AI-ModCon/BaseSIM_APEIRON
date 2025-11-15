import sys
from tqdm import tqdm

from src.logging import get_logger
from src.config.configuration import build_config, Config
from src.training.continual_learning import continual_learning_loop

from examples.mnist.model import MNIST_CNN

def main(argv=None) -> int:

    cfg: Config = build_config(argv)
    modelHarness = MNIST_CNN(cfg=cfg)

    logger = get_logger(enabled=True, csv_path="./output/cl_only.csv")
    logger.init(cfg, project="cl_only")

    progress_bar = tqdm(range(10), desc="CL Tasks", leave=True)
    for i in progress_bar:

        continual_learning_loop(cfg=cfg, modelHarness=modelHarness, logger=logger, global_iter=i)

    print("\nLogged Metrics:\n", logger.to_dataframe())

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
