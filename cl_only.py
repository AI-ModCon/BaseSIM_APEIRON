import sys


from src.config.configuration import build_config, Config

from tqdm import tqdm

from examples.utils import get_example

from src.training.continual_learning import continual_learning_loop


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = get_example(cfg=cfg)

    progress_bar = tqdm(range(10), desc="CL Tasks", leave=True)

    for i in progress_bar:
        continual_learning_loop(cfg=cfg, modelHarness=modelHarness)

    return 0


if __name__ == "__main__":
    sys.exit(main())
