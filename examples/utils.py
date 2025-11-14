from examples.cifar.model import CIFAR_VISION
from examples.mnist.model import MNIST_CNN

from src.config.configuration import Config
from src.model.torch_model_harness import BaseModelHarness


def get_example(cfg: Config) -> BaseModelHarness:

    if cfg.data.name == "mnist":
        return MNIST_CNN(cfg=cfg)
    elif cfg.data.name == "cifar10":
        return CIFAR_VISION(cfg=cfg)
    else:
        raise NotImplementedError
