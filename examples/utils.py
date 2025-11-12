from examples.cifar10.model import CIFAR10_VIT16B
from examples.mnist.model import MNIST_CNN

from src.config.configuration import Config
from src.model.torch_model_harness import BaseModelHarness


def get_example(cfg: Config) -> BaseModelHarness:

    if cfg.data.name == "mnist":
        return MNIST_CNN(cfg=cfg)
    elif cfg.data.name == "cifar10":
        if cfg.model.name == "vit16b":
            return CIFAR10_VIT16B(cfg=cfg)
        raise NotImplementedError
    else:
        raise NotImplementedError
