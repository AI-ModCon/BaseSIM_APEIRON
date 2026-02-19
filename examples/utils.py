from examples.cifar.model import CIFAR_VISION
from examples.imagenet.model import IMAGENET_VISION
from examples.mnist.model import MNIST_CNN

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness


def get_example(cfg: Config) -> BaseModelHarness:
    if cfg.data.name == "mnist":
        return MNIST_CNN(cfg=cfg)
    elif cfg.data.name == "cifar10":
        return CIFAR_VISION(cfg=cfg)
    elif cfg.data.name == "imagenet":
        return IMAGENET_VISION(cfg=cfg)
    elif cfg.data.name == "matey":
        from examples.matey.model import MATEYHarness

        return MATEYHarness(cfg=cfg)
    else:
        raise NotImplementedError(
            f"Example for dataset {cfg.data.name} is not implemented."
        )
