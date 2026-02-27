from config.configuration import Config
from model.torch_model_harness import BaseModelHarness


def get_example(cfg: Config) -> BaseModelHarness:
    if cfg.data.name == "mnist":
        from examples.mnist.model import MNIST_CNN

        return MNIST_CNN(cfg=cfg)
    elif cfg.data.name == "cifar10":
        from examples.cifar.model import CIFAR_VISION

        return CIFAR_VISION(cfg=cfg)
    elif cfg.data.name == "imagenet":
        from examples.imagenet.model import IMAGENET_VISION

        return IMAGENET_VISION(cfg=cfg)
    elif cfg.data.name == "prometheus":
        from examples.prometheus.model import PrometheusHarness

        return PrometheusHarness(cfg=cfg)
    else:
        raise NotImplementedError(
            f"Example for dataset {cfg.data.name} is not implemented."
        )
