import torch

from src.config.configuration import Config
from src.model.DummyCNN_MNIST import DummyCNN_MNIST


def load_model(cfg: Config) -> torch.nn.Module:

    if cfg.model.name == "dummy":
        return DummyCNN_MNIST()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    return 0
