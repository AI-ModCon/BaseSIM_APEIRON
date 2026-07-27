from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.training.updater.base import BaseUpdater


def create_updater(
    cfg: Config,
    modelHarness: BaseModelHarness,
    optimizer: Optional[Optimizer] = None,
) -> BaseUpdater:
    """Create an updater instance based on the configuration.

    Args:
        cfg: Configuration object with training settings.
        modelHarness: Model harness containing model and data loaders.
        optimizer: The trainer's optimizer. Only used by the ``retrain`` mode,
            which clears its state on a from-scratch rebuild.

    Returns:
        An instance of the specified updater class.

    Raises:
        NotImplementedError: If the specified updater mode is not implemented.
    """
    if cfg.continual_learning.update_mode == "base":
        return BaseUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "ewc_online":
        from apeiron.training.updater.ewc import OnlineEWCUpdater

        return OnlineEWCUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "kfac_online":
        from apeiron.training.updater.kfac import OnlineKFACUpdater

        return OnlineKFACUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "jvp_reg":
        from apeiron.training.updater.jvp_reg import JVPRegUpdater

        return JVPRegUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "retrain":
        from apeiron.training.updater.retrain import RetrainUpdater

        return RetrainUpdater(cfg=cfg, modelHarness=modelHarness, optimizer=optimizer)

    if cfg.continual_learning.update_mode == "none":
        from apeiron.training.updater.no_updater import NoUpdater

        return NoUpdater(cfg=cfg, modelHarness=modelHarness)

    raise NotImplementedError(
        f"Unknown update_mode: {cfg.continual_learning.update_mode}"
    )
