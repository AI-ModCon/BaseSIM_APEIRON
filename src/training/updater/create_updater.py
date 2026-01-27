from __future__ import annotations

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness
from training.updater.base import BaseUpdater


def create_updater(cfg: Config, modelHarness: BaseModelHarness) -> BaseUpdater:
    """Create an updater instance based on the configuration.

    Args:
        cfg: Configuration object with training settings.
        modelHarness: Model harness containing model and data loaders.

    Returns:
        An instance of the specified updater class.

    Raises:
        NotImplementedError: If the specified updater mode is not implemented.
    """
    if cfg.continual_learning.update_mode == "base":
        return BaseUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "ewc_online":
        from training.updater.ewc import OnlineEWCUpdater

        return OnlineEWCUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "kfac_online":
        from training.updater.kfac import OnlineKFACUpdater

        return OnlineKFACUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "jvp_reg":
        from training.updater.jvp_reg import JVPRegUpdater

        return JVPRegUpdater(cfg=cfg, modelHarness=modelHarness)

    if cfg.continual_learning.update_mode == "none":
        from training.updater.no_updater import NoUpdater

        return NoUpdater(cfg=cfg, modelHarness=modelHarness)

    raise NotImplementedError(
        f"Unknown update_mode: {cfg.continual_learning.update_mode}"
    )
