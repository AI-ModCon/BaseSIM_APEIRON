from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.training.updater.base import BaseUpdater
from apeiron.logger import get_logger


class RetrainUpdater(BaseUpdater):
    """Full retrain-from-scratch updater.

    Unlike the incremental CL updaters, this one discards the current weights on
    a drift event and re-learns the model from a random initialization. It is the
    escalation path for severe drift (``LearningRegime.RETRAIN``), where anchoring
    to the old parameters is no longer worthwhile.

    On ``cl_preprocessing()`` (called once, before the outer training loop) it:

    1. Re-initializes every module's parameters via ``reset_parameters()``, so
       training starts from scratch rather than from the drifted weights.
    2. Clears the optimizer state (momentum / Adam moment buffers), which would
       otherwise carry stale statistics from the pre-drift model.

    The per-step forward/backward is inherited from :class:`BaseUpdater`. Historic
    replay mixing is forced on so the rebuild spans the full distribution
    (historical base regimes + the current drifted stream) rather than
    specializing to the latest drift alone. When the harness supplies no
    historical dataloaders, it degrades to a from-scratch fit on the current
    stream only.

    The number of optimization steps is governed by ``train.max_iter`` like the
    other modes -- set it high enough that a from-scratch fit can converge.
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Initialize the retrain updater.

        Args:
            cfg: Configuration object.
            modelHarness: Model harness containing the model to rebuild.
            optimizer: The trainer's optimizer, whose state is cleared on
                retrain. If ``None``, only the model weights are reset.
        """
        super().__init__(cfg=cfg, modelHarness=modelHarness)
        self.optimizer = optimizer
        # A from-scratch rebuild should see the full distribution, not just the
        # drifted stream, so historic replay is always mixed in when available.
        self.mix_historic_data = True

    @torch.no_grad()
    def cl_preprocessing(self) -> None:
        """Reset model weights and optimizer state for a from-scratch retrain."""
        logger = get_logger(__name__)

        reset_count = self._reinitialize_weights()
        logger.info(
            f"\tRetrain: re-initialized {reset_count} module(s) from scratch",
            level=1,
        )

        if self.optimizer is not None:
            self.optimizer.state.clear()
            logger.info("\tRetrain: cleared optimizer state", level=1)

    @torch.no_grad()
    def _reinitialize_weights(self) -> int:
        """Re-initialize every submodule that exposes ``reset_parameters()``.

        Returns:
            The number of modules that were reset.
        """
        reset_count = 0

        def _reset(module: torch.nn.Module) -> None:
            nonlocal reset_count
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_count += 1

        self.model.apply(_reset)
        return reset_count
