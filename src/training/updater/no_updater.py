from __future__ import annotations


from training.updater.base import BaseUpdater

import torch


class NoUpdater(BaseUpdater):
    """No-op updater that skips training updates.

    Used when continuous learning updates should be disabled.
    """

    def fwd_bwd(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        hist_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> float:
        """Skip forward/backward pass, return sentinel value."""
        return -1.0
