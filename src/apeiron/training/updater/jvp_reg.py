"""First-order SAM/Bertsimas robust updater for continual learning.

Despite the historical ``jvp_reg`` name (and ``update_mode = "jvp_reg"``), this
updater implements a first-order Sharpness-Aware / Bertsimas robust step rather
than the original Jacobian-Vector-Product regularizer.

The cost combines the current and historical batches into ONE loss (half/half,
like base replay), and the robustness is the gradient of that combined loss at a
parameter-perturbed point -- first-order only (no Hessian, no third order):

    grad   = grad_theta  L_mix( theta + rho_theta * u_new ;  X_mix )
    X_mix  = concat( X_cur[:n_cur],  X_hist[:n_hist] + rho_x * dx_hat )   (n_hist = n//2)
    u_new  = -grad L_cur / ||grad L_cur||          (param direction -> new task)
    dx_hat =  unit( mean(X_cur) - mean(X_hist) )   (batch-mean drift direction)

Keeping the current batch inside the loss every step anchors the online accuracy
so it cannot degrade. ``rho_x = 0`` reduces this to param-SAM only. Radii come
from ``[continual_learning]``: ``jvp_rho_theta``, ``jvp_rho_x``, ``jvp_data_sign``.

Like the previous JVP updater, this manages the historical batch itself, so it is
independent of the ``mix_historic_data`` replay flag.
"""

from __future__ import annotations

import torch

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.training.updater.base import BaseUpdater

_EPS = 1e-12


class JVPRegUpdater(BaseUpdater):
    """SAM-based robust updater on a combined current+historical batch."""

    # fwd_bwd reads hist_batch (memory buffer) directly — JVP term and
    # the explicit historical-replay backward both depend on it. Signals
    # the trainer that prioritizing hist_train_loader will actually move
    # the gradient for this updater.
    uses_hist_batch: bool = True

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        """Initialize the SAM-based updater with config and model harness."""
        super().__init__(cfg, modelHarness)
        self.rho_theta: float = cfg.continual_learning.jvp_rho_theta
        self.rho_x: float = cfg.continual_learning.jvp_rho_x
        # +1: perturb old inputs toward current dist (drift dir); -1: toward old
        self.data_sign: float = cfg.continual_learning.jvp_data_sign
        self.loss_mem: float = 0.0

    def _data_dir(self, x_cur: torch.Tensor, x_old: torch.Tensor) -> torch.Tensor:
        """Unit batch-mean drift direction, broadcastable over the old batch."""
        mc = x_cur.reshape(x_cur.shape[0], -1).mean(0)
        mo = x_old.reshape(x_old.shape[0], -1).mean(0)
        # +1 toward current (drift dir), -1 toward old
        d = self.data_sign * (mc - mo)
        d = d / (d.norm() + _EPS)
        return d.reshape(x_old.shape[1:])  # e.g. [H, W], broadcasts over batch

    def fwd_bwd(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        hist_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> float:
        """Forward/backward for a single first-order SAM robust step.

        With no historical batch this is a plain current-batch step. Otherwise it
        combines both batches into one loss and back-propagates that loss at a
        parameter-perturbed point (SAM). Parameters are always restored.
        """
        x, y = batch
        A = self.cfg.train.grad_accumulation_steps
        params = [p for p in self.model.parameters() if p.requires_grad]

        if hist_batch is None:
            # no history yet: plain current-batch step
            loss = self.criterion(self.model(x), y) / A
            loss.backward()
            return float(loss.item())

        x_hist, y_hist = hist_batch

        # u_new: current-task descent direction (SAM param-perturbation direction)
        g_cur = torch.autograd.grad(self.criterion(self.model(x), y) / A, params)
        sq = torch.stack([(g.detach() ** 2).sum() for g in g_cur]).sum()
        gnorm = torch.sqrt(sq) + _EPS
        u = [(-g.detach() / gnorm) for g in g_cur]

        # --- COMBINE current + historical into ONE cost (half/half, like base replay) ---
        # Keeps the current task inside the loss every step, so the online accuracy
        # is anchored and cannot drift away.
        n_total = x.shape[0]
        n_hist = min(n_total // 2, x_hist.shape[0])
        n_cur = n_total - n_hist
        x_h = x_hist[:n_hist]
        if self.rho_x != 0.0 and n_hist > 0:  # data shift on the historical portion
            x_h = x_h + self.rho_x * self._data_dir(x, x_hist)
        x_mix = torch.cat([x[:n_cur], x_h], dim=0)
        y_mix = torch.cat([y[:n_cur], y_hist[:n_hist]], dim=0)

        # SAM: gradient of the COMBINED loss at the param-shifted point (first-order)
        try:
            with torch.no_grad():
                for p, ui in zip(params, u):
                    p.add_(self.rho_theta * ui)
            loss_mix = self.criterion(self.model(x_mix), y_mix) / A
            loss_mix.backward()
        finally:
            with torch.no_grad():  # always restore params
                for p, ui in zip(params, u):
                    p.sub_(self.rho_theta * ui)

        self.loss_mem += float(loss_mix.item())
        return float(loss_mix.item())

    @torch.no_grad()
    def update_post_fwd_bwd(self) -> float:
        """Return the accumulated combined-loss value and reset it."""
        lm = self.loss_mem
        self.loss_mem = 0.0
        return lm
