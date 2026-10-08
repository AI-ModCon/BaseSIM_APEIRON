import torch

from apeiron.training.updater.base import BaseUpdater
from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness


class OnlineEWCUpdater(BaseUpdater):
    """
    Online Elastic Weight Consolidation (EWC) updater. See work by
    Title: Overcoming catastrophic forgetting in neural networks
    Authors: James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwińska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell
    DOI: https://doi.org/10.1073/pnas.1611835114

    - Anchor (theta_star) and Fisher are held fixed during a CL loop.
    - Fisher is accumulated from gradients during CL.
    - Anchor and Fisher are updated exactly once in cl_postprocessing().

    All updater tensors are explicitly created on cfg.device.
    """

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        super().__init__(cfg, modelHarness)

        self.device = cfg.device
        self.lambda_ewc = float(cfg.continual_learning.ewc_lambda)
        self.fisher_decay = float(cfg.continual_learning.ewc_ema_decay)

        # Running anchor θ* (prior mean)
        self.theta_star: dict[str, torch.Tensor] = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Running diagonal Fisher F* (prior precision)
        self.fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # CL-event accumulators (allocated per CL loop)
        self._batch_fisher: dict[str, torch.Tensor] | None = None
        self._cl_fisher_accum: dict[str, torch.Tensor] | None = None
        self._cl_steps = 0

    @torch.no_grad()
    def cl_preprocessing(self) -> None:
        """Called once before the CL loop starts.

        Refreshes the EWC anchor θ* to the current model weights so that
        the EWC quadratic penalty during this round pulls toward the
        previous-task converged weights. Runs AFTER compute_sample_priorities
        in the outer CL loop, so importance sampling sees the one-round
        lag it needs.
        """
        # Allocate CL accumulators directly on correct device
        self._cl_fisher_accum = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._cl_steps = 0

        # Refresh anchor θ* = current model weights (previous-task end)
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.theta_star:
                self.theta_star[name].copy_(p.detach())

    @torch.no_grad()
    def cl_postprocessing(self) -> None:
        """
        Called once after the CL loop finishes.

        Commits the new Fisher prior:
            F* <- fisher_decay * F* + F_cl_avg

        Anchor θ* is NOT refreshed here — it is refreshed at the next
        cl_preprocessing, after compute_sample_priorities has had a chance
        to read the stale value.
        """
        if self._cl_steps == 0:
            return
        if self._cl_fisher_accum is None:
            return

        # Update Fisher
        for name in self.fisher:
            self.fisher[name].mul_(self.fisher_decay)
            self.fisher[name].add_(self._cl_fisher_accum[name] / float(self._cl_steps))

        # cleanup
        self._cl_fisher_accum = None
        self._cl_steps = 0

    @torch.no_grad()
    def update_pre_fwd_bwd(self) -> None:
        """Prepare per-step Fisher accumulator."""
        self._batch_fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update_post_fwd_bwd(self) -> float:
        """
        Called after backward(), before optimizer.step().

        - Adds EWC gradient using fixed anchor and Fisher.
        - Accumulates per-step Fisher estimate from gradients.
        """
        ewc_loss = 0.0

        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            diff = p - self.theta_star[name]

            # penalty scalar (for logging)
            ewc_loss += (self.fisher[name] * diff.pow(2)).sum().item()

            # Fisher estimate: grad^2
            if self._batch_fisher is not None:
                self._batch_fisher[name].add_(p.grad.detach().pow(2))

            # Add EWC gradient explicitly
            p.grad.add_(self.lambda_ewc * self.fisher[name] * diff)

        return float(0.5 * self.lambda_ewc * ewc_loss)

    @torch.no_grad()
    def update_post_optimizer_call(self) -> None:
        """
        After optimizer.step():
        - accumulate Fisher for this CL event
        - do NOT update anchor or running Fisher yet
        """
        if self._cl_fisher_accum is not None and self._batch_fisher is not None:
            for name in self._cl_fisher_accum:
                self._cl_fisher_accum[name].add_(self._batch_fisher[name])

        self._cl_steps += 1
        self._batch_fisher = None
