import torch
import torch.nn as nn

from apeiron.training.updater.base import BaseUpdater
from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness
import warnings

warnings.filterwarnings("ignore", "Full backward hook is firing")


class OnlineKFACUpdater(BaseUpdater):
    """
    Online K-FAC EWC-style regularizer. See work by
    Title: Optimizing neural networks with Kronecker-factored approximate curvature
    Authors: James Martens, Roger Grosse
    Link: http://proceedings.mlr.press/v37/martens15.html

    - Accumulates Kronecker factors during a CL loop
    - Commits them once at cl_postprocessing()
    - Applies KFAC-structured EWC gradient penalty
    """

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        super().__init__(cfg, modelHarness)

        self.device = cfg.device
        self.lambda_kfac = float(cfg.continual_learning.kfac_lambda)
        self.ema_decay = float(cfg.continual_learning.kfac_ema_decay)

        self.model = modelHarness.model

        # Per-layer anchor θ*
        self.theta_star: dict[str, torch.Tensor] = {}

        # Running Kronecker factors (the prior)
        self.A: dict[str, torch.Tensor] = {}
        self.G: dict[str, torch.Tensor] = {}

        # CL accumulators
        self._A_accum: dict[str, torch.Tensor] | None = None
        self._G_accum: dict[str, torch.Tensor] | None = None
        self._cl_steps = 0

        # Activation / gradient caches
        self._activations: dict[str, torch.Tensor] = {}
        self._grad_outputs: dict[str, torch.Tensor] = {}

        self._register_hooks()
        self._init_prior()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_prior(self):
        for name, module in self.model.named_modules():
            if self._supported(module):
                weight = module.weight
                self.theta_star[name] = weight.detach().clone().to(self.device)
                self.A[name] = torch.zeros(
                    self._a_dim(module),
                    self._a_dim(module),
                    device=self.device,
                )
                self.G[name] = torch.zeros(
                    self._g_dim(module),
                    self._g_dim(module),
                    device=self.device,
                )

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self._supported(module):
                module.register_forward_hook(self._save_activation(name))
                module.register_full_backward_hook(self._save_grad_output(name))

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activation(self, name):
        def hook(module, inp, out):
            self._activations[name] = inp[0].detach()

        return hook

    def _save_grad_output(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._grad_outputs[name] = grad_output[0].detach()

        return hook

    # ------------------------------------------------------------------
    # CL lifecycle
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cl_preprocessing(self):
        self._A_accum = {
            k: torch.zeros_like(v, device=self.device) for k, v in self.A.items()
        }
        self._G_accum = {
            k: torch.zeros_like(v, device=self.device) for k, v in self.G.items()
        }
        self._cl_steps = 0

    @torch.no_grad()
    def cl_postprocessing(self):
        if self._cl_steps == 0:
            return
        if self._A_accum is None or self._G_accum is None:
            return

        for name in self.A:
            self.A[name].mul_(self.ema_decay)
            self.G[name].mul_(self.ema_decay)

            self.A[name].add_(self._A_accum[name] / self._cl_steps)
            self.G[name].add_(self._G_accum[name] / self._cl_steps)

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if self._supported(module):
                    self.theta_star[name].copy_(module.weight.detach())

        self._A_accum = None
        self._G_accum = None
        self._cl_steps = 0

    # ------------------------------------------------------------------
    # Per-step updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_post_fwd_bwd(self) -> float:
        kfac_loss = 0.0

        for name, module in self.model.named_modules():
            if not self._supported(module):
                continue

            if name not in self._activations or name not in self._grad_outputs:
                continue

            a = self._extract_a(module, self._activations[name])
            g = self._extract_g(module, self._grad_outputs[name])

            # Accumulate KFAC statistics
            if self._A_accum is not None and self._G_accum is not None:
                self._A_accum[name].add_(a.T @ a / a.shape[0])
                self._G_accum[name].add_(g.T @ g / g.shape[0])

            # EWC-style gradient penalty
            W = module.weight
            W_star = self.theta_star[name]

            if isinstance(module, nn.Conv2d):
                # Reshape to 2D: (out_channels, in_channels*kH*kW)
                diff = W.view(W.shape[0], -1) - W_star.view(W_star.shape[0], -1)
            else:
                diff = W - W_star

            grad_penalty = self.G[name] @ diff @ self.A[name]
            kfac_loss += (diff * grad_penalty).sum().item()

            if isinstance(module, nn.Conv2d):
                grad_penalty = grad_penalty.view_as(W)

            W.grad.add_(self.lambda_kfac * grad_penalty)

        return 0.5 * self.lambda_kfac * kfac_loss

    @torch.no_grad()
    def update_post_optimizer_call(self):
        self._cl_steps += 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _supported(self, module):
        return isinstance(module, (nn.Linear, nn.Conv2d))

    @torch.no_grad()
    def _a_dim(self, module):
        if isinstance(module, nn.Linear):
            return module.in_features
        if isinstance(module, nn.Conv2d):
            return module.in_channels * module.kernel_size[0] * module.kernel_size[1]

    @torch.no_grad()
    def _g_dim(self, module):
        if isinstance(module, nn.Linear):
            return module.out_features
        if isinstance(module, nn.Conv2d):
            return module.out_channels

    @torch.no_grad()
    def _extract_a(self, module, a):
        if isinstance(module, nn.Linear):
            return a.view(a.shape[0], -1)
        if isinstance(module, nn.Conv2d):
            return (
                torch.nn.functional.unfold(
                    a,
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                )
                .transpose(1, 2)
                .reshape(-1, self._a_dim(module))
            )

    @torch.no_grad()
    def _extract_g(self, module, g):
        if isinstance(module, nn.Linear):
            return g.view(g.shape[0], -1)
        if isinstance(module, nn.Conv2d):
            return g.permute(0, 2, 3, 1).reshape(-1, self._g_dim(module))
