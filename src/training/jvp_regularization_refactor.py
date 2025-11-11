# -----------------------------------------
# Algorithm: JVP Regularization
# This function implements a continual learning method based on Jacobian-vector product regularization.
# It aims to minimize forgetting by penalizing changes in the model's output on the memory buffer


import torch

from torch.func import grad, jvp
from collections import OrderedDict
from typing import Mapping


class FunctionalAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # Initialize moment estimates
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grad_dict):
        self.t += 1
        lr = self.lr
        b1, b2 = self.betas
        eps = self.eps

        new_params = OrderedDict()
        for k, w in params.items():
            if k not in grad_dict:
                new_params[k] = w.clone()
                continue

            g = grad_dict[k]

            # Update moments
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)

            # Bias correction
            m_hat = self.m[k] / (1 - b1**self.t)
            v_hat = self.v[k] / (1 - b2**self.t)

            # Adam update
            new_params[k] = w - lr * m_hat / (torch.sqrt(v_hat) + eps)

        return new_params


def return_Hamiltonian(model, params: Mapping[str, torch.Tensor], data, cfg):
    (x, y, exp_x, exp_y, deltax, criterion) = data

    for p in params.values():
        if not p.requires_grad:
            p.requires_grad_(True)

    # Helper functions
    # Functional + batched forward
    def single_forward(p, xx):
        return torch.func.functional_call(model, p, (xx,))

    def model_batched(p, xx):
        return single_forward(p, xx)

    # loss function
    def V_star(p, xx, yy):
        preds = model_batched(p, xx)
        return criterion(preds, yy)

    # Useful helper
    def map_dict(d, fn):
        return {k: fn(v) for k, v in d.items()}

    # The gradient function
    grad_wrt_params = grad(V_star, argnums=0)

    def f(p, xx):
        return V_star(p, xx, exp_y)

    def tangents_from_params(params, tangent_seq):
        """Map tangent tensors to param OrderedDict structure."""
        return OrderedDict({k: t for (k, _), t in zip(params.items(), tangent_seq)})

    # def zero_like_params(params):
    #     return OrderedDict({k: torch.zeros_like(v) for k, v in params.items()})
    def jvp_func(p, tangents):
        return jvp(f, (p, exp_x), tangents)[1]

    # ------------------------------------------------
    # Core compute part
    # ------------------------------------------------
    # grad of the current task
    delta_theta = grad_wrt_params(params, x, y)
    # grad of the past task
    grad_V = grad_wrt_params(params, exp_x, exp_y)
    # JVP part
    wdot = map_dict(delta_theta, lambda v: v)
    wdot = tangents_from_params(params, wdot.values())
    grad_dV = grad(jvp_func)(params, (wdot, deltax))

    # Additional debug prints --- IGNORE ---
    # V = V_star(params, exp_x, exp_y)
    # _, fwd1 = jvp(f, (params, exp_x), (wdot, torch.zeros_like(deltax)))
    # _, fwd2 = jvp(f, (params, exp_x), (zero_dtheta, deltax))
    # print((V+dV).item(), V.item(), dV.item(), fwd1, fwd2)
    # _, dV = jvp(f, (params, exp_x), (wdot, deltax))

    # The final gradient calculation
    combined = {
        k: (delta_theta[k] + grad_V[k] + cfg.continuous_learning.jvp_reg * grad_dV[k])
        for k in params
    }
    return (combined, V_star(params, x, y), V_star(params, exp_x, exp_y))
