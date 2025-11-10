# -----------------------------------------
# Algorithm: JVP Regularization
# This function implements a continual learning method based on Jacobian-vector product regularization.
# It aims to minimize forgetting by penalizing changes in the model's output on the memory buffer


import copy
import torch

from torch.func import vmap, grad, jvp
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
        return vmap(lambda b: single_forward(p, b))(xx)

    # loss function
    def V_star(p, xx, yy):
        preds = model_batched(p, xx).squeeze(dim=1)
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


def update_CL_jvp_reg(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    mem_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    task: int,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> tuple[torch.nn.Module, float, float, float]:

    device = cfg.device
    # We set up the iterators for the memory loader and the train loader
    mem_iter = iter(mem_loader)
    # print("the length of the train loader is", len(mem_train_loader  ))
    task_iter = iter(train_loader)
    # The main loop over all the batch
    epoch_loss = 10000.0
    epoch_for_loss = 10000.0
    epoch_gen_loss = 10000.0
    num = -1
    params = OrderedDict(model.named_parameters())
    adam = FunctionalAdam(params, lr=1e-3)
    while (epoch_loss > 1e-05) or num < cfg.continuous_learning.max_iter:
        epoch_loss = 0.0
        num += 1
        for pp, data_m in enumerate(mem_iter):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)

            if task > 0:
                # ------------------------------------------------
                try:
                    data_t = next(task_iter)
                    (_, y) = data_t
                    if y.shape[0] < cfg.train.batch_size:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)
                optimizer.zero_grad()
                in_t, targets_t = data_t
                in_m, targets_m = data_m
                in_t = in_t.unsqueeze(dim=1).float().to(device)
                in_m = in_m.unsqueeze(dim=1).float().to(device)
                targets_t = targets_t.to(device)
                targets_m = targets_m.to(device)

                # ----------------------------------------
                # deltax direction calculation
                deltax = (
                    cfg.continuous_learning.deltax_norm
                    * (in_m - in_t)
                    / (torch.linalg.norm(in_m) + torch.linalg.norm(in_t))
                ).to(device)

                # ------------------------------------------------
                # Build data tuple for the actual gradient calculation
                data = (in_t, targets_t, in_m, targets_m, deltax, criterion)
                with torch.enable_grad():
                    grads_dict, J_P, J_M = return_Hamiltonian(model, params, data, cfg)

                # ------------------------------------------------
                # detach grads
                for k in grads_dict:
                    grads_dict[k] = grads_dict[k].detach()

                with torch.no_grad():
                    params = adam.step(params, grads_dict)
                    for k in params:
                        params[k] = params[k].detach()
                    model.load_state_dict(params, strict=False)

                # ------------------------------------------------
                torch.cuda.empty_cache()
                # send it to the model team's model class
                # J_P =criterion(model(in_t), targets_t.to(device))
                # J_M = criterion( model(in_m), targets_m.to(device))
                # # Experience replay loss calculation
                # Total_loss.backward()  # Derive gradients.
                # optimizer.step()  # Update parameters based on gradients.
                epoch_loss += (J_P + J_M).item()
                epoch_for_loss = J_P.item()
                epoch_gen_loss = J_M.item()
            else:
                try:
                    data_t = next(task_iter)
                    (_, y) = data_t
                    if y.shape[0] < cfg["batchsize"]:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)

                in_t, targets_t = data_t
                in_t = in_t.unsqueeze(dim=1).float()
                Total_loss = criterion(model(in_t.to(device)), targets_t.to(device))
                optimizer.zero_grad()
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.

    # return stuff
    if task > 0:
        # test_acc, loss_cal = test(model, mem_loader, criterion, device=device)
        epoch_loss = epoch_loss / len(mem_loader.dataset)
        epoch_for_loss = epoch_for_loss / len(mem_loader.dataset)
        epoch_gen_loss = epoch_gen_loss / len(mem_loader.dataset)
        # print("the loss at:", num, task, total, epoch_loss, loss_cal, test_acc)
        return (model, epoch_loss, epoch_for_loss, epoch_gen_loss)

    else:
        # test_acc, loss_cal = test(model, train_loader, criterion, device=device)
        epoch_loss = epoch_loss / len(train_loader.dataset)
        # print("the loss at:", num, task, total, epoch_loss, loss_cal, test_acc)
        return (
            model,
            epoch_loss,
            epoch_loss,
            epoch_loss,
        )


# ----------------------------------------------------------------------------------
