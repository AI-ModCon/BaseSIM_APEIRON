# -----------------------------------------
# Algorithm: JVP Regularization
# This function implements a continual learning method based on Jacobian-vector product regularization.
# It aims to minimize forgetting by penalizing changes in the model's output on the memory buffer

import torch


from src.config.configuration import Config
from src.training.profilers import FLOPSProfiler
from src.model.jvp_continual_learning import JVPRegularizedLoss, JVPAdam


def step_method_jvp_reg(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
    hist_batch: tuple,
    profiler: FLOPSProfiler,
    jvp_loss: JVPRegularizedLoss,
    jvp_adam: JVPAdam,
):
    if profiler and iter > profiler.warmup_iters:
        # Compute gradients
        with profiler.measure_flops(tag="hamiltonian"):
            grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)

        # Detach and assign gradients (outside profiling)
        # Memory operations add some latency.
        # Doing this externally to compare runtime w/ original implementation
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()

        # Optimizer step
        with profiler.measure_flops(tag="optim"):
            jvp_adam.step()

    else:
        grads_dict, J_P, J_M = jvp_loss(train_batch, hist_batch)
        for name, param in model.named_parameters():
            param.grad = grads_dict[name].detach()
        jvp_adam.step()

    return J_P.item(), J_M.item(), (J_P + J_M).item()
