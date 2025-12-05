"""
Docstring for training.updaters.basic

This module implements a baseline update step method for neural network training
"""

import torch

from config.configuration import Config
from training.profilers import FLOPSProfiler


def step_method_baseline(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
    profiler: FLOPSProfiler,
):
    """
    This function implements a baseline step method for continual learning.

    It takes in the model, criterion, optimizer, configuration object, iteration number, and the train batch.
    It then performs a single step of gradient descent on the current task.
    Returns the loss after the step.

    Args:
        model (torch.nn.Module): The model being used.
        criterion (torch.nn.Module): The loss function being used.
        optimizer (torch.optim.Optimizer): The optimizer being used.
        cfg (Config): The configuration object.
        iter (int): The iteration number.
        train_batch (tuple): The train batch.

    Returns:
        float: The loss after the step.
    """
    in_t, targets_t = train_batch
    optimizer.zero_grad()

    if (
        profiler and iter > profiler.warmup_iters
    ):  # Give warmup iterations, for accuracy.
        with profiler.measure_flops(tag="fwd"):
            outputs = model(in_t)
            loss = criterion(outputs, targets_t)

        with profiler.measure_flops(tag="bwd"):
            loss.backward()

        with profiler.measure_flops_optimizer(
            tag="optim", model=model, device=cfg.device
        ):
            # - Try profiling optimizer step agnostically.
            # profiler.count_optimizer_step(optimizer, model, cfg.device)
            optimizer.step()

    else:
        outputs = model(in_t)
        loss = criterion(outputs, targets_t)
        loss.backward()
        optimizer.step()

    return loss.item()
