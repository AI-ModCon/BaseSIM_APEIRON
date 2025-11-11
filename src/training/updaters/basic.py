"""
Docstring for src.training.updaters.basic

This module implements a baseline update step method for neural network training
"""

import torch

from src.config.configuration import Config


def step_method_baseline(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
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
    outputs = model(in_t)
    loss = criterion(outputs, targets_t)
    loss.backward()
    optimizer.step()

    return loss.item()
