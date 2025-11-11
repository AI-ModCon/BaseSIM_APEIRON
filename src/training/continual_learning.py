import torch

from typing import Dict

from torch.func import vmap, grad, jvp
from collections import OrderedDict

from torch.utils.data import DataLoader

from src.evaluation.evaluation import test
from src.config.configuration import Config
from examples.MNIST.data_utils import MyDataset


from src.model.torch_model_harness import BaseModelHarness
from src.training.jvp_regularization_refactor import return_Hamiltonian, FunctionalAdam


def continual_learning_loop(cfg: Config, modelHarness: BaseModelHarness):

    # 1) select the right cl update method #TODO

    # 2) Get loaders
    #  cur data loadershas to be called before hist data loaders
    train_loader = modelHarness.get_cur_data_loaders()[0]
    hist_loader = modelHarness.get_hist_data_loaders()[0]

    train_iter = iter(train_loader)
    if hist_loader is not None:
        hist_train_iter = iter(hist_loader)
    else:
        hist_train_iter = None

    criterion = modelHarness.get_criterion()
    model = modelHarness.model
    optimizer = modelHarness.get_optmizer()
    batch_size = cfg.train.batch_size

    # TODO: replace this - should be directly done with optimizer
    params = OrderedDict(model.named_parameters())
    adam = FunctionalAdam(params, lr=1e-3)

    # Generic "safe next" for any iterator/loader pair
    def _safe_next(current_iter, loader, min_batch=None):
        """
        Returns (possibly-updated-iter, batch) guaranteeing:
          - iterator restarts on StopIteration
          - optional min batch-size requirement (on y) if provided
        """
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, [b.to(cfg.device) for b in batch]

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, [b.to(cfg.device) for b in batch]
                # else: too small → loop to fetch a new batch/iterator
            except Exception:
                # If we cannot inspect batch size, just accept the batch
                return current_iter, [b.to(cfg.device) for b in batch]

    # 2) run the outer loop
    for iter_count in range(cfg.continuous_learning.max_iter):
        # Fetch valid batches from both streams
        train_iter, train_batch = _safe_next(
            train_iter, train_loader, min_batch=batch_size
        )

        if hist_train_iter is None:
            # Fall back to basic training if no historical data is available

            total_loss = step_method_baseline(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                cfg=cfg,
                iter=iter_count,
                train_batch=train_batch,
            )
        else:
            hist_train_iter, hist_batch = _safe_next(
                hist_train_iter,
                hist_loader,
                min_batch=batch_size,
            )

            forgetting_loss, generation_loss, total_loss = step_method_jvp_reg(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                cfg=cfg,
                iter=iter_count,
                train_batch=train_batch,
                hist_batch=hist_batch,
                adam=adam,  # TODO remove this.
                params=params,
            )

    return 0


def step_method_jvp_reg(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    iter: int,
    train_batch: tuple,
    hist_batch: tuple,
    adam: FunctionalAdam,
    params: OrderedDict,
):
    optimizer.zero_grad()
    in_t, targets_t = train_batch
    in_m, targets_m = hist_batch

    # ----------------------------------------
    # deltax direction calculation
    deltax = (
        cfg.continuous_learning.deltax_norm
        * (in_m - in_t)
        / (torch.linalg.norm(in_m) + torch.linalg.norm(in_t))
    )

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

    return (
        J_P.item(),
        J_M.item(),
        (J_P + J_M).item(),
    )  # forgetting loss, generation loss, total loss


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
