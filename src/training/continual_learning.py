import torch

from typing import Dict


from torch.utils.data import DataLoader

from src.evaluation.evaluation import test
from src.config.configuration import Config
from examples.MNIST.data_utils import MyDataset


from src.model.torch_model_harness import BaseModelHarness


def continual_learning_loop(cfg: Config, modelHarness: BaseModelHarness):

    # 1) select the right cl update method #TODO

    # 2) Get loaders
    hist_train_iter = iter(modelHarness.get_hist_train_loader())
    train_iter = iter(modelHarness.get_train_loader())
    criterion = modelHarness.get_criterion()
    model = modelHarness.model
    optimizer = modelHarness.get_optmizer()
    batch_size = cfg.train.batch_size

    # Generic "safe next" for any iterator/loader pair
    def _safe_next(current_iter, make_loader, min_batch=None):
        """
        Returns (possibly-updated-iter, batch) guaranteeing:
          - iterator restarts on StopIteration
          - optional min batch-size requirement (on y) if provided
        """
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(make_loader())
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, batch

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, batch
                # else: too small → loop to fetch a new batch/iterator
            except Exception:
                # If we cannot inspect batch size, just accept the batch
                return current_iter, batch

    # 2) run the outer loop
    for iter_count in range(cfg.continuous_learning.total_updates):
        # Fetch valid batches from both streams
        train_iter, train_batch = _safe_next(
            train_iter, modelHarness.get_train_loader, min_batch=batch_size
        )
        hist_train_iter, hist_batch = _safe_next(
            hist_train_iter, modelHarness.get_hist_train_loader, min_batch=batch_size
        )

        step_method_bcl(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            iter=iter_count,
            train_batch=train_batch,
            hist_batch=hist_batch,
        )

    pass


def step_method_bcl(model, criterion, optimizer, cfg, iter, train_batch, hist_batch):
    """Implements one step of BCL given ready-to-use batches.

    @article{raghavan5046289modelling,
      title={Modelling the Dynamics of Learning Continually Withgraph Neural Networks},
      author={Raghavan, Krishnan and Balaprakash, Prasanna},
      journal={Available at SSRN 5046289}
    }

    """
    # Example unpack (adjust as needed)
    (x_t, y_t) = train_batch.to(cfg.device)
    (x_hist, y_hist) = hist_batch.to(cfg.device)

    out = model(x_t)
    out_hist = model(x_hist)

    ############## The task cost and the memory cost
    #########################################################################################
    J_P = criterion(out, y_t)
    J_M = criterion(out_hist, y_hist)

    ############## This is the J_x loss
    #########################################################################################
    J_PN_x = criterion(model(x_hist), y_hist)
    x_PN = copy.copy(in_m).to(device)
    x_PN.requires_grad = True
    adv_grad = 0
    epsilon = params["x_lr"]

    for epoch in range(params["x_updates"]):
        x_PN = x_PN + epsilon * adv_grad
        crit = criterion(model(x_PN.float()), targets_m)
        loss = torch.mean(crit) + torch.var(crit)  # + skew(crit) + kurtosis(crit)
        adv_grad = torch.autograd.grad(loss, x_PN)[0]
        # Normalize the gradient values.
        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
    J_x_crit = criterion(model(x_PN.float()), targets_m) - J_PN_x
    # print(adv_grad.shape)
    # print("norm", torch.norm(adv_grad))
    ############### This is the loss J_th
    #########################################################################################
    cop = copy.deepcopy(model).to(device)
    opt_buffer = torch.optim.Adam(cop.parameters(), lr=params["th_lr"])
    J_PN_theta = criterion(model(in_m.float()), targets_m)
    for i in range(params["theta_updates"]):
        opt_buffer.zero_grad()
        loss_crit = criterion(cop(in_t.float()), targets_t)
        loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
        # + torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
        loss_m.backward(retain_graph=True)
        opt_buffer.step()
    J_th_crit = criterion(cop(in_m.float()), targets_m) - J_PN_theta

    # Now, put together  the loss fully
    Total_loss = torch.mean(
        J_M + J_P + params["factor"] * J_x_crit + params["factor"] * J_th_crit
    )
    # +torch.var(J_P+J_M+J_x_crit+params['factor']*J_th_crit)
    # adjoint_scores =
    optimizer.zero_grad()
    Total_loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.


# This function is my actual CL update. We can actually modify this function
# for both efficiency and effectiveness, however, I think this is a good starting point.
def update_CL_(
    model,
    criterion,
    mem_loader,
    train_loader,
    task,
    optimizer,
    cfg,
    Graph=1,
    params={
        "x_updates": 1,
        "theta_updates": 1,
        "factor": 0.00001,
        "x_lr": 0.00001,
        "th_lr": 0.00001,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batchsize": 64,
        "total_updates": 10000,
    },
):
    device = params["device"]

    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

    import copy

    # We set up the iterators for the memory loader and the train loader
    mem_iter = iter(mem_loader)
    task_iter = iter(train_loader)

    # The main loop over all the batch
    for i in range(params["total_updates"]):
        if task > 0:
            ###########################################
            try:
                data_t = next(task_iter)
                (_, y) = data_t
                if y.shape[0] < params["batchsize"]:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)
            except StopIteration:
                task_iter = iter(train_loader)
                data_t = next(task_iter)

            # Extract a batch from the memory
            try:
                data_m = next(mem_iter)
                (_, y) = data_m
                if y.shape[0] < params["batchsize"]:
                    mem_iter = iter(mem_loader)
                    data_m = next(mem_iter)
            except StopIteration:
                mem_iter = iter(mem_loader)
                data_m = next(mem_iter)

            in_t, targets_t = data_t
            in_m, targets_m = data_m

            in_t = in_t.unsqueeze(dim=1).float().to(device)
            in_m = in_m.unsqueeze(dim=1).float().to(device)
            targets_t = targets_t.to(device)
            targets_m = targets_m.to(device)

            out = model(in_t)
            out_m = model(in_m)

            ############## The task cost and the memory cost
            #########################################################################################
            J_P = criterion(out, targets_t.to(device))
            J_M = criterion(out_m, targets_m.to(device))

            ############## This is the J_x loss
            #########################################################################################
            J_PN_x = criterion(model(in_m), targets_m)
            x_PN = copy.copy(in_m).to(device)
            x_PN.requires_grad = True
            adv_grad = 0
            epsilon = params["x_lr"]

            for epoch in range(params["x_updates"]):
                x_PN = x_PN + epsilon * adv_grad
                crit = criterion(model(x_PN.float()), targets_m)
                loss = torch.mean(crit) + torch.var(
                    crit
                )  # + skew(crit) + kurtosis(crit)
                adv_grad = torch.autograd.grad(loss, x_PN)[0]
                # Normalize the gradient values.
                adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
            J_x_crit = criterion(model(x_PN.float()), targets_m) - J_PN_x
            # print(adv_grad.shape)
            # print("norm", torch.norm(adv_grad))
            ############### This is the loss J_th
            #########################################################################################
            cop = copy.deepcopy(model).to(device)
            opt_buffer = torch.optim.Adam(cop.parameters(), lr=params["th_lr"])
            J_PN_theta = criterion(model(in_m.float()), targets_m)
            for i in range(params["theta_updates"]):
                opt_buffer.zero_grad()
                loss_crit = criterion(cop(in_t.float()), targets_t)
                loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                # + torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                loss_m.backward(retain_graph=True)
                opt_buffer.step()
            J_th_crit = criterion(cop(in_m.float()), targets_m) - J_PN_theta

            # Now, put together  the loss fully
            Total_loss = torch.mean(
                J_M + J_P + params["factor"] * J_x_crit + params["factor"] * J_th_crit
            )
            # +torch.var(J_P+J_M+J_x_crit+params['factor']*J_th_crit)
            # adjoint_scores =
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        else:
            try:
                data_t = next(task_iter)
                (_, y) = data_t
                if y.shape[0] < params["batchsize"]:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)
            except StopIteration:
                task_iter = iter(train_loader)
                data_t = next(task_iter)

            in_t, targets_t = data_t
            in_t = in_t.unsqueeze(dim=1).float()

            critti = criterion(model(in_t.to(device)), targets_t.to(device))
            Total_loss = torch.mean(critti) + torch.var(critti)
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

    if task > 0:
        return (
            Total_loss.detach().cpu(),
            (
                torch.mean(J_M + params["factor"] * J_x_crit)
                + torch.var(
                    J_P
                    + J_M
                    + params["factor"] * J_x_crit
                    + params["factor"] * J_th_crit
                )
            )
            .detach()
            .cpu(),
            (
                torch.mean(J_P + params["factor"] * J_th_crit)
                + torch.var(
                    J_P
                    + J_M
                    + params["factor"] * J_x_crit
                    + params["factor"] * J_th_crit
                )
            )
            .detach()
            .cpu(),
        )

    else:
        return (
            Total_loss.detach().cpu(),
            Total_loss.detach().cpu(),
            Total_loss.detach().cpu(),
        )
