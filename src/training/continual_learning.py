import torch

from typing import Dict


from torch.utils.data import DataLoader
from src.data.data_utils import MyDataset
from src.validation.validation import test
from src.config.configuration import Config


def CL(
    data: tuple,
    task_id: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> torch.nn.Module:
    """
    This function is the main function for continual learning. It takes in the data, task id, model, criterion, and optimizer.
    It then constructs the dataloaders for the task and the memory, and sends them to the One_task_CL loop.
    The function returns the updated model.
    """
    (
        (xTrain, yTrain),
        (xTest, yTest),
        (memory_image, memory_label),
        (memory_test, memory_label_test),
    ) = data

    train_dataset = MyDataset(xTrain, yTrain)
    mem_train_dataset = MyDataset(memory_image, memory_label)

    test_dataset = MyDataset(xTest, yTest)
    mem_test_dataset = MyDataset(memory_test, memory_label_test)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False
    )

    mem_train_loader = DataLoader(
        mem_train_dataset, batch_size=cfg.train.batch_size, shuffle=False
    )
    mem_test_loader = DataLoader(
        mem_test_dataset, batch_size=cfg.train.batch_size, shuffle=False
    )

    # For now, I am recording all these, we can modify improve these things.
    accuracies_mem: list = []
    accuracies_one: list = []
    Total_loss: list = []
    Gen_loss: list = []
    For_loss: list = []
    dict: Dict = {}

    n_epoch: int = cfg.train.epochs

    print("Task id is", task_id)
    print("-------")
    # Send to the actual CL loop
    (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        _,  # scores
    ) = One_task_CL(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        n_epoch=n_epoch,
        criterion=criterion,
        test_loader=test_loader,
        mem_train_loader=mem_train_loader,
        mem_test_loader=mem_test_loader,
        Total_loss=Total_loss,
        Gen_loss=Gen_loss,
        For_loss=For_loss,
        accuracies_mem=accuracies_mem,
        accuracies_one=accuracies_one,
        i=task_id,
        dict=dict,
        cfg=cfg,
    )

    return model


def One_task_CL(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epoch: int,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    mem_train_loader: torch.utils.data.DataLoader,
    mem_test_loader: torch.utils.data.DataLoader,
    Total_loss: list,
    Gen_loss: list,
    For_loss: list,
    accuracies_mem: list,
    accuracies_one: list,
    i: int,
    dict: Dict,
    cfg: Config,
) -> tuple[torch.nn.Module, list, list, list, list, list, Dict, int]:
    """
    This function is the main continual learning loop.
    It takes in the train loader, model, optimizer, number of epochs, criterion, test loader,
    memory train loader, memory test loader, total loss, generation loss, forgetting loss,
    the accuracy of the memory set, the accuracy of the one task, and the task id.
    It first gathers all the dataloaders for our task.
    Then it records all the losses and accuracies.
    Finally, it sends all the information to the One_task_CL loop and returns the updated model.

    Args:
        train_loader (torch.utils.data.DataLoader): The loader for the current task.
        model (torch.nn.Module): The model being used.
        optimizer (torch.optim.Optimizer): The optimizer being used.
        n_epoch (int): The number of epochs to train for.
        criterion (torch.nn.Module): The loss function being used.
        test_loader (torch.utils.data.DataLoader): The loader for the test data.
        mem_train_loader (torch.utils.data.DataLoader): The loader for the memory train data.
        mem_test_loader (torch.utils.data.DataLoader): The loader for the memory test data.
        Total_loss (list): The list of total losses.
        Gen_loss (list): The list of generation losses.
        For_loss (list): The list of forgetting losses.
        accuracies_mem (list): The list of accuracies of the memory set.
        accuracies_one (list): The list of accuracies of the one task.
        i (int): The task id.
        dict (Dict): The dictionary of scores.
        device (torch.device): The device being used.
        cfg (Config): The configuration object.

    Returns:
        tuple[torch.nn.Module, list, list, list, list, list, Dict, int]: The updated model, total loss, generation loss, forgetting loss,
            accuracy of the memory set, accuracy of the one task, and the task id.
    """

    for epoch in range(n_epoch):
        Total, Gen, For = update_CL_(
            model,
            criterion,
            mem_train_loader,
            train_loader,
            i,
            optimizer,
            Graph=1,
            params={  # Keep for now, until Krishnan has updated his code. Then put everythin into cfg.
                "x_updates": cfg.continuous_learning.x_updates,
                "theta_updates": cfg.continuous_learning.theta_updates,
                "factor": cfg.continuous_learning.factor,
                "x_lr": cfg.continuous_learning.x_lr,
                "th_lr": cfg.continuous_learning.th_lr,
                "batchsize": cfg.train.batch_size,
                "total_updates": cfg.continuous_learning.total_updates,
                "device": cfg.device,
            },
            cfg=cfg,
        )
        # Add the losses
        Total_loss.append(Total)
        Gen_loss.append(Gen)
        For_loss.append(For)

        # Add the accuracies
        test_acc = test(model, test_loader, Graph=1, cfg=cfg)
        mem_test_acc = (
            test(model, mem_test_loader, Graph=1, cfg=cfg)
            if len(mem_test_loader) > 0
            else -1
        )
        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)

    # Print things when required

    mem_train_acc = (
        test(model, mem_train_loader, Graph=1, cfg=cfg)
        if len(mem_train_loader) > 0
        else -1
    )
    train_acc = test(model, train_loader, Graph=1, cfg=cfg)
    print("#########################################################################")
    print("Finished training images of class : ", i)
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}")
    print("#########################################################################")

    return (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        -1,  # dummy for scores later
    )


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
