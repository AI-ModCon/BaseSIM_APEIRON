import torch
from src.config.configuration import Config


# -----------------------------------------
# Basic continual learning methods can be added here.
# -----------------------------------------


# -----------------------------------------
# Algorithm: Total Retraining
# This function retrains the model with a huge memory buffer
# that maintains all the data seen so far. Obviously, this is not scalable,
# but it provides a good baseline for continual learning methods.
# This method is called "total retraining" in the literature.
# and focuses on minimizing forgetting, that is keeping the loss on the memory buffer low.
def update_CL_total_retraining(
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
    while (epoch_loss > 1e-05) and num < cfg.continuous_learning.total_updates:
        epoch_loss = 0.0
        num += 1
        for pp, data_m in enumerate(mem_iter):
            if task > 0:
                # ###########################################
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
                # send it to the model team's model class
                J_P = criterion(model(in_t), targets_t.to(device))
                J_M = criterion(model(in_m), targets_m.to(device))
                # Experience replay loss calculation
                Total_loss = J_P + J_M
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                epoch_loss += Total_loss.item()
                epoch_for_loss = J_P.item()
                epoch_gen_loss = J_M.item()
            else:
                try:
                    data_t = next(task_iter)
                    (_, y) = data_t
                    if y.shape[0] < cfg.train.batch_size:
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


# -----------------------------------------
# Additional continual learning methods can be added here.
# -----------------------------------------
