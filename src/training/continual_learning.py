import torch
from typing import Dict
from torch.utils.data import DataLoader
from src.data.data_utils import MyDataset
from src.validation.validation import test
from src.config.configuration import Config
from tqdm import  tqdm

def CL(
    data: tuple,
    task_id: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress_bar: tqdm,
    cfg: Config
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


    Total_loss: list = []
    Gen_loss: list = []
    For_loss: list = []
    n_epoch: int = cfg.train.epochs

    # print("Task id is", task_id)
    # print("-------")
    # Send to the actual CL loop
    (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        _,  # scores
    ) = One_task_CL(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        test_loader=test_loader,
        mem_train_loader=mem_train_loader,
        mem_test_loader=mem_test_loader,
        Total_loss=Total_loss,
        Gen_loss=Gen_loss,
        For_loss=For_loss,
        i=task_id,
        cfg=cfg,
        progress_bar=progress_bar
    )

    return model


def One_task_CL(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    mem_train_loader: torch.utils.data.DataLoader,
    mem_test_loader: torch.utils.data.DataLoader,
    Total_loss: list,
    Gen_loss: list,
    For_loss: list,
    i: int,
    cfg: Config,
    progress_bar: tqdm,
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

    # model.train()
    # model, Total, Gen, For = update_CL_total_retraining(
    #     model,
    #     criterion,
    #     mem_train_loader,
    #     train_loader,
    #     i,
    #     optimizer,
    #     params={
    #         "batchsize":  cfg.train.batch_size,
    #         "device": cfg.device,
    #     },
    # )
    
    
    # model, Total, Gen, For = update_CL_jvp_reg(
    #     model,
    #     criterion,
    #     mem_train_loader,
    #     train_loader,
    #     i,
    #     optimizer,
    #     param={
    #         "batchsize":  cfg.train.batch_size,
    #         "device": cfg.device,
    #         "max_iter": cfg.continuous_learning.total_updates,
    #     },
    #     cfg=cfg
    # )
    
    Total=0
    Gen=0
    For=0  # Dummy values for illustration
    Total_loss.append(Total)
    Gen_loss.append(Gen)
    For_loss.append(For)
    # Add the accuracies
    train_acc,_ = test(model, train_loader,  criterion, cfg=cfg)
    if i > 0:
        mem_train_acc,_ = (
            test(model, mem_train_loader,  criterion, cfg=cfg)
            if len(mem_train_loader) > 0
            else -1
        )
    # Print things when required
    if i> 0:
        mem_test_acc,_ = (
            test(model, mem_test_loader, criterion, cfg=cfg)
            if len(mem_train_loader) > 0
            else -1
        )
        test_acc,_ = test(model, test_loader, criterion, cfg=cfg)
        if i == 0:
            mem_train_acc = -1
            mem_test_acc = -1
            
            
        progress_bar.set_postfix({
            "Losses (Total)": f"{Total:.6f}",   
            "Prior(te)": f"{mem_test_acc:.1f}%"
        })
        tqdm.write("\n".join([
            f"Task {i} Summary:",
            f"Task Loss  : {Gen:.6f}",
            f"Prior Loss : {For:.6f}",
            f"Train Acc  : {train_acc:.1f}%",
            f"Test Acc   : {test_acc:.1f}%",
            f"Prior Tr   : {mem_train_acc:.1f}%",
            "-" * 40
        ]))
        

    return (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        -1,  # dummy for scores later
    )