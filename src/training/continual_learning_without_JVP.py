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
    model.train()
    # model, Total, Gen, For = update_CL_total_retraining(
    #     model,
    #     criterion,
    #     mem_train_loader,
    #     train_loader,
    #     i,
    #     optimizer,
    #     cfg=cfg
    # )
    
    
    model, Total, Gen, For = update_CL_jvp_reg(
        model,
        criterion,
        mem_train_loader,
        train_loader,
        i,
        optimizer,
        cfg=cfg
    )
    
    
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
    



#-----------------------------------------
# Basic continual learning methods can be added here.
#-----------------------------------------

#-----------------------------------------
# Algorithm: Total Retraining
# This function retrains the model with a huge memory buffer 
# that maintains all the data seen so far. Obviously, this is not scalable,
# but it provides a good baseline for continual learning methods. 
# This method is called "total retraining" in the literature.
# and focuses on minimizing forgetting, that is keeping the loss on the memory buffer low.
def update_CL_total_retraining(
    model: torch.nn.Module,
    criterion:torch.nn.Module,
    mem_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    task: int,
    optimizer: torch.optim.Optimizer,
    cfg: Config
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
    while(epoch_loss>1e-05) and num<cfg.continuous_learning.total_updates:
        epoch_loss=0.0
        num += 1
        for pp, data_m in enumerate(mem_iter):
            if task > 0:
            # ###########################################
                try:
                    data_t = next(task_iter)
                    (_, y) = data_t
                    if y.shape[0] <  cfg.train.batch_size:
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
                J_P =criterion(model(in_t), targets_t.to(device))
                J_M = criterion( model(in_m), targets_m.to(device)) 
                # Experience replay loss calculation
                Total_loss = (J_P+J_M)
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
        return (model, 
            epoch_loss,
            epoch_for_loss,
            epoch_gen_loss
        )

    else:
        # test_acc, loss_cal = test(model, train_loader, criterion, device=device)
        epoch_loss = epoch_loss / len(train_loader.dataset)
        # print("the loss at:", num, task, total, epoch_loss, loss_cal, test_acc)
        return (model, 
            epoch_loss,
            epoch_loss,
            epoch_loss,
        )



#-----------------------------------------
# Additional continual learning methods can be added here.
#-----------------------------------------


#-----------------------------------------
# Algorithm: JVP Regularization
# This function implements a continual learning method based on Jacobian-vector product regularization.
# It aims to minimize forgetting by penalizing changes in the model's output on the memory buffer by considering their 
# directional derivatives along the parameter updates computed from the current task as well as the memory buffer.
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
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)

            # Adam update
            new_params[k] = w - lr * m_hat / (torch.sqrt(v_hat) + eps)

        return new_params
    
    
    
def return_Hamiltonian(model, params: Mapping[str, torch.Tensor], data):
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
        return OrderedDict({
            k: t for (k, _), t in zip(params.items(), tangent_seq)
        })
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
    _, dV = jvp(f, (params, exp_x), (wdot, deltax))
    
    
    # The final gradient calculation
    combined = {k: (delta_theta[k]+grad_V[k]+1*grad_dV[k]) for k in params}
    return (combined, V_star(params, x, y), V_star(params, exp_x, exp_y))

def update_CL_jvp_reg(
    model: torch.nn.Module,
    criterion:torch.nn.Module,
    mem_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    task: int,
    optimizer: torch.optim.Optimizer,
    cfg: Config
    ) -> tuple[torch.nn.Module, float, float, float]:
    
    
    device =cfg.device
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
    while(epoch_loss>1e-05) or  num<cfg.continuous_learning.total_updates:
        epoch_loss=0.0
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

                #----------------------------------------
                # deltax direction calculation
                deltax = ((in_m-in_t)/(torch.linalg.norm(in_m)\
                        +torch.linalg.norm(in_t)) ).to(device)
                
                # ------------------------------------------------
                # Build data tuple for the actual gradient calculation
                data = (in_t, targets_t, in_m, targets_m, deltax, criterion)
                with torch.enable_grad():
                    grads_dict, J_P, J_M  = return_Hamiltonian(model, params, data)   
                                 
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
                epoch_loss    += (J_P+J_M).item()
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
        return (model, 
            epoch_loss,
            epoch_for_loss,
            epoch_gen_loss
        )

    else:
        # test_acc, loss_cal = test(model, train_loader, criterion, device=device)
        epoch_loss = epoch_loss / len(train_loader.dataset)
        # print("the loss at:", num, task, total, epoch_loss, loss_cal, test_acc)
        return (model, 
            epoch_loss,
            epoch_loss,
            epoch_loss,
        )
#----------------------------------------------------------------------------------