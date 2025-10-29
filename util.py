import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
        
def return_task_list(X, y, n_samples, n_task=20):
    # print(X.shape, y.shape)
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    import random
    torch.manual_seed(12345)
    random.seed(12345)
    tasks=[]
    task_order =[]
    total_k = k = np.random.randint(0,3,n_task) 
    for k in total_k:
        # print("The randomly selected task is", k)
        task_X= X[y == k]
        task_y= y[y == k]
        import numpy as np
        low, high, n = 0, task_y.shape[0], n_samples
        sample_n = int(high/n)
        rng = np.random.default_rng()  # or previously instantiated RNG
        space = np.array([i for i in range(high)]).reshape([-1])
        samples = np.stack( [ rng.choice(space, size=n, replace=False) for _ in range(sample_n) ] )
        for idx in samples:
            tasks.append((task_X[idx], task_y[idx] ))
            task_order.append(k)

    # printing lists    
    # Shuffle two lists with same order
    # Using zip() + * operator + shuffle()
    temp = list(zip(tasks, task_order))
    random.shuffle(temp)
    tasks, task_order= zip(*temp)
    ## res1 and res2 come out as tuples, and so must be converted to lists.
    tasks, task_order = list(tasks), list(task_order)
    
    #ids =np.random.randint(0, len(tasks), len(tasks)).reshape([-1]).tolist()
    #print(ids)
    #tasks = tasks[ids]
    #task_order = task_order[ids]
    for loc, k in enumerate([1, 0, 2]):
        task_X= X[y == k]
        task_y= y[y == k]
        import numpy as np
        low, high, n = 0, task_y.shape[0], n_samples
        sample_n = int(high/n)
        rng = np.random.default_rng()  # or previously instantiated RNG
        space = np.array([i for i in range(high)]).reshape([-1])
        samples = np.stack( [ rng.choice(space, size=n, replace=False) for _ in range(1) ] )
        for idx in samples:
            tasks.insert(loc, (task_X[idx], task_y[idx] ))
            task_order.insert(loc, k)
            break
    return tasks, task_order



def normalize_grad(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

def return_score(model, criterion, loader,\
     params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.00001, 'x_lr': 0.00001,'th_lr':0.00001,\
               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
              'batchsize': 64, 'total_updates': 1000 } ):
    
    device = params['device']
    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    list = []
    for data_m in loader:
        
        in_m, targets_m = data_m
        in_m = in_m.unsqueeze(dim=1).float().to(device)
        targets_m=targets_m.to(device)
        in_m.requires_grad = True
        out_m = model(in_m)
        crit = criterion(out_m, targets_m)
        loss = torch.mean(crit) + torch.var(crit) # + skew(crit) + kurtosis(crit)
        adv_grad = torch.autograd.grad(loss,in_m)[0]
        # Normalize the gradient values.
        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
        adv_grad = torch.mean(adv_grad, dim = (1,2,3) )

        # print(in_m.shape, adv_grad.shape)

        list.append(
            (
                in_m.detach().cpu().numpy(),
                adv_grad.detach().cpu().numpy(),
            )
        )
    
    return list    



# This function is my actual CL update. We can actually modify this function 
# for both efficiency and effectiveness, however, I think this is a good starting point.
def update_CL_(model, criterion, mem_loader, train_loader, task, optimizer, Graph = 1,\
     params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.00001, 'x_lr': 0.00001,'th_lr':0.00001,\
               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
              'batchsize': 64, 'total_updates': 1000 } ):
    device = params['device']
    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    import copy
    # We set up the iterators for the memory loader and the train loader
    mem_iter = iter(mem_loader)
    task_iter = iter(train_loader)
    
    # The main loop over all the batch
    for i in range(params['total_updates']): 
        if task>0:
            ###########################################
            try:
                data_t= next(task_iter)
                (_, y) = data_t
                if y.shape[0]<params['batchsize']:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)
            except StopIteration:
                task_iter = iter(train_loader)
                data_t= next(task_iter)
            
            
            # Extract a batch from the memory
            try:
                data_m= next(mem_iter)
                (_, y) = data_m
                if y.shape[0]<params['batchsize']:
                    mem_iter = iter(mem_loader)
                    data_m= next(mem_iter)    
            except StopIteration:
                mem_iter = iter(mem_loader)
                data_m= next(mem_iter)
                
            in_t, targets_t= data_t
            in_m, targets_m = data_m

            in_t = in_t.unsqueeze(dim=1).float().to(device)
            in_m = in_m.unsqueeze(dim=1).float().to(device)
            targets_t=targets_t.to(device)
            targets_m=targets_m.to(device)
                        
            out = model(in_t)
            out_m = model(in_m)
            
            
            ############## The task cost and the memory cost
            #########################################################################################
            J_P = criterion(out, targets_t.to(device))
            J_M = criterion(out_m, targets_m.to(device))

            ############## This is the J_x loss
            #########################################################################################
            J_PN_x=criterion(model(in_m), targets_m)
            x_PN = copy.copy(in_m).to(device)
            x_PN.requires_grad = True
            adv_grad = 0
            epsilon =params['x_lr']
            
            for epoch in range(params["x_updates"]):
                x_PN = x_PN+ epsilon*adv_grad
                crit = criterion(model(x_PN.float() ), targets_m)
                loss = torch.mean(crit) + torch.var(crit) # + skew(crit) + kurtosis(crit)
                adv_grad = torch.autograd.grad(loss,x_PN)[0]
                # Normalize the gradient values.
                adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
            J_x_crit = (criterion(model(x_PN.float()), targets_m) -J_PN_x)
            # print(adv_grad.shape)
            # print("norm", torch.norm(adv_grad))
            ############### This is the loss J_th
            #########################################################################################
            cop = copy.deepcopy(model).to(device)
            opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
            J_PN_theta = criterion(model(in_m.float()), targets_m)
            for i in range(params["theta_updates"]):
                opt_buffer.zero_grad()
                loss_crit = criterion(cop(in_t.float()), targets_t)
                loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                #+ torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                loss_m.backward(retain_graph=True)
                opt_buffer.step()
            J_th_crit = (criterion(cop(in_m.float()), targets_m) - J_PN_theta)
            
            # Now, put together  the loss fully 
            Total_loss= torch.mean(J_M+J_P+params['factor']*J_x_crit+params['factor']*J_th_crit)
            # +torch.var(J_P+J_M+J_x_crit+params['factor']*J_th_crit)
            #adjoint_scores =  
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        else:
            try:
                data_t= next(task_iter)
                (_, y) = data_t
                if y.shape[0]<params['batchsize']:
                    task_iter = iter(train_loader)
                    data_t = next(task_iter)
            except StopIteration:
                task_iter = iter(train_loader)
                data_t= next(task_iter)
                
            in_t, targets_t = data_t 
            in_t = in_t.unsqueeze(dim=1).float()
    
            critti= criterion(model(in_t.to(device)), targets_t.to(device))
            Total_loss = torch.mean(critti) + torch.var(critti)
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.


    if task>0:
        return Total_loss.detach().cpu(),\
              (torch.mean(J_M+ params['factor']*J_x_crit)+torch.var(J_P+J_M+params['factor']*J_x_crit+params['factor']*J_th_crit)).detach().cpu(),\
              (torch.mean(J_P+params['factor']*J_th_crit)+torch.var(J_P+J_M+params['factor']*J_x_crit+params['factor']*J_th_crit)).detach().cpu()

    else:
        return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()
