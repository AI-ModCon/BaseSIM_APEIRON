import torch


from torch.utils.data import DataLoader
from src.utils.data_utils import MyDataset
from src.validation.valiation_utils import return_score
from src.validation.validation import test


def CL(data, task_id, model, criterion, optimizer, device):
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    mem_train_loader = DataLoader(mem_train_dataset, batch_size=64, shuffle=True)
    mem_test_loader = DataLoader(mem_test_dataset, batch_size=64, shuffle=True)

    # For now, I am recording all these, we can modify improve these things.
    accuracies_mem = []
    accuracies_one = []
    Total_loss = []
    Gen_loss = []
    For_loss = []
    dict = {}
    n_epoch = 5

    # Send to the actual CL loop
    print("Starting the task ", task_id)
    (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        scores,
    ) = One_task_CL(
        train_loader,
        model,
        optimizer,
        n_epoch,
        criterion,
        test_loader,
        mem_train_loader,
        mem_test_loader,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        task_id,
        dict,
        device,
    )
    print("Finished the task ", task_id)

    return model


def One_task_CL(
    train_loader,
    model,
    optimizer,
    n_epoch,
    criterion,
    test_loader,
    mem_train_loader,
    mem_test_loader,
    Total_loss,
    Gen_loss,
    For_loss,
    accuracies_mem,
    accuracies_one,
    i,
    dict,
    device,
):
    """
    This function is the main continual learning loop.
    It takes in the train loader, model, optimizer, number of epochs, criterion, test loader,
    memory train loader, memory test loader, total loss, generation loss, forgetting loss,
    the accuracy of the memory set, the accuracy of the one task, and the task id.
    It first gathers all the dataloaders for our task.
    Then it records all the losses and accuracies.
    Finally, it sends all the information to the One_task_CL loop and returns the updated model.


    Original:    # Now I calculate the score (In a way, this is the sensitivity of the model output with respect to the data
    # ) after the task has been met. Note that this is an approximate calculation on the whole memory data.
    # With information from the domain team, we can make this calculation more precise and efficient.
    """

    scores = return_score(
        model,
        criterion,
        mem_train_loader,
        params={
            "x_updates": 1,
            "theta_updates": 1,
            "factor": 0.00001,
            "x_lr": 0.00001,
            "th_lr": 0.00001,
            "batchsize": 64,
            "total_updates": 1000,
            "device": device,
        },
    )
    (input, grad) = scores[0]
    print("evaluate scores after a task:", len(scores), input.shape, grad.shape)
    # For now, I do not know what to do with this information, so I just return it.
    # Going forward we can do something useful with it.

    # Note here that, I could have used this score to trigger this learning process,
    # But, since we do not know this condition, I am just running a fixed number of epochs.
    # Here I run a predefined number of epochs for a single task
    for epoch in range(n_epoch):
        Total, Gen, For = update_CL_(
            model,
            criterion,
            mem_train_loader,
            train_loader,
            i,
            optimizer,
            Graph=1,
            params={
                "x_updates": 10,
                "theta_updates": 10,
                "factor": 0.1,
                "x_lr": 0.001,
                "th_lr": 0.001,
                "batchsize": 64,
                "total_updates": 10,
                "device": device,
            },
        )
        # Add the losses
        Total_loss.append(Total)
        Gen_loss.append(Gen)
        For_loss.append(For)

        # Add the accuracies
        test_acc = test(model, test_loader, Graph=1, device=device)
        mem_test_acc = test(model, mem_test_loader, Graph=1, device=device)
        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)

        # Print things when required
        if epoch % 5 == 0:
            # scheduler.step()
            mem_train_acc = test(model, mem_train_loader, Graph=1, device=device)
            train_acc = test(model, train_loader, Graph=1, device=device)
            print(
                "#########################################################################"
            )
            print("The task is ", i, "I DUMPED THE DATA")
            print(
                f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )
            print(
                f"Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}"
            )
            print(
                "#########################################################################"
            )

    return (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        scores,
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
    Graph=1,
    params={
        "x_updates": 1,
        "theta_updates": 1,
        "factor": 0.00001,
        "x_lr": 0.00001,
        "th_lr": 0.00001,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batchsize": 64,
        "total_updates": 1000,
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
