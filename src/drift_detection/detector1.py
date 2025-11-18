import torch

"""
This is the start of a playground of drift detection functions. Each need to be tested separately. 
They should be independent of model and data. 
They should return a signal:
    a) continuous learning regime
    b) finetuning regime
    c) retrain regime 
"""


def return_score(  # TODO: Name paper/publication/method name. needs proper documentation. Filename needs to be named appropriately
    model,
    criterion,
    loader,
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
    """
    This function computes the gradient of the loss with respect to the input data.
    It takes in the model, criterion, data loader, and a dictionary of parameters.
    The parameters dictionary should contain the number of updates for x and theta,
    a factor for the loss, learning rates for x and theta, the device to use,
    the batch size, and the total number of updates.

    The function returns a list of tuples, where each tuple contains the input data
    and the corresponding gradient of the loss with respect to the input data.
    """
    device = params["device"]

    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

    list = []
    for data_m in loader:
        in_m, targets_m = data_m
        in_m = in_m.unsqueeze(dim=1).float().to(device)
        targets_m = targets_m.to(device)
        in_m.requires_grad = True
        out_m = model(in_m)
        crit = criterion(out_m, targets_m)
        loss = torch.mean(crit) + torch.var(crit)  # + skew(crit) + kurtosis(crit)
        adv_grad = torch.autograd.grad(loss, in_m)[0]
        # Normalize the gradient values.
        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
        adv_grad = torch.mean(adv_grad, dim=(1, 2, 3))

        # print(in_m.shape, adv_grad.shape)

        list.append(
            (
                in_m.detach().cpu().numpy(),
                adv_grad.detach().cpu().numpy(),
            )
        )

    return list
