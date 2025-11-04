import torch


def return_score(
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
