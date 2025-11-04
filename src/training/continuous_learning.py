# src/training/continuous_learning.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import stateless

from src.utils.data_utils import MyDataset
from src.validation.validation import test


def _ensure_4d(x):
    # expects [B, H, W] or [B, 1, H, W]
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _l2_normalize_like_images(t):
    # normalize over all non-batch dims
    flat = t.flatten(1)
    n = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    n = n.view(-1, *([1] * (t.dim() - 1)))
    return t / n


def _next_batch(it, loader):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


@torch.no_grad()
def _accuracy(model, loader, device):
    return test(model, loader, Graph=1, device=device)


def _adv_gap(model, x, y, eps, steps):
    """G(θ): increase memory loss in input space via L2-PGD"""
    x = x.detach()
    x_adv = x.requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x_adv), y, reduction="mean")
        g = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        g = _l2_normalize_like_images(g)
        x_adv = (x_adv + eps * g).detach().requires_grad_(True)
    with torch.enable_grad():
        return F.cross_entropy(model(x_adv), y, reduction="mean") - F.cross_entropy(
            model(x), y, reduction="mean"
        )


def _one_step_lookahead_params(model, x, y, inner_lr):
    """θ' = θ - inner_lr * ∇_θ L_t(θ) (differentiable)"""
    loss_t = F.cross_entropy(model(x), y, reduction="mean")
    params = dict(model.named_parameters())
    grads = torch.autograd.grad(loss_t, params.values(), create_graph=True)
    theta_prime = {
        name: p - inner_lr * g for (name, p), g in zip(params.items(), grads)
    }
    return theta_prime


def update_CL_(
    model,
    criterion,
    mem_loader,
    train_loader,
    task,
    optimizer,
    Graph=1,
    params=None,
):
    """
    One BCL-style inner loop:
      - Lt = current task loss
      - Lm = memory loss
      - F  = forgetting proxy = Lm(θ') - Lm(θ)  with θ' = θ - α∇Lt
      - G  = generalization proxy via adversarial memory batch
      - lam = dual variable, projected gradient ascent
      - minimize Lt + Lm + lam * (F - G)
    """
    if params is None:
        params = {}
    device = params.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    total_updates = params.get("total_updates", 200)
    x_updates = params.get("x_updates", 3)
    th_lr = params.get("th_lr", 1e-3)
    x_eps = params.get("x_lr", 1e-3)
    lam_lr = params.get("lam_lr", 0.1)

    # Create fresh iterators
    task_iter = iter(train_loader)
    mem_iter = iter(mem_loader) if (task > 0 and len(mem_loader.dataset) > 0) else None

    lam = 0.0  # learned balance (dual variable)
    last_total = None
    last_gen = None
    last_for = None

    for _ in range(total_updates):
        # --- current task batch
        (in_t, targets_t), task_iter = _next_batch(task_iter, train_loader)
        in_t = _ensure_4d(in_t).float().to(device)
        targets_t = targets_t.to(device)

        Lt = criterion(model(in_t), targets_t).mean()

        if mem_iter is not None:
            # --- memory batch
            (in_m, targets_m), mem_iter = _next_batch(mem_iter, mem_loader)
            in_m = _ensure_4d(in_m).float().to(device)
            targets_m = targets_m.to(device)

            Lm = criterion(model(in_m), targets_m).mean()

            # Generalization proxy G via adversarial memory examples
            G = _adv_gap(model, in_m, targets_m, eps=x_eps, steps=x_updates)

            # Forgetting proxy F via differentiable look-ahead on task step
            theta_prime = _one_step_lookahead_params(
                model, in_t, targets_t, inner_lr=th_lr
            )
            out_m_prime = stateless.functional_call(model, theta_prime, (in_m,))
            Lm_prime = F.cross_entropy(out_m_prime, targets_m, reduction="mean")
            F_forget = Lm_prime - Lm

            # Dual ascent on lam (no grad)
            with torch.no_grad():
                lam = max(0.0, lam + lam_lr * (F_forget.item() - G.item()))

            Total = Lt + Lm + lam * (F_forget - G)
            Gen_term = Lm + lam * G
            For_term = Lt + lam * F_forget
        else:
            # First task (no memory yet): pure ERM
            Total = Lt
            Gen_term = Lt
            For_term = Lt

        optimizer.zero_grad()
        Total.backward()
        optimizer.step()

        last_total = Total.detach().cpu()
        last_gen = Gen_term.detach().cpu()
        last_for = For_term.detach().cpu()

    return last_total, last_gen, last_for


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
    metrics_dict,
    device,
):
    """
    One task training using the BCL-style inner loop above.
    """
    # Optional: compute sensitivity scores on memory (unchanged)
    # Only compute scores if memory exists
    has_memory = len(mem_train_loader.dataset) > 0

    if has_memory:
        scores = return_score(
            model,
            criterion,
            mem_train_loader,
            params={
                "x_updates": 1,
                "theta_updates": 1,
                "factor": 1e-5,
                "x_lr": 1e-5,
                "th_lr": 1e-5,
                "batchsize": 64,
                "total_updates": 1000,
                "device": device,
            },
        )
    else:
        scores = None

    for epoch in range(n_epoch):
        Total, Gen, For = update_CL_(
            model,
            criterion,
            mem_train_loader,
            train_loader,
            task=i,
            optimizer=optimizer,
            Graph=1,
            params={
                "x_updates": 3,
                "th_lr": 1e-3,
                "x_lr": 1e-3,
                "lam_lr": 0.1,
                "total_updates": 100,  # tune
                "device": device,
            },
        )
        Total_loss.append(Total)
        Gen_loss.append(Gen)
        For_loss.append(For)

        test_acc = _accuracy(model, test_loader, device)
        mem_test_acc = (
            _accuracy(model, mem_test_loader, device) if has_memory else float("nan")
        )

        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)

    mem_train_acc = (
        _accuracy(model, mem_train_loader, device) if has_memory else float("nan")
    )
    train_acc = _accuracy(model, train_loader, device)
    print("#########################################################################")
    print("Finished training images of class : ", i)
    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}")
    print("#########################################################################")

    return (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        metrics_dict,
        scores,
    )


def CL(data, task_id, model, criterion, optimizer, device, cfg):
    """
    Wraps the dataloaders and delegates to One_task_CL.
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

    bs = cfg.data.batch_size

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    mem_train_loader = DataLoader(mem_train_dataset, batch_size=bs, shuffle=False)
    mem_test_loader = DataLoader(mem_test_dataset, batch_size=bs, shuffle=False)

    accuracies_mem, accuracies_one = [], []
    Total_loss, Gen_loss, For_loss = [], [], []
    metrics_dict = {}

    n_epoch = cfg.train.epochs
    print("Task id is", task_id)
    print("-------")

    return One_task_CL(
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
        metrics_dict=metrics_dict,
        device=device,
    )[
        0
    ]  # return updated model


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
