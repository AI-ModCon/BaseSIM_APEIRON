import torch


@torch.no_grad()
def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # maxk, dim, look_for_largest, results_sorted
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def accuracy(output, target):
    return accuracy_topk(output, target, topk=(1,))[0]


@torch.no_grad()
def vrmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Variance-normalised RMSE for pixel-level regression.

    VRMSE = RMSE / std(target).  A scale-free error measure: 1.0 means the
    model's error equals the natural spread of the target field.
    """
    mse = torch.mean((output - target) ** 2)
    rmse = torch.sqrt(mse)
    target_std = torch.std(target)
    # Guard against zero-variance targets (constant fields).
    if target_std < 1e-8:
        return rmse
    return rmse / target_std
