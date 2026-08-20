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


# ---------------------------------------------------------------------------
# Regression metrics (lower is better) -- for field-prediction / surrogate tasks
# ---------------------------------------------------------------------------


@torch.no_grad()
def mae(output, target):
    """Mean absolute error over all elements."""
    return (output - target).abs().mean()


@torch.no_grad()
def mse(output, target):
    """Mean squared error over all elements."""
    return ((output - target) ** 2).mean()


@torch.no_grad()
def vrmse(output, target, eps: float = 1e-8):
    """Variance-scaled RMSE, averaged over batch and channels.

    The Well's headline metric: per-sample, per-channel RMSE divided by the
    target field's spatial standard deviation, so channels on different physical
    scales contribute comparably and a value of 1.0 means "no better than
    predicting the field's own mean". Assumes ``[B, C, *spatial]`` layout; falls
    back to a plain relative RMSE for lower-rank tensors.
    """
    if output.ndim >= 3:
        spatial = tuple(range(2, output.ndim))
    else:
        spatial = tuple(range(output.ndim))
    num = ((output - target) ** 2).mean(dim=spatial)
    denom = target.var(dim=spatial, unbiased=False) + eps
    return torch.sqrt(num / denom).mean()
