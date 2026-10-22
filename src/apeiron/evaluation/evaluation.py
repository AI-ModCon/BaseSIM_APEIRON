import torch


def test(model, loader, criterion, cfg):
    """
    Evaluate the model on the given dataset.

    Nothing in the framework calls this; ``BaseModelHarness.eval()`` is the
    supported path. It is also only arithmetically correct for a
    ``reduction="sum"`` criterion: batch losses are summed and divided by the
    sample count, so a mean-reduction criterion (what every harness returns)
    under-reports the loss by the number of batches.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    loader (torch.utils.data.DataLoader): The dataset loader.
    criterion (Callable): Loss function; see the caveat above on reduction.
    cfg: Configuration object supplying ``device``.

    Returns:
    tuple[float, float]: Accuracy as a percentage and the average loss --
    except on an empty loader, where a bare ``0.0`` is returned instead.
    """
    model.eval()
    correct = 0
    total = len(loader.dataset)
    if total == 0:
        return 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            input, target = data
            # ensure inputs have channel dimension and are floats
            input = input.to(cfg.device)
            out = model(input)
            # move targets to same device as outputs to avoid device-mismatch errors
            target = target.to(cfg.device)
            # use sum reduction so we can average correctly over dataset size
            test_loss += criterion(out, target).item()
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    # average test loss over all examples and compute accuracy
    if total > 0:
        test_loss = test_loss / total
        accuracy = 100.0 * correct / total
    else:
        test_loss = 0.0
        accuracy = 0.0

    return accuracy, test_loss  # Derive ratio of correct predictions.
