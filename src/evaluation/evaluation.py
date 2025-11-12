import torch


def test(model, loader, criterion, cfg):
    """
    Evaluate the model on the given dataset.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    loader (torch.utils.data.DataLoader): The dataset loader.
    Graph (int): Whether the data is a graph (0) or image (1). Defaults to 0.

    # KR: I am removing graph support for now, because, we want this to be
    # transferred to the model team eventually.

    Returns:
    float: The ratio of correct predictions.
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
