def test(model, loader, device, Graph=0):
    """
    Evaluate the model on the given dataset.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    loader (torch.utils.data.DataLoader): The dataset loader.
    Graph (int): Whether the data is a graph (0) or image (1). Defaults to 0.

    Returns:
    float: The ratio of correct predictions.

    """
    model.eval()
    correct = 0
    if Graph == 0:
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(
                data.x.float().to(device),
                data.edge_index.to(device),
                data.batch.to(device),
            )
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int(
                (pred == data.y.to(device)).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    else:
        for data in loader:  # Iterate in batches over the training/test dataset.
            in_t, targets = data
            in_t = in_t.unsqueeze(dim=1).float()
            out = model(in_t.to(device))
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int(
                (pred == targets.to(device)).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
