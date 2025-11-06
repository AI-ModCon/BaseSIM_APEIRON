import torch


def test(model, loader, device):
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
    for data in loader:  # Iterate in batches over the training/test dataset.
        in_t, targets = data
        in_t = in_t.unsqueeze(dim=1).float()
        out = model(in_t.to(device))
        pred = torch.max(out,1)[1]
        correct += int(
            (pred == targets.to(device)).sum()
        )  # Check against ground-truth labels.
        
    print(correct, len(loader.dataset))
    
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
