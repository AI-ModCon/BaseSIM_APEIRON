import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List


def get_mnist_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Download and load the MNIST dataset with normalization.
    """
    my_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=my_transforms
    )
    [images, labels] = [list(t) for t in zip(*dataset)]
    images = torch.stack(images, dim=0)
    images = images.view(-1, 28, 28).float()
    labels = torch.tensor(labels)

    return images, labels


def filter_mnist_by_classes(
    images: torch.Tensor, labels: torch.Tensor, classes: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter MNIST dataset to include only specified classes.

    Args:
        images: Image tensor of shape [N, 28, 28]
        labels: Label tensor of shape [N]
        classes: List of class labels to keep (e.g., [0, 1, 2, 3, 4])

    Returns:
        tuple: Filtered (images, labels) containing only the specified classes
    """
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for cls in classes:
        mask |= labels == cls

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    return filtered_images, filtered_labels


def split_train_test(
    images: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.8
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data into training and test sets.

    Args:
        images: Image tensor of shape [N, 28, 28]
        labels: Label tensor of shape [N]
        train_ratio: Fraction of data to use for training (default: 0.8)

    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    n = len(images)
    n_train = int(train_ratio * n)

    # Random permutation for splitting
    perm = torch.randperm(n)
    idx_train = perm[:n_train]
    idx_test = perm[n_train:]

    train_images = images[idx_train]
    train_labels = labels[idx_train]
    test_images = images[idx_test]
    test_labels = labels[idx_test]

    return (train_images, train_labels), (test_images, test_labels)


class MNISTDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.images)
