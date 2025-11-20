from torch.utils.data import Dataset
import torchvision
import torch

from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List

def augment_and_split(
    images: torch.Tensor, labels: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies a single random affine transform to a batch of images and labels, then splits the batch into two non-overlapping sets of training and testing data.

    Parameters
    ----------
    images : torch.Tensor
        The batch of images to be augmented and split.
    labels : torch.Tensor
        The batch of labels corresponding with the images.

    Returns
    -------
    tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        A tuple containing two tuples. The first tuple contains the training images and labels, and the second tuple contains the testing images and labels.
    """

    # Random augmentation parameters (same behavior as original)
    rot_angle = torch.rand(1).item() * 180.0  # [0, 180]

    scale = float(1.0 + torch.rand(1).item())  # [1, 2)

    # Apply a single affine transform to the entire batch
    X_aug = TF.affine(
        images,
        angle=rot_angle,
        translate=(scale, scale),  # pixels (kept as in original)
        scale=scale,
        shear=rot_angle,
    )

    # Clean 80/20 split: non-overlapping, no duplicates
    n = X_aug.shape[0]
    n_train = int(0.8 * n)
    perm = torch.randperm(n)  # permutation without replacement
    idx_train = perm[:n_train]
    idx_test = perm[n_train:]

    xtrain = (X_aug[idx_train], labels[idx_train])
    xtest = (X_aug[idx_test], labels[idx_test])
    return xtrain, xtest


def get_mnist_cl_data() -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function downloads the MNIST dataset, applies a normalization transformation to the data,
    stacks the images and labels, and returns the images and labels as a tensor and a numpy array.

    Returns:
        tuple: A tuple containing the images and labels.
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


class CustomMnistData(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


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