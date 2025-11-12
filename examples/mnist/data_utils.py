from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch

from torchvision import transforms
import torchvision.transforms.functional as TF


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
    rot_angle = float(np.random.random() * 180.0)  # [0, 180)

    scale = float(1.0 + np.random.random())  # [1, 2)

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


def get_mnist_cl_data() -> tuple[torch.Tensor, np.array]:
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
    labels = np.array(labels)

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
