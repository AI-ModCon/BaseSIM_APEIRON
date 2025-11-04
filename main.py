from src.util import *
from data import *

import torch
from torch.nn import Linear
import torch.nn.functional as F

import numpy as np
from model import CNN
from torchvision import transforms
from torch.utils.data import DataLoader

from src.utils.general_utils import get_available_device
from src.training.continuous_learning import CL


def main():

    device = get_available_device(multi_gpu=False)
    print(device)

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
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # The dataloaders that keep the memory.
    memory_image = []
    memory_label = []
    memory_test = []
    memory_label_test = []

    # The main loop for continual learning
    # I pull the data for each task and then send it to the CL function
    for i in range(10):
        (xTrain, yTrain), (xTest, yTest) = mnist(images, labels, i)
        memory_image.extend(xTrain)
        memory_label.extend(yTrain)
        memory_test.extend(xTest)
        memory_label_test.extend(yTest)
        # Send the data and get continual learning.
        model = CL(
            (
                (xTrain, yTrain),
                (memory_image, memory_label),
                (xTest, yTest),
                (memory_test, memory_label_test),
            ),
            i,
            model,
            criterion,
            optimizer,
            device,
        )


if __name__ == "__main__":
    main()
