import sys
import torch
from torchvision import transforms
import torchvision

import numpy as np
from src.model.model import CNN


from src.utils.general_utils import get_available_device
from src.training.continuous_learning import CL
from src.config.configuration import build_config, Config
from src.data.data import mnist


def main(argv=None) -> int:

    cfg: Config = build_config(argv)

    print(cfg)

    device = get_available_device(
        multi_gpu=False
    )  # Todo: put this in config file. Once we determine how to handle multi-gpu

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
