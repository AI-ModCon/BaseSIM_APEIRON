import torch
import torch.nn.functional as F

from typing import Any, Callable, Iterable, Tuple, List
from torch import nn
from torch.optim import Optimizer

from src.model.torch_model_harness import BaseModelHarness
from src.config.configuration import Config
from examples.MNIST.data_utils import class_selector, get_mnist_cl_data


class Cnn(torch.nn.Module):
    # Simple CNN model as example

    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.unsqueeze(dim=1).float()
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_CNN(BaseModelHarness):

    def __init__(self, cfg: Config, model: nn.Module = Cnn()):
        super(MNIST_CNN, self).__init__(model, cfg)

        # To emulate a drifting data stream, we sort the MNISt data by label and stream the data in order
        self.task_counter = 0
        self.images, self.labels = get_mnist_cl_data()

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_cur_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a training and validation dataloader compatible with the model input
        """

        (xTrain, yTrain), (xTest, yTest) = class_selector(
            self.images, self.labels, self.task_counter
        )

        raise NotImplementedError

    def get_hist_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a training and validation dataloader with historical data (to measure drift) compatible with the model input
        """
        raise NotImplementedError

    def get_criterion(self) -> CriterionFn:
        """Return a loss function compatible with model output and dataloader labels"""
        raise NotImplementedError
