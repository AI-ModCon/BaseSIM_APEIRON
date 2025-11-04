# main.py (or your runner)

import sys
import torch

from src.utils.general_utils import get_available_device
from src.training.continuous_learning import CL
from src.config.configuration import build_config, Config
from src.data.mnist_cl import class_selector, get_mnist_cl_data
from src.model.model_utils import load_model


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    print(cfg)

    device = get_available_device(multi_gpu=False)

    model = load_model(cfg).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Memory holds ONLY past tasks
    memory_image, memory_label = [], []
    memory_test, memory_label_test = [], []

    images, labels = get_mnist_cl_data()

    for i in range(10):
        class_id = i % 10
        (xTrain, yTrain), (xTest, yTest) = class_selector(images, labels, class_id)

        # Train on current task using *past* memory only
        model = CL(
            data=(
                (xTrain, yTrain),
                (xTest, yTest),
                (memory_image, memory_label),
                (memory_test, memory_label_test),
            ),
            task_id=class_id,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )

        # AFTER training, add the current task to memory
        memory_image.extend(xTrain)
        memory_label.extend(yTrain)
        memory_test.extend(xTest)
        memory_label_test.extend(yTest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
