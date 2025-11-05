import sys
import torch


from src.training.continuous_learning import CL
from src.config.configuration import build_config, Config
from src.data.mnist_cl import class_selector, get_mnist_cl_data
from src.model.model_utils import load_model


def main(argv=None) -> int:

    cfg: Config = build_config(argv)

    model = load_model(cfg).to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )  # needs to be put into cl loop

    # The dataloaders that keep the memory.
    memory_image = []
    memory_label = []
    memory_test = []
    memory_label_test = []

    images, labels = get_mnist_cl_data()

    # The main loop for continual learning
    # I pull the data for each task=mnist class and then send it to the CL function

    for i in range(10):

        class_id = i % 10
        (xTrain, yTrain), (xTest, yTest) = class_selector(images, labels, class_id)

        # Send the data and get continual learning.
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
            cfg=cfg,
        )

        memory_image.extend(xTrain)
        memory_label.extend(yTrain)
        memory_test.extend(xTest)
        memory_label_test.extend(yTest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
