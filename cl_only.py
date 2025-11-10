import sys
import torch


from src.config.configuration import build_config, Config
from src.data.mnist_cl import class_selector, get_mnist_cl_data
from src.model.model_utils import load_model
from src.training.continual_learning import CL
from tqdm import tqdm


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    model = load_model(cfg).to(cfg.device)

    criterion = torch.nn.NLLLoss()
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
    print("The factor being used is", cfg.continuous_learning.jvp_reg)

    progress_bar = tqdm(range(10), desc=f"CL Tasks", leave=True)
    for i in progress_bar:
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
            progress_bar=progress_bar,
            cfg=cfg,
        )

        memory_image.extend(xTrain)
        memory_label.extend(yTrain)
        memory_test.extend(xTest)
        memory_label_test.extend(yTest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
