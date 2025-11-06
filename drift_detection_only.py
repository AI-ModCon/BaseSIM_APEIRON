import sys
import torch


from src.config.configuration import build_config, Config
from src.data.mnist_cl import class_selector, get_mnist_cl_data
from src.model.model_utils import load_model


from torch.utils.data import DataLoader
from data.data_utils import MyDataset
from src.drift_detection.detector1 import return_score


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

        memory_image.extend(xTrain)
        memory_label.extend(yTrain)
        memory_test.extend(xTest)
        memory_label_test.extend(yTest)

        mem_train_dataset = MyDataset(memory_image, memory_label)

        mem_train_loader = DataLoader(
            mem_train_dataset, batch_size=cfg.data.batch_size, shuffle=True
        )
        # Send the data and get continual learning.
        scores = return_score(
            model,
            criterion,
            mem_train_loader,
            params={
                "x_updates": 1,
                "theta_updates": 1,
                "factor": 0.00001,
                "x_lr": 0.00001,
                "th_lr": 0.00001,
                "batchsize": 64,
                "total_updates": 1000,
                "device": device,
            },  # todo: put these params to the toml file
        )
        print(scores)
        # input("Press Enter to continue with the next task...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
