import sys


from src.config.configuration import build_config, Config

from tqdm import tqdm

from examples.MNIST.mnist_cnn import MNIST_CNN
from src.training.continual_learning import continual_learning_loop


def main(argv=None) -> int:
    cfg: Config = build_config(argv)
    modelHarness = MNIST_CNN(cfg=cfg)

    # The main loop for continual learning
    # I pull the data for each task=mnist class and then send it to the CL function
    print("The factor being used is", cfg.continuous_learning.jvp_reg)

    progress_bar = tqdm(range(10), desc="CL Tasks", leave=True)
    for i in progress_bar:

        continual_learning_loop(cfg=cfg, modelHarness=modelHarness)

        # class_id = i % 10

        # (xTrain, yTrain), (xTest, yTest) = class_selector(images, labels, class_id)
        # # Send the data and get continual learning.
        # model = CL(
        #     data=(
        #         (xTrain, yTrain),
        #         (xTest, yTest),
        #         (memory_image, memory_label),
        #         (memory_test, memory_label_test),
        #     ),
        #     task_id=class_id,
        #     model=model,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     progress_bar=progress_bar,
        #     cfg=cfg,
        # )

        # memory_image.extend(xTrain)
        # memory_label.extend(yTrain)
        # memory_test.extend(xTest)
        # memory_label_test.extend(yTest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
