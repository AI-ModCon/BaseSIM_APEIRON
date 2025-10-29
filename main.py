from util import *
from data import *

import torch
from torch.nn import Linear
import torch.nn.functional as F

import numpy as np
from model import CNN
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(device):
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
    model = CNN().float().to(device)
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
        )


def test(model, loader, Graph=0):
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    if Graph == 0:
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(
                data.x.float().to(device),
                data.edge_index.to(device),
                data.batch.to(device),
            )
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int(
                (pred == data.y.to(device)).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    else:
        for data in loader:  # Iterate in batches over the training/test dataset.
            in_t, targets = data
            in_t = in_t.unsqueeze(dim=1).float()
            out = model(in_t.to(device))
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int(
                (pred == targets.to(device)).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def One_task_CL(
    train_loader,
    model,
    optimizer,
    n_epoch,
    criterion,
    test_loader,
    mem_train_loader,
    mem_test_loader,
    Total_loss,
    Gen_loss,
    For_loss,
    accuracies_mem,
    accuracies_one,
    i,
    dict,
):
    """
    This function is the main continual learning loop.
    It takes in the train loader, model, optimizer, number of epochs, criterion, test loader,
    memory train loader, memory test loader, total loss, generation loss, forgetting loss,
    the accuracy of the memory set, the accuracy of the one task, and the task id.
    It first gathers all the dataloaders for our task.
    Then it records all the losses and accuracies.
    Finally, it sends all the information to the One_task_CL loop and returns the updated model.


    Original:    # Now I calculate the score (In a way, this is the sensitivity of the model output with respect to the data
    # ) after the task has been met. Note that this is an approximate calculation on the whole memory data.
    # With information from the domain team, we can make this calculation more precise and efficient.
    """

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
        },
    )
    (input, grad) = scores[0]
    print("evaluate scores after a task:", len(scores), input.shape, grad.shape)
    # For now, I do not know what to do with this information, so I just return it.
    # Going forward we can do something useful with it.

    # Note here that, I could have used this score to trigger this learning process,
    # But, since we do not know this condition, I am just running a fixed number of epochs.
    # Here I run a predefined number of epochs for a single task
    for epoch in range(n_epoch):
        Total, Gen, For = update_CL_(
            model,
            criterion,
            mem_train_loader,
            train_loader,
            i,
            optimizer,
            Graph=1,
            params={
                "x_updates": 10,
                "theta_updates": 10,
                "factor": 0.1,
                "x_lr": 0.001,
                "th_lr": 0.001,
                "batchsize": 64,
                "total_updates": 10,
                "device": device,
            },
        )
        # Add the losses
        Total_loss.append(Total)
        Gen_loss.append(Gen)
        For_loss.append(For)

        # Add the accuracies
        test_acc = test(model, test_loader, Graph=1)
        mem_test_acc = test(model, mem_test_loader, Graph=1)
        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)

        # Print things when required
        if epoch % 5 == 0:
            # scheduler.step()
            mem_train_acc = test(model, mem_train_loader, Graph=1)
            train_acc = test(model, train_loader, Graph=1)
            print(
                "#########################################################################"
            )
            print("The task is ", i, "I DUMPED THE DATA")
            print(
                f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )
            print(
                f"Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}"
            )
            print(
                "#########################################################################"
            )

    return (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        scores,
    )


# I gather all the dataloaders here and then send them to the actual CL function.
def CL(data, task_id, model, criterion, optimizer):
    """
    This is the main function that runs the continual learning loop.
    It takes in the data, task_id, model, criterion, and optimizer as input.
    It first gathers all the dataloaders for our task.
    Then it records all the losses and accuracies.
    Finally, it sends all the information to the One_task_CL loop and returns the updated model.
    """
    (
        (xTrain, yTrain),
        (xTest, yTest),
        (memory_image, memory_label),
        (memory_test, memory_label_test),
    ) = data
    train_dataset = MyDataset(xTrain, yTrain)
    mem_train_dataset = MyDataset(memory_image, memory_label)
    test_dataset = MyDataset(xTest, yTest)
    mem_test_dataset = MyDataset(memory_test, memory_label_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    mem_train_loader = DataLoader(mem_train_dataset, batch_size=64, shuffle=True)
    mem_test_loader = DataLoader(mem_test_dataset, batch_size=64, shuffle=True)

    # For now, I am recording all these, we can modify improve these things.
    accuracies_mem = []
    accuracies_one = []
    Total_loss = []
    Gen_loss = []
    For_loss = []
    dict = {}
    n_epoch = 5

    # Send to the actual CL loop
    print("Starting the task ", task_id)
    (
        model,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        dict,
        scores,
    ) = One_task_CL(
        train_loader,
        model,
        optimizer,
        n_epoch,
        criterion,
        test_loader,
        mem_train_loader,
        mem_test_loader,
        Total_loss,
        Gen_loss,
        For_loss,
        accuracies_mem,
        accuracies_one,
        task_id,
        dict,
    )
    print("Finished the task ", task_id)

    return model


if __name__ == "__main__":
    main(device)
