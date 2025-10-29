import torchvision
import numpy as np
import torch


def mnist(images, labels, task_id):
    """
    This function takes in a dataset of images and labels, and a task id.
    It then randomly selects a class and applies a random transformation to the images.
    The transformation is an affine transformation with a random rotation and scaling.
    The function then splits the data into a training set and a test set.

    Parameters:
    images (torch.Tensor): The tensor of images.
    labels (numpy.array): The array of labels.
    task_id (int): The id of the task.

    Returns:
    tuple: A tuple containing the training data and the test data.
    """
    imp = np.random.randint(0, 9)
    # #print(imp, task_id)
    idx = labels == imp
    X = images[idx]
    y = labels[idx]
    # #print("We have to apply the transformation now.")
    rot_angle = np.random.random() * 180
    scaling = np.random.random() + 1
    # #print(rot_angle)
    X = torchvision.transforms.functional.affine(
        X, rot_angle, translate=(scaling, scaling), scale=1, shear=rot_angle
    )
    # #print("Just after the data is defined", X.shape, y.shape)
    # Split the data
    print(X.shape, y.shape)
    index = np.random.randint(0, X.shape[0], int(0.8 * X.shape[0]))
    xtrain = X[index], y[index]
    index = np.random.randint(0, X.shape[0], int(0.2 * X.shape[0]))
    X_test = X[index]
    y_test = y[index]
    return xtrain, (X_test, y_test)
