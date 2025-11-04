import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch

# "TODO: clean this file"


def return_task_list(X, y, n_samples, n_task=20):
    # print(X.shape, y.shape)
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    import random

    torch.manual_seed(12345)
    random.seed(12345)
    tasks = []
    task_order = []
    total_k = k = np.random.randint(0, 3, n_task)
    for k in total_k:
        # print("The randomly selected task is", k)
        task_X = X[y == k]
        task_y = y[y == k]
        import numpy as np

        low, high, n = 0, task_y.shape[0], n_samples
        sample_n = int(high / n)
        rng = np.random.default_rng()  # or previously instantiated RNG
        space = np.array([i for i in range(high)]).reshape([-1])
        samples = np.stack(
            [rng.choice(space, size=n, replace=False) for _ in range(sample_n)]
        )
        for idx in samples:
            tasks.append((task_X[idx], task_y[idx]))
            task_order.append(k)

    # printing lists
    # Shuffle two lists with same order
    # Using zip() + * operator + shuffle()
    temp = list(zip(tasks, task_order))
    random.shuffle(temp)
    tasks, task_order = zip(*temp)
    ## res1 and res2 come out as tuples, and so must be converted to lists.
    tasks, task_order = list(tasks), list(task_order)

    # ids =np.random.randint(0, len(tasks), len(tasks)).reshape([-1]).tolist()
    # print(ids)
    # tasks = tasks[ids]
    # task_order = task_order[ids]
    for loc, k in enumerate([1, 0, 2]):
        task_X = X[y == k]
        task_y = y[y == k]
        import numpy as np

        low, high, n = 0, task_y.shape[0], n_samples
        sample_n = int(high / n)
        rng = np.random.default_rng()  # or previously instantiated RNG
        space = np.array([i for i in range(high)]).reshape([-1])
        samples = np.stack([rng.choice(space, size=n, replace=False) for _ in range(1)])
        for idx in samples:
            tasks.insert(loc, (task_X[idx], task_y[idx]))
            task_order.insert(loc, k)
            break
    return tasks, task_order
