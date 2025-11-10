import torch

from typing import Dict


from torch.utils.data import DataLoader

from src.evaluation.evaluation import test
from src.config.configuration import Config
from examples.MNIST.data_utils import MyDataset


from src.model.torch_model_harness import BaseModelHarness


def continual_learning_loop(cfg: Config, modelHarness: BaseModelHarness):

    # 1) select the right cl update method #TODO

    # 2) Get loaders
    hist_train_iter = iter(modelHarness.get_hist_train_loader())
    train_iter = iter(modelHarness.get_train_loader())
    criterion = modelHarness.get_criterion()
    model = modelHarness.model
    optimizer = modelHarness.get_optmizer()
    batch_size = cfg.train.batch_size

    # Generic "safe next" for any iterator/loader pair
    def _safe_next(current_iter, make_loader, min_batch=None):
        """
        Returns (possibly-updated-iter, batch) guaranteeing:
          - iterator restarts on StopIteration
          - optional min batch-size requirement (on y) if provided
        """
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(make_loader())
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, batch

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, batch
                # else: too small → loop to fetch a new batch/iterator
            except Exception:
                # If we cannot inspect batch size, just accept the batch
                return current_iter, batch

    # 2) run the outer loop
    for iter_count in range(cfg.continuous_learning.total_updates):
        # Fetch valid batches from both streams
        train_iter, train_batch = _safe_next(
            train_iter, modelHarness.get_train_loader, min_batch=batch_size
        )
        hist_train_iter, hist_batch = _safe_next(
            hist_train_iter, modelHarness.get_hist_train_loader, min_batch=batch_size
        )

        step_method_bcl(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
            iter=iter_count,
            train_batch=train_batch,
            hist_batch=hist_batch,
        )

    return 0
