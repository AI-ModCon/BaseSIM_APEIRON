from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from apeiron.training.updater.create_updater import create_updater
from apeiron.logger import get_logger


class ContinuousTrainer:
    """Trainer for continuous/continual learning with drift handling."""

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        logger: Any,
        profiler: Optional[FLOPSProfiler],
    ) -> None:
        """Initialize the continuous trainer with config, model, logger, and profiler."""
        self.modelHarness = modelHarness
        self.cfg = cfg
        self.logger = logger

        self.profiler = profiler
        self.criterion = modelHarness.get_criterion()
        self.optimizer = modelHarness.get_optmizer()

        self.cl_updater = create_updater(cfg=self.cfg, modelHarness=self.modelHarness)

    def _safe_next(
        self,
        current_iter: Iterator,
        loader: DataLoader,
        min_batch: Optional[int] = None,
    ) -> tuple[Iterator, list[torch.Tensor]]:
        """Get next batch from iterator, restarting on exhaustion and enforcing min batch size."""
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, [b.to(self.cfg.device) for b in batch]

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, [b.to(self.cfg.device) for b in batch]
            except (IndexError, TypeError):
                # If we cannot inspect batch size, just accept the batch
                return current_iter, [b.to(self.cfg.device) for b in batch]

    def outer_cl_training_loop(
        self,
        drift_event_id: int = 0,
    ) -> int:
        """Run the outer continuous learning training loop for a drift event."""
        logger = get_logger(__name__)
        cur_train_loader, cur_test_loader = self.modelHarness.get_cur_data_loaders()
        hist_train_loader, hist_test_loader = self.modelHarness.get_hist_data_loaders()

        train_iter = iter(cur_train_loader)
        if hist_train_loader is not None:
            hist_train_iter = iter(hist_train_loader)
        else:
            hist_train_iter = None

        # TODO: need to find away to explicitly match the metrics to their name/label
        cur_validation_metrics = self.modelHarness.eval()
        hist_validation_metrics = self.modelHarness.history_eval()

        logger.info("==== Continual Learning ====")
        logger.info("\tInitial test acc: {}".format(cur_validation_metrics[0]), level=1)
        if hist_validation_metrics is not None:
            logger.info(
                "\tInitial historical test acc: {}".format(hist_validation_metrics[0]),
                level=1,
            )
        else:
            logger.info("\tNo historical data available for evaluation", level=1)

        self.modelHarness.model.train()
        # 2) run the outer loop
        desc = "CL Updates (drift_event_id={})".format(drift_event_id)
        progress_bar = tqdm(range(self.cfg.train.max_iter), desc=desc, leave=True)
        self.cl_updater.cl_preprocessing()

        iter_count = self.cfg.train.max_iter
        if self.cl_updater is not None:  # default: do nothing
            for iter_count in progress_bar:
                generation_loss, forgetting_loss = self.inner_cl_training_loop(
                    iter_count=iter_count,
                    cur_train_loader=cur_train_loader,
                    train_iter=train_iter,
                    hist_train_loader=hist_train_loader,
                    hist_train_iter=hist_train_iter,
                )

                logger.stage("cl")
                logger.log(
                    {
                        "jvp_reg_total_loss": generation_loss + forgetting_loss,
                        "jvp_reg_forgetting_loss": forgetting_loss,
                        "jvp_reg_generation_loss": generation_loss,
                        "drift_event_id": drift_event_id,
                    },
                    # commit=iter_count < (cfg.continuous_learning.max_iter - 1),
                )

                # Explicitly cleanup batch tensors to free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.cl_updater.cl_postprocessing()

        cur_validation_metrics = self.modelHarness.eval()
        hist_validation_metrics = self.modelHarness.history_eval()

        logger.info(f"\tTest Accuracy: {cur_validation_metrics[0]:.1f}%", level=1)
        if hist_validation_metrics is not None:
            logger.info(
                f"\tHist Test Accuracy: {hist_validation_metrics[0]:.1f}%",
                level=1,
            )

        else:
            logger.info("\tNo historical data available for evaluation", level=1)

        logger.stage("eval")
        if hist_validation_metrics is not None:
            logger.log(
                {
                    "test_curr_acc": cur_validation_metrics[0],
                    "test_hist_acc": hist_validation_metrics[0],
                },
                commit=False,
            )
        else:
            logger.log(
                {
                    "test_curr_acc": cur_validation_metrics[0],
                },
                commit=False,
            )

        if self.profiler:
            flops_perf = self.profiler.get_performance()
            self.profiler.print_performance()
            logger.stage("cl")
            self.logger.log(
                {
                    **{f"cperf_{k}": v for k, v in flops_perf.items()},
                },
            )

        return 0

    def inner_cl_training_loop(
        self,
        iter_count: int,
        cur_train_loader: DataLoader,
        train_iter: Iterator,
        hist_train_loader: Optional[DataLoader] = None,
        hist_train_iter: Optional[Iterator] = None,
    ) -> tuple[float, float]:
        """Run a single inner training iteration with forward/backward and optimizer step."""
        self.optimizer.zero_grad()
        self.cl_updater.update_pre_fwd_bwd()

        # Forward and backward
        loss = 0.0
        for step in range(self.cfg.train.grad_accumulation_steps):
            train_iter, train_batch = self._safe_next(
                train_iter,
                cur_train_loader,
                min_batch=self.cfg.train.batch_size,
            )
            if hist_train_iter is not None and hist_train_loader is not None:
                hist_train_iter, hist_train_batch = self._safe_next(
                    hist_train_iter,
                    hist_train_loader,
                    min_batch=self.cfg.train.batch_size,
                )
            else:
                hist_train_batch = None

            # Cast batches to tuple type expected by fwd_bwd
            train_batch_tuple = (train_batch[0], train_batch[1])
            hist_batch_tuple = (
                (hist_train_batch[0], hist_train_batch[1])
                if hist_train_batch is not None
                else None
            )

            # Run profiler for forward and backward after warmup for one of the grad acc steps.
            if self.profiler and iter_count > self.profiler.warmup_iters and step == 0:
                with self.profiler.measure_flops(tag="update_fwd_bwd"):
                    loss += self.cl_updater.fwd_bwd(train_batch_tuple, hist_batch_tuple)
            else:
                loss += self.cl_updater.fwd_bwd(train_batch_tuple, hist_batch_tuple)

        reg_loss = self.cl_updater.update_post_fwd_bwd()

        # 3) Update with optimizer
        if self.profiler and iter_count > self.profiler.warmup_iters:
            with self.profiler.measure_flops_optimizer(
                tag="optimizer", model=self.modelHarness.model, device=self.cfg.device
            ):
                self.optimizer.step()
        else:
            self.optimizer.step()

        self.cl_updater.update_post_optimizer_call()

        return loss, reg_loss
