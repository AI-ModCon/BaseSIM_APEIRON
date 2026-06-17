from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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

    def _rebuild_with_priorities(
        self,
        loader: DataLoader,
        drift_event_id: int,
        tag: str,
    ) -> DataLoader:
        """Rebuild a DataLoader so its sampler draws by per-sample priority.

        Computes priorities = (L_current − L_anchor)^alpha over the loader's
        dataset, logs the distribution, and returns a new DataLoader wired
        to a WeightedRandomSampler.
        """
        logger = get_logger(__name__)
        priorities = self.cl_updater.compute_sample_priorities(loader, self.cfg.device)

        # Diagnostic: per-round priority distribution. Non-trivial std means
        # theta_star has drifted from the model and prioritized sampling will
        # actually re-weight samples. Near-zero std → sampling collapses to
        # uniform (the bug-fix regression signal).
        with torch.no_grad():
            p = priorities.float()
            p_mean = p.mean().item()
            p_std = p.std().item()
            p_min = p.min().item()
            p_max = p.max().item()
            # ESS = (sum w)^2 / sum w^2, ranges from 1 (degenerate) to N (uniform).
            ess = (p.sum().item() ** 2) / (p.pow(2).sum().item() + 1e-30)
            ess_frac = ess / p.numel()
        logger.info(
            f"[priority/{tag}] drift_event_id={drift_event_id} "
            f"n={p.numel()} min={p_min:.3e} mean={p_mean:.3e} "
            f"max={p_max:.3e} std={p_std:.3e} ess_frac={ess_frac:.3f}"
        )

        sampler = WeightedRandomSampler(
            weights=priorities.tolist(),
            num_samples=len(priorities),
            replacement=True,
        )

        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size or self.cfg.train.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            drop_last=loader.drop_last,
        )

    def outer_cl_training_loop(
        self,
        drift_event_id: int = 0,
    ) -> int:
        """Run the outer continuous learning training loop for a drift event."""
        logger = get_logger(__name__)
        cur_train_loader, cur_test_loader = self.modelHarness.get_train_dataloaders()
        hist_train_loader, hist_test_loader = self.modelHarness.get_hist_dataloaders()

        # Prioritized sampling. Always rebuild the current-task loader with
        # importance-based weights when the gate is on. Additionally rebuild
        # the historical loader for updaters that actually consume hist_batch
        # in fwd_bwd (uses_hist_batch=True — jvp_reg). For EWC/KFAC the
        # historical signal is expected to enter via mixing into the current
        # loader, so reweighting hist_train_loader for them would be wasted
        # work.
        if self.cl_updater.importance_weighting and self.cl_updater.theta_star:
            cur_train_loader = self._rebuild_with_priorities(
                cur_train_loader,
                drift_event_id=drift_event_id,
                tag="cur",
            )
            if self.cl_updater.uses_hist_batch and hist_train_loader is not None:
                hist_train_loader = self._rebuild_with_priorities(
                    hist_train_loader,
                    drift_event_id=drift_event_id,
                    tag="hist",
                )

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
