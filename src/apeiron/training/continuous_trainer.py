from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apeiron.config.configuration import Config
from apeiron.distributed import comm
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from apeiron.training.updater.create_updater import create_updater


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

    def _allreduce_gradients(self) -> None:
        """Average parameter gradients across ranks (manual data parallel).

        No-op in a single-process run. Called after backward and any updater
        regularization gradients, before ``optimizer.step()``.
        """
        if not comm.is_distributed:
            return
        for p in self.modelHarness.model.parameters():
            if p.grad is not None:
                comm.all_reduce_mean_(p.grad)

    def _safe_next(
        self,
        current_iter: Iterator,
        loader: DataLoader,
        min_batch: Optional[int] = None,
    ) -> tuple[Iterator, list[torch.Tensor]]:
        """Get the next batch, restarting on exhaustion and preferring a full batch.

        When ``min_batch`` is set, undersized batches (e.g. the trailing partial
        batch of an epoch) are skipped in favour of one with at least
        ``min_batch`` samples. If the loader simply *cannot* produce one -- the
        split has fewer than ``min_batch`` samples -- the search is bounded to
        roughly one epoch and the largest batch seen is returned. Without that
        bound a split smaller than ``train.batch_size`` sends this loop spinning
        forever; degrading to a smaller step is the safe fallback.
        """
        best_batch: Optional[list[torch.Tensor]] = None
        best_size = -1
        attempts = 0
        try:
            # One epoch of headroom: enough to skip the lone trailing partial in
            # a healthy loader, but a hard ceiling when no full batch exists.
            max_attempts = max(1, len(loader))
        except TypeError:  # loader without a known length
            max_attempts = 128

        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, [b.to(self.cfg.device) for b in batch]

            # Inspect the batch's sample count via the label tensor.
            try:
                y = batch[1]
                size = (
                    int(y.shape[0]) if getattr(y, "shape", None) is not None else None
                )
            except (IndexError, TypeError):
                size = None

            if size is None or size >= min_batch:
                # Cannot inspect the size (accept it) or it is already big enough.
                return current_iter, [b.to(self.cfg.device) for b in batch]

            # Undersized: remember the largest seen and bound the search so a
            # split with fewer than min_batch samples cannot spin forever.
            if size > best_size:
                best_size, best_batch = size, batch
            attempts += 1
            if attempts >= max_attempts and best_batch is not None:
                return current_iter, [b.to(self.cfg.device) for b in best_batch]

    def _log_validation(
        self,
        tag: str,
        cur: Optional[list],
        hist: Optional[list],
        drift_event_id: int,
    ) -> None:
        """Record every eval metric by name, for both domains.

        ``eval()`` returns a positional list; ``eval_metrics`` holds the labels
        in the same order.
        """
        logger = self.logger
        names = self.modelHarness.eval_metrics
        payload: dict[str, float] = {"drift_event_id": drift_event_id}
        for domain, values in (("cur", cur), ("hist", hist)):
            for name, value in zip(names, values or ()):
                payload[f"val_{tag}_{domain}_{name}"] = float(value)
        logger.stage("eval")
        # increment=False: annotate the CL round's step, do not advance it.
        logger.log(payload, commit=False, increment=False)

    def compute_bwt(self, metric_index: int = 0) -> Optional[float]:
        """Backward transfer for the task that just finished adapting.

        ``BWT = (1/(T-1)) * sum_{i<T} ( R[T][i] - R[i][i] )``

        where a *task* is one drift event, ``R[T][i]`` is the current model's
        score on task ``i``'s validation split and ``R[i][i]`` is the score on
        that same split recorded right after adapting to it. It therefore
        compares one task across two model states -- the definition of
        forgetting -- rather than comparing different tasks at one state.

        Sign follows the raw metric, so the reading depends on the metric's
        direction: with a higher-is-better metric (accuracy) negative means
        forgetting, while with a lower-is-better one (SLAC-FEL's MAE) *positive*
        means forgetting.

        :param metric_index: which entry of ``eval_metrics`` to use, matching the
            index behind ``test_curr_acc``/``test_hist_acc``.
        :type metric_index: int

        :return: BWT over the retained past tasks, or None before any task has
            been registered (the first drift event, where the sum is empty).
        :rtype: Optional[float]
        """
        past_metrics = self.modelHarness.eval_past_tasks()
        if not past_metrics:
            return None

        diagonals = self.modelHarness.task_diagonals
        deltas = [
            row[metric_index] - diagonal[metric_index]
            for row, diagonal in zip(past_metrics, diagonals)
        ]
        return sum(deltas) / len(deltas)

    def outer_cl_training_loop(
        self,
        drift_event_id: int = 0,
    ) -> int:
        """Run the outer continuous learning training loop for a drift event."""
        logger = self.logger
        # Data-parallel training keeps ranks in sync via gradient all-reduce, but
        # that only holds if they start identical -- so broadcast rank 0's weights
        # before adapting (no-op single-process).
        comm.broadcast_module_(self.modelHarness.model)
        cur_train_loader, cur_test_loader = self.modelHarness.get_train_dataloaders()
        hist_train_loader, hist_test_loader = self.modelHarness.get_hist_dataloaders()

        train_iter = iter(cur_train_loader)
        if hist_train_loader is not None:
            hist_train_iter = iter(hist_train_loader)
        else:
            hist_train_iter = None

        cur_validation_metrics = self.modelHarness.eval()
        hist_validation_metrics = self.modelHarness.history_eval()

        # R[i-1][i]: this window scored by the model that has not yet adapted to
        # it. Kept for the FWT delta once the post-CL score (R[i][i]) is in.
        pre_cl_validation_metrics = cur_validation_metrics

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
        self._log_validation(
            "post", cur_validation_metrics, hist_validation_metrics, drift_event_id
        )

        logger.info(f"\tTest Accuracy: {cur_validation_metrics[0]:.1f}%", level=1)
        if hist_validation_metrics is not None:
            logger.info(
                f"\tHist Test Accuracy: {hist_validation_metrics[0]:.1f}%",
                level=1,
            )

        else:
            logger.info("\tNo historical data available for evaluation", level=1)

        # FWT = R[i][i] - R[i-1][i]: how much adapting to this window moved the
        # score on it, i.e. the gain CL delivered on the task that triggered.
        # Available at every drift event, including the first.
        fwt = cur_validation_metrics[0] - pre_cl_validation_metrics[0]
        logger.info(f"\tFWT: {fwt:.4g}", level=1)

        bwt = self.compute_bwt()
        if bwt is not None:
            logger.info(f"\tBWT: {bwt:.4g}", level=1)

        logger.stage("eval")
        eval_metrics: dict[str, float] = {
            "test_curr_acc": cur_validation_metrics[0],
            "test_pre_cl_acc": pre_cl_validation_metrics[0],
            "fwt": fwt,
        }
        if hist_validation_metrics is not None:
            eval_metrics["test_hist_acc"] = hist_validation_metrics[0]
        if bwt is not None:
            eval_metrics["bwt"] = bwt
        logger.log(eval_metrics, commit=False)

        # Hand the round's metrics to the harness so checkpoint retention and
        # promotion rules can select by quality (e.g. best historical accuracy)
        # rather than only by recency.
        self.modelHarness.last_metrics = dict(eval_metrics)

        # Register *after* BWT so this event's window becomes task T only for
        # subsequent events -- R[T][T] belongs on the diagonal, not in the sum.
        self.modelHarness.register_task(cur_validation_metrics)

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

        # Data-parallel step: average the (accumulated) gradients across ranks
        # before the optimizer applies them, so every rank takes the same step on
        # an effective batch of world_size * per-rank samples. No-op single-process.
        self._allreduce_gradients()

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
