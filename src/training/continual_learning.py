import torch
from evaluation.evaluation import test
from config.configuration import Config
from model.torch_model_harness import BaseModelHarness
from training.updaters.basic import step_method_baseline

from training.updaters.jvp_reg import step_method_jvp_reg, JVPRegularizedLoss

from profilers import FLOPSProfiler
from tqdm import tqdm


def continual_learning_loop(
    cfg: Config,
    modelHarness: BaseModelHarness,
    logger,
    global_step=0,
    basic_only=False,  # Needed to test drift_detection, will remove in future PR.
    drift_event_id: int = 0,
):
    # 1) select the right cl update method #TODO

    # 2) Get loaders
    cur_train_loader, cur_test_loader = modelHarness.get_cur_data_loaders()
    hist_train_loader, hist_test_loader = modelHarness.get_hist_data_loaders()

    train_iter = iter(cur_train_loader)
    if hist_train_loader is not None:
        hist_train_iter = iter(hist_train_loader)
    else:
        hist_train_iter = None

    criterion = modelHarness.get_criterion()
    model = modelHarness.model
    optimizer = modelHarness.get_optmizer()
    batch_size = cfg.train.batch_size

    # JVP continual learning setup
    # Should be done outside of update call to keep optimizer state.
    jvp_loss = JVPRegularizedLoss(
        model=model,
        criterion=criterion,
        jvp_reg=cfg.continuous_learning.jvp_reg,
        deltax_norm=cfg.continuous_learning.deltax_norm,
    )

    # Generic "safe next" for any iterator/loader pair
    def _safe_next(current_iter, loader, min_batch=None):
        """
        Returns (possibly-updated-iter, batch) guaranteeing:
          - iterator restarts on StopIteration
          - optional min batch-size requirement (on y) if provided
        """
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                batch = next(current_iter)

            if min_batch is None:
                return current_iter, [b.to(cfg.device) for b in batch]

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, [b.to(cfg.device) for b in batch]
                # else: too small → loop to fetch a new batch/iterator
            except Exception:
                # If we cannot inspect batch size, just accept the batch
                return current_iter, [b.to(cfg.device) for b in batch]

    flops_profiler = FLOPSProfiler()

    validation_metrics = (
        modelHarness.eval()
    )  # TODO: need to find away to explicitly match the metrics to their name/label

    print("Initial test acc:", validation_metrics[0])
    print("-------------")

    # 2) run the outer loop
    progress_bar = tqdm(
        range(cfg.continuous_learning.max_iter), desc="Continuous Learning", leave=True
    )

    for iter_count in progress_bar:
        # Fetch valid batches from both streams
        train_iter, train_batch = _safe_next(
            train_iter, cur_train_loader, min_batch=batch_size
        )

        if hist_train_iter is None or basic_only:
            # Fall back to basic training if no historical data is available

            # - Count Flops
            total_loss = step_method_baseline(
                model=model,
                criterion=criterion,  # type: ignore[arg-type]
                optimizer=optimizer,
                cfg=cfg,
                iter=iter_count,
                train_batch=train_batch,
                profiler=flops_profiler,
            )

            logger.log(
                {
                    "cl/step": iter_count + global_step,
                    "cl/basic/total_loss": total_loss,
                    "cl/drift_event_id": drift_event_id,
                },
                step=iter_count + global_step,
                # commit=iter_count < (cfg.continuous_learning.max_iter - 1),
            )

        else:
            hist_train_iter, hist_batch = _safe_next(
                hist_train_iter,
                hist_train_loader,
                min_batch=batch_size,
            )

            # - Count Flops
            forgetting_loss, generation_loss, total_loss = step_method_jvp_reg(
                model=model,
                criterion=criterion,  # type: ignore[arg-type]
                optimizer=optimizer,
                cfg=cfg,
                iter=iter_count,
                train_batch=train_batch,
                hist_batch=hist_batch,
                profiler=flops_profiler,
                jvp_loss=jvp_loss,
            )

            logger.log(
                {
                    "cl/step": iter_count + global_step,
                    "cl/jvp_reg/total_loss": total_loss,
                    "cl/jvp_reg/forgetting_loss": forgetting_loss,
                    "cl/jvp_reg/generation_loss": generation_loss,
                    "cl/drift_event_id": drift_event_id,
                },
                step=iter_count + global_step,
                # commit=iter_count < (cfg.continuous_learning.max_iter - 1),
            )

            # Explicitly cleanup batch tensors to free GPU memory
            del train_batch, hist_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if hist_train_iter is None:
        mem_test_acc = -1

    else:
        mem_test_acc, _ = test(model, hist_test_loader, criterion, cfg=cfg)

    test_acc = modelHarness.eval()[0]

    print(
        "Task Summary:",
        f"Test Acc      : {test_acc:.1f}%",
        f"Hist Test Acc : {mem_test_acc:.1f}%",
        "-" * 40,
        sep="\n",
    )

    # logger.log(
    #     {
    #         "cl/step": iter_count + global_step,
    #         "cl/test_curr/acc": test_acc,
    #         "cl/test_hist/acc": mem_test_acc,
    #     },
    #     step=iter_count + global_step,
    #     commit=False,
    # )

    # if flops_profiler:
    #     flops_perf = flops_profiler.get_performance()
    #     flops_profiler.print_performance()
    #     logger.log(
    #         {
    #             "cl/step": iter_count + global_step,
    #             **{f"cl/cperf/{k}": v for k, v in flops_perf.items()},
    #         },
    #         step=iter_count + global_step,
    #     )

    return 0
