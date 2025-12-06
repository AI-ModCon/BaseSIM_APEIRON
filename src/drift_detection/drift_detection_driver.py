import torch

from config.configuration import Config
from drift_detection.detectors.base import DriftSignal
from drift_detection.load_drift_detector import load_drift_detector
from model.torch_model_harness import BaseModelHarness
from profilers import FLOPSProfiler

def drift_detection_driver(
    cfg: Config, modelHarness: BaseModelHarness, logger, global_step=0
) -> DriftSignal:
    cur_train_loader, cur_test_loader = modelHarness.get_cur_data_loaders()
    criterion = modelHarness.get_criterion()
    model = modelHarness.model
    model.eval()

    detector = load_drift_detector(cfg)

    train_iter = iter(cur_train_loader)

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

    for iter_count in range(cfg.drift_detection.detection_steps):
        train_iter, batch = _safe_next(
            train_iter, cur_train_loader, min_batch=cfg.train.batch_size
        )
        inputs, targets = batch

        if (
            iter_count > flops_profiler.warmup_iters
        ):  # Give warmup iterations, for accuracy.
            with flops_profiler.measure_flops(tag="infer"):
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

            # NOTE: Profiler only covers Pytorch operations
            # Even so, we measure runtime to see if its a potential bottleneck
            with flops_profiler.measure_flops(tag="detector"):
                drift_signal = detector.update(loss.item())
                any_drift = drift_signal.drift_detected
                if any_drift:
                    break
        else:
            # Get model predictions
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Update the drift detector with new data
            drift_signal = detector.update(loss.item())
            any_drift = drift_signal.drift_detected
            if any_drift:
                break

    # -
    logger.log(
        {"drift/regime": drift_signal.regime.value}, step=global_step, commit=False
    )
    logger.log(
        {"drift/detected": drift_signal.drift_detected}, step=global_step, commit=False
    )
    logger.log(
        {"drift/score": drift_signal.drift_score}, step=global_step, commit=False
    )
    logger.log(
        {"drift/confidence": drift_signal.confidence}, step=global_step, commit=False
    )

    # -
    if flops_profiler:
        flops_perf = flops_profiler.get_performance()
        flops_profiler.print_performance()
        logger.log(
            {f"drift/cperf/{k}": v for k, v in flops_perf.items()},
            step=global_step,
            commit=False,
        )

    return drift_signal
