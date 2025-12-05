import torch
from config.configuration import Config
from .detectors.base import DriftSignal
from drift_detection.load_drift_detector import load_drift_detector
from model.torch_model_harness import BaseModelHarness


def drift_detection_driver(
    cfg: Config, modelHarness: BaseModelHarness, logger
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

    for iter_count in range(cfg.drift_detection.detection_steps):
        train_iter, batch = _safe_next(
            train_iter, cur_train_loader, min_batch=cfg.train.batch_size
        )
        inputs, targets = batch

        # Get model predictions

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Update the drift detector with new data
        drift_signal = detector.update(loss.item())
        any_drift = drift_signal.drift_detected
        if any_drift:
            break

    return drift_signal
