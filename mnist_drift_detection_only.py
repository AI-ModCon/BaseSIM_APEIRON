"""
MNIST Drift Detection Example.

This script demonstrates drift detection on MNIST by:
1. Training on classes 0-4
2. Running 2 epochs with 10 evaluations per epoch
3. Switching test sets after epoch 1 to simulate distribution drift
4. Using drift detectors to identify the distribution shift

Usage:
    python drift_detection_demo_mnist.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Tuple, Union
from torch.utils.data import DataLoader
from src.drift_detection import (
    ADWINDetector,
    ModelPerformanceDetector,
    DriftSignal,
    LearningRegime,
)
from examples.mnist.model import Cnn
from examples.mnist.utils import (
    get_mnist_data,
    filter_mnist_by_classes,
    split_train_test,
    CustomMnistData,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_features: bool = False,
) -> Union[
    Tuple[float, float], Tuple[float, float, pd.DataFrame, np.ndarray, np.ndarray]
]:
    """
    Evaluate model on the test set.

    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        return_features: If True, also return features and predictions for drift detection

    Returns:
        tuple: (accuracy, average_loss) or (accuracy, average_loss, features_df, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if return_features:
        all_images = []
        all_predictions = []
        all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_features:
                all_images.append(images.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    if return_features:
        images_array = np.concatenate(all_images, axis=0)
        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)
        images_flat = images_array.reshape(images_array.shape[0], -1)
        n_pixels = images_flat.shape[1]
        # Sample pixels to reduce dimensionality (use every 8th pixel)
        pixel_indices = np.arange(0, n_pixels, 8)
        sampled_pixels = images_flat[:, pixel_indices]
        # Add small jitter to avoid zero-variance features
        sampled_pixels_jittered = sampled_pixels + np.random.normal(
            0, 1e-8, sampled_pixels.shape
        )
        # Create DataFrame with pixel features
        feature_cols = [f"pixel_{i}" for i in range(sampled_pixels_jittered.shape[1])]
        features_df = pd.DataFrame(sampled_pixels_jittered, columns=feature_cols)
        return accuracy, avg_loss, features_df, predictions_array, targets_array

    return accuracy, avg_loss


def batch_train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    print("""================================================================================
MNIST Drift Detection Example.

This script demonstrates drift detection on MNIST by:
1. Training on classes 0-4
2. Running 2 epochs with 10 evaluations per epoch
3. Switching test sets after epoch 1 to simulate distribution drift
4. Using drift detectors to identify the distribution shift

Usage:
    python drift_detection_demo_mnist.py
================================================================================
""")
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 2
    CHANGE_EPOCH = NUM_EPOCHS // 2  # Switch test sets after half the epochs
    EVALS_PER_EPOCH = 10
    TRAIN_CLASSES = [0, 1, 2, 3, 4]  # Training on classes 0-4
    TEST1_CLASSES = [0, 1, 2, 3, 4]  # First test set: classes 0-4
    TEST2_CLASSES = [5, 6, 7, 8, 9]  # Second test set: classes 5-9 (drift!)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MNIST data...")
    images, labels = get_mnist_data()
    (train_images, train_labels), (test_images, test_labels) = split_train_test(
        images, labels, train_ratio=0.8
    )
    print(f"Filtering training data which only has classes {TRAIN_CLASSES}...")
    train_images, train_labels = filter_mnist_by_classes(
        train_images, train_labels, TRAIN_CLASSES
    )

    # Filter test set 1 (classes 0-4)
    print(f"Filtering test set 1 which only has classes {TEST1_CLASSES}...")
    test1_images, test1_labels = filter_mnist_by_classes(
        test_images, test_labels, TEST1_CLASSES
    )
    # Filter test set 2 (classes 5-9)
    print(f"Filtering test set 2 which only has classes {TEST2_CLASSES}...")
    test2_images, test2_labels = filter_mnist_by_classes(
        test_images, test_labels, TEST2_CLASSES
    )

    print(
        f"\nDataset sizes:"
        f"\n  Training Set (classes {TRAIN_CLASSES}): {len(train_images)} images"
        f"\n  Test Set 1   (classes {TEST1_CLASSES}): {len(test1_images)} images"
        f"\n  Test Set 2   (classes {TEST2_CLASSES}): {len(test2_images)} images\n"
    )

    # Create data loaders
    train_dataset = CustomMnistData(train_images, train_labels)
    test1_dataset = CustomMnistData(test1_images, test1_labels)
    test2_dataset = CustomMnistData(test2_images, test2_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test1_loader = DataLoader(test1_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test2_loader = DataLoader(test2_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer
    model = Cnn().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize drift detectors
    # Feature-based detector for monitoring image/prediction distributions
    feature_detector = ModelPerformanceDetector(
        drift_share_threshold=0.2,  # Trigger if 20% of features drift
        minor_threshold=0.3,
        moderate_threshold=0.6,
        name="Feature Drift Detector",
    )

    # Keep one simple statistical detector for comparison - monitoring test loss changes
    # Note: This simple detector cannot detect significant changes in loss, but serves as a baseline for comparison
    loss_detector = ADWINDetector(
        delta=0.01,
        minor_threshold=0.1,
        moderate_threshold=0.3,
        name="Loss Drift (ADWIN)",
    )

    reference_set = False

    print("=" * 80)
    print("Starting training and drift detection...")
    print("=" * 80)
    print(
        f"Training for {NUM_EPOCHS} epochs with {EVALS_PER_EPOCH} evaluations per epoch"
    )
    print(f"Epochs 1: Evaluate on test set 1 (classes {TEST1_CLASSES})")
    print(f"Epochs 2: Evaluate on test set 2 (classes {TEST2_CLASSES})")
    print("\nDetectors:")
    print(f"  - {feature_detector.name}: Monitors pixel feature distributions")
    print(f"  - {loss_detector.name}: Monitors test loss changes")
    print("\nExpected: Feature drift should be detected when switching to test set 2\n")

    # Training loop
    total_iterations = 0
    batches_per_eval = len(train_loader) // EVALS_PER_EPOCH

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'=' * 80}")

        if epoch <= CHANGE_EPOCH:
            test_loader = test1_loader
            print(f"Evaluating on: Test Set 1 (classes {TEST1_CLASSES})\n")
        else:
            test_loader = test2_loader
            print(f"Evaluating on: Test Set 2 (classes {TEST2_CLASSES})\n")

        for eval_num in range(1, EVALS_PER_EPOCH + 1):
            train_losses = []
            for _ in range(batches_per_eval):
                loss = batch_train(model, train_loader, criterion, optimizer, device)
                train_losses.append(loss)
                total_iterations += 1

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Evaluate on test set with features for drift detection
            test_acc, test_loss, features_df, predictions, targets = evaluate_model(
                model, test_loader, criterion, device, return_features=True
            )

            if epoch == 1 and eval_num == 1:
                if not reference_set:
                    feature_detector.set_reference(
                        data=features_df, predictions=predictions, targets=targets
                    )
                    reference_set = True
                    print(
                        f"[INFO] Set reference data: {len(features_df)} samples, {len(features_df.columns)} features"
                    )

            if reference_set:
                feature_signal = feature_detector.update(
                    data=features_df, predictions=predictions, targets=targets
                )
            else:
                feature_signal = DriftSignal(
                    regime=LearningRegime.STABLE,
                    drift_detected=False,
                    drift_score=0.0,
                )

            loss_signal = loss_detector.update(test_loss)
            any_drift = feature_signal.drift_detected or loss_signal.drift_detected

            feature_info = ""
            if reference_set and "drift_share" in feature_signal.metadata:
                drift_share = feature_signal.metadata["drift_share"]
                n_drifted = feature_signal.metadata.get("n_drifted_columns", 0)
                feature_info = (
                    f"Feature Drift Share: {drift_share:.2%} ({n_drifted} cols) | "
                )

            print(
                f"Eval {eval_num:2d}/{EVALS_PER_EPOCH} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:6.2f}% | "
                f"{feature_info}"
                f"Loss Drift Score: {loss_signal.drift_score:.3f}"
            )

            # Check for drift
            if any_drift:
                print(f"🚨 DRIFT DETECTED at Epoch {epoch}, Eval {eval_num}")
                if feature_signal.drift_detected:
                    drift_share = feature_signal.metadata.get("drift_share", 0)
                    n_drifted = feature_signal.metadata.get("n_drifted_columns", 0)
                    n_total = feature_signal.metadata.get("n_columns", 0)
                    print(
                        f"[{feature_detector.name}] "
                        f"Recommended Regime: {feature_signal.regime.value} | "
                        f"Drift Share: {drift_share:.2%} | "
                        f"Drifted Features: {n_drifted}/{n_total} | "
                        f"Confidence: {feature_signal.confidence:.4f}"
                    )

                if loss_signal.drift_detected:
                    print(
                        f"[{loss_detector.name}] "
                        f"Recommended Regime: {loss_signal.regime.value} | "
                        f"Score: {loss_signal.drift_score:.4f} | "
                        f"Confidence: {loss_signal.confidence:.4f}"
                    )

    # Final summary
    print(f"\n{'=' * 80}")
    print("Experiment Complete")
    print(f"{'=' * 80}")

    feature_detections = (
        sum(
            1
            for drift_share in feature_detector._drift_history
            if drift_share > feature_detector.drift_share_threshold
        )
        if feature_detector._drift_history
        else 0
    )

    print(
        f"\nDrift Detection Summary:"
        f"\n  [{feature_detector.name}] Total detections: {feature_detections}"
        f"\n  [{loss_detector.name}] Total detections: {sum(loss_detector._drift_history)}"
    )

    if feature_detector._drift_history:
        avg_drift_share = np.mean(feature_detector._drift_history)
        max_drift_share = np.max(feature_detector._drift_history)
        print(
            f"\nFeature Drift Statistics:"
            f"\n  Average drift share: {avg_drift_share:.2%}"
            f"\n  Maximum drift share: {max_drift_share:.2%}"
        )
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
