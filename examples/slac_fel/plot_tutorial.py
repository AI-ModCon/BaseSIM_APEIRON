"""Reproduce the focused SLAC-FEL drift-adaptation figure."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch

from examples.slac_fel.model import FELNet
from examples.slac_fel.utils import (
    discover_window_files,
    load_feature_config,
    load_scalers,
    load_window_file,
)

TRIGGER_FILE = "data_740.pkl"
ROLLING_SAMPLES = 500


def load_adapted_model(
    checkpoint_path: Path,
    input_size: int,
    output_size: int,
) -> FELNet:
    """Load one drift-adapted FEL checkpoint."""
    model = FELNet(
        input_size=input_size,
        output_size=output_size,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(state)
    model.eval()
    return model


def build_predictions(
    data_dir: Path,
    model_dir: Path,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Evaluate static and adapted models on the focused stream."""
    input_columns, output_columns = load_feature_config(str(model_dir))

    input_scaler, output_scaler = load_scalers(
        str(model_dir),
        device="cpu",
    )

    # BoTorch exposes this as a method at runtime, but its current
    # type information describes the attribute as a Tensor.
    untransform = cast(
        Callable[[torch.Tensor], torch.Tensor],
        output_scaler.untransform,
    )

    loaded_static_model = torch.load(
        model_dir / "final_lcls_fel_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        loaded_static_model,
        torch.nn.Module,
    ):
        raise TypeError("The pretrained FEL artifact is not a Torch module")

    static_model = loaded_static_model
    static_model.eval()

    adapted_model = load_adapted_model(
        checkpoint_path,
        input_size=len(input_columns),
        output_size=len(output_columns),
    )

    records = []
    trigger_time = None

    for path_string in discover_window_files(str(data_dir)):
        data_path = Path(path_string)

        inputs, targets_scaled, timestamps = load_window_file(
            str(data_path),
            input_columns,
            output_columns,
            input_scaler,
            output_scaler,
        )

        with torch.no_grad():
            measured = untransform(targets_scaled)
            static_prediction = untransform(static_model(inputs))
            adapted_prediction = untransform(adapted_model(inputs))

        use_adapted = data_path.name >= TRIGGER_FILE

        if data_path.name == TRIGGER_FILE:
            trigger_time = pd.Timestamp(timestamps[0])

        segmented_prediction = adapted_prediction if use_adapted else static_prediction

        frame = pd.DataFrame(
            {
                "measurement": (measured.squeeze(-1).cpu().numpy()),
                "static_prediction": (static_prediction.squeeze(-1).cpu().numpy()),
                "segmented_prediction": (
                    segmented_prediction.squeeze(-1).cpu().numpy()
                ),
                "file": data_path.name,
            },
            index=pd.DatetimeIndex(timestamps),
        )

        records.append(frame)

    if trigger_time is None:
        raise ValueError(f"Trigger file {TRIGGER_FILE} was not found")

    result = pd.concat(records).sort_index()

    result["static_absolute_error"] = (
        result["static_prediction"] - result["measurement"]
    ).abs()

    result["segmented_absolute_error"] = (
        result["segmented_prediction"] - result["measurement"]
    ).abs()

    return result, trigger_time


def plot_focused_replay(
    predictions: pd.DataFrame,
    trigger_time: pd.Timestamp,
    output_path: Path,
) -> None:
    """Plot measured intensity, predictions, and absolute error."""
    rolling = (
        predictions[
            [
                "measurement",
                "static_prediction",
                "segmented_prediction",
                "static_absolute_error",
                "segmented_absolute_error",
            ]
        ]
        .rolling(
            ROLLING_SAMPLES,
            min_periods=1,
        )
        .mean()
    )

    static_mae = predictions["static_absolute_error"].mean()

    segmented_mae = predictions["segmented_absolute_error"].mean()

    improvement = static_mae / segmented_mae

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    signal_axis, error_axis = axes

    signal_axis.scatter(
        predictions.index,
        predictions["measurement"],
        color="#9ca3af",
        s=2,
        alpha=0.08,
        rasterized=True,
        label="Measured pulse energy (raw)",
    )

    signal_axis.plot(
        rolling.index,
        rolling["measurement"],
        color="#111827",
        linewidth=1.6,
        label="Measured pulse energy (rolling mean)",
    )

    signal_axis.plot(
        rolling.index,
        rolling["static_prediction"],
        color="#3b82f6",
        linewidth=1.5,
        label="Static pretrained model",
    )

    signal_axis.plot(
        rolling.index,
        rolling["segmented_prediction"],
        color="#e11d48",
        linewidth=1.8,
        label="Paper-style segmented replay",
    )

    error_axis.plot(
        rolling.index,
        rolling["static_absolute_error"],
        color="#3b82f6",
        linewidth=1.4,
        label="Static absolute error",
    )

    error_axis.plot(
        rolling.index,
        rolling["segmented_absolute_error"],
        color="#e11d48",
        linewidth=1.6,
        label="Segmented absolute error",
    )

    for axis in axes:
        axis.axvline(
            trigger_time,
            color="#f97316",
            linestyle="--",
            linewidth=1.6,
            label=("Detected transition" if axis is signal_axis else None),
        )
        axis.grid(alpha=0.25)

    signal_axis.annotate(
        "Detected transition\ndata_740.pkl",
        xy=(trigger_time, 0.98),
        xycoords=(
            "data",
            "axes fraction",
        ),
        xytext=(8, -8),
        textcoords="offset points",
        color="#c2410c",
        fontsize=9,
        ha="left",
        va="top",
    )

    signal_axis.set_title("APEIRON SLAC-FEL focused drift-adaptation tutorial")
    signal_axis.set_ylabel("FEL pulse energy (mJ)")
    signal_axis.legend(
        loc="upper left",
        ncol=2,
    )

    error_axis.set_ylabel("Absolute error (mJ)")
    error_axis.set_xlabel("Time")
    error_axis.legend(loc="upper left")

    locator = mdates.AutoDateLocator(
        minticks=4,
        maxticks=9,
    )
    formatter = mdates.ConciseDateFormatter(locator)

    error_axis.xaxis.set_major_locator(locator)
    error_axis.xaxis.set_major_formatter(formatter)

    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.94,
        bottom=0.13,
        hspace=0.06,
    )

    figure.text(
        0.5,
        0.065,
        (
            f"Static MAE: {static_mae:.3f} mJ   "
            f"Segmented replay MAE: "
            f"{segmented_mae:.3f} mJ   "
            f"Improvement: {improvement:.1f}x"
        ),
        fontsize=11,
        ha="center",
    )

    figure.text(
        0.5,
        0.025,
        (
            "Paper-style offline replay: the adapted checkpoint is "
            "applied from the start of the trigger window."
        ),
        color="#4b5563",
        fontsize=9,
        ha="center",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)

    print("Samples:", len(predictions))
    print("Trigger time:", trigger_time)
    print("Static MAE:", static_mae, "mJ")
    print(
        "Segmented replay MAE:",
        segmented_mae,
        "mJ",
    )
    print("Improvement:", improvement, "x")
    print("Figure:", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Generate the focused SLAC-FEL tutorial figure.")
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing data_730.pkl through data_743.pkl.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Adapted drift_adaptation_1.pt checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("examples/slac_fel/model"),
        help="Directory containing the pretrained model and scalers.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the focused replay and plotting workflow."""
    args = parse_args()

    predictions, trigger_time = build_predictions(
        data_dir=args.data,
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
    )

    plot_focused_replay(
        predictions=predictions,
        trigger_time=trigger_time,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
