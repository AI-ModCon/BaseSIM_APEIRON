"""Tests for the SLAC-FEL scientific example."""

from pathlib import Path

import pandas as pd
import torch

from apeiron.config.configuration import build_config
from examples.slac_fel.plot_tutorial import plot_focused_replay
from examples.slac_fel.utils import (
    discover_window_files,
    load_feature_config,
    load_scalers,
    split_into_windows,
)

MODEL_DIR = Path("examples/slac_fel/model")
TUTORIAL_CONFIG = Path("examples/slac_fel/slac-fel-tutorial.toml")


def test_tutorial_configuration():
    cfg = build_config(
        [
            "--config",
            str(TUTORIAL_CONFIG),
        ]
    )

    assert cfg.seed == 42
    assert cfg.device == "cpu"
    assert cfg.data.name == "slac-fel"
    assert cfg.data.batch_size == 5000
    assert cfg.train.max_iter == 200
    assert cfg.drift_detection.detector_name == ("PageHinkleyDetector")
    assert cfg.drift_detection.max_stream_updates == 13
    assert cfg.drift_detection.ph_min_instances == 10
    assert cfg.drift_detection.ph_threshold == 2.0
    assert cfg.drift_detection.reset_after_learning is True


def test_feature_configuration_and_scalers():
    input_columns, output_columns = load_feature_config(str(MODEL_DIR))

    input_scaler, output_scaler = load_scalers(
        str(MODEL_DIR),
        device="cpu",
    )

    assert len(input_columns) == 341
    assert output_columns == ["GDET:FEE1:241:ENRC"]
    assert input_scaler.state_dict()["_coefficient"].shape == (341,)
    assert output_scaler.state_dict()["_coefficient"].shape == (1,)


def test_pretrained_model_dimensions():
    model = torch.load(
        MODEL_DIR / "final_lcls_fel_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 341))

    assert output.shape == (2, 1)


def test_window_files_are_sorted_by_numeric_suffix(
    tmp_path: Path,
):
    for name in [
        "data_10.pkl",
        "data_2.pkl",
        "data_1.pkl",
        "unrelated.pkl",
    ]:
        (tmp_path / name).touch()

    discovered = discover_window_files(str(tmp_path))

    assert [Path(item).name for item in discovered] == [
        "data_1.pkl",
        "data_2.pkl",
        "data_10.pkl",
    ]


def test_small_trailing_window_is_merged():
    inputs = torch.arange(18).reshape(9, 2)
    targets = torch.arange(9).reshape(9, 1)

    windows = split_into_windows(
        inputs,
        targets,
        window_size=4,
        min_window_size=2,
    )

    assert [len(window_inputs) for window_inputs, _ in windows] == [
        4,
        5,
    ]

    reconstructed_inputs = torch.cat([window_inputs for window_inputs, _ in windows])
    reconstructed_targets = torch.cat([window_targets for _, window_targets in windows])

    assert torch.equal(reconstructed_inputs, inputs)
    assert torch.equal(reconstructed_targets, targets)


def test_focused_replay_plot(tmp_path: Path):
    timestamps = pd.date_range(
        "2025-12-04T00:00:00-08:00",
        periods=12,
        freq="min",
    )

    frame = pd.DataFrame(
        {
            "measurement": [
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
            ],
            "static_prediction": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                1.5,
                1.5,
                1.5,
                1.5,
                1.5,
                1.5,
            ],
            "segmented_prediction": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
            ],
        },
        index=timestamps,
    )

    frame["static_absolute_error"] = (
        frame["static_prediction"] - frame["measurement"]
    ).abs()

    frame["segmented_absolute_error"] = (
        frame["segmented_prediction"] - frame["measurement"]
    ).abs()

    output = tmp_path / "focused-replay.png"

    plot_focused_replay(
        predictions=frame,
        trigger_time=timestamps[6],
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000
