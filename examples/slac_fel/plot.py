from pathlib import Path
import gc
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


DATA_DIR = Path("data")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("march_2026_3_drifts_base_full.png")
MAX_FILES = 230 # not using (plotting all)
PREDICTION_BATCH_SIZE = 8192
WINDOW_SIZE = 50
GROUND_TRUTH_COLUMN = "GDET:FEE1:241:ENRC"

DRIFTS = [
    ("2026-03-15 10:46:05", "2026-03-15 12:07:17"),
    ("2026-03-21 08:22:42", "2026-03-21 09:45:13"),
    ("2026-03-26 16:54:16", "2026-03-26 18:42:51"),
]

UPDATED_CHECKPOINTS = [
    (Path("../../output/slac-fel/drift_adaptation_1.pt"), "Updated Chkpt 1", DRIFTS[0][0]),
    (Path("../../output/slac-fel/drift_adaptation_2.pt"), "Updated Chkpt 2", DRIFTS[1][0]),
    (Path("../../output/slac-fel/drift_adaptation_3.pt"), "Updated Chkpt 3", DRIFTS[2][0]),
]

UPDATED_PLOT_RANGES = [
    (DRIFTS[0][0], DRIFTS[1][0]),
    (DRIFTS[1][0], DRIFTS[2][0]),
    (DRIFTS[2][0], None),
]


def sorted_data_files(limit: int) -> list[str]:
    files = glob.glob(str(DATA_DIR / "*.pkl"))
    files.sort(key=lambda file_name: int(re.search(r"\d+", file_name).group()))
    return files#[:limit]


def load_feature_columns() -> tuple[list[str], list[str]]:
    with (MODEL_DIR / "feature_config.yml").open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    input_columns = list(config["input_variables"].keys())
    output_columns = list(config["output_variables"].keys())
    return input_columns, output_columns


def load_selected_frame(files: list[str], columns: list[str]) -> pd.DataFrame:
    selected_columns = list(dict.fromkeys(columns))
    frames = []
    for file_name in files:
        frame = pd.read_pickle(file_name)
        frames.append(frame.loc[:, selected_columns])
    return pd.concat(frames, copy=False)


def predict_in_batches(model: torch.nn.Module, scaled_inputs: torch.Tensor, output_scaler) -> np.ndarray:
    batches = []
    model.eval()
    with torch.inference_mode():
        for start_index in range(0, scaled_inputs.shape[0], PREDICTION_BATCH_SIZE):
            batch_inputs = scaled_inputs[start_index:start_index + PREDICTION_BATCH_SIZE]
            batch_output = model(batch_inputs)
            batch_unscaled = output_scaler.untransform(batch_output)
            batches.append(batch_unscaled.detach().cpu().numpy().reshape(-1))
    return np.concatenate(batches, axis=0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    cleaned_state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)


def align_timestamp(timestamp_text: str, index: pd.Index) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp_text)
    index_timezone = getattr(index, "tz", None)
    if index_timezone is not None and timestamp.tzinfo is None:
        return timestamp.tz_localize(index_timezone)
    if index_timezone is None and timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def main() -> None:
    input_columns, output_columns = load_feature_columns()
    files = sorted_data_files(MAX_FILES)
    selected_frame = load_selected_frame(files, input_columns + output_columns + [GROUND_TRUTH_COLUMN])

    model = torch.load(MODEL_DIR / "final_lcls_fel_model.pt", weights_only=False, map_location="cpu")
    input_scaler = torch.load(MODEL_DIR / "lcls_fel_input_scaler.pt", weights_only=False, map_location="cpu")
    output_scaler = torch.load(MODEL_DIR / "lcls_fel_output_scaler.pt", weights_only=False, map_location="cpu")

    raw_inputs = torch.as_tensor(selected_frame[input_columns].to_numpy(dtype=np.float32, copy=False))
    scaled_inputs = input_scaler.transform(raw_inputs)

    pretrained_prediction = predict_in_batches(model, scaled_inputs, output_scaler)

    date_format = mdates.DateFormatter("%m-%d %H:%M")
    fontsize = 12
    fig, ax = plt.subplots(figsize=(12, 7))

    ground_truth = selected_frame[GROUND_TRUTH_COLUMN]
    moving_avg = ground_truth.rolling(window=WINDOW_SIZE).mean()

    ax.scatter(selected_frame.index, ground_truth.to_numpy(), label="Measurement", color="salmon", marker="x", s=12, rasterized=True)
    ax.plot(selected_frame.index, moving_avg, label="Moving Mean", color="red", linewidth=2)
    ax.scatter(selected_frame.index, pretrained_prediction, label="Pretrained Model Prediction", color="dodgerblue", marker=".", s=15, rasterized=True)

    colors = ["blueviolet", "lightblue", "lightgreen"]
    for (checkpoint_path, label, _), (range_start_text, range_end_text) in zip(UPDATED_CHECKPOINTS, UPDATED_PLOT_RANGES, strict=True):
        load_checkpoint(model, checkpoint_path)
        updated_prediction = predict_in_batches(model, scaled_inputs, output_scaler)
        range_start = align_timestamp(range_start_text, selected_frame.index)
        mask = selected_frame.index >= range_start
        if range_end_text is not None:
            range_end = align_timestamp(range_end_text, selected_frame.index)
            mask &= selected_frame.index < range_end
        ax.scatter(selected_frame.index[mask], updated_prediction[mask], label=label, color=colors.pop(0), marker=".", s=15, rasterized=True)
        del updated_prediction
        gc.collect()

    ax.set_xlabel("Time", fontsize=fontsize)
    ax.set_ylabel("HXR pulse intensity (mJ)", fontsize=fontsize)
    # ax.set_xlim(
    #     align_timestamp("2026-03-13 00:00", selected_frame.index),
    #     align_timestamp("2026-04-01 06:00", selected_frame.index),
    # )
    # ax.set_ylim([0, 6])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(date_format)

    for drift_index, (start_text, end_text) in enumerate(DRIFTS, start=1):
        ax.axvspan(
            align_timestamp(start_text, selected_frame.index),
            align_timestamp(end_text, selected_frame.index),
            color="green",
            alpha=0.3,
            #label=f"Drift {drift_index}",
        )

    

    ax.legend(fontsize=12, loc="upper left")
    ax.tick_params(labelsize=fontsize)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
