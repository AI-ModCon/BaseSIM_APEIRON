"""
reproduce_prometheus.py

Concise reimplementation of the Prometheus temporal-prediction model
(originally TemporalPredict), with Keras and PyTorch backends.

Framework is auto-detected from file extension:
    .keras / .h5   -> keras
    .pt   / .pth   -> torch
(or override with --framework keras|torch)

Modes:
    # Eval a saved model (framework detected from extension)
    python reproduce_prometheus.py eval --model ./data/npm1_pwr_model.keras
    python reproduce_prometheus.py eval --model ./output/reproduce_prometheus/retrained.pt

    # Train a fresh model (framework detected from --save extension)
    python reproduce_prometheus.py train --save ./output/reproduce_prometheus/retrained.keras
    python reproduce_prometheus.py train --save ./output/reproduce_prometheus/retrained.pt

    # Eval baseline then train
    python reproduce_prometheus.py both  --model ./data/npm1_pwr_model.keras \\
                                         --save  ./output/reproduce_prometheus/retrained.pt

    # Overlay a Keras baseline and a PyTorch reproduction on the test set
    python reproduce_prometheus.py compare --model ./data/npm1_pwr_model.keras \\
                                           --torch-model ./output/reproduce_prometheus/retrained.pt

Training uses a continual / per-case scheme: one fit() per training file,
with normalization statistics recomputed from cases seen so far (0..N) at
the start of each case. Each per-case checkpoint is written alongside a
JSON stats sidecar so eval/compare can reproduce the exact normalization
used at training time.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score


# ---------- Config --------------------------------------------------------

@dataclass
class Config:
    train_dir: str = "./data/train"
    test_dir: str = "./data/test"
    out_dir: str = "./output/prometheus_torch/"
    sequence_length: int = 10
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    feature_cols: List[str] = field(default_factory=lambda: [
        "NRAD_RX_REG_POS", "NRAD_RX_SHIM1_POS", "NRAD_RX_SHIM2_POS",
        "total_rod_position", "NRAD_RX_PERIOD_Inverse",
        "NRAD_RX_REG_POS_dt", "NRAD_RX_REG_POS_dt2",
        "NRAD_RX_SHIM1_POS_dt", "NRAD_RX_SHIM1_POS_dt2",
        "NRAD_RX_SHIM2_POS_dt", "NRAD_RX_SHIM2_POS_dt2",
        "NRAD_RX_NMP1_PWR_integral",
    ])
    target_cols: List[str] = field(default_factory=lambda: ["NRAD_RX_NMP1_PWR"])


# ---------- Data ----------------------------------------------------------

def load_csvs(folder: str) -> List[pd.DataFrame]:
    """Read every .csv in folder (sorted) and stash the filename stem on
    df.attrs['source'] so downstream plots can label the data by when it
    was generated (file names encode the collection date).
    """
    files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(folder, f))
        df.attrs["source"] = os.path.splitext(f)[0]
        dfs.append(df)
    return dfs


def df_label(df: pd.DataFrame, fallback: str = "") -> str:
    """Return the source-file stem recorded by load_csvs, or a fallback."""
    return df.attrs.get("source", fallback)


def compute_train_stats(dfs: List[pd.DataFrame], cols: List[str]) -> dict:
    """Return {col: (mu, std)} from concatenated training dataframes."""
    full = pd.concat([df[cols] for df in dfs], ignore_index=True)
    return {c: (float(full[c].mean()), float(full[c].std())) for c in cols}


def normalize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, std) in stats.items():
        if c in out.columns:
            out[c] = (out[c] - mu) / std
    return out


def make_windows(
    dfs: List[pd.DataFrame],
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    stats: dict,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sliding-window sequences. Returns per-file lists of arrays:
    X[k] has shape (n_windows, seq_len, n_features),
    Y[k] has shape (n_windows, seq_len, n_targets).
    """
    X_list, Y_list = [], []
    for df in dfs:
        if len(df) <= seq_len:
            continue
        data = normalize(df[feature_cols + target_cols], stats)
        feat = data[feature_cols].to_numpy(dtype=np.float32)
        targ = data[target_cols].to_numpy(dtype=np.float32)
        n = len(data) - seq_len
        X = np.stack([feat[i:i + seq_len] for i in range(n)])
        Y = np.stack([targ[i:i + seq_len] for i in range(n)])
        X_list.append(X)
        Y_list.append(Y)
    return X_list, Y_list


# ---------- Models (Keras + Torch) ----------------------------------------

def build_keras_model(seq_len: int, n_features: int, n_targets: int, lr: float) -> tf.keras.Model:
    """LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> LSTM(32) -> TimeDistributed(Dense).
    Output shape: (batch, seq_len, n_targets).
    """
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_targets))(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


class TorchTemporalModel(nn.Module):
    """PyTorch mirror of the Keras architecture above. A Linear layer applied
    to the (batch, seq_len, hidden) tensor is equivalent to Keras'
    TimeDistributed(Dense(...)).
    """

    def __init__(self, n_features: int, n_targets: int, dropout: float = 0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.head = nn.Linear(32, n_targets)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        return self.head(x)


# ---------- Framework dispatch --------------------------------------------

def framework_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".keras", ".h5"):
        return "keras"
    if ext in (".pt", ".pth"):
        return "torch"
    raise ValueError(f"Cannot detect framework from extension: {ext!r}. "
                     f"Use --framework keras|torch to override.")


def load_saved_model(path: str, n_features: int, n_targets: int, framework: str):
    if framework == "keras":
        return tf.keras.models.load_model(path)
    if framework == "torch":
        model = TorchTemporalModel(n_features, n_targets)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    raise ValueError(f"Unknown framework: {framework}")


def save_trained_model(model, path: str, framework: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if framework == "keras":
        model.save(path)
    elif framework == "torch":
        torch.save(model.state_dict(), path)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def predict_np(model, X: np.ndarray, framework: str) -> np.ndarray:
    """Return model predictions as a numpy array of shape (n_windows, seq_len, n_targets)."""
    if framework == "keras":
        return model.predict(X, verbose=0)
    if framework == "torch":
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32))
            return model(t).cpu().numpy()
    raise ValueError(f"Unknown framework: {framework}")


# ---------- Stats sidecar (per-checkpoint persistence) -------------------

def stats_sidecar_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return base + ".stats.json"


def save_stats(stats: dict, model_path: str) -> str:
    path = stats_sidecar_path(model_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {k: [float(mu), float(std)] for k, (mu, std) in stats.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_stats(model_path: str, required_cols: Optional[List[str]] = None) -> Optional[dict]:
    """Load a stats sidecar next to model_path. Returns None if missing."""
    path = stats_sidecar_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    stats = {k: (float(v[0]), float(v[1])) for k, v in payload.items()}
    if required_cols is not None:
        missing = [c for c in required_cols if c not in stats]
        if missing:
            raise ValueError(f"Stats sidecar {path} is missing columns: {missing}")
    return stats


def stats_for_model(model_path: str, train_dfs: List[pd.DataFrame], cols: List[str]) -> dict:
    """Return per-checkpoint stats if a sidecar exists, else fall back to
    global training stats computed over all train_dfs. Logs which path was taken.
    """
    stats = load_stats(model_path, required_cols=cols)
    if stats is not None:
        print(f"  using stats sidecar: {stats_sidecar_path(model_path)}")
        return stats
    print(f"  no stats sidecar for {model_path}; falling back to global train stats")
    return compute_train_stats(train_dfs, cols)


# ---------- Denormalize + plot helpers ------------------------------------

def denorm_last_step(pred_norm: np.ndarray, stats: dict, target_cols: List[str]) -> pd.DataFrame:
    """Take the last timestep of each window and invert the train-stats normalization."""
    last = pred_norm[:, -1, :]
    df = pd.DataFrame(last, columns=target_cols)
    for c in target_cols:
        mu, std = stats[c]
        df[c] = df[c] * std + mu
    return df


def plot_pred_vs_truth(
    preds: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    truth: pd.DataFrame,
    target_cols: List[str],
    title: str,
    save_path: str,
) -> None:
    """preds can be a single DataFrame or a dict {label: DataFrame} for overlays."""
    if isinstance(preds, pd.DataFrame):
        preds = {"Prediction": preds}
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
              "tab:brown", "tab:pink", "tab:cyan"]
    n = len(target_cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    for i, var in enumerate(target_cols):
        ax = axes[i, 0]
        ax.plot(truth.index, truth[var].values, color="black", linewidth=1.5, label="Ground Truth")
        for j, (label, pred) in enumerate(preds.items()):
            ax.plot(pred.index, pred[var].values,
                    color=colors[j % len(colors)],
                    linewidth=1.2, label=label)
        ax.set_xlabel("time step")
        ax.set_ylabel(var)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------- Evaluation ----------------------------------------------------

def evaluate(model, test_dfs: List[pd.DataFrame], X_test: List[np.ndarray],
             stats: dict, cfg: Config, tag: str, framework: str) -> None:
    """Predict on each test file, compute metrics, save plots."""
    out_subdir = os.path.join(cfg.out_dir, f"eval_{tag}")
    os.makedirs(out_subdir, exist_ok=True)
    print(f"\n=== Evaluation [{tag}] ({framework}) ===")
    for i, (df, X) in enumerate(zip(test_dfs, X_test)):
        label = df_label(df, fallback=f"test{i:02d}")
        pred_norm = predict_np(model, X, framework)
        pred = denorm_last_step(pred_norm, stats, cfg.target_cols).reset_index(drop=True)
        truth = df[cfg.target_cols].iloc[cfg.sequence_length:].reset_index(drop=True)
        m = min(len(pred), len(truth))
        pred, truth = pred.iloc[:m], truth.iloc[:m]
        r2 = r2_score(truth, pred)
        mae = mean_absolute_error(truth, pred)
        print(f"  test {i} [{label}]: R2={r2:.4f}  MAE={mae:.3f}")
        plot_pred_vs_truth(
            pred, truth, cfg.target_cols,
            title=f"{tag} — {label}  (R²={r2:.3f}, MAE={mae:.3f})",
            save_path=os.path.join(out_subdir, f"test{i:02d}_{label}.png"),
        )


# ---------- Torch per-case training loop ---------------------------------

def torch_fit_case(model: nn.Module, X: np.ndarray, Y: np.ndarray,
                   epochs: int, batch_size: int, lr: float,
                   val_split: float = 0.2, patience: int = 8) -> nn.Module:
    """Mirror of keras model.fit(...) for one case, with early stopping and
    best-weight restoration on val loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n = len(X)
    n_val = max(1, int(n * val_split))
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.from_numpy(X[tr_idx]).to(device)
    Y_tr = torch.from_numpy(Y[tr_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    Y_val = torch.from_numpy(Y[val_idx]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr), device=device)
        total = 0.0
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_tr[b])
            loss = loss_fn(pred, Y_tr[b])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        train_loss = total / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        print(f"  Epoch {epoch + 1:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    return model


# ---------- Training ------------------------------------------------------

def train(cfg: Config, train_dfs: List[pd.DataFrame], test_dfs: List[pd.DataFrame],
          save_path: str, framework: str):
    """Per-case training with stats-so-far normalization.

    At the start of each case N:
      * stats_N = compute_train_stats(train_dfs[:N+1])
      * The case N windows and the test windows are (re)built with stats_N.
      * The model (weights carried over from case N-1) is fit on case N.
      * Predictions are denormalized with stats_N for the post-fit plot and
        the test eval pass.
      * A per-case checkpoint is saved alongside a stats sidecar so later
        eval/compare runs can reproduce the same normalization.

    This is an intentional continual-learning setup where the input scale
    drifts between cases, and the "final" model is just the model after the
    last case (saved to save_path with its own stats sidecar).
    """
    all_cols = cfg.feature_cols + cfg.target_cols
    n_features, n_targets = len(cfg.feature_cols), len(cfg.target_cols)

    if framework == "keras":
        model = build_keras_model(cfg.sequence_length, n_features, n_targets, cfg.learning_rate)
        model.summary()
    else:
        model = TorchTemporalModel(n_features, n_targets)
        print(model)

    train_plot_dir = os.path.join(cfg.out_dir, "train_fits")
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_ext = os.path.splitext(save_path)[1] or (".keras" if framework == "keras" else ".pt")

    stats_n = None
    for case_idx in range(len(train_dfs)):
        # Stats use only cases seen so far (inclusive of the current one)
        stats_n = compute_train_stats(train_dfs[:case_idx + 1], all_cols)

        # Rebuild just this case's windows with stats_n
        Xc_list, Yc_list = make_windows(
            [train_dfs[case_idx]], cfg.feature_cols, cfg.target_cols, cfg.sequence_length, stats_n)
        if not Xc_list:
            print(f"\n--- Case {case_idx}: empty after windowing, skipping ---")
            continue
        Xc, Yc = Xc_list[0], Yc_list[0]

        # Rebuild test windows with the same stats so the eval pass is consistent
        X_test, _ = make_windows(
            test_dfs, cfg.feature_cols, cfg.target_cols, cfg.sequence_length, stats_n)

        case_label = df_label(train_dfs[case_idx], fallback=f"case{case_idx:02d}")
        print(f"\n--- Case {case_idx} [{case_label}] ({framework}): X={Xc.shape}, Y={Yc.shape} "
              f"(stats from {case_idx + 1} case(s)) ---")
        if framework == "keras":
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True)
            model.fit(Xc, Yc, epochs=cfg.epochs, batch_size=cfg.batch_size,
                      validation_split=0.2, callbacks=[early_stop], verbose=1)
        else:
            model = torch_fit_case(model, Xc, Yc, cfg.epochs, cfg.batch_size, cfg.learning_rate)

        # Single post-fit plot: predictions on the case we just trained on
        pred_norm = predict_np(model, Xc, framework)
        pred = denorm_last_step(pred_norm, stats_n, cfg.target_cols).reset_index(drop=True)
        truth = train_dfs[case_idx][cfg.target_cols].iloc[cfg.sequence_length:].reset_index(drop=True)
        m = min(len(pred), len(truth))
        plot_pred_vs_truth(
            pred.iloc[:m], truth.iloc[:m], cfg.target_cols,
            title=f"train case {case_idx} — {case_label} — post-fit ({framework})",
            save_path=os.path.join(train_plot_dir, f"{framework}_case_{case_idx:02d}_{case_label}.png"),
        )

        # Per-case checkpoint + stats sidecar
        ckpt_path = os.path.join(ckpt_dir, f"{framework}_case_{case_idx:02d}{ckpt_ext}")
        save_trained_model(model, ckpt_path, framework)
        save_stats(stats_n, ckpt_path)
        print(f"  checkpoint -> {ckpt_path}")
        print(f"  stats      -> {stats_sidecar_path(ckpt_path)}")

        evaluate(model, test_dfs, X_test, stats_n, cfg,
                 tag=f"{framework}_case_{case_idx:02d}", framework=framework)

    # Final save uses the last case's stats
    save_trained_model(model, save_path, framework)
    if stats_n is not None:
        save_stats(stats_n, save_path)
    print(f"\nSaved final {framework} model to {save_path}")
    if stats_n is not None:
        print(f"Saved final stats sidecar to {stats_sidecar_path(save_path)}")
    return model


# ---------- Compare mode --------------------------------------------------

def compare(cfg: Config, train_dfs: List[pd.DataFrame], test_dfs: List[pd.DataFrame],
            keras_path: str, torch_paths: List[str]) -> None:
    """Load a Keras baseline and one or more PyTorch models, then overlay their
    predictions against the ground truth on each test file. Each model uses
    its own stats sidecar (if present) for normalization and denormalization.
    Labels in plots and titles use the .pt filename stem.
    """
    n_features, n_targets = len(cfg.feature_cols), len(cfg.target_cols)
    all_cols = cfg.feature_cols + cfg.target_cols

    # --- Keras baseline ---
    print(f"Loading keras model: {keras_path}")
    keras_model = load_saved_model(keras_path, n_features, n_targets, "keras")
    k_stats = stats_for_model(keras_path, train_dfs, all_cols)
    X_test_k, _ = make_windows(test_dfs, cfg.feature_cols, cfg.target_cols,
                               cfg.sequence_length, k_stats)
    keras_stem = os.path.splitext(os.path.basename(keras_path))[0]

    # --- Torch models ---
    torch_entries: List[Tuple[str, object, dict, List[np.ndarray]]] = []
    for tp in torch_paths:
        stem = os.path.splitext(os.path.basename(tp))[0]
        print(f"Loading torch model: {tp}")
        model = load_saved_model(tp, n_features, n_targets, "torch")
        stats = stats_for_model(tp, train_dfs, all_cols)
        X_test_t, _ = make_windows(test_dfs, cfg.feature_cols, cfg.target_cols,
                                   cfg.sequence_length, stats)
        torch_entries.append((stem, model, stats, X_test_t))

    stems = [keras_stem] + [e[0] for e in torch_entries]
    pair_name = "__vs__".join(stems)
    out_subdir = os.path.join(cfg.out_dir, "compare", pair_name)
    os.makedirs(out_subdir, exist_ok=True)
    print(f"  writing comparison plots to {out_subdir}")
    print(f"\n=== Comparison: {' vs '.join(stems)} ===")

    for i, df in enumerate(test_dfs):
        label = df_label(df, fallback=f"test{i:02d}")
        truth = df[cfg.target_cols].iloc[cfg.sequence_length:].reset_index(drop=True)

        # Keras prediction
        k_pred = denorm_last_step(
            predict_np(keras_model, X_test_k[i], "keras"), k_stats, cfg.target_cols
        ).reset_index(drop=True)

        # Torch predictions
        overlay: Dict[str, pd.DataFrame] = {keras_stem: k_pred}
        min_len = min(len(k_pred), len(truth))
        metrics_parts: List[str] = []

        k_r2 = r2_score(truth.iloc[:min_len], k_pred.iloc[:min_len])
        k_mae = mean_absolute_error(truth.iloc[:min_len], k_pred.iloc[:min_len])
        metrics_parts.append(f"{keras_stem} R²={k_r2:.3f}")
        print(f"  test {i} [{label}]: {keras_stem} R2={k_r2:.4f} MAE={k_mae:.3f}", end="")

        for stem, model, stats, X_test_t in torch_entries:
            t_pred = denorm_last_step(
                predict_np(model, X_test_t[i], "torch"), stats, cfg.target_cols
            ).reset_index(drop=True)
            overlay[stem] = t_pred
            min_len = min(min_len, len(t_pred))
            t_r2 = r2_score(truth.iloc[:min_len], t_pred.iloc[:min_len])
            t_mae = mean_absolute_error(truth.iloc[:min_len], t_pred.iloc[:min_len])
            metrics_parts.append(f"{stem} R²={t_r2:.3f}")
            print(f"  |  {stem} R2={t_r2:.4f} MAE={t_mae:.3f}", end="")
        print()

        # Trim all to common length
        truth_trimmed = truth.iloc[:min_len]
        overlay_trimmed = {k: v.iloc[:min_len] for k, v in overlay.items()}

        plot_pred_vs_truth(
            overlay_trimmed, truth_trimmed, cfg.target_cols,
            title=f"compare — {label}  ({', '.join(metrics_parts)})",
            save_path=os.path.join(out_subdir, f"test{i:02d}_{label}.png"),
        )


# ---------- Main ----------------------------------------------------------

def resolve_framework(explicit: str, path: str) -> str:
    return explicit if explicit != "auto" else framework_from_path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "train", "both", "compare"])
    parser.add_argument("--model", default="./data/npm1_pwr_model.keras",
                        help="Saved model path for eval/both, or the Keras baseline for compare.")
    parser.add_argument("--save", default="./output/reproduce_prometheus/retrained.keras",
                        help="Where to save the retrained model (for train/both). "
                             "Extension determines framework (.keras/.h5 or .pt/.pth).")
    parser.add_argument("--torch-model", nargs="+", default=None,
                        help="Path(s) to PyTorch model(s) for compare mode. "
                             "Multiple paths overlay all models on the same plot.")
    parser.add_argument("--framework", choices=["keras", "torch", "auto"], default="auto",
                        help="Override framework detection.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("Loading data ...")
    train_dfs = load_csvs(cfg.train_dir)
    test_dfs = load_csvs(cfg.test_dir)
    print(f"  {len(train_dfs)} training files, {len(test_dfs)} test files")

    n_features, n_targets = len(cfg.feature_cols), len(cfg.target_cols)
    all_cols = cfg.feature_cols + cfg.target_cols

    if args.mode in ("eval", "both"):
        framework = resolve_framework(args.framework, args.model)
        print(f"Loading {framework} model: {args.model}")
        model = load_saved_model(args.model, n_features, n_targets, framework)
        stats = stats_for_model(args.model, train_dfs, all_cols)
        X_test, _ = make_windows(test_dfs, cfg.feature_cols, cfg.target_cols,
                                 cfg.sequence_length, stats)
        evaluate(model, test_dfs, X_test, stats, cfg,
                 tag=f"{framework}_baseline", framework=framework)

    if args.mode in ("train", "both"):
        framework = resolve_framework(args.framework, args.save)
        train(cfg, train_dfs, test_dfs, args.save, framework)

    if args.mode == "compare":
        if not args.torch_model:
            parser.error("compare mode requires --torch-model")
        compare(cfg, train_dfs, test_dfs,
                keras_path=args.model, torch_paths=args.torch_model)


if __name__ == "__main__":
    main()
