import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
import os
import glob
from pathlib import Path


class ModelGeneration:
    """
    LSTM model class for handling batched time series data from CSV files.
    Designed for reactor control rod position tracking and similar applications.
    Supports variable-length and rolling-window sequence modes, optional
    log-space target transformation, and relative/log-space loss functions
    for accurate prediction across wide dynamic ranges (e.g. 0 W – 250 kW).
    """

    def __init__(self, model_path, testing_path, training_path, text_file_name,
                 feature_variables, target_variable, epochs=100,
                 lstm_units=[64, 32], dropout=0.2, normalization_method='minmax',
                 batch_size=32, validation_split=0.2,
                 use_rolling_windows=False, window_size=100, window_stride=10,
                 forecast_horizon=0,
                 target_log_transform=False,
                 loss_fn='mse',
                 random_seed=None):
        """
        Initialize the ModelGeneration class.

        Parameters
        ----------
        model_path : str
            Path where the trained model will be saved.
        testing_path : str
            Directory containing testing CSV files.
        training_path : str
            Directory containing training CSV files.
        text_file_name : str
            Path for saving model summary and training information.
        feature_variables : list
            Column names to use as input features (e.g. ['Shim_1', 'Shim_2', 'Reg']).
        target_variable : list
            Column names to use as targets (e.g. ['power']).
        epochs : int
            Number of training epochs.
        lstm_units : list
            Units for each LSTM layer.
        dropout : float
            Dropout rate for regularization.
        normalization_method : str
            'minmax' | 'standard' | 'robust' | 'none'
        batch_size : int
            Batch size for training.
        validation_split : float
            Fraction of training data reserved for validation.
        use_rolling_windows : bool
            If True, creates overlapping windows from each sequence.
            If False, uses entire variable-length sequences.
        window_size : int
            Rolling window length in timesteps.
        window_stride : int
            Stride between successive windows.
        forecast_horizon : int
            0  → predict current y_t (aligned / nowcasting).
            >0 → predict y_{t+h} (forecasting h steps ahead).
        target_log_transform : bool
            If True, applies np.log1p() to y before normalization and
            np.expm1() after denormalization.  Compresses wide dynamic
            ranges (e.g. 50 W – 250 kW) so low-power values are not
            washed out by the dominant high-power training signal.
            Requires all target values to be non-negative.
        loss_fn : str
            'mse'          – standard mean squared error (default).
            'relative_mse' – MSE / |y_true|²; equal relative weight
                             across all power levels.
            'log_mse'      – MSE in log-space; implicit relative weighting
                             without modifying the normalization pipeline.
            Note: combining target_log_transform=True with loss_fn='mse'
            already trains in log-space and is the recommended pairing for
            NRAD power prediction.
        random_seed : int
            Ability to set random seed when splitting training/validation data set
        """
        self.model_path = model_path
        self.testing_path = testing_path
        self.training_path = training_path
        self.text_file_name = text_file_name
        self.feature_variables = feature_variables
        self.target_variable = target_variable
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.normalization_method = normalization_method
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_rolling_windows = use_rolling_windows
        self.window_size = window_size
        self.window_stride = window_stride
        self.forecast_horizon = int(forecast_horizon)
        self.target_log_transform = target_log_transform
        self.loss_fn = loss_fn
        self.random_seed = random_seed        

        if self.forecast_horizon < 0:
            raise ValueError("forecast_horizon must be >= 0")
        if self.loss_fn not in ('mse', 'relative_mse', 'log_mse'):
            raise ValueError(f"Unknown loss_fn '{self.loss_fn}'. "
                             "Choose 'mse', 'relative_mse', or 'log_mse'.")

        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_val   = None
        self.y_val   = None
        self.X_test  = None
        self.y_test  = None

        # Model components
        self.model    = None
        self.scaler_X = None
        self.scaler_y = None
        self.history  = None

        # Metadata
        self.train_files  = []
        self.test_files   = []
        self.max_timesteps = None

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.text_file_name).parent.mkdir(parents=True, exist_ok=True)

        self._initialize_scalers()

    # ─────────────────────────────────────────────────────────────────────────
    # Scalers
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_scalers(self):
        """Initialize scalers based on chosen normalization method."""
        if self.normalization_method == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        elif self.normalization_method == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.normalization_method == 'robust':
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        elif self.normalization_method == 'none':
            self.scaler_X = None
            self.scaler_y = None
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

    # ─────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_csv_files(self, directory_path):
        """Read all CSV files from a directory."""
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory_path}")

        dataframes, filenames = [], []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                filenames.append(os.path.basename(csv_file))
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")

        return dataframes, filenames

    # ─────────────────────────────────────────────────────────────────────────
    # Sequence extraction & windowing
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_sequences(self, dataframes, feature_cols, target_cols):
        """Extract feature and target sequences from DataFrames."""
        X, y = [], []
        for df in dataframes:
            missing_f = set(feature_cols) - set(df.columns)
            missing_t = set(target_cols)  - set(df.columns)
            if missing_f:
                raise ValueError(f"Missing feature columns: {missing_f}")
            if missing_t:
                raise ValueError(f"Missing target columns: {missing_t}")
            X.append(df[feature_cols].values)
            y.append(df[target_cols].values)
        return X, y

    def _create_rolling_windows(self, X_sequences, y_sequences):
        """
        Create rolling windows from sequences.
        forecast_horizon=0 → aligned window targets.
        forecast_horizon>0 → forecast_horizon future timesteps as targets.
        """
        X_windows, y_windows = [], []
        for X_seq, y_seq in zip(X_sequences, y_sequences):
            seq_len = len(X_seq)
            min_req = self.window_size + self.forecast_horizon
            if seq_len < min_req:
                print(f"Warning: Skipping sequence of length {seq_len} (< {min_req})")
                continue
            for start in range(0, seq_len - min_req + 1, self.window_stride):
                end = start + self.window_size
                X_windows.append(X_seq[start:end])
                if self.forecast_horizon == 0:
                    y_windows.append(y_seq[start:end])
                else:
                    y_windows.append(y_seq[end:end + self.forecast_horizon])
        return X_windows, y_windows

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def data_preprocessing(self):
        """
        Load and preprocess data from CSV files.
        Reads training and testing data, extracts sequences, and prepares
        for modelling. Applies rolling windows when enabled.
        """
        print("=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)

        print(f"\nReading training data from: {self.training_path}")
        train_dfs, self.train_files = self._read_csv_files(self.training_path)
        print(f"  Found {len(train_dfs)} training files")

        print(f"\nReading testing data from: {self.testing_path}")
        test_dfs, self.test_files = self._read_csv_files(self.testing_path)
        print(f"  Found {len(test_dfs)} testing files")

        print("\nExtracting sequences...")
        X_train_raw, y_train_raw = self._extract_sequences(
            train_dfs, self.feature_variables, self.target_variable)
        X_test_raw, y_test_raw = self._extract_sequences(
            test_dfs, self.feature_variables, self.target_variable)

        # Forecast-horizon shift for full-sequence (non-rolling) mode
        if (not self.use_rolling_windows) and (self.forecast_horizon > 0):
            X_train_raw = [s[:-self.forecast_horizon] for s in X_train_raw
                           if len(s) > self.forecast_horizon]
            y_train_raw = [s[self.forecast_horizon:]  for s in y_train_raw
                           if len(s) > self.forecast_horizon]
            X_test_raw  = [s[:-self.forecast_horizon] for s in X_test_raw
                           if len(s) > self.forecast_horizon]
            y_test_raw  = [s[self.forecast_horizon:]  for s in y_test_raw
                           if len(s) > self.forecast_horizon]

        if self.use_rolling_windows:
            print(f"\nApplying rolling windows (size={self.window_size}, "
                  f"stride={self.window_stride}, horizon={self.forecast_horizon})")
            X_train_raw, y_train_raw = self._create_rolling_windows(X_train_raw, y_train_raw)
            X_test_raw,  y_test_raw  = self._create_rolling_windows(X_test_raw,  y_test_raw)
            print(f"  Training windows : {len(X_train_raw)}")
            print(f"  Testing  windows : {len(X_test_raw)}")
            self.max_timesteps = self.window_size
        else:
            all_lengths = [len(s) for s in X_train_raw + X_test_raw]
            self.max_timesteps = max(all_lengths)

        # Train / validation split
        n_train  = len(X_train_raw)
        n_val    = int(n_train * self.validation_split)

        # Set seed if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        indices  = np.random.permutation(n_train)
        val_idx  = indices[:n_val]
        train_idx = indices[n_val:]

        self.X_train = [X_train_raw[i] for i in train_idx]
        self.y_train = [y_train_raw[i] for i in train_idx]
        self.X_val   = [X_train_raw[i] for i in val_idx]
        self.y_val   = [y_train_raw[i] for i in val_idx]
        self.X_test  = X_test_raw
        self.y_test  = y_test_raw

        print("\nData Statistics:")
        print(f"  Training   : {len(self.X_train)} samples")
        print(f"  Validation : {len(self.X_val)} samples")
        print(f"  Testing    : {len(self.X_test)} samples")
        print(f"  Features   : {len(self.feature_variables)} {self.feature_variables}")
        print(f"  Targets    : {len(self.target_variable)} {self.target_variable}")
        if self.target_log_transform:
            print(f"  Target log-transform : ENABLED (log1p/expm1)")

        if not self.use_rolling_windows:
            tl = [len(s) for s in self.X_train]
            el = [len(s) for s in self.X_test]
            print(f"\nSequence lengths — Train: min={min(tl)}, max={max(tl)}, "
                  f"mean={np.mean(tl):.1f} | Test: min={min(el)}, max={max(el)}")
            print(f"  Max timesteps for model: {self.max_timesteps}")

        print(f"\nTarget Variable Statistics (across all timesteps):")
        for i, var in enumerate(self.target_variable):
            tv = np.concatenate([s[:, i] for s in self.y_train])
            ev = np.concatenate([s[:, i] for s in self.y_test])
            print(f"  {var}:")
            print(f"    Train — min={tv.min():.4f}, max={tv.max():.4f}, mean={tv.mean():.4f}")
            print(f"    Test  — min={ev.min():.4f}, max={ev.max():.4f}, mean={ev.mean():.4f}")

        print("\nData preprocessing complete!")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_data(self, X, y=None, fit=True):
        """
        Normalize batched variable-length sequences.
        If target_log_transform=True, applies log1p to y before scaling
        so the scaler fits in log-space, expanding the low-value regime.
        """
        if self.normalization_method == 'none':
            return X, y
        if not X or len(X) == 0:
            return X, y

        # ── X ──────────────────────────────────────────────────────────────
        X_flat = np.vstack(X)
        X_scaled = (self.scaler_X.fit_transform(X_flat) if fit
                    else self.scaler_X.transform(X_flat))
        X_norm, idx = [], 0
        for seq in X:
            l = len(seq)
            X_norm.append(X_scaled[idx:idx+l])
            idx += l

        # ── y ──────────────────────────────────────────────────────────────
        y_norm = None
        if y is not None and len(y) > 0:
            y_flat = np.vstack(y)

            if self.target_log_transform:
                if np.any(y_flat < 0):
                    raise ValueError(
                        "target_log_transform=True requires non-negative targets. "
                        "Found negative values in y.")
                y_flat = np.log1p(y_flat)

            if self.scaler_y is not None:
                y_scaled = (self.scaler_y.fit_transform(y_flat) if fit
                            else self.scaler_y.transform(y_flat))
            else:
                y_scaled = y_flat

            y_norm, idx = [], 0
            for seq in y:
                l = len(seq)
                y_norm.append(y_scaled[idx:idx+l])
                idx += l

        return X_norm, y_norm

    def _denormalize_data(self, data, is_target=False):
        """
        Denormalize data back to original scale.
        For targets with target_log_transform=True, applies expm1 after
        the scaler's inverse_transform to undo the log1p step.
        """
        if self.normalization_method == 'none':
            return data

        scaler = self.scaler_y if is_target else self.scaler_X
        result = scaler.inverse_transform(data) if scaler is not None else data

        if is_target and self.target_log_transform:
            result = np.expm1(result)
            result = np.clip(result, 0, None)   # guard against sub-zero float noise

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Padding helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _pad_sequences(self, sequences, max_length=None):
        """Pad feature sequences to uniform length for batching."""
        if max_length is None:
            max_length = self.max_timesteps
        if self.use_rolling_windows:
            return np.array(sequences)
        n, feats = len(sequences), sequences[0].shape[-1]
        padded = np.zeros((n, max_length, feats))
        for i, seq in enumerate(sequences):
            l = min(len(seq), max_length)
            padded[i, :l, :] = seq[:l]
        return padded

    def _pad_targets(self, targets, max_length=None):
        """Pad target sequences to uniform length for batching."""
        if max_length is None:
            max_length = self.max_timesteps
        if self.use_rolling_windows:
            return np.array(targets)
        n, feats = len(targets), targets[0].shape[-1]
        padded = np.zeros((n, max_length, feats))
        for i, seq in enumerate(targets):
            l = min(len(seq), max_length)
            padded[i, :l, :] = seq[:l]
        return padded

    # ─────────────────────────────────────────────────────────────────────────
    # Loss function
    # ─────────────────────────────────────────────────────────────────────────

    def _get_loss(self):
        """
        Return the Keras-compatible loss for model.compile().

        'mse'          – standard mean squared error.
        'relative_mse' – MSE / |y_true|²; equal relative penalty at all
                         power levels.
        'log_mse'      – MSE in log-space; operates on normalized model
                         outputs so it complements (but does not duplicate)
                         target_log_transform.
        """
        if self.loss_fn == 'mse':
            return 'mse'

        elif self.loss_fn == 'relative_mse':
            eps = tf.constant(1e-6, dtype=tf.float32)
            def relative_mse(y_true, y_pred):
                denom = tf.square(tf.abs(y_true) + eps)
                return tf.reduce_mean(tf.square(y_true - y_pred) / denom)
            return relative_mse

        elif self.loss_fn == 'log_mse':
            eps = tf.constant(1.0, dtype=tf.float32)
            def log_mse(y_true, y_pred):
                log_t = tf.math.log(tf.abs(y_true) + eps)
                log_p = tf.math.log(tf.abs(y_pred) + eps)
                return tf.reduce_mean(tf.square(log_t - log_p))
            return log_mse

    # ─────────────────────────────────────────────────────────────────────────
    # Model construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self):
        """Build the LSTM model architecture."""
        print("\nBuilding LSTM model...")

        inputs = layers.Input(shape=(self.max_timesteps, len(self.feature_variables)))

        x = inputs if self.use_rolling_windows else layers.Masking(mask_value=0.0)(inputs)

        if self.forecast_horizon == 0:
            # Sequence-to-sequence (aligned prediction)
            for i, units in enumerate(self.lstm_units):
                x = layers.LSTM(units, return_sequences=True, dropout=self.dropout)(x)
                if i < len(self.lstm_units) - 1:
                    x = layers.Dropout(self.dropout)(x)
            outputs = layers.TimeDistributed(layers.Dense(len(self.target_variable)))(x)
            out_shape = f"(batch, {self.max_timesteps}, {len(self.target_variable)})"
            arch_type = "Sequence-to-Sequence (aligned prediction)"
        else:
            # Encoder-Decoder (forecasting)
            for i, units in enumerate(self.lstm_units[:-1]):
                x = layers.LSTM(units, return_sequences=True, dropout=self.dropout)(x)
                x = layers.Dropout(self.dropout)(x)
            x = layers.LSTM(self.lstm_units[-1], return_sequences=False, dropout=self.dropout)(x)
            x = layers.RepeatVector(self.forecast_horizon)(x)
            x = layers.LSTM(self.lstm_units[-1], return_sequences=True, dropout=self.dropout)(x)
            outputs = layers.TimeDistributed(layers.Dense(len(self.target_variable)))(x)
            out_shape = f"(batch, {self.forecast_horizon}, {len(self.target_variable)})"
            arch_type = f"Encoder-Decoder (forecast {self.forecast_horizon} steps ahead)"

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self._get_loss(),
            metrics=['mae', 'mse']
        )

        print(f"  Input  shape : (batch, {self.max_timesteps}, {len(self.feature_variables)})")
        print(f"  Output shape : {out_shape}")
        print(f"  LSTM layers  : {self.lstm_units}")
        print(f"  Parameters   : {self.model.count_params():,}")
        print(f"  Architecture : {arch_type}")
        print(f"  Loss function: {self.loss_fn}")
        if self.use_rolling_windows:
            print(f"  Mode: Rolling Windows (size={self.window_size}, stride={self.window_stride})")
        else:
            print(f"  Mode: Variable-length sequences (with masking)")

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        """Train the LSTM model on preprocessed data."""
        if self.X_train is None:
            raise ValueError("Data not preprocessed. Call data_preprocessing() first.")

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        print("\nNormalizing data...")
        X_train_norm, y_train_norm = self._normalize_data(self.X_train, self.y_train, fit=True)

        has_val = bool(self.X_val and len(self.X_val) > 0)
        if has_val:
            X_val_norm, y_val_norm = self._normalize_data(self.X_val, self.y_val, fit=False)
        else:
            print("Warning: No validation data. Training without validation.")
            X_val_norm = y_val_norm = None

        print("Padding sequences...")
        X_tr = self._pad_sequences(X_train_norm)
        y_tr = self._pad_targets(y_train_norm)
        val_data = None
        if has_val:
            val_data = (self._pad_sequences(X_val_norm), self._pad_targets(y_val_norm))

        self._build_model()

        # Shape verification
        print("\n=== SHAPE VERIFICATION ===")
        exp_y = (len(self.y_train),
                 self.forecast_horizon if self.forecast_horizon > 0 else self.max_timesteps,
                 len(self.target_variable))
        if y_tr.shape != exp_y:
            print(f"WARNING: target shape mismatch — expected {exp_y}, got {y_tr.shape}")
        else:
            print(f"✓ Target shape verified: {y_tr.shape}")
        print("=" * 26 + "\n")

        monitor = 'val_loss' if has_val else 'loss'
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=15, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            keras.callbacks.ModelCheckpoint(
                self.model_path, monitor=monitor, save_best_only=True, verbose=1),
        ]

        print(f"Training for up to {self.epochs} epochs...")
        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\nTraining complete!")
        print("=" * 60)
        self._save_model_info()

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, verbose=1):
        """Evaluate the model on test data and return per-variable metrics."""
        if self.X_test is None:
            raise ValueError("Test data not loaded. Call data_preprocessing() first.")
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        X_test_norm, y_test_norm = self._normalize_data(self.X_test, self.y_test, fit=False)
        X_tp = self._pad_sequences(X_test_norm)
        y_tp = self._pad_targets(y_test_norm)

        print("\nEvaluating on test data...")
        self.model.evaluate(X_tp, y_tp, verbose=verbose)

        y_pred_norm = self.model.predict(X_tp, verbose=0)
        bs, ts, feats = y_pred_norm.shape
        y_pred = self._denormalize_data(y_pred_norm.reshape(-1, feats),
                                        is_target=True).reshape(bs, ts, feats)

        print("\nTest Metrics (Original Scale):")
        metrics = {}
        for i, var in enumerate(self.target_variable):
            true_all, pred_all = [], []
            for j in range(len(self.y_test)):
                sl = len(self.y_test[j])
                true_all.extend(self.y_test[j][:, i])
                pred_all.extend(y_pred[j, :sl, i])
            t, p = np.array(true_all), np.array(pred_all)
            metrics[var] = self._compute_metrics(t, p)
            self._print_metrics(var, metrics[var])

        print("\n" + "=" * 60)
        return metrics, y_pred

    def evaluate_full_sequence(self, verbose=1):
        """
        Evaluate on full test sequences by aggregating rolling-window predictions.
        Falls back to standard evaluate() when use_rolling_windows=False.
        """
        if self.X_test is None:
            raise ValueError("Test data not loaded. Call data_preprocessing() first.")
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 60)
        print("FULL SEQUENCE EVALUATION")
        print("=" * 60)

        if not self.use_rolling_windows:
            print("\nNot using rolling windows — falling back to standard evaluation.")
            return self.evaluate(verbose=verbose)

        print(f"\nReconstructing full sequences from rolling windows...")
        all_preds, all_true = [], []
        for tf_ in self.test_files:
            df = pd.read_csv(os.path.join(self.testing_path, tf_))
            all_preds.append(self.predict_full_sequence(df))
            all_true.append(df[self.target_variable].values)

        print("\nTest Metrics (Full Sequences, Original Scale):")
        metrics = {}
        for i, var in enumerate(self.target_variable):
            t = np.concatenate([s[:, i] for s in all_true])
            p = np.concatenate([s[:, i] for s in all_preds])
            valid = ~np.isnan(t) & ~np.isnan(p)
            t, p = t[valid], p[valid]
            if len(t) == 0:
                print(f"\nWarning: No valid predictions for {var}.")
                continue
            metrics[var] = self._compute_metrics(t, p)
            self._print_metrics(var, metrics[var])
            print(f"    Timesteps evaluated: {len(t):,}")

        print("\n" + "=" * 60)
        return metrics, all_preds

    def evaluate_power_bands(self, y_true_flat, y_pred_flat, bands=None):
        """
        Break out evaluation metrics by power band.

        Useful for verifying low-power capture independently of full-power
        accuracy. Default bands are tuned for the NRAD 0–250 kW range with
        emphasis on the startup regime.

        Parameters
        ----------
        y_true_flat : 1-D array   (original-scale, same units as training data)
        y_pred_flat : 1-D array
        bands : list of (label, low, high), optional
            Custom power bands. Default NRAD bands used if None.

        Returns
        -------
        dict keyed by band label with keys: n, RMSE, MAPE, R2
        """
        if bands is None:
            bands = [
                ("startup   0–50 W",      0,        50),
                ("low      50 W–1 kW",   50,     1_000),
                ("mid       1–10 kW",  1_000,   10_000),
                ("high    10–250 kW", 10_000,  250_000),
            ]

        y_true_flat = np.asarray(y_true_flat)
        y_pred_flat = np.asarray(y_pred_flat)

        results = {}
        print("\nPower-Band Evaluation")
        print(f"  {'Band':<24} {'N':>8}  {'RMSE':>12}  {'MAPE %':>8}  {'R²':>7}")
        print("  " + "-" * 66)

        for label, lo, hi in bands:
            mask = (y_true_flat >= lo) & (y_true_flat < hi)
            n = mask.sum()
            if n == 0:
                print(f"  {label:<24} {'—':>8}")
                continue
            t, p = y_true_flat[mask], y_pred_flat[mask]
            rmse = np.sqrt(np.mean((t - p) ** 2))
            nz   = t != 0
            mape = np.mean(np.abs((t[nz] - p[nz]) / t[nz])) * 100 if nz.any() else np.nan
            ss_r = np.sum((t - p) ** 2)
            ss_t = np.sum((t - t.mean()) ** 2)
            r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
            results[label] = {'n': n, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
            mape_s = f"{mape:8.1f}" if not np.isnan(mape) else "     N/A"
            r2_s   = f"{r2:7.4f}"   if not np.isnan(r2)   else "    N/A"
            print(f"  {label:<24} {n:>8,}  {rmse:>12.4f}  {mape_s}  {r2_s}")

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(t, p):
        """Compute standard regression metrics given flat true/pred arrays."""
        mse  = np.mean((t - p) ** 2)
        mae  = np.mean(np.abs(t - p))
        rmse = np.sqrt(mse)
        me   = np.max(np.abs(t - p))
        nz   = t != 0
        mape = np.mean(np.abs((t[nz] - p[nz]) / t[nz])) * 100 if nz.any() else np.nan
        ss_r = np.sum((t - p) ** 2)
        ss_t = np.sum((t - t.mean()) ** 2)
        r2   = 1 - ss_r / ss_t
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'ME': me, 'MAPE': mape, 'R²': r2}

    @staticmethod
    def _print_metrics(var, m):
        print(f"\n  {var}:")
        print(f"    MSE  : {m['MSE']:.6f}")
        print(f"    MAE  : {m['MAE']:.6f}")
        print(f"    RMSE : {m['RMSE']:.6f}")
        print(f"    ME   : {m['ME']:.6f}")
        if not np.isnan(m['MAPE']):
            print(f"    MAPE : {m['MAPE']:.2f}%")
        else:
            print(f"    MAPE : N/A (zero values present)")
        print(f"    R²   : {m['R²']:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict_full_sequence(self, csv_file_or_dataframe):
        """
        Predict the target variable for an entire sequence by aggregating
        rolling-window predictions. Overlapping windows are averaged.

        Returns
        -------
        np.ndarray, shape (timesteps, n_targets)
            forecast_horizon=0: predictions align with all input timesteps.
            forecast_horizon>0: first window_size timesteps are NaN.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 70)
        print("DEBUG: predict_full_sequence()")
        print("=" * 70)

        if isinstance(csv_file_or_dataframe, str):
            df = pd.read_csv(csv_file_or_dataframe)
            print(f"Loaded CSV: {os.path.basename(csv_file_or_dataframe)}")
        else:
            df = csv_file_or_dataframe
            print("Using DataFrame")

        X_full   = df[self.feature_variables].values
        seq_len  = len(X_full)
        print(f"Sequence length: {seq_len}")

        if not self.use_rolling_windows:
            print("Mode: Full sequence (non-rolling)")
            print("=" * 70 + "\n")
            return self.predict(csv_file_or_dataframe)

        print(f"\nWindow parameters: size={self.window_size}, "
              f"stride={self.window_stride}, horizon={self.forecast_horizon}")

        max_start = seq_len - self.window_size - self.forecast_horizon
        print(f"Max start index : {max_start}")

        X_windows, window_starts = [], []
        for start in range(0, max_start + 1, self.window_stride):
            X_windows.append(X_full[start:start + self.window_size])
            window_starts.append(start)

        print(f"Windows created : {len(X_windows)}")
        if not X_windows:
            raise ValueError(f"No windows created. Seq length: {seq_len}, "
                             f"need >= {self.window_size + self.forecast_horizon}")

        X_norm, _ = self._normalize_data(X_windows, fit=False)
        X_pad     = self._pad_sequences(X_norm)
        pred_norm = self.model.predict(X_pad, verbose=0)

        print(f"\nModel I/O — Input: {X_pad.shape}, Output: {pred_norm.shape}")

        bs, out_ts, feats = pred_norm.shape
        pred_denorm = self._denormalize_data(
            pred_norm.reshape(-1, feats), is_target=True).reshape(bs, out_ts, feats)

        full_pred = np.full((seq_len, len(self.target_variable)), np.nan, dtype=float)
        counts    = np.zeros(seq_len, dtype=float)

        if self.forecast_horizon == 0:
            for i, s in enumerate(window_starts):
                for j in range(out_ts):
                    t = s + j
                    if t < seq_len:
                        full_pred[t] = (pred_denorm[i, j] if np.isnan(full_pred[t, 0])
                                        else full_pred[t] + pred_denorm[i, j])
                        counts[t] += 1
        else:
            for i, s in enumerate(window_starts):
                p_start = s + self.window_size
                for j in range(out_ts):
                    t = p_start + j
                    if t < seq_len:
                        full_pred[t] = (pred_denorm[i, j] if np.isnan(full_pred[t, 0])
                                        else full_pred[t] + pred_denorm[i, j])
                        counts[t] += 1

        for t in range(seq_len):
            if counts[t] > 1:
                full_pred[t] /= counts[t]

        coverage = int(np.sum(~np.isnan(full_pred[:, 0])))
        print(f"Coverage: {coverage}/{seq_len} timesteps")
        print("=" * 70 + "\n")
        return full_pred

    def predict(self, csv_file_or_dataframe, return_all_windows=False):
        """
        Make predictions on new data.

        Parameters
        ----------
        csv_file_or_dataframe : str or DataFrame
        return_all_windows : bool
            Rolling-window mode only. If True, returns all window predictions
            with shape (n_windows, output_timesteps, n_targets).
            If False, returns the final window's predictions.

        Returns
        -------
        np.ndarray
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df = pd.read_csv(csv_file_or_dataframe) if isinstance(
            csv_file_or_dataframe, str) else csv_file_or_dataframe
        X_full = df[self.feature_variables].values

        if self.use_rolling_windows:
            seq_len = len(X_full)
            if seq_len < self.window_size:
                raise ValueError(f"Input length ({seq_len}) < window size ({self.window_size})")
            X_windows = [X_full[s:s + self.window_size]
                         for s in range(0, seq_len - self.window_size + 1, self.window_stride)]
            X_norm, _ = self._normalize_data(X_windows, fit=False)
            X_pad     = self._pad_sequences(X_norm)
            pn        = self.model.predict(X_pad, verbose=0)
            bs, ts, f = pn.shape
            preds = self._denormalize_data(pn.reshape(-1, f),
                                           is_target=True).reshape(bs, ts, f)
            return preds if return_all_windows else preds[-1]
        else:
            X_norm, _ = self._normalize_data([X_full], fit=False)
            X_pad     = self._pad_sequences(X_norm)
            pn        = self.model.predict(X_pad, verbose=0)
            bs, ts, f = pn.shape
            preds = self._denormalize_data(pn.reshape(-1, f),
                                           is_target=True).reshape(bs, ts, f)
            actual = len(df)
            if self.forecast_horizon > 0:
                out = np.full((actual, f), np.nan, dtype=float)
                out[self.forecast_horizon:, :] = preds[0, :actual - self.forecast_horizon, :]
                return out
            return preds[0, :actual, :]

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, base_path=None):
        """
        Save the model, scalers, and configuration.

        Creates:
            {base_path}_model.keras
            {base_path}_model.h5
            {base_path}_config.pkl
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if base_path is None:
            base_path = str(Path(self.model_path).with_suffix(''))

        model_keras = f"{base_path}_model.keras"
        self.model.save(model_keras)
        print(f"Saved Keras model : {model_keras}")

        model_h5 = f"{base_path}_model.h5"
        self.model.save(model_h5, save_format='h5')
        print(f"Saved H5 model    : {model_h5}")

        config = {
            'feature_variables':    self.feature_variables,
            'target_variable':      self.target_variable,
            'lstm_units':           self.lstm_units,
            'dropout':              self.dropout,
            'normalization_method': self.normalization_method,
            'use_rolling_windows':  self.use_rolling_windows,
            'window_size':          self.window_size,
            'window_stride':        self.window_stride,
            'forecast_horizon':     self.forecast_horizon,
            'target_log_transform': self.target_log_transform,
            'loss_fn':              self.loss_fn,
            'max_timesteps':        self.max_timesteps,
            'scaler_X':             self.scaler_X,
            'scaler_y':             self.scaler_y,
            'epochs':               self.epochs,
            'batch_size':           self.batch_size,
            'validation_split':     self.validation_split,
        }
        cfg_file = f"{base_path}_config.pkl"
        with open(cfg_file, 'wb') as fh:
            pickle.dump(config, fh)
        print(f"Saved config      : {cfg_file}")
        print(f"\nTo reload: ModelGeneration.load('{base_path}')")

    @classmethod
    def load(cls, base_path):
        """
        Load a saved ModelGeneration instance.

        Parameters
        ----------
        base_path : str
            Base path used when save() was called (without extension).

        Returns
        -------
        ModelGeneration ready for prediction.
        """
        cfg_file = f"{base_path}_config.pkl"
        with open(cfg_file, 'rb') as fh:
            cfg = pickle.load(fh)

        print(f"Loading model from : {base_path}")
        print(f"  Features         : {cfg['feature_variables']}")
        print(f"  Targets          : {cfg['target_variable']}")
        print(f"  Rolling windows  : {cfg['use_rolling_windows']}")
        print(f"  Forecast horizon : {cfg.get('forecast_horizon', 0)}")
        print(f"  Log transform    : {cfg.get('target_log_transform', False)}")
        print(f"  Loss function    : {cfg.get('loss_fn', 'mse')}")

        instance = cls(
            model_path=f"{base_path}_model.keras",
            testing_path="",
            training_path="",
            text_file_name="",
            feature_variables=cfg['feature_variables'],
            target_variable=cfg['target_variable'],
            epochs=cfg['epochs'],
            lstm_units=cfg['lstm_units'],
            dropout=cfg['dropout'],
            normalization_method=cfg['normalization_method'],
            batch_size=cfg['batch_size'],
            validation_split=cfg['validation_split'],
            use_rolling_windows=cfg['use_rolling_windows'],
            window_size=cfg['window_size'],
            window_stride=cfg['window_stride'],
            forecast_horizon=cfg.get('forecast_horizon', 0),
            target_log_transform=cfg.get('target_log_transform', False),
            loss_fn=cfg.get('loss_fn', 'mse'),
        )

        instance.model     = keras.models.load_model(f"{base_path}_model.keras")
        instance.scaler_X  = cfg['scaler_X']
        instance.scaler_y  = cfg['scaler_y']
        instance.max_timesteps = cfg['max_timesteps']

        print("Model loaded successfully!")
        return instance

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _save_model_info(self):
        """Save model configuration and training history to text file."""
        with open(self.text_file_name, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LSTM MODEL CONFIGURATION\n")
            f.write("=" * 60 + "\n\n")

            f.write("Model Architecture:\n")
            f.write(f"  LSTM Units        : {self.lstm_units}\n")
            f.write(f"  Dropout           : {self.dropout}\n")
            f.write(f"  Total Parameters  : {self.model.count_params():,}\n\n")

            f.write("Training Configuration:\n")
            f.write(f"  Epochs            : {self.epochs}\n")
            f.write(f"  Batch Size        : {self.batch_size}\n")
            f.write(f"  Normalization     : {self.normalization_method}\n")
            f.write(f"  Validation Split  : {self.validation_split}\n")
            f.write(f"  Loss Function     : {self.loss_fn}\n\n")

            f.write("Target Configuration:\n")
            f.write(f"  Log Transform     : {self.target_log_transform}\n")
            if self.target_log_transform:
                f.write(f"  Transform         : log1p (forward) / expm1 (inverse)\n\n")
            else:
                f.write("\n")

            f.write("Data Mode:\n")
            if self.use_rolling_windows:
                f.write(f"  Mode              : Rolling Windows\n")
                f.write(f"  Window Size       : {self.window_size}\n")
                f.write(f"  Window Stride     : {self.window_stride}\n")
                f.write(f"  Forecast Horizon  : {self.forecast_horizon}\n")
                pred_type = ("Aligned (nowcasting)" if self.forecast_horizon == 0
                             else f"Forecasting ({self.forecast_horizon} steps ahead)")
                f.write(f"  Prediction Type   : {pred_type}\n")
                f.write(f"  Sequence Length   : {self.max_timesteps} (fixed)\n")
            else:
                f.write(f"  Mode              : Variable-length sequences\n")
                f.write(f"  Max Seq Length    : {self.max_timesteps}\n")
                if self.forecast_horizon > 0:
                    f.write(f"  Forecast Horizon  : {self.forecast_horizon}\n")
            f.write("\n")

            f.write("Data Configuration:\n")
            f.write(f"  Feature Variables : {self.feature_variables}\n")
            f.write(f"  Target Variables  : {self.target_variable}\n")
            f.write(f"  Training Samples  : {len(self.X_train)}\n")
            f.write(f"  Validation Samples: {len(self.X_val) if self.X_val else 0}\n")
            f.write(f"  Testing Samples   : {len(self.X_test)}\n\n")

            f.write("Paths:\n")
            f.write(f"  Model             : {self.model_path}\n")
            f.write(f"  Training Data     : {self.training_path}\n")
            f.write(f"  Testing Data      : {self.testing_path}\n\n")

            if self.history:
                f.write("Training History (Final Epoch):\n")
                f.write(f"  Loss              : {self.history.history['loss'][-1]:.6f}\n")
                if 'val_loss' in self.history.history:
                    f.write(f"  Val Loss          : {self.history.history['val_loss'][-1]:.6f}\n")
                f.write(f"  MAE               : {self.history.history['mae'][-1]:.6f}\n")
                if 'val_mae' in self.history.history:
                    f.write(f"  Val MAE           : {self.history.history['val_mae'][-1]:.6f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("=" * 60 + "\n")
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        print(f"\nModel information saved to: {self.text_file_name}")