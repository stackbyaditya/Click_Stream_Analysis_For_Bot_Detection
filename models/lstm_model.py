"""LSTM utilities for behavioural bot detection."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


RANDOM_SEED = 42
SEQUENCE_FEATURES = [
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "mouse_acceleration_std",
    "movement_curvature",
    "click_interval_entropy",
    "click_interval_mean",
    "requests_per_minute",
    "clicks_per_minute",
]


def set_lstm_seed(seed: int = RANDOM_SEED) -> None:
    """Set deterministic seeds for the LSTM stage."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def build_session_sequences(
    df: pd.DataFrame,
    session_ids: pd.Series,
    feature_columns: list[str] | None = None,
    sequence_length: int = 10,
) -> np.ndarray:
    """Build padded per-session sequences from session-level features.

    The final modelling dataset is one row per session. When only one row is available,
    the session vector is repeated to create a fixed-length sequence for the LSTM stage.
    """
    feature_columns = feature_columns or SEQUENCE_FEATURES
    available = [column for column in feature_columns if column in df.columns]
    working = df.copy()
    working["session_id"] = session_ids.to_numpy()

    sequences: list[np.ndarray] = []
    for _, group in working.groupby("session_id", sort=False):
        values = group[available].to_numpy(dtype=np.float32)
        if len(values) >= sequence_length:
            seq = values[:sequence_length]
        elif len(values) == 1:
            seq = np.repeat(values, sequence_length, axis=0)
        else:
            padding = np.repeat(values[-1:, :], sequence_length - len(values), axis=0)
            seq = np.vstack([values, padding])
        sequences.append(seq.astype(np.float32))

    return np.asarray(sequences, dtype=np.float32)


def create_lstm_model(input_shape: tuple[int, int], n_classes: int = 3) -> tf.keras.Model:
    """Build the requested LSTM architecture."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    class_weight: dict[int, float],
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train and save the LSTM model."""
    set_lstm_seed()
    model = create_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), n_classes=len(np.unique(y_train)))
    history = model.fit(
        X_train_seq,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight,
        verbose=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return model, history


def predict_lstm_probabilities(model: tf.keras.Model, X_test_seq: np.ndarray) -> np.ndarray:
    """Generate class probabilities for the test split."""
    return model.predict(X_test_seq, verbose=0)
