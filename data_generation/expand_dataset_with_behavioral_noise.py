"""Expand the final modelling dataset with realistic behavioural noise."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "final_clickstream_dataset_model_ready.csv"
OUTPUT_DIR = ROOT_DIR / "data_balanced"
BALANCED_PATH = OUTPUT_DIR / "balanced_clickstream_dataset.csv"
REPORT_PATH = OUTPUT_DIR / "balanced_dataset_report.json"
PLOT_DIR = ROOT_DIR / "analysis_outputs_balanced"
PLOT_PATH = PLOT_DIR / "class_distribution.png"

TARGET_COUNTS = {0: 5013, 1: 1000, 2: 1000}
PROTECTED_COLUMNS = {
    "label",
    "label_name",
    "session_id",
    "ip_address",
    "country",
    "region",
    "browser",
    "device_type",
    "operating_system",
    "data_source",
    "bot_type",
    "user_agent",
}
NOISE_FEATURES = [
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "coordinate_entropy",
    "click_interval_entropy",
    "requests_per_minute",
    "session_idle_ratio",
    "burstiness",
    "movement_std",
    "mouse_acceleration_std",
    "click_interval_mean",
    "request_interval_std",
    "session_duration_sec",
    "click_count",
    "total_movements",
]
BOT_NOISE = {
    "mouse_speed_mean": 0.2,
    "mouse_speed_std": 0.1,
    "mouse_path_length": 0.5,
    "coordinate_entropy": 0.1,
    "click_interval_entropy": 0.05,
    "requests_per_minute": 10.0,
    "session_idle_ratio": 0.02,
    "burstiness": 0.05,
    "movement_std": 0.1,
    "mouse_acceleration_std": 0.1,
    "click_interval_mean": 0.08,
    "request_interval_std": 0.08,
    "session_duration_sec": 0.2,
    "click_count": 0.5,
    "total_movements": 0.5,
}
HUMAN_NOISE = {
    "mouse_speed_mean": 0.1,
    "click_interval_entropy": 0.02,
    "requests_per_minute": 5.0,
}


def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the final modelling dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def get_noise_features(df: pd.DataFrame) -> list[str]:
    """Return only noise-eligible numeric features present in the dataset."""
    return [
        column
        for column in NOISE_FEATURES
        if column in df.columns and column not in PROTECTED_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def compute_clip_ranges(df: pd.DataFrame, noise_features: list[str]) -> dict[str, tuple[float, float]]:
    """Build quantile-based clipping ranges from the original dataset."""
    clip_ranges = {}
    for column in noise_features:
        series = df[column].astype(float)
        lower = float(series.quantile(0.01))
        upper = float(series.quantile(0.99))
        span = max(upper - lower, 1e-6)
        clip_ranges[column] = (max(0.0, lower - 0.1 * span), upper + 0.1 * span)
    return clip_ranges


def apply_noise(
    df: pd.DataFrame,
    noise_parameters: dict[str, float],
    clip_ranges: dict[str, tuple[float, float]],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Apply Gaussian behavioural noise and clip to valid ranges."""
    noisy = df.copy()
    for column, std_dev in noise_parameters.items():
        if column not in noisy.columns:
            continue
        noisy[column] = noisy[column].astype(float) + rng.normal(0, std_dev, size=len(noisy))
        lower, upper = clip_ranges[column]
        noisy[column] = noisy[column].clip(lower=lower, upper=upper)
        if column in {"click_count", "total_movements"}:
            noisy[column] = np.rint(noisy[column]).astype(int)
    return noisy


def expand_bot_class(
    df: pd.DataFrame,
    label: int,
    target_count: int,
    noise_features: list[str],
    clip_ranges: dict[str, tuple[float, float]],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Duplicate and perturb bot samples until the target class count is reached."""
    subset = df[df["label"] == label].copy()
    needed = max(0, target_count - len(subset))
    if needed == 0:
        return subset

    synthetic_rows = []
    while len(synthetic_rows) < needed:
        sample = subset.sample(n=min(needed - len(synthetic_rows), len(subset)), replace=True, random_state=rng.randint(0, 1_000_000)).copy()
        sample = apply_noise(sample, {key: BOT_NOISE[key] for key in noise_features if key in BOT_NOISE}, clip_ranges, rng)
        synthetic_rows.append(sample)

    augmented = pd.concat([subset] + synthetic_rows, ignore_index=True)
    return augmented.iloc[:target_count].copy()


def apply_human_noise(
    df: pd.DataFrame,
    noise_features: list[str],
    clip_ranges: dict[str, tuple[float, float]],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Apply light behavioural noise to a random subset of human rows."""
    humans = df[df["label"] == 0].copy()
    if humans.empty:
        return humans

    sampled_idx = humans.sample(frac=0.30, random_state=RANDOM_SEED).index
    noisy_slice = humans.loc[sampled_idx].copy()
    noisy_slice = apply_noise(noisy_slice, {key: HUMAN_NOISE[key] for key in noise_features if key in HUMAN_NOISE}, clip_ranges, rng)
    humans.loc[sampled_idx, noisy_slice.columns] = noisy_slice
    return humans


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates while preserving schema."""
    return df.drop_duplicates().reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate class counts, schema integrity, and missing values."""
    missing = int(df.isna().sum().sum())
    class_distribution = df["label"].value_counts().sort_index().to_dict()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "final_dataset_size": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "feature_count": int(df.shape[1]),
        "missing_values": missing,
        "numeric_columns": numeric_columns,
    }


def save_plot(df: pd.DataFrame) -> None:
    """Optionally save a quick class-distribution plot."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    counts = df["label"].map({0: "human", 1: "moderate_bot", 2: "advanced_bot"}).value_counts()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=["#2e8b57", "#f4a261", "#d62828"])
    plt.title("Balanced Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = check_random_state(RANDOM_SEED)

    original = load_dataset()
    schema = original.columns.tolist()
    noise_features = get_noise_features(original)
    clip_ranges = compute_clip_ranges(original, noise_features)

    humans = apply_human_noise(original, noise_features, clip_ranges, rng)
    moderate = expand_bot_class(original, 1, TARGET_COUNTS[1], noise_features, clip_ranges, rng)
    advanced = expand_bot_class(original, 2, TARGET_COUNTS[2], noise_features, clip_ranges, rng)

    balanced = pd.concat([humans, moderate, advanced], ignore_index=True)
    balanced = balanced[schema].copy()
    balanced = remove_exact_duplicates(balanced)

    for column in original.select_dtypes(include=[np.number]).columns:
        balanced[column] = pd.to_numeric(balanced[column], errors="coerce")
        if balanced[column].isna().any():
            balanced[column] = balanced[column].fillna(original[column].median())

    validation = validate_dataset(balanced)
    balanced.to_csv(BALANCED_PATH, index=False)
    save_plot(balanced)

    report = {
        "original_rows": int(len(original)),
        "new_rows": int(len(balanced)),
        "human_count": int((balanced["label"] == 0).sum()),
        "moderate_bot_count": int((balanced["label"] == 1).sum()),
        "advanced_bot_count": int((balanced["label"] == 2).sum()),
        "noise_features_used": noise_features,
        "noise_parameters": {
            "bot_noise": {key: value for key, value in BOT_NOISE.items() if key in noise_features},
            "human_noise": {key: value for key, value in HUMAN_NOISE.items() if key in noise_features},
        },
        "validation": validation,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Final dataset size:", len(balanced))
    print("Class distribution:", balanced["label"].value_counts().sort_index().to_dict())
    print("Feature count:", balanced.shape[1])
    print("Missing values:", int(balanced.isna().sum().sum()))
    print("Balanced dataset path:", BALANCED_PATH)


if __name__ == "__main__":
    main()
