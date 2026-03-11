"""Final preprocessing pipeline for modelling-ready clickstream data."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    ROOT_DIR / "preprocessing_output" / "combined_clickstream_dataset.csv",
    ROOT_DIR / "combined_clickstream_dataset.csv",
]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
FINAL_DATASET_PATH = OUTPUT_DIR / "final_clickstream_dataset.csv"
FINAL_REPORT_PATH = OUTPUT_DIR / "dataset_quality_report_final.json"
PREPROCESSOR_PATH = OUTPUT_DIR / "final_preprocessing_artifacts.pkl"

REDUNDANT_FEATURES = [
    "logs_count",
    "request_interval_entropy",
    "request_interval_mean",
    "click_interval_std",
]
NUMERIC_CAPS = {
    "clicks_per_minute": 1500,
    "requests_per_minute": 1000,
    "mouse_path_length": 50000,
    "direction_change_count": 5000,
    "total_movements": 5000,
    "session_duration_sec": 10000,
}
CATEGORICAL_FILL_COLUMNS = ["browser", "device_type", "country", "region", "operating_system"]
PROTECTED_COLUMNS = {
    "label",
    "bot_likelihood_score",
    "anomaly_score",
    "click_interval_mean",
    "click_interval_entropy",
    "request_interval_std",
    "click_count",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "movement_curvature",
    "mouse_acceleration_std",
    "coordinate_entropy",
    "total_movements",
    "session_duration_sec",
    "session_idle_ratio",
    "burstiness",
    "successful_requests",
    "success_rate",
    "clicks_per_minute",
    "requests_per_minute",
    "browser",
    "operating_system",
    "device_type",
    "country",
    "region",
    "is_proxy",
    "device_entropy",
    "app_entropy",
    "channel_entropy",
}
PREFERRED_KEEP = {
    frozenset({"click_count", "total_requests"}): "click_count",
    frozenset({"original_device", "device"}): "device",
    frozenset({"original_os", "os"}): "os",
}
FEATURE_GROUPS = {
    "behavioral_features": [
        "mouse_speed_mean",
        "mouse_speed_std",
        "mouse_path_length",
        "direction_change_count",
        "movement_std",
        "movement_curvature",
        "mouse_acceleration_std",
        "coordinate_entropy",
        "total_movements",
    ],
    "temporal_features": [
        "click_interval_mean",
        "click_interval_entropy",
        "request_interval_std",
        "session_duration_sec",
        "session_idle_ratio",
        "burstiness",
    ],
    "activity_features": [
        "click_count",
        "total_requests",
        "successful_requests",
        "success_rate",
        "clicks_per_minute",
        "requests_per_minute",
    ],
    "device_network_features": [
        "browser",
        "operating_system",
        "device_type",
        "country",
        "region",
        "is_proxy",
        "device_entropy",
        "app_entropy",
        "channel_entropy",
    ],
}
NON_MODELLING_COLUMNS = [
    "data_source_detail",
    "candidate_human",
    "session_start",
    "session_end",
    "activity_date",
    "time_range",
    "click_interval_median",
]


def resolve_input_path() -> Path:
    """Resolve the latest combined dataset path."""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("combined_clickstream_dataset.csv not found in expected locations.")


def load_dataset() -> pd.DataFrame:
    """Load the combined dataset and add derived temporal burstiness."""
    df = pd.read_csv(resolve_input_path())
    if "burstiness" not in df.columns:
        df["burstiness"] = (
            (df["request_interval_std"].fillna(0) - df["click_interval_mean"].fillna(0))
            / (df["request_interval_std"].fillna(0) + df["click_interval_mean"].fillna(0) + 1e-9)
        )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with 'unknown' and numeric values with median."""
    cleaned = df.copy()
    for column in CATEGORICAL_FILL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("unknown")

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    remaining_object_columns = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    for column in remaining_object_columns:
        cleaned[column] = cleaned[column].fillna("unknown")
    return cleaned


def remove_redundant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop known manually identified redundant features."""
    removable = [column for column in REDUNDANT_FEATURES if column in df.columns]
    trimmed = df.drop(columns=removable, errors="ignore").copy()
    return trimmed, removable


def cap_extreme_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winsorize selected high-variance numeric columns using fixed caps."""
    capped = df.copy()
    applied_caps = {}
    for column, upper_bound in NUMERIC_CAPS.items():
        if column in capped.columns:
            capped[column] = np.clip(capped[column], a_min=None, a_max=upper_bound)
            applied_caps[column] = upper_bound
    return capped, applied_caps


def _choose_column_to_drop(col_a: str, col_b: str) -> str:
    pair = frozenset({col_a, col_b})
    if pair in PREFERRED_KEEP:
        keep = PREFERRED_KEEP[pair]
        return col_b if keep == col_a else col_a
    if col_a in PROTECTED_COLUMNS and col_b not in PROTECTED_COLUMNS:
        return col_b
    if col_b in PROTECTED_COLUMNS and col_a not in PROTECTED_COLUMNS:
        return col_a
    return sorted([col_a, col_b])[1]


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.98) -> tuple[pd.DataFrame, dict]:
    """Drop one column from highly correlated numeric pairs and perfectly duplicated columns."""
    working = df.copy()
    removed_numeric = []
    removed_duplicate = []
    retained_protected_pairs = []

    numeric = working.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    for column in upper.columns:
        correlated = [idx for idx, value in upper[column].dropna().items() if value > threshold]
        for other in correlated:
            if column in PROTECTED_COLUMNS and other in PROTECTED_COLUMNS:
                retained_protected_pairs.append(
                    {"feature_a": other, "feature_b": column, "correlation": float(upper.loc[other, column])}
                )
                continue
            to_drop = _choose_column_to_drop(column, other)
            if to_drop in working.columns and to_drop not in removed_numeric:
                removed_numeric.append(to_drop)
    working = working.drop(columns=sorted(set(removed_numeric)), errors="ignore")

    columns = working.columns.tolist()
    for idx, column in enumerate(columns):
        for other in columns[idx + 1 :]:
            if column not in working.columns or other not in working.columns:
                continue
            if working[column].equals(working[other]):
                to_drop = _choose_column_to_drop(column, other)
                if to_drop in working.columns:
                    working = working.drop(columns=[to_drop])
                    removed_duplicate.append(to_drop)

    summary = {
        "threshold": threshold,
        "removed_numeric_correlation": sorted(set(removed_numeric)),
        "removed_duplicate_columns": sorted(set(removed_duplicate)),
        "retained_protected_pairs": retained_protected_pairs,
    }
    return working, summary


def define_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return only feature-group members that remain after cleaning."""
    return {
        group_name: [column for column in columns if column in df.columns]
        for group_name, columns in FEATURE_GROUPS.items()
    }


def drop_non_modelling_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove helper columns that are useful for auditability but not for modelling."""
    removable = [column for column in NON_MODELLING_COLUMNS if column in df.columns]
    return df.drop(columns=removable, errors="ignore").copy(), removable


def build_preprocessing_pipeline(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the sklearn-compatible preprocessing stack for future train-only fitting."""
    categorical_features = [
        column
        for column in ["browser", "operating_system", "device_type", "country", "region", "data_source"]
        if column in df.columns
    ]
    numeric_features = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"label", "bot_likelihood_score", "anomaly_score"}
    ]

    pipeline = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return pipeline, numeric_features, categorical_features


def scale_numerical_features(df: pd.DataFrame, numeric_features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Apply RobustScaler to numeric features in-place for export."""
    scaled = df.copy()
    scaler = RobustScaler()
    scaled[numeric_features] = scaler.fit_transform(scaled[numeric_features])
    summary = {
        "scaled_numeric_features": numeric_features,
        "excluded_from_scaling": ["label", "bot_likelihood_score"],
        "scaler": "RobustScaler",
    }
    return scaled, summary


def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add IsolationForest anomaly scores without removing any rows."""
    enriched = df.copy()
    numeric_features = enriched.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")
    model = IsolationForest(contamination=0.03, random_state=RANDOM_SEED)
    model.fit(numeric_features)
    scores = -model.decision_function(numeric_features)
    enriched["anomaly_score"] = scores
    summary = {
        "outlier_count": int((model.predict(numeric_features) == -1).sum()),
        "anomaly_score_min": float(scores.min()),
        "anomaly_score_mean": float(scores.mean()),
        "anomaly_score_max": float(scores.max()),
    }
    return enriched, summary


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset integrity after preprocessing."""
    missing_total = int(df.isna().sum().sum())
    class_distribution = df["label"].value_counts().sort_index().to_dict()
    print("dataset shape:", df.shape)
    print("class distribution:", class_distribution)
    print("missing value count:", missing_total)
    print("no NaNs remain:", missing_total == 0)
    return {
        "dataset_shape": list(df.shape),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "missing_value_count": missing_total,
        "no_nans_remain": missing_total == 0,
    }


def save_outputs(df: pd.DataFrame, preprocessor: ColumnTransformer, report: dict) -> None:
    """Persist the cleaned dataset, reusable preprocessing object, and report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINAL_DATASET_PATH, index=False)
    with PREPROCESSOR_PATH.open("wb") as handle:
        pickle.dump(preprocessor, handle)
    with FINAL_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    df = load_dataset()
    original_shape = list(df.shape)
    original_distribution = df["label"].value_counts().sort_index().to_dict()

    df = handle_missing_values(df)
    df, removed_manual = remove_redundant_columns(df)
    df, cap_summary = cap_extreme_values(df)
    df, corr_summary = remove_highly_correlated_features(df, threshold=0.98)
    df, dropped_auxiliary = drop_non_modelling_columns(df)
    feature_groups = define_feature_groups(df)
    preprocessor, numeric_features, categorical_features = build_preprocessing_pipeline(df)
    df, scaling_summary = scale_numerical_features(df, numeric_features)
    df, outlier_summary = detect_anomalies(df)
    validation_summary = validate_dataset(df)

    preprocessor.fit(df.drop(columns=["label"], errors="ignore"))

    report = {
        "input_dataset_shape": original_shape,
        "output_dataset_shape": list(df.shape),
        "class_distribution_before": {str(k): int(v) for k, v in original_distribution.items()},
        "class_distribution_after": validation_summary["class_distribution"],
        "removed_features": {
            "manual": removed_manual,
            "correlation_filter": corr_summary,
            "auxiliary_non_modelling": dropped_auxiliary,
        },
        "caps_applied": cap_summary,
        "scaling_summary": scaling_summary,
        "categorical_features_for_encoding": categorical_features,
        "feature_groups": feature_groups,
        "correlation_summary": corr_summary,
        "outlier_statistics": outlier_summary,
        "validation_summary": validation_summary,
    }
    save_outputs(df, preprocessor, report)


if __name__ == "__main__":
    main()
