"""Minimal feature cleanup step for the final modelling dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
INPUT_PATH = ROOT_DIR / "data" / "processed" / "final_clickstream_dataset.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "final_clickstream_dataset_model_ready.csv"
REPORT_PATH = ROOT_DIR / "data" / "processed" / "final_modelling_dataset_report.json"

EXPECTED_COUNTS = {0: 5013, 1: 263, 2: 263}
EXPECTED_ROWS = 5539
DROP_COLUMNS = [
    "device_entropy",
    "original_ip",
    "app",
    "device",
    "os",
    "channel",
    "success_rate",
]


def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the already-processed dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def remove_problematic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove only the explicitly requested modelling-risk features."""
    removable = [column for column in DROP_COLUMNS if column in df.columns]
    cleaned = df.drop(columns=removable, errors="ignore").copy()
    return cleaned, removable


def verify_dataset(original: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Verify rows, labels, and missing-value integrity."""
    original_counts = original["label"].value_counts().sort_index().to_dict()
    cleaned_counts = cleaned["label"].value_counts().sort_index().to_dict()
    verification = {
        "row_count_unchanged": len(cleaned) == len(original) == EXPECTED_ROWS,
        "label_counts_unchanged": cleaned_counts == original_counts == EXPECTED_COUNTS,
        "label_column_unchanged": original["label"].equals(cleaned["label"]),
        "missing_values_zero": int(cleaned.isna().sum().sum()) == 0,
    }
    verification["verification_passed"] = all(verification.values())
    return verification


def save_outputs(df: pd.DataFrame, report: dict) -> None:
    """Save the final modelling dataset and report."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    original = load_dataset()
    print("dataset shape:", original.shape)
    print("column list:", original.columns.tolist())
    print("class distribution:", original["label"].value_counts().sort_index().to_dict())

    cleaned, removed = remove_problematic_features(original)
    verification = verify_dataset(original, cleaned)

    report = {
        "dataset_shape": list(cleaned.shape),
        "remaining_features": cleaned.columns.tolist(),
        "removed_features": removed,
        "class_distribution": {str(k): int(v) for k, v in cleaned["label"].value_counts().sort_index().to_dict().items()},
        "missing_values": int(cleaned.isna().sum().sum()),
        "verification_passed": verification["verification_passed"],
        "verification_details": verification,
    }
    save_outputs(cleaned, report)

    print("Original feature count:", len(original.columns))
    print("Final feature count:", len(cleaned.columns))
    print("Removed features:", removed)
    print("Final dataset path:", OUTPUT_PATH)
    print("Final dataset ready for modelling.")


if __name__ == "__main__":
    main()
