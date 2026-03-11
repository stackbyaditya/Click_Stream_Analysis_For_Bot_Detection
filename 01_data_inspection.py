"""
STEP 1: DATA INSPECTION
========================
Load all CSV datasets and analyze structure, column names, data types, and record counts.
Detect linking keys and check for existing label columns.
"""

import pandas as pd
import os
import json
from pathlib import Path
import sys

# Fix encoding issues on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASET_DIR = Path("Datasets")
OUTPUT_DIR = Path("preprocessing_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define datasets to load
DATASETS = {
    "humans_and_advanced_bots_behavior_summary": DATASET_DIR / "humans_and_advanced_bots_behavior_summary.csv",
    "humans_and_advanced_bots_behavioral_detailed": DATASET_DIR / "humans_and_advanced_bots_behavioral_detailed.csv",
    "humans_and_advanced_bots_temporal_detailed": DATASET_DIR / "humans_and_advanced_bots_temporal_detailed.csv",
    "humans_and_advanced_bots_web_activity_summary": DATASET_DIR / "humans_and_advanced_bots_web_activity_summary.csv",
    "humans_and_advanced_bots_combined_report": DATASET_DIR / "humans_and_advanced_bots_combined_report.csv",
    "humans_and_moderate_bots_behavior_summary": DATASET_DIR / "humans_and_moderate_bots_behavior_summary.csv",
    "humans_and_moderate_bots_behavioral_detailed": DATASET_DIR / "humans_and_moderate_bots_behavioral_detailed.csv",
    "humans_and_moderate_bots_combined_report": DATASET_DIR / "humans_and_moderate_bots_combined_report.csv",
}

def inspect_dataset(name, filepath):
    """Load and inspect a single dataset"""
    print(f"\n{'='*80}")
    print(f"DATASET: {name}")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(filepath, nrows=5 if "detailed" in name else None)
        print(f"[OK] Successfully loaded from: {filepath}")
        print(f"\nSHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nCOLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col:40s} -> {str(df[col].dtype):15s}")

        print(f"\nDATA TYPE SUMMARY:")
        print(df.dtypes.value_counts())

        print(f"\nPOTENTIAL LINKING KEYS:")
        linking_keys = ["user_id", "session_id", "timestamp", "category", "label",
                       "user", "session", "time", "type", "class"]
        found_keys = [col for col in df.columns if any(key in col.lower() for key in linking_keys)]
        if found_keys:
            for key in found_keys:
                n_unique = df[key].nunique()
                n_null = df[key].isna().sum()
                print(f"  - {key:30s} -> {n_unique:8d} unique, {n_null:8d} nulls")
        else:
            print("  [WARN] No obvious linking keys detected")

        print(f"\nFIRST 3 ROWS:")
        print(df.head(3).to_string())

        # Check for label column
        label_cols = [col for col in df.columns if "label" in col.lower() or "category" in col.lower()]
        has_label = len(label_cols) > 0
        print(f"\nLABEL COLUMN FOUND: {has_label}")
        if label_cols:
            for col in label_cols:
                print(f"  - {col}: {df[col].unique()}")

        return {
            "name": name,
            "filepath": str(filepath),
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "linking_keys": found_keys,
            "has_label": has_label,
            "label_columns": label_cols,
            "loaded": True
        }
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return {
            "name": name,
            "filepath": str(filepath),
            "loaded": False,
            "error": str(e)
        }

def main():
    print("="*80)
    print("FRAUD DETECTION PREPROCESSING - STEP 1: DATA INSPECTION")
    print("="*80)

    inspection_results = {}

    for name, filepath in DATASETS.items():
        inspection_results[name] = inspect_dataset(name, filepath)

    # Save inspection report
    report_path = OUTPUT_DIR / "01_inspection_report.json"
    with open(report_path, 'w') as f:
        json.dump(inspection_results, f, indent=2, default=str)
    print(f"\n[OK] Inspection report saved to: {report_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in inspection_results.values() if r.get("loaded"))
    print(f"[OK] Successfully loaded: {successful}/{len(DATASETS)} datasets")

    has_labels = sum(1 for r in inspection_results.values() if r.get("has_label"))
    print(f"[OK] Datasets with label columns: {has_labels}/{successful}")

    if has_labels > 0:
        print("\n[INFO] LABEL LINKAGE STATUS: POSSIBLE")
        print("   Next step: Validate and merge datasets using linking keys")
    else:
        print("\n[INFO] LABEL LINKAGE STATUS: NOT FOUND")
        print("   Next step: Implement heuristic classification and record linkage")

if __name__ == "__main__":
    main()
