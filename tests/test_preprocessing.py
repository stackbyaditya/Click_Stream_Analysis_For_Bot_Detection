from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing_module import REQUIRED_COLUMNS
from talkingdata_preprocessor import derive_temporal_features, sessionize_talkingdata

COMBINED_PATH = ROOT / "preprocessing_output" / "combined_clickstream_dataset.csv"


def test_talkingdata_sessionize() -> None:
    sample = pd.DataFrame(
        {
            "ip": [1, 1, 1, 1],
            "app": [3, 3, 3, 3],
            "device": [1, 1, 1, 1],
            "os": [13, 13, 13, 13],
            "channel": [111, 111, 111, 111],
            "click_time": pd.to_datetime(
                ["2017-11-07 09:30:00", "2017-11-07 09:30:05", "2017-11-07 10:15:00", "2017-11-07 10:15:03"]
            ),
            "is_attributed": [1, 0, 0, 1],
        }
    )
    sessionized = sessionize_talkingdata(sample, session_gap_minutes=30)
    temporal = derive_temporal_features(sessionized)
    assert temporal["session_id"].nunique() == 2
    assert (temporal["session_duration_sec"] >= 1.0).all()


def test_schema_alignment() -> None:
    assert COMBINED_PATH.exists(), "combined_clickstream_dataset.csv was not generated"
    combined = pd.read_csv(COMBINED_PATH, nrows=10)
    missing = [column for column in REQUIRED_COLUMNS if column not in combined.columns]
    assert not missing, f"missing required columns: {missing}"


def test_label_distribution() -> None:
    combined = pd.read_csv(COMBINED_PATH)
    counts = combined["label"].value_counts().to_dict()
    assert counts.get(0, 0) > 0, "human count is zero"
    assert counts.get(1, 0) > 0, "moderate bot count is zero"
    assert counts.get(2, 0) > 0, "advanced bot count is zero"
