"""TalkingData preprocessing utilities for human-session integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_module import (
    LABEL_MAPPING,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    ensure_feature_schema,
    entropy,
    generate_synthetic_device_features,
    safe_mode,
    seeded_faker,
    seeded_rng,
)


LOGGER = logging.getLogger(__name__)
SESSION_GAP_MINUTES = 30


def load_talkingdata(path_or_df) -> pd.DataFrame:
    """Load TalkingData train/test files or accept a preloaded DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        path = Path(path_or_df)
        if path.is_dir():
            train_path = path / "train.csv"
            sample_path = path / "train_sample.csv"
            if not train_path.exists() and sample_path.exists():
                train_path = sample_path
            if not train_path.exists():
                raise FileNotFoundError(f"No TalkingData train file found under {path}.")

            usecols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
            positive_parts: list[pd.DataFrame] = []
            negative_parts: list[pd.DataFrame] = []
            positive_rows = 0
            negative_rows = 0
            for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=750_000):
                chunk["is_attributed"] = chunk["is_attributed"].fillna(0).astype(int)
                positives = chunk[chunk["is_attributed"] == 1]
                negatives = chunk[chunk["is_attributed"] == 0]
                if not positives.empty:
                    positive_parts.append(positives.copy())
                    positive_rows += len(positives)
                if negative_rows < 120_000 and not negatives.empty:
                    take = min(120_000 - negative_rows, len(negatives))
                    negative_parts.append(negatives.head(take).copy())
                    negative_rows += take
                if positive_rows >= 5_000 and negative_rows >= 120_000:
                    break

            df = pd.concat(positive_parts + negative_parts, ignore_index=True)
        else:
            df = pd.read_csv(path)

    if "click_time" not in df.columns:
        raise ValueError("TalkingData input must contain click_time.")

    loaded = df.copy()
    loaded["click_time"] = pd.to_datetime(loaded["click_time"], errors="coerce")
    loaded = loaded.dropna(subset=["click_time"]).reset_index(drop=True)
    loaded["is_attributed"] = loaded.get("is_attributed", 0)
    loaded["is_attributed"] = loaded["is_attributed"].fillna(0).astype(int)
    loaded["year"] = loaded["click_time"].dt.year
    loaded["month"] = loaded["click_time"].dt.month
    loaded["day"] = loaded["click_time"].dt.day
    loaded["hour"] = loaded["click_time"].dt.hour
    loaded["minute"] = loaded["click_time"].dt.minute
    loaded["second"] = loaded["click_time"].dt.second
    return loaded


def sessionize_talkingdata(
    df: pd.DataFrame,
    group_keys: list[str] | None = None,
    session_gap_minutes: int = SESSION_GAP_MINUTES,
) -> pd.DataFrame:
    """Create deterministic session ids from ordered TalkingData events."""
    group_keys = group_keys or ["ip", "device", "os"]
    sessionized = df.copy().sort_values(group_keys + ["click_time"]).reset_index(drop=True)
    gaps = sessionized.groupby(group_keys)["click_time"].diff().dt.total_seconds()
    sessionized["session_gap_sec"] = gaps.fillna((session_gap_minutes * 60) + 1)
    sessionized["new_session"] = (sessionized["session_gap_sec"] > session_gap_minutes * 60).astype(int)
    sessionized["session_index"] = sessionized.groupby(group_keys)["new_session"].cumsum()
    sessionized["session_id"] = (
        "td_"
        + sessionized["ip"].astype(str)
        + "_"
        + sessionized["device"].astype(str)
        + "_"
        + sessionized["os"].astype(str)
        + "_"
        + sessionized["session_index"].astype(str)
    )
    return sessionized


def label_talkingdata_sessions(sessions_df: pd.DataFrame) -> pd.Series:
    """Label sessions as human only when install evidence exists; otherwise mark unknown."""
    labels = pd.Series(pd.NA, index=sessions_df.index, dtype="object")
    human_mask = sessions_df["install_count"].fillna(0).astype(int) > 0
    labels.loc[human_mask] = LABEL_MAPPING["human"]
    LOGGER.info(
        "TalkingData session labels: humans=%s unknown=%s",
        int(human_mask.sum()),
        int((~human_mask).sum()),
    )
    return labels


def derive_temporal_features(session_events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session-level temporal features from raw click events."""
    grouped = session_events_df.groupby("session_id", sort=False)
    temporal = grouped.agg(
        session_start=("click_time", "min"),
        session_end=("click_time", "max"),
        click_count=("session_id", "size"),
        total_requests=("session_id", "size"),
        install_count=("is_attributed", "sum"),
        successful_requests=("is_attributed", "sum"),
        original_ip=("ip", "first"),
        original_device=("device", "first"),
        original_os=("os", "first"),
        app=("app", safe_mode),
        device=("device", safe_mode),
        os=("os", safe_mode),
        channel=("channel", safe_mode),
    ).reset_index()

    intervals = grouped["click_time"].apply(lambda s: s.sort_values().diff().dt.total_seconds().fillna(0.0))
    interval_stats = intervals.groupby(level=0).agg(["mean", "std", "median"]).reset_index()
    interval_stats.columns = ["session_id", "click_interval_mean", "click_interval_std", "click_interval_median"]
    temporal = temporal.merge(interval_stats, on="session_id", how="left")

    temporal["session_duration_sec"] = (
        (temporal["session_end"] - temporal["session_start"]).dt.total_seconds().clip(lower=1.0)
    )
    temporal["request_interval_mean"] = temporal["click_interval_mean"].fillna(0.0)
    temporal["request_interval_std"] = temporal["click_interval_std"].fillna(0.0)
    temporal["clicks_per_minute"] = temporal["click_count"] / (temporal["session_duration_sec"] / 60.0)
    temporal["requests_per_minute"] = temporal["total_requests"] / (temporal["session_duration_sec"] / 60.0)
    temporal["request_interval_entropy"] = grouped["click_time"].apply(
        lambda s: entropy(s.sort_values().diff().dt.total_seconds().fillna(0.0), n_bins=10)
    ).to_numpy()
    temporal["click_interval_entropy"] = temporal["request_interval_entropy"]
    temporal["device_entropy"] = grouped["device"].apply(entropy).to_numpy()
    temporal["channel_entropy"] = grouped["channel"].apply(entropy).to_numpy()
    temporal["app_entropy"] = grouped["app"].apply(entropy).to_numpy()
    temporal["session_idle_ratio"] = np.clip(
        temporal["click_interval_std"].fillna(0.0) / (temporal["session_duration_sec"] + 1e-6),
        0.0,
        1.0,
    )
    temporal["activity_date"] = temporal["session_start"].dt.strftime("%Y-%m-%d")
    temporal["time_range"] = (
        temporal["session_start"].dt.strftime("%H:%M:%S") + "-" + temporal["session_end"].dt.strftime("%H:%M:%S")
    )
    temporal["logs_count"] = temporal["click_count"]
    temporal["success_rate"] = temporal["successful_requests"] / temporal["total_requests"].clip(lower=1)
    return temporal


def synthesize_behavioral_proxies(session_events_df: pd.DataFrame) -> pd.DataFrame:
    """Map click timing patterns into movement-like proxies without uniform random sampling."""
    temporal = derive_temporal_features(session_events_df)
    engineered = temporal.copy()

    median_interval = engineered["click_interval_mean"].fillna(0.0).clip(lower=0.05)
    interval_std = engineered["click_interval_std"].fillna(0.0)
    cadence = np.log1p(engineered["requests_per_minute"].clip(lower=0.0))
    variability = interval_std / (median_interval + 1.0)
    entropy_signal = engineered["click_interval_entropy"].fillna(0.0)

    # Deterministic transforms preserve observed cadence and dispersion instead of drawing arbitrary values.
    engineered["mouse_speed_mean"] = np.clip(120.0 / (median_interval + 0.5) + 18.0 * entropy_signal, 15.0, 320.0)
    engineered["mouse_speed_std"] = np.clip(4.0 + 14.0 * variability + 3.0 * entropy_signal, 0.5, 120.0)
    engineered["mouse_path_length"] = np.clip(
        engineered["mouse_speed_mean"] * np.sqrt(engineered["click_count"].clip(lower=1.0)) * (1.0 + 0.25 * cadence),
        20.0,
        None,
    )
    engineered["direction_change_count"] = np.clip(
        np.rint((engineered["click_count"] - 1) * (0.65 + 0.35 * entropy_signal)),
        0,
        None,
    ).astype(int)
    engineered["coordinate_entropy"] = np.clip(0.9 + 1.2 * entropy_signal + 0.4 * variability, 0.1, 8.0)
    engineered["movement_std"] = np.clip(0.35 * engineered["mouse_speed_std"] + 0.2 * cadence, 0.2, None)
    engineered["mouse_acceleration_std"] = np.clip(
        engineered["mouse_speed_std"] / (median_interval + 1.0),
        0.1,
        None,
    )
    # Curvature is proxied by directional variability normalized by path length.
    engineered["movement_curvature"] = np.clip(
        engineered["direction_change_count"] / engineered["mouse_path_length"].clip(lower=1.0),
        0.0,
        5.0,
    )
    return engineered


def synthesize_device_network_features(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic device/network features using controlled distributions and Faker."""
    return generate_synthetic_device_features(
        sessions_df,
        rng=seeded_rng(RANDOM_SEED),
        faker=seeded_faker(RANDOM_SEED),
    )


def align_schema_and_export(sessions_agg_df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    """Align to the canonical schema and export TalkingData session features."""
    aligned = ensure_feature_schema(sessions_agg_df)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_file, index=False)
    return aligned


@dataclass
class TalkingDataArtifacts:
    events: pd.DataFrame
    sessions: pd.DataFrame
    unknown_sessions: pd.DataFrame


class TalkingDataPreprocessor:
    """Compatibility wrapper that exposes the new function-based TalkingData pipeline."""

    def __init__(
        self,
        data_dir: str | Path,
        random_seed: int = RANDOM_SEED,
        sequence_length: int = SEQUENCE_LENGTH,
        session_gap_minutes: int = SESSION_GAP_MINUTES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.sequence_length = sequence_length
        self.session_gap_minutes = session_gap_minutes

    def run(self) -> TalkingDataArtifacts:
        loaded = load_talkingdata(self.data_dir)
        sessionized = sessionize_talkingdata(loaded, session_gap_minutes=self.session_gap_minutes)
        temporal = derive_temporal_features(sessionized)
        temporal["label"] = label_talkingdata_sessions(temporal)
        temporal["candidate_human"] = temporal["label"].eq(LABEL_MAPPING["human"])
        temporal["bot_type"] = temporal["label"].map({LABEL_MAPPING["human"]: "human"}).fillna("unknown")
        temporal["data_source"] = "talkingdata"
        temporal["data_source_detail"] = "real_human_candidates"
        temporal = synthesize_behavioral_proxies(sessionized).drop(columns=["session_start", "session_end"], errors="ignore").merge(
            temporal[["session_id", "session_start", "session_end", "label", "candidate_human", "bot_type", "data_source", "data_source_detail"]],
            on="session_id",
            how="left",
        )
        temporal["label"] = temporal["label"].fillna(pd.NA)
        temporal = synthesize_device_network_features(temporal)
        human_sessions = temporal[temporal["candidate_human"]].copy().reset_index(drop=True)
        unknown_sessions = temporal[~temporal["candidate_human"]].copy().reset_index(drop=True)
        LOGGER.info("TalkingData humans retained: %s", len(human_sessions))
        return TalkingDataArtifacts(events=sessionized, sessions=human_sessions, unknown_sessions=unknown_sessions)
