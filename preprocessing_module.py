"""Shared preprocessing utilities for the clickstream fraud pipeline."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
LABEL_MAPPING = {"human": 0, "moderate_bot": 1, "advanced_bot": 2}
LABEL_TO_BOT_TYPE = {value: key for key, value in LABEL_MAPPING.items()}
COMMON_SEQUENCE_FEATURES = [
    "timestamp_delta",
    "click_frequency",
    "session_progress",
    "device_entropy",
    "channel_entropy",
]


def seeded_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Create a reproducible RNG."""
    return np.random.default_rng(seed)


def stable_fraction(value: str) -> float:
    """Map a string to a reproducible float in [0, 1)."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def entropy(values: Iterable, n_bins: int | None = None) -> float:
    """Shannon entropy for categorical or numeric values."""
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return 0.0

    if np.issubdtype(series.dtype, np.number) and n_bins:
        counts, _ = np.histogram(series.to_numpy(dtype=float), bins=n_bins)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0
        probs = counts / counts.sum()
    else:
        probs = series.value_counts(normalize=True).to_numpy(dtype=float)

    return float(-(probs * np.log2(probs + 1e-12)).sum())


def safe_mode(series: pd.Series):
    """Return the first mode or first non-null value."""
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode()
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def parse_clickstream_timestamp(value: str | float | int) -> pd.Timestamp:
    """Parse Apache-style timestamps from the legacy clickstream logs."""
    if pd.isna(value):
        return pd.NaT
    cleaned = str(value).strip("[]")
    cleaned = cleaned.split("+")[0].strip()
    return pd.to_datetime(cleaned, format="%d/%b/%Y:%H:%M:%S", errors="coerce")


def calculate_mouse_features(x_coords: np.ndarray, y_coords: np.ndarray) -> dict[str, float]:
    """Aggregate mouse movement statistics for one session."""
    if len(x_coords) <= 1 or len(y_coords) <= 1:
        return {
            "mouse_speed_mean": 0.0,
            "mouse_speed_std": 0.0,
            "mouse_path_length": 0.0,
            "direction_change_count": 0,
            "movement_std": 0.0,
            "coordinate_entropy": 0.0,
        }

    x_diff = np.diff(x_coords)
    y_diff = np.diff(y_coords)
    distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
    headings = np.arctan2(y_diff, x_diff)
    direction_changes = np.abs(np.diff(headings))

    return {
        "mouse_speed_mean": float(np.mean(distances)),
        "mouse_speed_std": float(np.std(distances)),
        "mouse_path_length": float(np.sum(distances)),
        "direction_change_count": int(np.sum(direction_changes > (np.pi / 4))),
        "movement_std": float(np.std(np.concatenate([x_diff, y_diff]))),
        "coordinate_entropy": float(
            (entropy(x_coords, n_bins=16) + entropy(y_coords, n_bins=16)) / 2.0
        ),
    }


def calculate_bot_likelihood_score(row: pd.Series) -> float:
    """Heuristic bot score shared by legacy and TalkingData sessions."""
    score = 0.0

    rpm = float(row.get("requests_per_minute", 0.0))
    interval_std = float(row.get("request_interval_std", 0.0))
    coord_entropy = float(row.get("coordinate_entropy", 0.0))
    success_rate = float(row.get("success_rate", 0.0))
    proxy = int(row.get("is_proxy", 0))
    label = int(row.get("label", 0))

    if rpm > 120:
        score += 2.0
    elif rpm > 60:
        score += 1.0

    if interval_std < 0.35:
        score += 1.2
    elif interval_std < 1.0:
        score += 0.5

    if coord_entropy < 1.6:
        score += 1.5
    elif coord_entropy < 2.6:
        score += 0.8

    if success_rate < 0.02:
        score += 0.6
    if proxy:
        score += 0.9

    score += 0.5 * label
    return float(round(score, 4))


def _weighted_choice(
    rng: np.random.Generator, mapping: dict[str, float], size: int
) -> np.ndarray:
    keys = list(mapping)
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=size, p=probs)


def _build_user_agent(browser: str, operating_system: str, label: int, key: str) -> str:
    frac = stable_fraction(key)
    major = int(118 + frac * 8)
    build = int(4100 + frac * 900)

    if label > 0 and browser == "Chrome":
        return (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) HeadlessChrome/"
            f"{major}.0.{build}.0 Safari/537.36"
        )
    if browser == "Safari":
        return (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            f"Version/{max(15, major // 8)}.0 Safari/605.1.15"
        )
    if browser == "Firefox":
        token = "Windows NT 10.0; Win64; x64" if operating_system != "Linux" else "X11; Linux x86_64"
        return (
            f"Mozilla/5.0 ({token}; rv:{major}.0) "
            f"Gecko/20100101 Firefox/{major}.0"
        )
    if browser == "Edge":
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.{build}.0 Safari/537.36 "
            f"Edg/{major}.0.{build}.0"
        )
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.{build}.0 Safari/537.36"
    )


def add_device_network_features(
    df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Add reproducible synthetic device and network fields."""
    if df.empty:
        return df.copy()

    rng = rng or seeded_rng()
    enriched = df.copy()
    size = len(enriched)
    labels = enriched["label"].astype(int).to_numpy()

    browsers = _weighted_choice(
        rng,
        {"Chrome": 0.60, "Safari": 0.14, "Firefox": 0.12, "Edge": 0.10, "Opera": 0.04},
        size,
    )
    operating_systems = _weighted_choice(
        rng,
        {"Windows": 0.38, "Android": 0.23, "iOS": 0.14, "Linux": 0.13, "MacOS": 0.12},
        size,
    )
    device_types = _weighted_choice(
        rng,
        {"desktop": 0.44, "mobile": 0.47, "tablet": 0.09},
        size,
    )

    bot_mask = labels > 0
    operating_systems = np.where(bot_mask & (rng.random(size) < 0.42), "Linux", operating_systems)
    device_types = np.where(np.isin(operating_systems, ["Android", "iOS"]), "mobile", device_types)
    device_types = np.where((labels == 2) & (rng.random(size) < 0.78), "desktop", device_types)
    device_types = np.where((labels == 1) & (rng.random(size) < 0.55), "desktop", device_types)

    countries = _weighted_choice(
        rng,
        {
            "United States": 0.28,
            "India": 0.18,
            "United Kingdom": 0.10,
            "Germany": 0.08,
            "Canada": 0.08,
            "France": 0.07,
            "Japan": 0.06,
            "Brazil": 0.05,
            "Singapore": 0.05,
            "Australia": 0.05,
        },
        size,
    )
    regions_map = {
        "United States": ["California", "New York", "Texas", "Florida", "Illinois"],
        "India": ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Telangana"],
        "United Kingdom": ["England", "Scotland", "Wales"],
        "Germany": ["Berlin", "Bavaria", "Hamburg"],
        "Canada": ["Ontario", "Quebec", "British Columbia"],
        "France": ["Ile-de-France", "Auvergne-Rhone-Alpes"],
        "Japan": ["Tokyo", "Osaka", "Kanagawa"],
        "Brazil": ["Sao Paulo", "Rio de Janeiro"],
        "Singapore": ["Central Singapore"],
        "Australia": ["New South Wales", "Victoria"],
    }
    regions = [rng.choice(regions_map[country]) for country in countries]

    proxy = np.where(labels == 0, rng.random(size) < 0.06, rng.random(size) < 0.68).astype(int)
    dc_prefixes = np.array([1, 8, 34, 35, 44, 52, 54, 64, 66, 104, 128, 136, 138, 143, 159, 167, 185, 198, 207])
    res_prefixes = np.array([24, 47, 73, 86, 98, 101, 103, 115, 117, 122, 149, 172, 174, 188, 203])

    ip_addresses = []
    user_agents = []
    for idx, (label, browser, os_name, proxy_flag) in enumerate(
        zip(labels, browsers, operating_systems, proxy)
    ):
        key = str(enriched.iloc[idx]["session_id"])
        prefix_pool = dc_prefixes if proxy_flag or label > 0 else res_prefixes
        prefix = int(rng.choice(prefix_pool))
        ip_addresses.append(
            f"{prefix}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"
        )
        user_agents.append(_build_user_agent(browser, os_name, label, key))

    enriched["browser"] = browsers
    enriched["operating_system"] = operating_systems
    enriched["device_type"] = device_types
    enriched["user_agent"] = user_agents
    enriched["ip_address"] = ip_addresses
    enriched["country"] = countries
    enriched["region"] = regions
    enriched["is_proxy"] = proxy
    enriched["bot_likelihood_score"] = enriched.apply(calculate_bot_likelihood_score, axis=1)
    return enriched


def ensure_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add and order the required session-level columns."""
    ordered_columns = [
        "session_id",
        "label",
        "bot_type",
        "data_source",
        "session_start",
        "session_end",
        "activity_date",
        "time_range",
        "mouse_speed_mean",
        "mouse_speed_std",
        "mouse_path_length",
        "direction_change_count",
        "coordinate_entropy",
        "movement_std",
        "requests_per_minute",
        "clicks_per_minute",
        "session_duration_sec",
        "request_interval_mean",
        "request_interval_std",
        "total_movements",
        "total_requests",
        "successful_requests",
        "success_rate",
        "browser",
        "operating_system",
        "device_type",
        "user_agent",
        "ip_address",
        "country",
        "region",
        "is_proxy",
        "bot_likelihood_score",
        "device_entropy",
        "channel_entropy",
        "app_entropy",
        "click_count",
        "install_count",
        "original_ip",
        "original_device",
        "original_os",
        "app",
        "device",
        "os",
        "channel",
        "logs_count",
        "total_response_size",
        "avg_response_size",
    ]

    aligned = df.copy()
    for column in ordered_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan

    extra_columns = [column for column in aligned.columns if column not in ordered_columns]
    return aligned[ordered_columns + extra_columns]


def _pad_sequence(sequence: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    if sequence.shape[0] >= sequence_length:
        return sequence[:sequence_length]
    padding = np.zeros((sequence_length - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
    return np.vstack([sequence.astype(np.float32), padding])


def build_mouse_sequences(
    behavioral_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a shared temporal sequence representation from mouse movement logs."""
    if behavioral_df is None or behavioral_df.empty or sessions_df.empty:
        return (
            np.zeros((0, sequence_length, len(COMMON_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
        )

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    session_ids: list[str] = []

    label_lookup = sessions_df.set_index("session_id")["label"].astype(int).to_dict()
    grouped = behavioral_df.sort_values(["session_id", "movement_index"]).groupby("session_id", sort=False)

    for session_id, frame in grouped:
        if session_id not in label_lookup or len(frame) < 2:
            continue

        x = frame["x_coordinate"].to_numpy(dtype=float)
        y = frame["y_coordinate"].to_numpy(dtype=float)
        order = frame["movement_index"].to_numpy(dtype=float)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dist = np.sqrt(dx ** 2 + dy ** 2)
        delta = np.diff(order, prepend=order[0])
        delta[0] = 0.0
        delta = np.clip(delta, 0.0, None)
        progress = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
        coord_entropy = float(
            (entropy(x, n_bins=16) + entropy(y, n_bins=16)) / 2.0
        )
        headings = np.arctan2(dy, dx)
        rolling_heading_entropy = np.array(
            [entropy(headings[max(0, idx - 4): idx + 1], n_bins=6) for idx in range(len(headings))],
            dtype=np.float32,
        )
        click_frequency = dist / np.clip(delta + 1.0, 1.0, None)
        device_entropy = np.full(len(frame), coord_entropy, dtype=np.float32)

        seq = np.column_stack(
            [delta, click_frequency, progress, device_entropy, rolling_heading_entropy]
        ).astype(np.float32)
        sequences.append(_pad_sequence(seq, sequence_length))
        labels.append(int(label_lookup[session_id]))
        session_ids.append(str(session_id))

    if not sequences:
        return (
            np.zeros((0, sequence_length, len(COMMON_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
        )

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), session_ids


def standardize_sequences(sequences: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Scale sequence features across all time steps."""
    scaler = StandardScaler()
    if sequences.size == 0:
        return sequences.astype(np.float32), scaler
    reshaped = sequences.reshape(-1, sequences.shape[-1])
    scaled = scaler.fit_transform(reshaped).reshape(sequences.shape)
    return scaled.astype(np.float32), scaler


def stratified_sample(
    df: pd.DataFrame,
    label_targets: dict[int, int],
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Down-sample overrepresented classes while preserving minority classes."""
    rng = seeded_rng(random_seed)
    parts = []
    for label, group in df.groupby("label", sort=False):
        target = label_targets.get(int(label), len(group))
        if len(group) <= target:
            parts.append(group)
            continue

        strata = (
            group["requests_per_minute"].rank(method="first", pct=True).round(1).astype(str)
            + "|"
            + group["session_duration_sec"].rank(method="first", pct=True).round(1).astype(str)
            + "|"
            + group["coordinate_entropy"].rank(method="first", pct=True).round(1).astype(str)
        )
        sampled_indices = []
        for _, stratum_group in group.groupby(strata, sort=False):
            stratum_target = max(1, round(target * len(stratum_group) / len(group)))
            take = min(len(stratum_group), stratum_target)
            if take == len(stratum_group):
                sampled_indices.extend(stratum_group.index.tolist())
            else:
                sampled_indices.extend(
                    rng.choice(stratum_group.index.to_numpy(), size=take, replace=False).tolist()
                )

        if len(sampled_indices) > target:
            sampled_indices = rng.choice(np.asarray(sampled_indices), size=target, replace=False).tolist()
        elif len(sampled_indices) < target:
            remaining = group.index.difference(sampled_indices)
            top_up = rng.choice(remaining.to_numpy(), size=target - len(sampled_indices), replace=False)
            sampled_indices.extend(top_up.tolist())

        parts.append(group.loc[sorted(sampled_indices)])

    combined = pd.concat(parts, ignore_index=False)
    return combined.sort_values(["label", "session_id"]).reset_index(drop=True)


@dataclass
class PipelineOutputs:
    """Container for final pipeline artifacts."""

    sessions: pd.DataFrame
    sequences: np.ndarray
    labels: np.ndarray
    session_ids: list[str]
    scaler: StandardScaler


def save_sequence_payload(
    path: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    session_ids: list[str],
    scaler: StandardScaler,
) -> None:
    """Persist the combined sequence payload."""
    payload = {
        "sequences": sequences,
        "labels": labels.astype(np.int64),
        "session_ids": session_ids,
        "feature_names": COMMON_SEQUENCE_FEATURES,
        "sequence_length": int(sequences.shape[1]) if sequences.ndim == 3 else SEQUENCE_LENGTH,
        "n_features": int(sequences.shape[2]) if sequences.ndim == 3 else len(COMMON_SEQUENCE_FEATURES),
        "scaler": scaler,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_metadata_report(path: Path, report: dict) -> None:
    """Write pipeline metadata as JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)


class FraudDetectionPreprocessor:
    """Compatibility wrapper used by legacy scripts and notebooks."""

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        sequence_length: int = SEQUENCE_LENGTH,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sequence_length = sequence_length
        self.random_seed = random_seed

    def run(self) -> dict:
        return {
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
            "sequence_length": self.sequence_length,
            "random_seed": self.random_seed,
            "created_at": datetime.utcnow().isoformat(),
        }
