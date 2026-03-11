"""Shared preprocessing utilities for the clickstream fraud pipeline.

Updated on 2026-03-11:
- added category extraction, schema alignment, dual-mode sequence preparation
- added deterministic human augmentation, validation reporting, and merge helpers
- added richer behavioral features and a unified bot likelihood score
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from faker import Faker
except ImportError:  # pragma: no cover - exercised only in stripped environments
    Faker = None  # type: ignore


RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
MIN_HUMAN_SESSIONS = 1000
LABEL_MAPPING = {"human": 0, "moderate_bot": 1, "advanced_bot": 2}
LABEL_TO_BOT_TYPE = {value: key for key, value in LABEL_MAPPING.items()}
CLICK_SEQUENCE_FEATURES = [
    "timestamp_delta",
    "click_count_window",
    "session_progress",
    "click_interval_std",
    "device_entropy",
    "channel_entropy",
]
MOUSE_SEQUENCE_FEATURES = [
    "timestamp_delta",
    "mouse_velocity",
    "session_progress",
    "device_entropy",
    "channel_entropy",
]
COMMON_SEQUENCE_FEATURES = CLICK_SEQUENCE_FEATURES
REQUIRED_COLUMNS = [
    "session_id",
    "label",
    "bot_type",
    "data_source",
    "data_source_detail",
    "candidate_human",
    "session_start",
    "session_end",
    "activity_date",
    "time_range",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "mouse_acceleration_std",
    "movement_curvature",
    "session_idle_ratio",
    "click_interval_entropy",
    "device_entropy",
    "channel_entropy",
    "app_entropy",
    "request_interval_entropy",
    "requests_per_minute",
    "clicks_per_minute",
    "session_duration_sec",
    "request_interval_mean",
    "request_interval_std",
    "click_interval_mean",
    "click_interval_std",
    "total_movements",
    "total_requests",
    "click_count",
    "install_count",
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


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def seeded_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Create a reproducible RNG."""
    return np.random.default_rng(seed)


def seeded_faker(seed: int = RANDOM_SEED) -> Faker:
    """Create a reproducible Faker instance."""
    if Faker is None:
        class _FallbackFaker:
            def country(self) -> str:
                return "United States"

            def state(self) -> str:
                return "California"

        return _FallbackFaker()  # type: ignore[return-value]

    faker = Faker()
    Faker.seed(seed)
    faker.seed_instance(seed)
    return faker


def stable_fraction(value: str) -> float:
    """Map a string to a reproducible float in [0, 1)."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


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


def extract_category_label(category: str | int | float | None) -> int | None:
    """Normalize category text into the shared label mapping."""
    if category is None or pd.isna(category):
        return None
    if isinstance(category, (int, np.integer)):
        return int(category)

    text = str(category).strip().lower()
    if "advanced" in text:
        return LABEL_MAPPING["advanced_bot"]
    if "moderate" in text:
        return LABEL_MAPPING["moderate_bot"]
    if "human" in text:
        return LABEL_MAPPING["human"]
    return None


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
            "mouse_acceleration_std": 0.0,
            "movement_curvature": 0.0,
        }

    x_diff = np.diff(x_coords)
    y_diff = np.diff(y_coords)
    distances = np.sqrt(x_diff**2 + y_diff**2)
    headings = np.arctan2(y_diff, x_diff)
    direction_changes = np.abs(np.diff(headings))
    acceleration = np.diff(distances, prepend=distances[0])
    curvature = direction_changes / np.clip(distances[1:], 1.0, None)

    return {
        "mouse_speed_mean": float(np.mean(distances)),
        "mouse_speed_std": float(np.std(distances)),
        "mouse_path_length": float(np.sum(distances)),
        "direction_change_count": int(np.sum(direction_changes > (np.pi / 4))),
        "movement_std": float(np.std(np.concatenate([x_diff, y_diff]))),
        "coordinate_entropy": float((entropy(x_coords, n_bins=16) + entropy(y_coords, n_bins=16)) / 2.0),
        "mouse_acceleration_std": float(np.std(acceleration)),
        "movement_curvature": float(np.mean(curvature) if len(curvature) else 0.0),
    }


def compute_bot_likelihood_score(row: pd.Series) -> float:
    """Heuristic bot score combining cadence, entropy, motion, and proxy signals."""
    score = 0.0
    rpm = float(row.get("requests_per_minute", 0.0))
    cpm = float(row.get("clicks_per_minute", 0.0))
    interval_std = float(row.get("request_interval_std", row.get("click_interval_std", 0.0)))
    entropy_val = float(row.get("coordinate_entropy", 0.0))
    device_entropy = float(row.get("device_entropy", 0.0))
    click_entropy = float(row.get("click_interval_entropy", 0.0))
    curvature = float(row.get("movement_curvature", 0.0))
    idle_ratio = float(row.get("session_idle_ratio", 0.0))
    proxy = int(row.get("is_proxy", 0))

    if rpm > 120:
        score += 2.0
    elif rpm > 60:
        score += 1.0
    if cpm > 120:
        score += 0.8
    if interval_std < 0.35:
        score += 1.4
    elif interval_std < 1.0:
        score += 0.6
    if entropy_val < 1.6:
        score += 1.6
    elif entropy_val < 2.6:
        score += 0.8
    if device_entropy < 0.2:
        score += 0.6
    if click_entropy < 0.4:
        score += 0.9
    if curvature < 0.02:
        score += 0.5
    if idle_ratio < 0.02 and rpm > 80:
        score += 0.7
    if proxy:
        score += 1.0

    return float(round(score, 4))


calculate_bot_likelihood_score = compute_bot_likelihood_score


def _weighted_choice(rng: np.random.Generator, mapping: dict[str, float], size: int) -> np.ndarray:
    keys = list(mapping)
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=size, p=probs)


def _build_user_agent(browser: str, operating_system: str, headless: bool, key: str) -> str:
    frac = stable_fraction(key)
    major = int(118 + frac * 8)
    build = int(4200 + frac * 900)
    if headless and browser == "Chrome":
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
        return f"Mozilla/5.0 ({token}; rv:{major}.0) Gecko/20100101 Firefox/{major}.0"
    if browser == "Edge":
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.{build}.0 Safari/537.36 Edg/{major}.0.{build}.0"
        )
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.{build}.0 Safari/537.36"
    )


def generate_synthetic_device_features(
    df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    faker: Faker | None = None,
) -> pd.DataFrame:
    """Generate realistic device/network metadata while preserving class correlations."""
    if df.empty:
        return df.copy()

    rng = rng or seeded_rng()
    faker = faker or seeded_faker()
    enriched = df.copy()
    labels = pd.to_numeric(enriched["label"], errors="coerce").fillna(LABEL_MAPPING["human"]).astype(int).to_numpy()
    size = len(enriched)

    browsers = _weighted_choice(
        rng,
        {"Chrome": 0.65, "Safari": 0.18, "Firefox": 0.10, "Edge": 0.07},
        size,
    )
    operating_systems = _weighted_choice(
        rng,
        {"Windows": 0.40, "Android": 0.22, "iOS": 0.15, "Linux": 0.11, "MacOS": 0.12},
        size,
    )
    device_types = _weighted_choice(
        rng,
        {"desktop": 0.46, "mobile": 0.45, "tablet": 0.09},
        size,
    )

    bot_mask = labels > 0
    adv_mask = labels == LABEL_MAPPING["advanced_bot"]
    operating_systems = np.where(bot_mask & (rng.random(size) < 0.38), "Linux", operating_systems)
    device_types = np.where(np.isin(operating_systems, ["Android", "iOS"]), "mobile", device_types)
    device_types = np.where(adv_mask & (rng.random(size) < 0.70), "desktop", device_types)
    device_types = np.where(bot_mask & (rng.random(size) < 0.25), "server", device_types)

    proxy_flags = np.where(labels == 0, rng.random(size) < 0.05, rng.random(size) < 0.52).astype(int)
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
    country_regions = {
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
    regions = [rng.choice(country_regions[country]) for country in countries]
    dc_prefixes = np.array([1, 8, 34, 35, 44, 52, 54, 64, 66, 104, 128, 136, 138, 143, 159, 167, 185, 198, 207])
    res_prefixes = np.array([24, 47, 73, 86, 98, 101, 103, 115, 117, 122, 149, 172, 174, 188, 203])

    human_ua_cache: dict[tuple[str, str], str] = {}
    bot_ua_cache: dict[str, str] = {}
    ip_addresses: list[str] = []
    user_agents: list[str] = []
    generated_regions: list[str] = []
    generated_countries: list[str] = []

    for idx, (label, browser, os_name, proxy_flag) in enumerate(zip(labels, browsers, operating_systems, proxy_flags)):
        session_id = str(enriched.iloc[idx].get("session_id", idx))
        key = f"{session_id}|{browser}|{os_name}"
        prefix_pool = dc_prefixes if proxy_flag or label > 0 else res_prefixes
        prefix = int(rng.choice(prefix_pool))
        ip_addresses.append(
            f"{prefix}.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"
        )

        if label == 0:
            ua_key = (browser, os_name)
            if ua_key not in human_ua_cache:
                human_ua_cache[ua_key] = _build_user_agent(browser, os_name, False, key)
            user_agents.append(human_ua_cache[ua_key])
        else:
            cache_key = browser if label == LABEL_MAPPING["moderate_bot"] else f"adv-{browser}"
            if cache_key not in bot_ua_cache:
                bot_ua_cache[cache_key] = _build_user_agent(browser, os_name, browser == "Chrome", key)
            user_agents.append(bot_ua_cache[cache_key])

        country = countries[idx]
        region = regions[idx]
        if label == 0 and rng.random() < 0.15:
            country = faker.country()
            region = faker.state() if country == "United States" else region
        generated_countries.append(country)
        generated_regions.append(region)

    enriched["browser"] = browsers
    enriched["operating_system"] = operating_systems
    enriched["device_type"] = device_types
    enriched["user_agent"] = user_agents
    enriched["ip_address"] = ip_addresses
    enriched["country"] = generated_countries
    enriched["region"] = generated_regions
    enriched["is_proxy"] = proxy_flags
    enriched["bot_likelihood_score"] = enriched.apply(compute_bot_likelihood_score, axis=1)
    return enriched


add_device_network_features = generate_synthetic_device_features


def ensure_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add and order the required session-level columns."""
    aligned = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan

    aligned["label"] = pd.to_numeric(aligned["label"], errors="coerce").fillna(-1).astype(int)
    aligned["bot_type"] = aligned["bot_type"].fillna(aligned["label"].map(LABEL_TO_BOT_TYPE))
    aligned["data_source"] = aligned["data_source"].fillna("unknown")
    aligned["candidate_human"] = aligned["candidate_human"].fillna(False).astype(bool)
    aligned["click_count"] = aligned["click_count"].fillna(aligned["total_requests"]).fillna(0).astype(int)
    aligned["total_requests"] = aligned["total_requests"].fillna(aligned["click_count"]).fillna(0).astype(int)
    aligned["total_movements"] = aligned["total_movements"].fillna(aligned["click_count"]).fillna(0).astype(int)
    aligned["successful_requests"] = aligned["successful_requests"].fillna(aligned["install_count"]).fillna(0).astype(int)
    aligned["logs_count"] = aligned["logs_count"].fillna(aligned["click_count"]).fillna(0).astype(int)
    aligned["install_count"] = aligned["install_count"].fillna(0).astype(int)
    aligned["is_proxy"] = aligned["is_proxy"].fillna(0).astype(int)
    aligned["success_rate"] = aligned["success_rate"].fillna(
        aligned["successful_requests"] / aligned["total_requests"].clip(lower=1)
    )
    aligned["bot_likelihood_score"] = aligned.apply(compute_bot_likelihood_score, axis=1)

    extra_columns = [column for column in aligned.columns if column not in REQUIRED_COLUMNS]
    return aligned[REQUIRED_COLUMNS + extra_columns]


def aggregate_by_session(
    events_df: pd.DataFrame,
    session_col: str = "session_id",
    timestamp_col: str = "click_time",
) -> pd.DataFrame:
    """Generic session aggregator used for click-mode sequence prep and validation."""
    if events_df.empty:
        return pd.DataFrame(columns=[session_col, "event_count", "session_start", "session_end", "session_duration_sec"])

    ordered = events_df.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], errors="coerce")
    grouped = ordered.groupby(session_col, sort=False)
    agg = grouped.agg(
        event_count=(session_col, "size"),
        session_start=(timestamp_col, "min"),
        session_end=(timestamp_col, "max"),
    ).reset_index()
    agg["session_duration_sec"] = (agg["session_end"] - agg["session_start"]).dt.total_seconds().clip(lower=1.0)
    return agg


def _pad_sequence(sequence: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    if sequence.shape[0] >= sequence_length:
        return sequence[:sequence_length]
    padding = np.zeros((sequence_length - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
    return np.vstack([sequence.astype(np.float32), padding])


def build_mouse_sequences(
    behavioral_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build mouse-mode sequence tensors from x/y movement logs."""
    if behavioral_df is None or behavioral_df.empty or sessions_df.empty:
        return (
            np.zeros((0, sequence_length, len(MOUSE_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
            [],
        )

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    session_ids: list[str] = []
    sequence_types: list[str] = []

    lookup = sessions_df.set_index("session_id")[["label", "device_entropy", "channel_entropy"]].copy()
    grouped = behavioral_df.sort_values(["session_id", "movement_index"]).groupby("session_id", sort=False)

    for session_id, frame in grouped:
        if session_id not in lookup.index or len(frame) < 2:
            continue

        x = frame["x_coordinate"].to_numpy(dtype=float)
        y = frame["y_coordinate"].to_numpy(dtype=float)
        idx = frame["movement_index"].to_numpy(dtype=float)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dist = np.sqrt(dx**2 + dy**2)
        delta = np.diff(idx, prepend=idx[0]).astype(float)
        delta[0] = 0.0
        progress = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
        device_entropy = np.full(len(frame), float(lookup.loc[session_id, "device_entropy"]), dtype=np.float32)
        channel_entropy = np.full(len(frame), float(lookup.loc[session_id, "channel_entropy"]), dtype=np.float32)
        velocity = dist / np.clip(delta + 1.0, 1.0, None)
        seq = np.column_stack([delta, velocity, progress, device_entropy, channel_entropy]).astype(np.float32)
        sequences.append(_pad_sequence(seq, sequence_length))
        labels.append(int(lookup.loc[session_id, "label"]))
        session_ids.append(str(session_id))
        sequence_types.append("mouse-mode")

    if not sequences:
        return (
            np.zeros((0, sequence_length, len(MOUSE_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
            [],
        )

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), session_ids, sequence_types


def build_click_sequences(
    event_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build click-mode sequence tensors when mouse x/y coordinates are absent."""
    if event_df is None or event_df.empty or sessions_df.empty:
        return (
            np.zeros((0, sequence_length, len(CLICK_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
            [],
        )

    ordered = event_df.copy()
    ordered["click_time"] = pd.to_datetime(ordered["click_time"], errors="coerce")
    ordered = ordered.dropna(subset=["click_time"]).sort_values(["session_id", "click_time"])
    lookup = sessions_df.set_index("session_id")[["label", "device_entropy", "channel_entropy", "click_interval_std"]]

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    session_ids: list[str] = []
    sequence_types: list[str] = []

    for session_id, frame in ordered.groupby("session_id", sort=False):
        if session_id not in lookup.index or frame.empty:
            continue
        deltas = frame["click_time"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)
        progress = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
        rolling_count = pd.Series(np.ones(len(frame))).rolling(window=5, min_periods=1).sum().to_numpy(dtype=float)
        click_interval_std = pd.Series(deltas).rolling(window=5, min_periods=1).std().fillna(0.0).to_numpy(dtype=float)
        device_entropy = np.full(len(frame), float(lookup.loc[session_id, "device_entropy"]), dtype=np.float32)
        channel_entropy = np.full(len(frame), float(lookup.loc[session_id, "channel_entropy"]), dtype=np.float32)
        seq = np.column_stack(
            [deltas, rolling_count, progress, click_interval_std, device_entropy, channel_entropy]
        ).astype(np.float32)
        sequences.append(_pad_sequence(seq, sequence_length))
        labels.append(int(lookup.loc[session_id, "label"]))
        session_ids.append(str(session_id))
        sequence_types.append("click-mode")

    if not sequences:
        return (
            np.zeros((0, sequence_length, len(CLICK_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
            [],
        )

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), session_ids, sequence_types


def prepare_lstm_sequences(
    sessions_df: pd.DataFrame,
    event_df: pd.DataFrame | None = None,
    behavioral_df: pd.DataFrame | None = None,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare mouse-mode or click-mode sequences automatically."""
    if behavioral_df is not None and {"x_coordinate", "y_coordinate"}.issubset(behavioral_df.columns):
        seq, labels, session_ids, sequence_types = build_mouse_sequences(behavioral_df, sessions_df, seq_len)
        metadata = pd.DataFrame({"session_id": session_ids, "label": labels, "sequence_type": sequence_types})
        return seq, labels, metadata

    if event_df is None:
        raise ValueError("click-mode sequence preparation requires event_df when mouse coordinates are absent")

    seq, labels, session_ids, sequence_types = build_click_sequences(event_df, sessions_df, seq_len)
    metadata = pd.DataFrame({"session_id": session_ids, "label": labels, "sequence_type": sequence_types})
    return seq, labels, metadata


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
                sampled_indices.extend(rng.choice(stratum_group.index.to_numpy(), size=take, replace=False).tolist())

        if len(sampled_indices) > target:
            sampled_indices = rng.choice(np.asarray(sampled_indices), size=target, replace=False).tolist()
        elif len(sampled_indices) < target:
            remaining = group.index.difference(sampled_indices)
            if len(remaining) >= target - len(sampled_indices):
                top_up = rng.choice(remaining.to_numpy(), size=target - len(sampled_indices), replace=False)
                sampled_indices.extend(top_up.tolist())

        parts.append(group.loc[sorted(sampled_indices)])

    combined = pd.concat(parts, ignore_index=False)
    return combined.sort_values(["label", "session_id"]).reset_index(drop=True)


def bootstrap_human_sessions(
    human_df: pd.DataFrame,
    target_count: int,
    random_seed: int = RANDOM_SEED,
    augment_method: str = "bootstrap",
) -> tuple[pd.DataFrame, dict]:
    """Bootstrap additional human sessions using correlated parametric perturbations."""
    if len(human_df) >= target_count:
        return human_df.copy(), {"performed": False, "method": augment_method, "generated": 0}
    if human_df.empty:
        raise ValueError("Cannot augment humans because no real human sessions are available.")

    rng = seeded_rng(random_seed)
    base = human_df.copy().reset_index(drop=True)
    needed = target_count - len(base)
    numeric_cols = [
        "session_duration_sec",
        "request_interval_mean",
        "request_interval_std",
        "click_interval_mean",
        "click_interval_std",
        "clicks_per_minute",
        "requests_per_minute",
        "mouse_speed_mean",
        "mouse_speed_std",
        "mouse_path_length",
        "movement_std",
        "coordinate_entropy",
        "mouse_acceleration_std",
        "movement_curvature",
        "session_idle_ratio",
        "click_interval_entropy",
        "device_entropy",
        "channel_entropy",
        "request_interval_entropy",
    ]
    available_numeric = [column for column in numeric_cols if column in base.columns]
    covariance = base[available_numeric].fillna(base[available_numeric].median()).cov().to_numpy(dtype=float)
    mean_vector = np.zeros(len(available_numeric), dtype=float)

    samples = []
    for idx in range(needed):
        source = base.iloc[idx % len(base)].copy()
        jitter = rng.multivariate_normal(mean_vector, covariance * 0.015 + np.eye(len(available_numeric)) * 1e-6)
        clone = source.copy()
        clone["session_id"] = f"aug_human_{idx:05d}"
        clone["data_source"] = "synthetic_human"
        clone["data_source_detail"] = augment_method
        clone["candidate_human"] = True
        for column, delta in zip(available_numeric, jitter):
            base_value = float(source[column]) if not pd.isna(source[column]) else 0.0
            clone[column] = max(0.0, base_value + delta)

        clone["click_count"] = int(max(2, round(clone.get("click_count", source.get("click_count", 2)))))
        clone["total_requests"] = int(max(clone["click_count"], round(source.get("total_requests", clone["click_count"]))))
        clone["total_movements"] = int(max(clone["click_count"], round(source.get("total_movements", clone["click_count"]))))
        clone["install_count"] = int(max(0, round(source.get("install_count", 1))))
        clone["successful_requests"] = int(max(clone["install_count"], round(source.get("successful_requests", clone["install_count"]))))
        clone["success_rate"] = float(np.clip(clone["successful_requests"] / max(clone["total_requests"], 1), 0.0, 1.0))
        clone["session_duration_sec"] = float(max(1.0, clone["session_duration_sec"]))
        clone["clicks_per_minute"] = float(clone["click_count"] / (clone["session_duration_sec"] / 60.0))
        clone["requests_per_minute"] = float(clone["total_requests"] / (clone["session_duration_sec"] / 60.0))
        clone["bot_likelihood_score"] = compute_bot_likelihood_score(pd.Series(clone))
        samples.append(clone)

    augmented = pd.concat([base, pd.DataFrame(samples)], ignore_index=True)
    augmented = ensure_feature_schema(augmented)
    return augmented, {"performed": True, "method": augment_method, "generated": needed}


def merge_and_sanity_check(
    existing_sessions_path: str | Path,
    talkingdata_sessions_path: str | Path,
    out_path: str | Path = "preprocessing_output/combined_clickstream_dataset.csv",
) -> pd.DataFrame:
    """Merge legacy bot sessions with TalkingData human sessions after schema alignment."""
    existing_path = Path(existing_sessions_path)
    talking_path = Path(talkingdata_sessions_path)
    out_file = Path(out_path)

    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    talking = pd.read_csv(talking_path) if talking_path.exists() else pd.DataFrame()
    combined = pd.concat([ensure_feature_schema(talking), ensure_feature_schema(existing)], ignore_index=True)
    combined = ensure_feature_schema(combined).drop_duplicates(subset=["session_id"]).reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_file, index=False)
    return combined


def save_sequence_payload(
    path: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    scaler: StandardScaler,
) -> None:
    """Persist the combined sequence payload."""
    payload = {
        "sequences": sequences,
        "labels": labels.astype(np.int64),
        "metadata": metadata.to_dict(orient="records"),
        "feature_names": metadata.attrs.get("feature_names", COMMON_SEQUENCE_FEATURES),
        "sequence_length": int(sequences.shape[1]) if sequences.ndim == 3 else SEQUENCE_LENGTH,
        "n_features": int(sequences.shape[2]) if sequences.ndim == 3 else len(COMMON_SEQUENCE_FEATURES),
        "scaler": scaler,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_metadata_report(path: Path, report: dict) -> None:
    """Write metadata as JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)


def build_validation_report(df: pd.DataFrame) -> dict:
    """Build dataset validation statistics."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    label_counts = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()}
    feature_summary = {}
    for column in numeric.columns:
        feature_summary[column] = {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std(ddof=0)),
            "min": float(numeric[column].min()),
            "max": float(numeric[column].max()),
        }

    corr_pairs = []
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False).head(10)
        corr_pairs = [
            {"feature_a": idx[0], "feature_b": idx[1], "abs_correlation": float(value)}
            for idx, value in pairs.items()
        ]

    outlier_counts = {}
    for column in numeric.columns:
        series = numeric[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[column] = int(((series < lower) | (series > upper)).sum())

    return {
        "label_counts": label_counts,
        "feature_summary": feature_summary,
        "top_feature_correlations": corr_pairs,
        "outlier_counts": outlier_counts,
    }


def build_integration_report(df: pd.DataFrame) -> dict:
    """Build a concise integration report with sample rows and source notes."""
    samples = {}
    for label, label_name in LABEL_TO_BOT_TYPE.items():
        subset = df[df["label"] == label].head(3)
        samples[label_name] = subset[["session_id", "data_source", "bot_type", "candidate_human"]].to_dict(orient="records")

    talking_count = int((df["data_source"] == "talkingdata").sum())
    return {
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        "sample_rows": samples,
        "talkingdata_note": f"{talking_count} sessions originated from TalkingData.",
    }


@dataclass
class PipelineOutputs:
    """Container for final pipeline artifacts."""

    sessions: pd.DataFrame
    sequences: np.ndarray
    labels: np.ndarray
    metadata: pd.DataFrame
    scaler: StandardScaler


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
