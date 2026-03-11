"""TalkingData and clickstream preprocessing pipeline for fraud detection."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from faker import Faker
except ImportError:  # pragma: no cover
    Faker = None


RANDOM_SEED = 42
SEQUENCE_LENGTH = 50
SESSION_GAP_SECONDS = 1800
BASE_DIR = Path(__file__).resolve().parent
TALKINGDATA_PATH = BASE_DIR / "talkingdata-adtracking-fraud-detection (1)" / "train_sample.csv"
EXISTING_AGG_PATH = BASE_DIR / "preprocessing_output" / "session_aggregated_dataset.csv"
EXISTING_SEQ_PATH = BASE_DIR / "preprocessing_output" / "session_sequence_dataset.pkl"

OUTPUTS = {
    "candidates": BASE_DIR / "talkingdata_human_candidates.csv",
    "sessions": BASE_DIR / "talkingdata_session_features.csv",
    "combined": BASE_DIR / "combined_fraud_detection_dataset.csv",
    "sequence": BASE_DIR / "combined_sequence_dataset.pkl",
    "report": BASE_DIR / "dataset_integration_report.json",
}

np.random.seed(RANDOM_SEED)
RNG = np.random.default_rng(RANDOM_SEED)
FAKE = Faker() if Faker is not None else None
if FAKE is not None:
    Faker.seed(RANDOM_SEED)
    FAKE.seed_instance(RANDOM_SEED)


def entropy(values) -> float:
    values = pd.Series(list(values))
    if values.empty:
        return 0.0
    probs = values.value_counts(normalize=True)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def rolling_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=min(6, max(2, values.size)))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def weighted_choice(mapping: dict[str, float], size: int) -> np.ndarray:
    keys = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return RNG.choice(keys, size=size, p=probs)


def safe_mode(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode()
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def load_talkingdata() -> pd.DataFrame:
    return pd.read_csv(
        TALKINGDATA_PATH,
        dtype={"ip": "int64", "app": "int32", "device": "int32", "os": "int32", "channel": "int32", "is_attributed": "int8"},
        parse_dates=["click_time", "attributed_time"],
    )


def inspect_talkingdata(df: pd.DataFrame) -> dict:
    return {
        "row_count": int(len(df)),
        "column_names": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "timestamp_range": {
            "click_time_min": df["click_time"].min().isoformat(),
            "click_time_max": df["click_time"].max().isoformat(),
        },
        "categorical_distributions": {
            col: df[col].value_counts(normalize=True).head(10).round(6).to_dict()
            for col in ["app", "device", "os", "channel", "is_attributed"]
        },
        "unique_counts": {col: int(df[col].nunique()) for col in ["ip", "device", "app", "channel"]},
        "top_ip_distribution": df["ip"].value_counts().head(10).to_dict(),
        "top_device_distribution": df["device"].value_counts().head(10).to_dict(),
        "top_app_distribution": df["app"].value_counts().head(10).to_dict(),
    }


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["ip", "click_time"]).reset_index(drop=True)
    parts = df["click_time"].dt
    df["year"], df["month"], df["day"] = parts.year, parts.month, parts.day
    df["hour"], df["minute"], df["second"] = parts.hour, parts.minute, parts.second
    df["ip_click_frequency"] = df.groupby("ip")["ip"].transform("size")
    df["device_click_frequency"] = df.groupby("device")["device"].transform("size")
    df["ip_install_count"] = df.groupby("ip")["is_attributed"].transform("sum")
    df["device_install_count"] = df.groupby("device")["is_attributed"].transform("sum")
    df["ip_install_rate"] = df["ip_install_count"] / df["ip_click_frequency"].clip(lower=1)
    df["ip_time_gap_sec"] = df.groupby("ip")["click_time"].diff().dt.total_seconds().fillna(0.0)
    combo_counts = df.groupby(["ip", "device", "channel"])["channel"].transform("size")
    df["combo_repeat_rate"] = combo_counts / df["ip_click_frequency"].clip(lower=1)
    df["non_repetitive_combo_score"] = 1.0 - df["combo_repeat_rate"].clip(upper=1.0)
    return df


def label_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hi_ip = df["ip_click_frequency"].quantile(0.99)
    hi_device = df["device_click_frequency"].quantile(0.99)
    low_gap = max(0.5, df["ip_time_gap_sec"].quantile(0.05))
    low_div = df["non_repetitive_combo_score"].quantile(0.10)
    possible_bot = (
        (df["is_attributed"] == 0)
        & (
            ((df["ip_click_frequency"] >= hi_ip) & (df["ip_install_count"] == 0) & (df["ip_time_gap_sec"] <= low_gap))
            | ((df["device_click_frequency"] >= hi_device) & (df["device_install_count"] == 0) & (df["non_repetitive_combo_score"] < low_div))
        )
    )
    df["human_heuristic"] = ((df["is_attributed"] == 1) | ~possible_bot).astype(int)
    df["possible_bot_heuristic"] = possible_bot.astype(int)
    df["label"] = possible_bot.astype(int)
    df["behavior_class"] = np.where(df["label"] == 1, "possible_bot", "human")
    return df


def sessionize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = df.copy().sort_values(["ip", "device", "os", "click_time"]).reset_index(drop=True)
    gaps = events.groupby(["ip", "device", "os"])["click_time"].diff().dt.total_seconds()
    events["session_gap_sec"] = gaps.fillna(SESSION_GAP_SECONDS + 1)
    events["new_session"] = (events["session_gap_sec"] > SESSION_GAP_SECONDS).astype(int)
    events["session_index"] = events.groupby(["ip", "device", "os"])["new_session"].cumsum()
    events["session_id"] = events.apply(
        lambda row: f"td_{int(row['ip'])}_{int(row['device'])}_{int(row['os'])}_{int(row['session_index'])}",
        axis=1,
    )
    events["event_interval_sec"] = events.groupby("session_id")["click_time"].diff().dt.total_seconds().fillna(0.0)
    grouped = events.groupby("session_id", sort=False)
    sessions = grouped.agg(
        data_source=("session_id", lambda _: "talkingdata"),
        original_ip=("ip", "first"),
        original_device=("device", "first"),
        original_os=("os", "first"),
        app=("app", safe_mode),
        device=("device", safe_mode),
        os=("os", safe_mode),
        channel=("channel", safe_mode),
        click_count=("session_id", "size"),
        total_requests=("session_id", "size"),
        successful_requests=("is_attributed", "sum"),
        install_count=("is_attributed", "sum"),
        session_start=("click_time", "min"),
        session_end=("click_time", "max"),
        year=("year", "first"),
        month=("month", "first"),
        day=("day", "first"),
        hour=("hour", "first"),
        minute=("minute", "first"),
        second=("second", "first"),
        ip_click_frequency=("ip_click_frequency", "max"),
        device_click_frequency=("device_click_frequency", "max"),
        ip_install_rate=("ip_install_rate", "max"),
        combo_repeat_rate=("combo_repeat_rate", "mean"),
        human_event_ratio=("human_heuristic", "mean"),
        possible_bot_ratio=("possible_bot_heuristic", "mean"),
    ).reset_index()
    sessions["session_duration_sec"] = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().clip(lower=1.0)
    intervals = grouped["event_interval_sec"].agg(["mean", "std"]).reset_index()
    intervals.columns = ["session_id", "click_interval_mean", "click_interval_std"]
    sessions = sessions.merge(intervals, on="session_id", how="left")
    sessions["click_interval_mean"] = sessions["click_interval_mean"].fillna(0.0)
    sessions["click_interval_std"] = sessions["click_interval_std"].fillna(0.0)
    sessions["clicks_per_minute"] = sessions["click_count"] / (sessions["session_duration_sec"] / 60.0)
    sessions["requests_per_minute"] = sessions["total_requests"] / (sessions["session_duration_sec"] / 60.0)
    sessions["device_entropy"] = grouped["device"].apply(entropy).values
    sessions["channel_entropy"] = grouped["channel"].apply(entropy).values
    sessions["app_entropy"] = grouped["app"].apply(entropy).values
    sessions["label"] = np.where(
        (sessions["possible_bot_ratio"] >= 0.50)
        | ((sessions["requests_per_minute"] >= sessions["requests_per_minute"].quantile(0.99)) & (sessions["install_count"] == 0)),
        1,
        0,
    )
    sessions.loc[sessions["install_count"] > 0, "label"] = 0
    sessions["bot_type"] = np.where(sessions["label"] == 1, "talkingdata_possible_bot", "talkingdata_human")
    sessions["success_rate"] = sessions["install_count"] / sessions["click_count"].clip(lower=1)
    sessions["request_interval_mean"] = sessions["click_interval_mean"]
    sessions["request_interval_std"] = sessions["click_interval_std"]
    sessions["activity_date"] = sessions["session_start"].dt.strftime("%Y-%m-%d")
    sessions["time_range"] = sessions["session_start"].dt.strftime("%H:%M:%S") + "-" + sessions["session_end"].dt.strftime("%H:%M:%S")
    sessions["logs_count"] = sessions["click_count"]
    sessions["total_response_size"] = np.nan
    sessions["avg_response_size"] = np.nan
    return events, sessions


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df[["clicks_per_minute", "requests_per_minute", "click_interval_mean", "click_interval_std", "channel_entropy", "app_entropy", "session_duration_sec", "install_count"]].fillna(0.0)
    z = (base - base.mean()) / base.std(ddof=0).replace(0, 1.0)
    human = 1.0 - df["label"].to_numpy()
    bot = df["label"].to_numpy()
    throughput = z["requests_per_minute"].to_numpy()
    irregular = z["click_interval_std"].to_numpy()
    diversity = (z["channel_entropy"] + z["app_entropy"]).to_numpy() / 2.0
    installs = z["install_count"].to_numpy()
    noise = RNG.normal(0.0, 1.0, size=(len(df), 10))
    speed_mean = 145 + 28 * human + 12 * throughput + 10 * diversity - 14 * bot + 8 * noise[:, 0]
    speed_std = 34 + 12 * human + 8 * irregular + 6 * diversity - 16 * bot + 4 * noise[:, 1]
    path_length = np.maximum(df["click_count"].to_numpy(), 1) * (0.72 * speed_mean + 18 * diversity + 6 * irregular + 4 * noise[:, 2])
    direction_changes = np.maximum(df["click_count"].to_numpy() - 1, 0) * (0.42 + 0.16 * human + 0.10 * diversity - 0.14 * bot) + 2 * noise[:, 3]
    hover_mean = 1.20 + 0.18 * human + 0.12 * installs - 0.20 * throughput - 0.15 * bot + 0.08 * noise[:, 4]
    hover_std = 0.55 + 0.24 * human + 0.18 * irregular - 0.22 * bot + 0.05 * noise[:, 5]
    scroll_count = 2.5 + 0.014 * df["session_duration_sec"].to_numpy() + 6 * human + 4 * diversity - 5.5 * bot + 1.5 * noise[:, 6]
    scroll_depth = 28 + 8.5 * np.log1p(np.maximum(scroll_count, 0)) + 16 * human + 7 * diversity - 14 * bot + 3 * noise[:, 7]
    coord_entropy = 2.6 + 1.1 * human + 0.45 * diversity + 0.20 * irregular - 1.15 * bot + 0.18 * noise[:, 8]
    accel_std = 0.85 + 0.018 * speed_std + 0.12 * coord_entropy + 0.08 * human - 0.12 * bot + 0.04 * noise[:, 9]
    df["mouse_speed_mean"] = np.clip(speed_mean, 35, None)
    df["mouse_speed_std"] = np.clip(speed_std, 4, None)
    df["mouse_path_length"] = np.clip(path_length, 50, None)
    df["direction_change_count"] = np.clip(np.rint(direction_changes), 0, None).astype(int)
    df["hover_time_mean"] = np.clip(hover_mean, 0.05, None)
    df["hover_time_std"] = np.clip(hover_std, 0.01, None)
    df["scroll_count"] = np.clip(np.rint(scroll_count), 0, None).astype(int)
    df["scroll_depth"] = np.clip(scroll_depth, 0, 100)
    df["coordinate_entropy"] = np.clip(coord_entropy, 0.05, 8.5)
    df["cursor_acceleration_std"] = np.clip(accel_std, 0.01, None)
    df["movement_std"] = np.clip(0.42 * df["mouse_speed_std"] + 4.0, 0.5, None)
    df["bot_likelihood_score"] = (
        2.2 * (df["requests_per_minute"] > df["requests_per_minute"].quantile(0.95)).astype(int)
        + 1.6 * (df["coordinate_entropy"] < df["coordinate_entropy"].quantile(0.10)).astype(int)
        + 1.4 * (df["mouse_speed_std"] < df["mouse_speed_std"].quantile(0.10)).astype(int)
        + 1.1 * df["label"]
    )
    return df


def add_device_network_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    browsers = weighted_choice({"Chrome": 0.65, "Safari": 0.18, "Firefox": 0.10, "Edge": 0.07}, len(df))
    operating_systems = weighted_choice({"Windows": 0.38, "MacOS": 0.16, "Linux": 0.10, "Android": 0.20, "iOS": 0.16}, len(df))
    device_types = weighted_choice({"desktop": 0.46, "mobile": 0.46, "tablet": 0.08}, len(df))
    bot_mask = df["label"].to_numpy() == 1
    operating_systems = np.where(bot_mask & (RNG.random(len(df)) < 0.45), "Linux", operating_systems)
    device_types = np.where(np.isin(operating_systems, ["Android", "iOS"]), "mobile", device_types)
    device_types = np.where(bot_mask & (RNG.random(len(df)) < 0.80), "desktop", device_types)
    countries = weighted_choice({"United States": 0.28, "India": 0.18, "United Kingdom": 0.10, "Germany": 0.09, "France": 0.07, "Canada": 0.08, "Japan": 0.07, "Brazil": 0.06, "Singapore": 0.04, "Australia": 0.03}, len(df))
    regions_map = {
        "United States": ["California", "New York", "Texas", "Illinois", "Florida"],
        "India": ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Telangana"],
        "United Kingdom": ["England", "Scotland", "Wales"],
        "Germany": ["Berlin", "Bavaria", "Hamburg"],
        "France": ["Ile-de-France", "Auvergne-Rhone-Alpes"],
        "Canada": ["Ontario", "Quebec", "British Columbia"],
        "Japan": ["Tokyo", "Osaka", "Kanagawa"],
        "Brazil": ["Sao Paulo", "Rio de Janeiro"],
        "Singapore": ["Central Singapore"],
        "Australia": ["New South Wales", "Victoria"],
    }
    regions = [RNG.choice(regions_map[c]) for c in countries]
    proxy = np.where(bot_mask, RNG.random(len(df)) < 0.72, RNG.random(len(df)) < 0.08).astype(int)
    bot_uas = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/121.0.6167.85 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.70 Safari/537.36",
    ]
    uas, ips = [], []
    dc_prefixes = [1, 8, 34, 35, 44, 52, 54, 64, 66, 104, 128, 136, 138, 143, 159, 167, 185, 198, 207]
    res_prefixes = [24, 47, 73, 86, 98, 101, 103, 115, 117, 122, 149, 172, 174, 188, 203]
    for idx, (label, browser, os_name, proxy_flag) in enumerate(zip(df["label"], browsers, operating_systems, proxy)):
        major = int(np.clip(round(RNG.normal(122, 3)), 110, 132))
        build = int(np.clip(round(RNG.normal(4500, 180)), 1000, 7000))
        if label == 1:
            uas.append(bot_uas[idx % len(bot_uas)])
        elif browser == "Safari":
            uas.append(f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{max(15, major // 8)}.0 Safari/605.1.15")
        elif browser == "Firefox":
            token = "Windows NT 10.0; Win64; x64" if os_name in {"Windows", "MacOS"} else "X11; Linux x86_64"
            uas.append(f"Mozilla/5.0 ({token}; rv:{major}.0) Gecko/20100101 Firefox/{major}.0")
        elif browser == "Edge":
            uas.append(f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.{build}.0 Safari/537.36 Edg/{major}.0.{build}.0")
        else:
            uas.append(f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.{build}.0 Safari/537.36")
        if FAKE is not None and proxy_flag == 0 and RNG.random() < 0.15:
            ips.append(FAKE.ipv4_public())
        else:
            pool = dc_prefixes if label == 1 else res_prefixes
            ips.append(f"{int(RNG.choice(pool))}.{int(RNG.integers(0,256))}.{int(RNG.integers(0,256))}.{int(RNG.integers(1,255))}")
    df["browser"], df["operating_system"], df["device_type"] = browsers, operating_systems, device_types
    df["user_agent"], df["ip_address"] = uas, ips
    df["country"], df["region"], df["is_proxy"] = countries, regions, proxy
    return df


def load_existing_sessions() -> pd.DataFrame:
    df = pd.read_csv(EXISTING_AGG_PATH).copy()
    df["data_source"] = "existing_clickstream"
    df["label"] = np.where(df["label"].fillna(0).astype(int) == 0, 0, 1)
    df["bot_type"] = np.where(df["label"] == 1, "existing_clickstream_bot", "existing_clickstream_human")
    df["click_count"] = df.get("click_count", df["total_requests"])
    df["install_count"] = df.get("install_count", 0)
    df["click_interval_mean"] = df.get("click_interval_mean", df["request_interval_mean"])
    df["click_interval_std"] = df.get("click_interval_std", df["request_interval_std"])
    df["ip_click_frequency"] = df.get("ip_click_frequency", df["total_requests"].fillna(0))
    df["device_click_frequency"] = df.get("device_click_frequency", df["total_movements"].fillna(df["total_requests"].fillna(0)))
    df["ip_install_rate"] = df.get("ip_install_rate", 0.0)
    df["combo_repeat_rate"] = df.get("combo_repeat_rate", (1.0 - df["coordinate_entropy"].rank(pct=True)).fillna(0.5))
    df["device_entropy"] = df.get("device_entropy", 0.0)
    df["channel_entropy"] = df.get("channel_entropy", df["coordinate_entropy"] / 8.0)
    df["app_entropy"] = df.get("app_entropy", df["channel_entropy"])
    for col in ["original_ip", "original_device", "original_os", "app", "device", "os", "channel"]:
        if col not in df.columns:
            df[col] = pd.NA
    start = pd.to_datetime(df["activity_date"].astype(str) + " " + df["time_range"].astype(str).str.split("-").str[0], format="%d/%b/%Y %H:%M:%S", errors="coerce")
    df["session_start"] = start
    df["session_end"] = df["session_start"] + pd.to_timedelta(df["session_duration_sec"].fillna(0), unit="s")
    for part in ["year", "month", "day", "hour", "minute", "second"]:
        if part not in df.columns:
            df[part] = getattr(df["session_start"].dt, part).fillna(0).astype(int)
    for col in ["hover_time_mean", "hover_time_std", "scroll_count", "scroll_depth", "cursor_acceleration_std"]:
        if col not in df.columns:
            df = add_behavioral_features(df)
            break
    if any(col not in df.columns for col in ["browser", "operating_system", "device_type", "user_agent", "ip_address", "country", "region", "is_proxy"]):
        enriched = add_device_network_features(df.copy())
        for col in ["browser", "operating_system", "device_type", "user_agent", "ip_address", "country", "region", "is_proxy"]:
            if col not in df.columns:
                df[col] = enriched[col]
    df["is_proxy"] = df["is_proxy"].astype(int)
    return df


def combine_sessions(talking_sessions: pd.DataFrame, existing_sessions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "session_id", "data_source", "bot_type", "label", "session_start", "session_end", "year", "month", "day", "hour", "minute", "second",
        "activity_date", "time_range", "click_count", "total_requests", "successful_requests", "install_count", "logs_count", "session_duration_sec",
        "click_interval_mean", "click_interval_std", "request_interval_mean", "request_interval_std", "clicks_per_minute", "requests_per_minute",
        "device_entropy", "channel_entropy", "app_entropy", "ip_click_frequency", "device_click_frequency", "ip_install_rate", "combo_repeat_rate",
        "success_rate", "mouse_speed_mean", "mouse_speed_std", "mouse_path_length", "direction_change_count", "hover_time_mean", "hover_time_std",
        "scroll_count", "scroll_depth", "coordinate_entropy", "cursor_acceleration_std", "movement_std", "bot_likelihood_score",
        "browser", "operating_system", "device_type", "user_agent", "ip_address", "country", "region", "is_proxy",
        "original_ip", "original_device", "original_os", "app", "device", "os", "channel", "total_response_size", "avg_response_size",
    ]
    frames = [talking_sessions.copy(), existing_sessions.copy()]
    for frame in frames:
        for col in cols:
            if col not in frame.columns:
                frame[col] = pd.NA
    combined = pd.concat([frames[0][cols], frames[1][cols]], ignore_index=True).sort_values(["session_start", "session_id"], na_position="last")
    num_cols = combined.select_dtypes(include=[np.number]).columns
    combined[num_cols] = combined[num_cols].replace([np.inf, -np.inf], np.nan).fillna(combined[num_cols].median(numeric_only=True))
    for col in combined.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns:
        combined[col] = combined[col].fillna("unknown")
    return combined.reset_index(drop=True)


def _strata(series: pd.Series, bins: int = 5) -> pd.Series:
    try:
        return pd.qcut(series.rank(method="first"), q=min(bins, series.nunique()), duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")


def balance_combined(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = combined["label"].value_counts()
    if len(counts) < 2 or counts.iloc[0] == counts.iloc[-1]:
        balanced = combined.copy().reset_index(drop=True)
        balanced.to_csv(OUTPUTS["combined"], index=False)
        return balanced, {"before": {str(k): int(v) for k, v in counts.to_dict().items()}, "after": {str(k): int(v) for k, v in counts.to_dict().items()}, "minority_count": int(counts.min()) if len(counts) else 0}

    minority_label = int(counts.idxmin())
    majority_label = int(counts.idxmax())
    minority = combined[combined["label"] == minority_label].copy()
    majority = combined[combined["label"] == majority_label].copy()
    target = len(minority)

    majority["rpm_bin"] = _strata(majority["requests_per_minute"])
    majority["dur_bin"] = _strata(majority["session_duration_sec"])
    majority["entropy_bin"] = _strata(majority["coordinate_entropy"])
    majority["stratum"] = (
        majority["data_source"].astype(str) + "|" +
        majority["is_proxy"].astype(str) + "|" +
        majority["rpm_bin"].astype(str) + "|" +
        majority["dur_bin"].astype(str) + "|" +
        majority["entropy_bin"].astype(str)
    )

    stratum_counts = majority["stratum"].value_counts()
    raw_targets = (stratum_counts / stratum_counts.sum()) * target
    floor_targets = np.floor(raw_targets).astype(int)
    floor_targets = floor_targets.clip(upper=stratum_counts)
    remainder = int(target - floor_targets.sum())
    if remainder > 0:
        fractional = (raw_targets - floor_targets).sort_values(ascending=False)
        for stratum in fractional.index:
            if remainder == 0:
                break
            if floor_targets[stratum] < stratum_counts[stratum]:
                floor_targets[stratum] += 1
                remainder -= 1

    sampled_parts = []
    for stratum, take_n in floor_targets.items():
        if take_n <= 0:
            continue
        sampled_parts.append(
            majority[majority["stratum"] == stratum].sample(n=int(take_n), random_state=RANDOM_SEED, replace=False)
        )

    sampled_majority = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else majority.sample(n=target, random_state=RANDOM_SEED, replace=False)
    if len(sampled_majority) < target:
        remaining = majority.drop(index=sampled_majority.index, errors="ignore")
        top_up = remaining.sample(n=target - len(sampled_majority), random_state=RANDOM_SEED, replace=False)
        sampled_majority = pd.concat([sampled_majority, top_up], ignore_index=False)

    balanced = pd.concat([minority, sampled_majority.drop(columns=["rpm_bin", "dur_bin", "entropy_bin", "stratum"], errors="ignore")], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=RANDOM_SEED).sort_values(["session_start", "session_id"], na_position="last").reset_index(drop=True)
    balanced.to_csv(OUTPUTS["combined"], index=False)
    return balanced, {
        "before": {str(k): int(v) for k, v in counts.to_dict().items()},
        "after": {str(k): int(v) for k, v in balanced["label"].value_counts().to_dict().items()},
        "minority_count": int(target),
        "majority_label_downsampled": majority_label,
    }


def talking_sequence(session_row: pd.Series, session_event_map: dict[str, pd.DataFrame]) -> np.ndarray:
    session_events = session_event_map.get(session_row["session_id"])
    if session_events is None or session_events.empty:
        return np.zeros((SEQUENCE_LENGTH, 6), dtype=np.float32)
    count = len(session_events)
    deltas = session_events["click_time"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    speeds = np.clip(
        float(session_row["mouse_speed_mean"]) + RNG.normal(0.0, max(float(session_row["mouse_speed_std"]), 1.0), size=count)
        - 0.12 * deltas + (0.15 if int(session_row["label"]) == 0 else -0.10) * float(session_row["coordinate_entropy"]),
        0.1,
        None,
    )
    directions = np.zeros(count, dtype=float)
    volatility = 0.85 if int(session_row["label"]) == 0 else 0.28
    for idx in range(1, count):
        directions[idx] = np.clip(directions[idx - 1] + RNG.normal(0.0, volatility), -math.pi, math.pi)
    scroll_probability = min(0.85, float(session_row["scroll_count"]) / max(count, 1))
    scroll_flags = RNG.binomial(1, scroll_probability, size=count).astype(float)
    movement_entropy = np.array([rolling_entropy(speeds[max(0, i - 4): i + 1]) for i in range(count)], dtype=float)
    features = np.column_stack([deltas, speeds, directions, np.ones(count), scroll_flags, movement_entropy]).astype(np.float32)
    if count >= SEQUENCE_LENGTH:
        return features[:SEQUENCE_LENGTH]
    return np.vstack([features, np.zeros((SEQUENCE_LENGTH - count, 6), dtype=np.float32)])


def existing_sequences(selected_session_ids: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not EXISTING_SEQ_PATH.exists():
        return np.zeros((0, SEQUENCE_LENGTH, 6), dtype=np.float32), np.array([], dtype=int), []
    with open(EXISTING_SEQ_PATH, "rb") as handle:
        payload = pickle.load(handle)
    seq = np.asarray(payload.get("sequences", np.zeros((0, SEQUENCE_LENGTH, 6), dtype=np.float32)))
    labels = np.where(np.asarray(payload.get("labels", np.array([], dtype=int))) == 0, 0, 1).astype(int)
    session_ids = list(payload.get("session_ids", []))
    if selected_session_ids is not None and session_ids:
        keep = [idx for idx, session_id in enumerate(session_ids) if session_id in selected_session_ids]
        seq = seq[keep] if len(keep) else np.zeros((0, SEQUENCE_LENGTH, 6), dtype=np.float32)
        labels = labels[keep] if len(keep) else np.array([], dtype=int)
        session_ids = [session_ids[idx] for idx in keep]
    if seq.size == 0:
        return np.zeros((0, SEQUENCE_LENGTH, 6), dtype=np.float32), labels, session_ids
    if seq.shape[2] < 6:
        padded = np.zeros((seq.shape[0], seq.shape[1], 6), dtype=np.float32)
        padded[:, :, :seq.shape[2]] = seq
        seq = padded
    dx = np.diff(seq[:, :, 0], axis=1, prepend=seq[:, :1, 0])
    dy = np.diff(seq[:, :, 1], axis=1, prepend=seq[:, :1, 1])
    timestamp_delta = np.diff(seq[:, :, 5], axis=1, prepend=seq[:, :1, 5])
    direction = np.arctan2(dy, dx)
    movement_entropy = np.zeros_like(seq[:, :, 2])
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            movement_entropy[i, j] = rolling_entropy(seq[i, max(0, j - 4): j + 1, 2])
    mapped = np.stack([timestamp_delta, seq[:, :, 2], direction, seq[:, :, 3], seq[:, :, 4], movement_entropy], axis=2).astype(np.float32)
    return mapped, labels, session_ids


def build_sequence_dataset(combined: pd.DataFrame, events: pd.DataFrame) -> dict:
    talking = combined[combined["data_source"] == "talkingdata"]
    session_event_map = {
        session_id: frame.sort_values("click_time")
        for session_id, frame in events.groupby("session_id", sort=False)
    }
    td_seq = (
        np.asarray([talking_sequence(row, session_event_map) for _, row in talking.iterrows()], dtype=np.float32)
        if len(talking)
        else np.zeros((0, SEQUENCE_LENGTH, 6), dtype=np.float32)
    )
    td_labels = talking["label"].to_numpy(dtype=int) if len(talking) else np.array([], dtype=int)
    td_ids = talking["session_id"].tolist()
    selected_session_ids = set(combined["session_id"].astype(str))
    ex_seq, ex_labels, ex_ids = existing_sequences(selected_session_ids)
    sequences = np.concatenate([td_seq, ex_seq], axis=0)
    labels = np.concatenate([td_labels, ex_labels], axis=0)
    session_ids = td_ids + ex_ids
    if labels.size:
        label_counts = pd.Series(labels).value_counts()
        if len(label_counts) > 1 and label_counts.max() != label_counts.min():
            target = int(label_counts.min())
            keep_indices = []
            for label_value in sorted(label_counts.index):
                label_idx = np.where(labels == label_value)[0]
                sampled_idx = RNG.choice(label_idx, size=target, replace=False)
                keep_indices.extend(sampled_idx.tolist())
            keep_indices = np.array(sorted(keep_indices))
            sequences = sequences[keep_indices]
            labels = labels[keep_indices]
            session_ids = [session_ids[idx] for idx in keep_indices]
    scaler = StandardScaler()
    if sequences.size:
        sequences = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape).astype(np.float32)
    payload = {
        "sequences": sequences,
        "labels": labels,
        "session_ids": session_ids,
        "feature_names": ["timestamp_delta", "mouse_speed", "mouse_direction", "click_flag", "scroll_flag", "movement_entropy"],
        "sequence_length": SEQUENCE_LENGTH,
        "n_features": 6,
        "source_counts": {"talkingdata_sequences": int(td_seq.shape[0]), "existing_sequences": int(ex_seq.shape[0]), "total_sequences": int(sequences.shape[0])},
        "scaler": scaler,
    }
    with open(OUTPUTS["sequence"], "wb") as handle:
        pickle.dump(payload, handle)
    return payload


def build_report(inspection: dict, candidates: pd.DataFrame, sessions: pd.DataFrame, combined: pd.DataFrame, sequence_payload: dict, balance_info: dict) -> dict:
    outliers = {}
    for col in combined.select_dtypes(include=[np.number]).columns:
        q1, q3 = combined[col].quantile(0.25), combined[col].quantile(0.75)
        iqr = q3 - q1
        count = int(((combined[col] < (q1 - 1.5 * iqr)) | (combined[col] > (q3 + 1.5 * iqr))).sum())
        if count:
            outliers[col] = count
    pairs = [("mouse_speed_mean", "mouse_path_length"), ("mouse_speed_std", "cursor_acceleration_std"), ("coordinate_entropy", "direction_change_count"), ("requests_per_minute", "label"), ("is_proxy", "label")]
    correlations = {f"{a}__{b}": round(float(combined[a].corr(combined[b])), 6) for a, b in pairs if a in combined.columns and b in combined.columns and not pd.isna(combined[a].corr(combined[b]))}
    report = {
        "random_seed": RANDOM_SEED,
        "talkingdata_inspection": inspection,
        "intermediate_output": {"talkingdata_human_candidates_rows": int(len(candidates)), "talkingdata_label_distribution": {str(k): int(v) for k, v in candidates["label"].value_counts().to_dict().items()}},
        "sessionization": {"talkingdata_session_rows": int(len(sessions)), "talkingdata_session_label_distribution": {str(k): int(v) for k, v in sessions["label"].value_counts().to_dict().items()}, "duplicate_session_ids": int(combined["session_id"].duplicated().sum())},
        "combined_dataset": {
            "rows": int(len(combined)),
            "columns": list(combined.columns),
            "class_balance": {str(k): int(v) for k, v in combined["label"].value_counts().to_dict().items()},
            "balance_operation": balance_info,
            "source_balance": combined["data_source"].value_counts().to_dict(),
            "missing_values": {k: int(v) for k, v in combined.isna().sum().to_dict().items() if int(v) > 0},
            "outliers": dict(sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:20]),
            "selected_correlations": correlations,
            "behavior_summary_by_label": combined.groupby("label")[["mouse_speed_mean", "mouse_speed_std", "coordinate_entropy", "requests_per_minute", "scroll_depth", "hover_time_mean"]].mean().round(4).to_dict(orient="index"),
        },
        "sequence_dataset": {"shape": list(sequence_payload["sequences"].shape), "label_distribution": {str(k): int(v) for k, v in pd.Series(sequence_payload["labels"]).value_counts().to_dict().items()}, "feature_names": sequence_payload["feature_names"]},
        "integrity_checks": {"combined_has_missing_labels": bool(combined["label"].isna().any()), "combined_has_duplicate_sessions": bool(combined["session_id"].duplicated().any()), "synthetic_feature_correlations_reasonable": all(abs(v) <= 0.999 for v in correlations.values())},
        "dependencies": {"faker_available": Faker is not None},
    }
    with open(OUTPUTS["report"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return report


def main() -> None:
    talking = load_talkingdata()
    inspection = inspect_talkingdata(talking)
    events = label_events(add_event_features(talking))
    events.to_csv(OUTPUTS["candidates"], index=False)
    session_events, talking_sessions = sessionize(events)
    talking_sessions = add_device_network_features(add_behavioral_features(talking_sessions))
    talking_sessions.to_csv(OUTPUTS["sessions"], index=False)
    existing = load_existing_sessions()
    combined_raw = combine_sessions(talking_sessions, existing)
    combined, balance_info = balance_combined(combined_raw)
    sequence_payload = build_sequence_dataset(combined, session_events)
    report = build_report(inspection, events, talking_sessions, combined, sequence_payload, balance_info)
    print(json.dumps({"combined_rows": len(combined), "sequence_shape": sequence_payload["sequences"].shape, "report": str(OUTPUTS["report"]), "integrity": report["integrity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
