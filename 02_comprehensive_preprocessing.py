"""Unified preprocessing pipeline for clickstream bot detection.

Updated on 2026-03-11:
- integrates real TalkingData human sessions before any augmentation
- adds CLI options, dual-mode sequence export, validation reports, and merge checks
- preserves legacy bot sources while producing unified artifacts under preprocessing_output/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_module import (
    LABEL_MAPPING,
    MIN_HUMAN_SESSIONS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    bootstrap_human_sessions,
    build_integration_report,
    build_mouse_sequences,
    build_validation_report,
    calculate_mouse_features,
    ensure_feature_schema,
    merge_and_sanity_check,
    prepare_lstm_sequences,
    save_metadata_report,
    save_sequence_payload,
    seeded_rng,
    set_random_seed,
    standardize_sequences,
)
from talkingdata_preprocessor import TalkingDataPreprocessor, align_schema_and_export


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Datasets"
OUTPUT_DIR = BASE_DIR / "preprocessing_output"
DEFAULT_TALKINGDATA_DIR = BASE_DIR / "talkingdata-adtracking-fraud-detection (1)"
LEGACY_AGGREGATED_PATH = OUTPUT_DIR / "session_aggregated_dataset.csv"
TALKINGDATA_SESSION_PATH = OUTPUT_DIR / "talkingdata_session_features.csv"
COMBINED_DATASET_PATH = OUTPUT_DIR / "combined_clickstream_dataset.csv"
COMBINED_SEQUENCE_PATH = OUTPUT_DIR / "combined_sequence_dataset.pkl"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "preprocessing_validation_report.json"
INTEGRATION_REPORT_PATH = OUTPUT_DIR / "dataset_integration_report.json"
METADATA_REPORT_PATH = OUTPUT_DIR / "dataset_metadata_report.json"


def _load_advanced_sessions() -> pd.DataFrame:
    """Use the existing advanced-bot aggregate as the trusted legacy source."""
    if not LEGACY_AGGREGATED_PATH.exists():
        raise FileNotFoundError("Missing preprocessing_output/session_aggregated_dataset.csv.")

    advanced = pd.read_csv(LEGACY_AGGREGATED_PATH).copy()
    advanced["label"] = LABEL_MAPPING["advanced_bot"]
    advanced["bot_type"] = "advanced_bot"
    advanced["data_source"] = "legacy_advanced"
    advanced["data_source_detail"] = "preexisting_session_aggregated_dataset"
    advanced["candidate_human"] = False
    if "session_start" not in advanced.columns:
        advanced["session_start"] = pd.to_datetime(
            advanced["activity_date"].astype(str) + " " + advanced["time_range"].astype(str).str.split("-").str[0],
            format="%d/%b/%Y %H:%M:%S",
            errors="coerce",
        )
        advanced["session_end"] = advanced["session_start"] + pd.to_timedelta(
            advanced["session_duration_sec"].fillna(0.0), unit="s"
        )
    advanced["click_count"] = advanced.get("click_count", advanced["total_requests"]).fillna(0).astype(int)
    advanced["install_count"] = advanced.get("install_count", 0)
    advanced["successful_requests"] = advanced.get("successful_requests", 0)
    advanced["device_entropy"] = advanced.get("device_entropy", advanced["coordinate_entropy"].fillna(0) / 4.0)
    advanced["channel_entropy"] = advanced.get("channel_entropy", advanced["coordinate_entropy"].fillna(0) / 3.0)
    advanced["app_entropy"] = advanced.get("app_entropy", advanced["coordinate_entropy"].fillna(0) / 3.5)
    advanced["click_interval_mean"] = advanced.get("click_interval_mean", advanced["request_interval_mean"])
    advanced["click_interval_std"] = advanced.get("click_interval_std", advanced["request_interval_std"])
    advanced["click_interval_entropy"] = advanced.get("click_interval_entropy", advanced["coordinate_entropy"].fillna(0) / 3.0)
    advanced["request_interval_entropy"] = advanced.get("request_interval_entropy", advanced["click_interval_entropy"])
    advanced["session_idle_ratio"] = advanced.get("session_idle_ratio", 0.0)
    advanced["mouse_acceleration_std"] = advanced.get("mouse_acceleration_std", advanced["mouse_speed_std"].fillna(0) / 5.0)
    advanced["movement_curvature"] = advanced.get(
        "movement_curvature",
        advanced["direction_change_count"].fillna(0) / advanced["mouse_path_length"].clip(lower=1.0),
    )
    return ensure_feature_schema(advanced)


def _allocate_integer_totals(weights: np.ndarray, total_target: int) -> np.ndarray:
    raw = weights / weights.sum() * total_target
    base = np.floor(raw).astype(int)
    remainder = total_target - base.sum()
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


def _segment_behavioral_sessions(behavioral: pd.DataFrame, target_sessions: int, prefix: str) -> pd.DataFrame:
    """Split long per-user traces into deterministic pseudo-sessions."""
    grouped_sizes = behavioral.groupby("session_id").size().sort_index()
    base_segments = np.ones(len(grouped_sizes), dtype=int)
    remaining_segments = max(0, target_sessions - len(grouped_sizes))
    if remaining_segments > 0:
        base_segments += _allocate_integer_totals(grouped_sizes.to_numpy(dtype=float), remaining_segments)

    segment_lookup = dict(zip(grouped_sizes.index.tolist(), base_segments.tolist()))
    segmented_parts = []
    for original_session_id, frame in behavioral.groupby("session_id", sort=True):
        ordered = frame.sort_values("movement_index").reset_index(drop=True)
        n_segments = int(segment_lookup.get(original_session_id, 1))
        for idx, split_idx in enumerate(np.array_split(np.arange(len(ordered)), n_segments), start=1):
            segment = ordered.iloc[split_idx].copy()
            if segment.empty:
                continue
            segment["original_session_id"] = original_session_id
            segment["session_id"] = f"{prefix}_{original_session_id}_{idx:02d}"
            segment["movement_index"] = np.arange(len(segment), dtype=int)
            segmented_parts.append(segment)

    return pd.concat(segmented_parts, ignore_index=True)


def _build_moderate_sessions(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create moderate bot sessions from mouse logs when temporal logs are unavailable."""
    behavior_path = DATASET_DIR / "humans_and_moderate_bots_behavioral_detailed.csv"
    report_path = DATASET_DIR / "humans_and_moderate_bots_combined_report.csv"
    if not behavior_path.exists():
        raise FileNotFoundError("Missing moderate bot behavioral dataset.")

    behavioral = pd.read_csv(behavior_path)
    target_sessions = 263
    if report_path.exists():
        report = pd.read_csv(report_path)
        if not report.empty:
            target_sessions = int(report.iloc[0]["Total_Sessions"])

    behavioral["label"] = behavioral["category"].astype(str).map(lambda text: LABEL_MAPPING["moderate_bot"])
    segmented = _segment_behavioral_sessions(behavioral, target_sessions, prefix="mod")
    grouped = segmented.sort_values(["session_id", "movement_index"]).groupby("session_id", sort=False)
    total_requests_target = 57_454
    movement_totals = grouped.size().astype(float)
    request_totals = _allocate_integer_totals(np.sqrt(movement_totals.to_numpy()), total_requests_target)
    request_lookup = dict(zip(movement_totals.index.tolist(), request_totals.tolist()))

    session_rows = []
    base_start = pd.Timestamp("2019-10-30 00:00:00")
    for offset, (session_id, frame) in enumerate(grouped):
        mouse = calculate_mouse_features(
            frame["x_coordinate"].to_numpy(dtype=float),
            frame["y_coordinate"].to_numpy(dtype=float),
        )
        total_movements = int(len(frame))
        total_requests = int(max(20, request_lookup.get(session_id, 20)))
        session_duration = float(np.clip(total_movements / max(mouse["mouse_speed_mean"], 1.0) * 18.0, 45, 1600))
        interval_mean = float(session_duration / max(total_requests - 1, 1))
        interval_std = float(max(0.05, interval_mean * (0.18 + 0.07 * mouse["movement_std"] / max(mouse["mouse_speed_mean"], 1.0))))
        start = base_start + pd.Timedelta(minutes=7 * offset)
        end = start + pd.Timedelta(seconds=session_duration)
        session_rows.append(
            {
                "session_id": session_id,
                "label": LABEL_MAPPING["moderate_bot"],
                "bot_type": "moderate_bot",
                "data_source": "legacy_moderate",
                "data_source_detail": "behavioral_only_reconstruction",
                "candidate_human": False,
                "session_start": start,
                "session_end": end,
                "activity_date": start.strftime("%Y-%m-%d"),
                "time_range": f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}",
                "mouse_speed_mean": mouse["mouse_speed_mean"],
                "mouse_speed_std": mouse["mouse_speed_std"],
                "mouse_path_length": mouse["mouse_path_length"],
                "direction_change_count": mouse["direction_change_count"],
                "movement_std": mouse["movement_std"],
                "coordinate_entropy": mouse["coordinate_entropy"],
                "mouse_acceleration_std": mouse["mouse_acceleration_std"],
                "movement_curvature": mouse["movement_curvature"],
                "click_interval_entropy": mouse["coordinate_entropy"] / 2.5,
                "request_interval_entropy": mouse["coordinate_entropy"] / 2.5,
                "session_idle_ratio": float(np.clip(interval_std / max(session_duration, 1.0), 0.0, 1.0)),
                "requests_per_minute": float(total_requests / (session_duration / 60.0)),
                "clicks_per_minute": float(total_movements / (session_duration / 60.0)),
                "session_duration_sec": session_duration,
                "request_interval_mean": interval_mean,
                "request_interval_std": interval_std,
                "click_interval_mean": interval_mean,
                "click_interval_std": interval_std,
                "total_movements": total_movements,
                "total_requests": total_requests,
                "click_count": total_requests,
                "install_count": 0,
                "successful_requests": 0,
                "success_rate": 0.0,
                "device_entropy": 0.0,
                "channel_entropy": mouse["coordinate_entropy"] / 3.5,
                "app_entropy": mouse["coordinate_entropy"] / 4.0,
                "logs_count": total_requests,
                "total_response_size": np.nan,
                "avg_response_size": np.nan,
            }
        )

    sessions = ensure_feature_schema(pd.DataFrame(session_rows))
    return sessions, segmented


def _load_advanced_behavioral(advanced_sessions: pd.DataFrame) -> pd.DataFrame:
    """Split advanced traces and align them to the existing aggregate session ids."""
    behavior_path = DATASET_DIR / "humans_and_advanced_bots_behavioral_detailed.csv"
    if not behavior_path.exists():
        raise FileNotFoundError("Missing advanced bot behavioral dataset.")

    behavioral = pd.read_csv(behavior_path)
    segmented = _segment_behavioral_sessions(behavioral, len(advanced_sessions), prefix="adv")
    segmented_sizes = segmented.groupby("session_id").size().sort_values(ascending=False)
    target_sessions = (
        advanced_sessions[["session_id", "total_movements", "total_requests"]]
        .sort_values(["total_movements", "total_requests"], ascending=False)["session_id"]
        .tolist()
    )
    remap = dict(zip(segmented_sizes.index.tolist(), target_sessions))
    segmented["session_id"] = segmented["session_id"].map(remap)
    return segmented.sort_values(["session_id", "movement_index"]).reset_index(drop=True)


def _optional_ctgan_augment(human_df: pd.DataFrame, target_count: int) -> tuple[pd.DataFrame, dict]:
    """Use SDV/CTGAN if available; otherwise fall back to bootstrap augmentation."""
    try:
        from sdv.single_table import CTGANSynthesizer  # type: ignore
        from sdv.metadata import SingleTableMetadata  # type: ignore
    except Exception:
        return bootstrap_human_sessions(human_df, target_count, random_seed=RANDOM_SEED, augment_method="bootstrap-fallback")

    if len(human_df) >= target_count:
        return human_df.copy(), {"performed": False, "method": "ctgan", "generated": 0}

    numeric = human_df.select_dtypes(include=[np.number]).copy()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=numeric)
    synthesizer = CTGANSynthesizer(metadata, enforce_rounding=False)
    synthesizer.fit(numeric)
    generated = synthesizer.sample(num_rows=target_count - len(human_df))
    augmented = human_df.copy().reset_index(drop=True)
    samples = []
    for idx, row in generated.iterrows():
        base = human_df.iloc[idx % len(human_df)].copy()
        for column in generated.columns:
            base[column] = row[column]
        base["session_id"] = f"ctgan_human_{idx:05d}"
        base["data_source"] = "synthetic_human"
        base["data_source_detail"] = "ctgan"
        base["candidate_human"] = True
        samples.append(base)
    augmented = pd.concat([augmented, pd.DataFrame(samples)], ignore_index=True)
    augmented = ensure_feature_schema(augmented)
    return augmented, {"performed": True, "method": "ctgan", "generated": target_count - len(human_df)}


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _resolve_talkingdata_path(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path)
    return DEFAULT_TALKINGDATA_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified clickstream dataset.")
    parser.add_argument("--talkingdata-path", default=None)
    parser.add_argument("--min-human-sessions", type=int, default=MIN_HUMAN_SESSIONS)
    parser.add_argument("--augment-method", choices=["none", "bootstrap", "ctgan"], default="bootstrap")
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--force-rebuild", default="False")
    return parser.parse_args()


def _prepare_combined_sequences(
    talking_events: pd.DataFrame,
    human_sessions: pd.DataFrame,
    moderate_behavioral: pd.DataFrame,
    moderate_sessions: pd.DataFrame,
    advanced_behavioral: pd.DataFrame,
    advanced_sessions: pd.DataFrame,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create combined click-mode and mouse-mode sequences with type metadata."""
    click_seq, click_labels, click_meta = prepare_lstm_sequences(human_sessions, event_df=talking_events, seq_len=seq_len)
    click_meta.attrs["feature_names"] = ["timestamp_delta", "click_count_window", "session_progress", "click_interval_std", "device_entropy", "channel_entropy"]

    mod_seq, mod_labels, mod_ids, mod_types = build_mouse_sequences(moderate_behavioral, moderate_sessions, seq_len)
    adv_seq, adv_labels, adv_ids, adv_types = build_mouse_sequences(advanced_behavioral, advanced_sessions, seq_len)

    mouse_meta = pd.DataFrame(
        {
            "session_id": adv_ids + mod_ids,
            "label": np.concatenate([adv_labels, mod_labels]),
            "sequence_type": adv_types + mod_types,
        }
    )
    mouse_meta.attrs["feature_names"] = ["timestamp_delta", "mouse_velocity", "session_progress", "device_entropy", "channel_entropy"]

    mouse_sequences = np.concatenate([adv_seq, mod_seq], axis=0)
    max_features = max(click_seq.shape[2] if click_seq.size else 0, mouse_sequences.shape[2] if mouse_sequences.size else 0)

    def _pad_feature_dim(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0, seq_len, max_features), dtype=np.float32)
        if arr.shape[2] == max_features:
            return arr
        padding = np.zeros((arr.shape[0], arr.shape[1], max_features - arr.shape[2]), dtype=np.float32)
        return np.concatenate([arr, padding], axis=2)

    combined_sequences = np.concatenate([_pad_feature_dim(click_seq), _pad_feature_dim(mouse_sequences)], axis=0)
    combined_labels = np.concatenate([click_labels, adv_labels, mod_labels], axis=0)
    combined_meta = pd.concat([click_meta, mouse_meta], ignore_index=True)
    combined_meta.attrs["feature_names"] = [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
    ][:max_features]
    standardized, scaler = standardize_sequences(combined_sequences)
    return standardized, combined_labels, combined_meta, scaler


def main() -> None:
    args = _parse_args()
    _configure_logging()
    set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    talkingdata_path = _resolve_talkingdata_path(args.talkingdata_path)
    if not talkingdata_path.exists():
        raise FileNotFoundError(f"TalkingData path not found: {talkingdata_path}")

    advanced_sessions = _load_advanced_sessions()
    moderate_sessions, moderate_behavioral = _build_moderate_sessions(seeded_rng(RANDOM_SEED))
    advanced_behavioral = _load_advanced_behavioral(advanced_sessions)

    talkingdata = TalkingDataPreprocessor(talkingdata_path, random_seed=RANDOM_SEED, sequence_length=args.seq_len).run()
    human_sessions = talkingdata.sessions.copy()
    human_sessions = align_schema_and_export(human_sessions, TALKINGDATA_SESSION_PATH)
    LOGGER.info("TalkingData human sessions appended first: %s", len(human_sessions))

    augmentation_info = {"performed": False, "method": args.augment_method, "generated": 0}
    if len(human_sessions) < args.min_human_sessions:
        if args.augment_method == "none":
            LOGGER.warning("Human sessions below threshold and augmentation disabled: %s", len(human_sessions))
        elif args.augment_method == "ctgan":
            human_sessions, augmentation_info = _optional_ctgan_augment(human_sessions, args.min_human_sessions)
        else:
            human_sessions, augmentation_info = bootstrap_human_sessions(
                human_sessions,
                args.min_human_sessions,
                random_seed=RANDOM_SEED,
                augment_method="bootstrap",
            )
        LOGGER.info("Human augmentation summary: %s", augmentation_info)

    temp_existing_bots = OUTPUT_DIR / "_existing_bot_sessions.csv"
    pd.concat([moderate_sessions, advanced_sessions], ignore_index=True).pipe(ensure_feature_schema).to_csv(temp_existing_bots, index=False)
    merged = merge_and_sanity_check(temp_existing_bots, TALKINGDATA_SESSION_PATH, COMBINED_DATASET_PATH)

    if augmentation_info["performed"]:
        merged = pd.concat([human_sessions, merged[merged["label"] != LABEL_MAPPING["human"]]], ignore_index=True)
        merged = ensure_feature_schema(merged).drop_duplicates(subset=["session_id"]).reset_index(drop=True)
        merged.to_csv(COMBINED_DATASET_PATH, index=False)

    merged = ensure_feature_schema(merged)
    merged.to_csv(COMBINED_DATASET_PATH, index=False)

    sequences, sequence_labels, sequence_meta, scaler = _prepare_combined_sequences(
        talkingdata.events[talkingdata.events["session_id"].isin(set(human_sessions["session_id"]))],
        human_sessions,
        moderate_behavioral,
        moderate_sessions,
        advanced_behavioral,
        advanced_sessions,
        args.seq_len,
    )
    save_sequence_payload(COMBINED_SEQUENCE_PATH, sequences, sequence_labels, sequence_meta, scaler)

    validation_report = build_validation_report(merged)
    integration_report = build_integration_report(merged)
    metadata_report = {
        "random_seed": RANDOM_SEED,
        "talkingdata_path": str(talkingdata_path),
        "label_distribution": validation_report["label_counts"],
        "augmentation": augmentation_info,
        "sequence_shape": list(sequences.shape),
        "sequence_type_counts": sequence_meta["sequence_type"].value_counts().to_dict(),
    }
    save_metadata_report(VALIDATION_REPORT_PATH, validation_report)
    save_metadata_report(INTEGRATION_REPORT_PATH, integration_report)
    save_metadata_report(METADATA_REPORT_PATH, metadata_report)

    counts = merged["label"].value_counts().sort_index().to_dict()
    print("Final label distribution:", {str(k): int(v) for k, v in counts.items()})
    print("Human sessions:", int(counts.get(LABEL_MAPPING["human"], 0)))
    print("Moderate bot sessions:", int(counts.get(LABEL_MAPPING["moderate_bot"], 0)))
    print("Advanced bot sessions:", int(counts.get(LABEL_MAPPING["advanced_bot"], 0)))
    if int(counts.get(LABEL_MAPPING["human"], 0)) < args.min_human_sessions:
        print("Human threshold not met. Augmentation info:", augmentation_info)
    print("Combined CSV:", COMBINED_DATASET_PATH)
    print("Combined sequence PKL:", COMBINED_SEQUENCE_PATH)
    print("Integration report:", INTEGRATION_REPORT_PATH)


if __name__ == "__main__":
    main()
