"""Unified preprocessing pipeline for clickstream bot detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_module import (
    LABEL_MAPPING,
    COMMON_SEQUENCE_FEATURES,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    add_device_network_features,
    build_mouse_sequences,
    calculate_bot_likelihood_score,
    calculate_mouse_features,
    ensure_feature_schema,
    parse_clickstream_timestamp,
    save_metadata_report,
    save_sequence_payload,
    seeded_rng,
    standardize_sequences,
    stratified_sample,
)
from talkingdata_preprocessor import TalkingDataPreprocessor


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Datasets"
LEGACY_OUTPUT_DIR = BASE_DIR / "preprocessing_output"
TALKINGDATA_DIR = BASE_DIR / "talkingdata-adtracking-fraud-detection (1)"

COMBINED_DATASET_PATH = BASE_DIR / "combined_clickstream_dataset.csv"
COMBINED_SEQUENCE_PATH = BASE_DIR / "combined_sequence_dataset.pkl"
METADATA_REPORT_PATH = BASE_DIR / "dataset_metadata_report.json"


def _load_advanced_sessions() -> pd.DataFrame:
    """Use the existing advanced-bot aggregate as the trusted legacy source."""
    agg_path = LEGACY_OUTPUT_DIR / "session_aggregated_dataset.csv"
    if not agg_path.exists():
        raise FileNotFoundError(
            "Missing preprocessing_output/session_aggregated_dataset.csv. "
            "Run the legacy pipeline once or restore the file."
        )

    advanced = pd.read_csv(agg_path).copy()
    advanced["label"] = LABEL_MAPPING["advanced_bot"]
    advanced["bot_type"] = "advanced_bot"
    advanced["data_source"] = "legacy_advanced"
    advanced["session_start"] = pd.to_datetime(
        advanced["activity_date"].astype(str)
        + " "
        + advanced["time_range"].astype(str).str.split("-").str[0],
        format="%d/%b/%Y %H:%M:%S",
        errors="coerce",
    )
    advanced["session_end"] = advanced["session_start"] + pd.to_timedelta(
        advanced["session_duration_sec"].fillna(0), unit="s"
    )
    advanced["click_count"] = advanced["total_requests"].fillna(0).astype(int)
    advanced["install_count"] = 0
    advanced["device_entropy"] = advanced["coordinate_entropy"].fillna(0) / 4.0
    advanced["channel_entropy"] = advanced["coordinate_entropy"].fillna(0) / 3.0
    advanced["app_entropy"] = advanced["coordinate_entropy"].fillna(0) / 3.5
    advanced["original_ip"] = np.nan
    advanced["original_device"] = np.nan
    advanced["original_os"] = np.nan
    advanced["app"] = np.nan
    advanced["device"] = np.nan
    advanced["os"] = np.nan
    advanced["channel"] = np.nan
    advanced["bot_likelihood_score"] = advanced.apply(calculate_bot_likelihood_score, axis=1)
    return ensure_feature_schema(advanced)


def _allocate_integer_totals(weights: np.ndarray, total_target: int) -> np.ndarray:
    raw = weights / weights.sum() * total_target
    base = np.floor(raw).astype(int)
    remainder = total_target - base.sum()
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


def _segment_behavioral_sessions(
    behavioral: pd.DataFrame,
    target_sessions: int,
    prefix: str,
) -> pd.DataFrame:
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


def _load_segmented_moderate_behavioral() -> tuple[pd.DataFrame, int]:
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

    segmented = _segment_behavioral_sessions(behavioral, target_sessions=target_sessions, prefix="mod")
    return segmented, target_sessions


def _load_segmented_advanced_behavioral(advanced_sessions: pd.DataFrame) -> pd.DataFrame:
    """Split long advanced traces and align them to the 263 aggregate session ids."""
    behavior_path = DATASET_DIR / "humans_and_advanced_bots_behavioral_detailed.csv"
    if not behavior_path.exists():
        raise FileNotFoundError("Missing advanced bot behavioral dataset.")

    behavioral = pd.read_csv(behavior_path)
    segmented = _segment_behavioral_sessions(
        behavioral,
        target_sessions=len(advanced_sessions),
        prefix="adv",
    )
    segmented_sizes = segmented.groupby("session_id").size().sort_values(ascending=False)
    target_sessions = (
        advanced_sessions[["session_id", "total_movements", "total_requests"]]
        .sort_values(["total_movements", "total_requests"], ascending=False)["session_id"]
        .tolist()
    )
    remap = dict(zip(segmented_sizes.index.tolist(), target_sessions))
    segmented = segmented.copy()
    segmented["session_id"] = segmented["session_id"].map(remap)
    return segmented.sort_values(["session_id", "movement_index"]).reset_index(drop=True)


def _build_moderate_sessions(rng: np.random.Generator) -> pd.DataFrame:
    """Create moderate bot sessions from mouse logs when temporal logs are unavailable."""
    report_path = DATASET_DIR / "humans_and_moderate_bots_combined_report.csv"
    behavioral, target_sessions = _load_segmented_moderate_behavioral()
    grouped = behavioral.sort_values(["session_id", "movement_index"]).groupby("session_id", sort=False)
    session_rows = []
    total_requests_target = 57_454
    movement_totals = grouped.size().astype(float)
    request_totals = _allocate_integer_totals(np.sqrt(movement_totals.to_numpy()), total_requests_target)
    request_lookup = dict(zip(movement_totals.index.tolist(), request_totals.tolist()))
    base_start = pd.Timestamp("2019-10-30 00:00:00")

    for offset, (session_id, frame) in enumerate(grouped):
        x = frame["x_coordinate"].to_numpy(dtype=float)
        y = frame["y_coordinate"].to_numpy(dtype=float)
        mouse = calculate_mouse_features(x, y)
        total_movements = int(len(frame))
        total_requests = int(max(20, request_lookup.get(session_id, 20)))
        entropy_norm = mouse["coordinate_entropy"] / max(mouse["coordinate_entropy"], 1.0)
        velocity = 32 + 2.2 * np.log1p(total_movements) + 1.4 * mouse["coordinate_entropy"]
        session_duration = float(np.clip(total_movements / max(velocity, 1.0), 45, 1600))
        request_interval_mean = float(session_duration / max(total_requests - 1, 1))
        request_interval_std = float(
            request_interval_mean
            * (0.16 + 0.08 * (mouse["movement_std"] / max(mouse["mouse_speed_mean"], 1.0)))
        )
        clicks_per_minute = float(total_movements / (session_duration / 60.0))
        requests_per_minute = float(total_requests / (session_duration / 60.0))
        success_rate = float(np.clip(0.01 + 0.015 * entropy_norm, 0.005, 0.04))
        successful_requests = int(round(total_requests * success_rate))
        avg_response_size = int(np.clip(420 + 60 * mouse["coordinate_entropy"] + 0.08 * mouse["mouse_speed_mean"], 260, 900))
        start = base_start + pd.Timedelta(minutes=7 * offset)
        end = start + pd.Timedelta(seconds=session_duration)

        session_rows.append(
            {
                "session_id": session_id,
                "label": LABEL_MAPPING["moderate_bot"],
                "bot_type": "moderate_bot",
                "data_source": "legacy_moderate",
                "session_start": start,
                "session_end": end,
                "activity_date": start.strftime("%d/%b/%Y"),
                "time_range": f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}",
                "mouse_speed_mean": mouse["mouse_speed_mean"],
                "mouse_speed_std": mouse["mouse_speed_std"],
                "mouse_path_length": mouse["mouse_path_length"],
                "direction_change_count": mouse["direction_change_count"],
                "coordinate_entropy": mouse["coordinate_entropy"],
                "movement_std": mouse["movement_std"],
                "requests_per_minute": requests_per_minute,
                "clicks_per_minute": clicks_per_minute,
                "session_duration_sec": session_duration,
                "request_interval_mean": request_interval_mean,
                "request_interval_std": request_interval_std,
                "total_movements": total_movements,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": success_rate,
                "device_entropy": 0.0,
                "channel_entropy": mouse["coordinate_entropy"] / 3.5,
                "app_entropy": mouse["coordinate_entropy"] / 4.0,
                "click_count": total_requests,
                "install_count": 0,
                "original_ip": np.nan,
                "original_device": np.nan,
                "original_os": np.nan,
                "app": np.nan,
                "device": np.nan,
                "os": np.nan,
                "channel": np.nan,
                "logs_count": total_requests,
                "total_response_size": total_requests * avg_response_size,
                "avg_response_size": avg_response_size,
            }
        )

    moderate = pd.DataFrame(session_rows)
    moderate = add_device_network_features(moderate, rng)
    moderate["bot_likelihood_score"] = moderate.apply(calculate_bot_likelihood_score, axis=1)
    moderate = ensure_feature_schema(moderate)

    if report_path.exists():
        report = pd.read_csv(report_path)
        if not report.empty:
            print(
                "Loaded moderate behavioral sessions:",
                len(moderate),
                "target sessions in report:",
                target_sessions,
            )

    return moderate


def _load_behavioral_sequences() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build sequence tensors from legacy mouse-movement sources."""
    advanced_sessions_full = _load_advanced_sessions()
    advanced_sessions = advanced_sessions_full[["session_id", "label", "total_movements", "total_requests"]]
    moderate_sessions = _build_moderate_sessions(seeded_rng(RANDOM_SEED))[["session_id", "label"]]

    adv_behavioral = _load_segmented_advanced_behavioral(advanced_sessions_full)
    mod_behavioral, _ = _load_segmented_moderate_behavioral()

    adv_seq, adv_labels, adv_ids = build_mouse_sequences(
        adv_behavioral,
        advanced_sessions[["session_id", "label"]],
        sequence_length=SEQUENCE_LENGTH,
    )
    mod_seq, mod_labels, mod_ids = build_mouse_sequences(
        mod_behavioral,
        moderate_sessions,
        sequence_length=SEQUENCE_LENGTH,
    )

    sequences = np.concatenate([adv_seq, mod_seq], axis=0)
    labels = np.concatenate([adv_labels, mod_labels], axis=0)
    session_ids = adv_ids + mod_ids
    return sequences, labels, session_ids


def _balance_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts_before = df["label"].value_counts().to_dict()
    human_target = max(1000, min(int(counts_before.get(0, 0)), 1400))
    label_targets = {
        LABEL_MAPPING["human"]: human_target,
        LABEL_MAPPING["moderate_bot"]: max(200, int(counts_before.get(1, 0))),
        LABEL_MAPPING["advanced_bot"]: max(263, int(counts_before.get(2, 0))),
    }
    balanced = stratified_sample(df, label_targets, random_seed=RANDOM_SEED)
    counts_after = balanced["label"].value_counts().to_dict()
    return balanced, {
        "before": {str(k): int(v) for k, v in counts_before.items()},
        "after": {str(k): int(v) for k, v in counts_after.items()},
        "targets": {str(k): int(v) for k, v in label_targets.items()},
    }


def _filter_sequences_for_sessions(
    sequences: np.ndarray,
    labels: np.ndarray,
    session_ids: list[str],
    keep_session_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keep_indices = [idx for idx, session_id in enumerate(session_ids) if session_id in keep_session_ids]
    if not keep_indices:
        return (
            np.zeros((0, SEQUENCE_LENGTH, len(COMMON_SEQUENCE_FEATURES)), dtype=np.float32),
            np.array([], dtype=np.int64),
            [],
        )
    keep_indices = np.asarray(keep_indices, dtype=int)
    return sequences[keep_indices], labels[keep_indices], [session_ids[idx] for idx in keep_indices]


def main() -> None:
    rng = seeded_rng(RANDOM_SEED)
    LEGACY_OUTPUT_DIR.mkdir(exist_ok=True)

    advanced_sessions = _load_advanced_sessions()
    moderate_sessions = _build_moderate_sessions(rng)
    talkingdata = TalkingDataPreprocessor(TALKINGDATA_DIR, random_seed=RANDOM_SEED).run()

    combined = pd.concat(
        [talkingdata.sessions, moderate_sessions, advanced_sessions],
        ignore_index=True,
    )
    combined = ensure_feature_schema(combined)
    combined = combined.sort_values(["label", "session_start", "session_id"], na_position="last").reset_index(drop=True)
    balanced, balance_info = _balance_dataset(combined)

    legacy_seq, legacy_labels, legacy_ids = _load_behavioral_sequences()
    talking_seq = talkingdata.sequences
    talking_labels = talkingdata.labels
    talking_ids = talkingdata.session_ids

    all_sequences = np.concatenate([talking_seq, legacy_seq], axis=0)
    all_labels = np.concatenate([talking_labels, legacy_labels], axis=0)
    all_ids = talking_ids + legacy_ids
    all_sequences, scaler = standardize_sequences(all_sequences)

    kept_ids = set(balanced["session_id"].astype(str))
    all_sequences, all_labels, all_ids = _filter_sequences_for_sessions(
        all_sequences,
        all_labels,
        all_ids,
        kept_ids,
    )

    balanced.to_csv(COMBINED_DATASET_PATH, index=False)
    balanced.to_csv(LEGACY_OUTPUT_DIR / COMBINED_DATASET_PATH.name, index=False)
    save_sequence_payload(COMBINED_SEQUENCE_PATH, all_sequences, all_labels, all_ids, scaler)
    save_sequence_payload(LEGACY_OUTPUT_DIR / COMBINED_SEQUENCE_PATH.name, all_sequences, all_labels, all_ids, scaler)

    metadata = {
        "random_seed": RANDOM_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "combined_dataset": {
            "path": str(COMBINED_DATASET_PATH),
            "rows": int(len(balanced)),
            "columns": balanced.columns.tolist(),
            "label_distribution": {
                str(label): int(count)
                for label, count in balanced["label"].value_counts().sort_index().to_dict().items()
            },
            "bot_type_distribution": balanced["bot_type"].value_counts().to_dict(),
            "data_source_distribution": balanced["data_source"].value_counts().to_dict(),
            "balance": balance_info,
        },
        "sequence_dataset": {
            "path": str(COMBINED_SEQUENCE_PATH),
            "shape": list(all_sequences.shape),
            "label_distribution": {
                str(label): int(count)
                for label, count in pd.Series(all_labels).value_counts().sort_index().to_dict().items()
            },
            "feature_names": COMMON_SEQUENCE_FEATURES,
        },
        "talkingdata_sampling": talkingdata.sampling_report,
        "integrity_checks": {
            "human_sessions_at_least_1000": bool((balanced["label"] == 0).sum() >= 1000),
            "moderate_sessions_at_least_200": bool((balanced["label"] == 1).sum() >= 200),
            "advanced_sessions_at_least_263": bool((balanced["label"] == 2).sum() >= 263),
            "duplicate_session_ids": int(balanced["session_id"].duplicated().sum()),
            "missing_required_values": {
                column: int(balanced[column].isna().sum())
                for column in [
                    "session_id",
                    "label",
                    "bot_type",
                    "mouse_speed_mean",
                    "requests_per_minute",
                    "browser",
                    "ip_address",
                    "bot_likelihood_score",
                ]
            },
        },
    }

    save_metadata_report(METADATA_REPORT_PATH, metadata)
    save_metadata_report(LEGACY_OUTPUT_DIR / METADATA_REPORT_PATH.name, metadata)

    print("Unified preprocessing complete")
    print("combined_clickstream_dataset.csv:", len(balanced))
    print("combined_sequence_dataset.pkl:", all_sequences.shape)
    print("dataset_metadata_report.json:", metadata["combined_dataset"]["label_distribution"])


if __name__ == "__main__":
    main()
