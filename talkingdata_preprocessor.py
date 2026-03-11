"""TalkingData session builder for the unified clickstream fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_module import (
    LABEL_MAPPING,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    add_device_network_features,
    calculate_bot_likelihood_score,
    ensure_feature_schema,
    entropy,
    safe_mode,
    seeded_rng,
)


SESSION_GAP_MINUTES = 30


@dataclass
class TalkingDataArtifacts:
    events: pd.DataFrame
    sessions: pd.DataFrame
    sequences: np.ndarray
    labels: np.ndarray
    session_ids: list[str]
    sampling_report: dict


class TalkingDataPreprocessor:
    """Load, sessionize, and transform TalkingData clicks into human sessions."""

    def __init__(
        self,
        data_dir: str | Path,
        random_seed: int = RANDOM_SEED,
        sequence_length: int = SEQUENCE_LENGTH,
        target_human_sessions: int = 1000,
        train_chunk_size: int = 750_000,
        test_chunk_size: int = 500_000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.sequence_length = sequence_length
        self.target_human_sessions = target_human_sessions
        self.train_chunk_size = train_chunk_size
        self.test_chunk_size = test_chunk_size
        self.rng = seeded_rng(random_seed)

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test_supplement.csv"

    def _sample_train_events(self) -> tuple[pd.DataFrame, dict]:
        """Read enough train rows to build a stable pool of human sessions."""
        usecols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]
        positive_parts: list[pd.DataFrame] = []
        negative_parts: list[pd.DataFrame] = []
        positive_rows = 0
        negative_rows = 0
        chunks_read = 0
        negative_target = 140_000

        for chunk in pd.read_csv(
            self.train_path,
            usecols=usecols,
            parse_dates=["click_time"],
            chunksize=self.train_chunk_size,
        ):
            chunks_read += 1
            chunk["is_attributed"] = chunk["is_attributed"].fillna(0).astype(np.int8)

            positives = chunk[chunk["is_attributed"] == 1].copy()
            if not positives.empty:
                positive_parts.append(positives)
                positive_rows += len(positives)

            negatives = chunk[chunk["is_attributed"] == 0].copy()
            if negative_rows < negative_target and not negatives.empty:
                remaining = negative_target - negative_rows
                take = min(remaining, max(25_000, remaining))
                if len(negatives) > take:
                    negatives = negatives.sample(n=take, random_state=self.random_seed)
                negative_parts.append(negatives)
                negative_rows += len(negatives)

            if positive_rows >= 6_000 and negative_rows >= negative_target:
                break

        sampled = pd.concat(positive_parts + negative_parts, ignore_index=True)
        sampled = sampled.sort_values(["ip", "device", "os", "click_time"]).reset_index(drop=True)
        report = {
            "train_chunks_read": chunks_read,
            "sampled_train_rows": int(len(sampled)),
            "sampled_train_positive_rows": int(positive_rows),
            "sampled_train_negative_rows": int(negative_rows),
        }
        return sampled, report

    def _sample_test_background(self) -> tuple[pd.DataFrame, dict]:
        """Read a bounded slice of test traffic for threshold calibration."""
        if not self.test_path.exists():
            return pd.DataFrame(), {"sampled_test_rows": 0, "test_chunks_read": 0}

        usecols = ["ip", "app", "device", "os", "channel", "click_time"]
        sampled_parts: list[pd.DataFrame] = []
        chunks_read = 0
        rows_kept = 0
        target_rows = 120_000

        for chunk in pd.read_csv(
            self.test_path,
            usecols=usecols,
            parse_dates=["click_time"],
            chunksize=self.test_chunk_size,
        ):
            chunks_read += 1
            chunk["is_attributed"] = np.int8(0)
            remaining = target_rows - rows_kept
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.sample(n=remaining, random_state=self.random_seed)
            sampled_parts.append(chunk)
            rows_kept += len(chunk)
            if rows_kept >= target_rows:
                break

        sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=usecols + ["is_attributed"])
        sampled = sampled.sort_values(["ip", "device", "os", "click_time"]).reset_index(drop=True)
        return sampled, {
            "sampled_test_rows": int(len(sampled)),
            "test_chunks_read": chunks_read,
        }

    def _sessionize(self, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if events.empty:
            empty_sessions = pd.DataFrame(
                columns=[
                    "session_id",
                    "session_start",
                    "session_end",
                    "click_count",
                    "install_count",
                    "session_duration_sec",
                    "request_interval_mean",
                    "request_interval_std",
                    "clicks_per_minute",
                    "requests_per_minute",
                    "device_entropy",
                    "channel_entropy",
                    "app_entropy",
                    "original_ip",
                    "original_device",
                    "original_os",
                    "app",
                    "device",
                    "os",
                    "channel",
                ]
            )
            return events.copy(), empty_sessions

        sessionized = events.copy().sort_values(["ip", "device", "os", "click_time"]).reset_index(drop=True)
        key_columns = ["ip", "device", "os"]
        gaps = sessionized.groupby(key_columns)["click_time"].diff().dt.total_seconds()
        sessionized["session_gap_sec"] = gaps.fillna((SESSION_GAP_MINUTES * 60) + 1)
        sessionized["new_session"] = (sessionized["session_gap_sec"] > SESSION_GAP_MINUTES * 60).astype(int)
        sessionized["session_index"] = sessionized.groupby(key_columns)["new_session"].cumsum()
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
        sessionized["event_interval_sec"] = (
            sessionized.groupby("session_id")["click_time"].diff().dt.total_seconds().fillna(0.0)
        )

        grouped = sessionized.groupby("session_id", sort=False)
        sessions = grouped.agg(
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

        interval_stats = grouped["event_interval_sec"].agg(["mean", "std"]).reset_index()
        interval_stats.columns = ["session_id", "request_interval_mean", "request_interval_std"]
        sessions = sessions.merge(interval_stats, on="session_id", how="left")
        sessions["request_interval_mean"] = sessions["request_interval_mean"].fillna(0.0)
        sessions["request_interval_std"] = sessions["request_interval_std"].fillna(0.0)
        sessions["session_duration_sec"] = (
            (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().clip(lower=1.0)
        )
        sessions["clicks_per_minute"] = sessions["click_count"] / (sessions["session_duration_sec"] / 60.0)
        sessions["requests_per_minute"] = sessions["total_requests"] / (sessions["session_duration_sec"] / 60.0)
        sessions["device_entropy"] = grouped["device"].apply(entropy).to_numpy()
        sessions["channel_entropy"] = grouped["channel"].apply(entropy).to_numpy()
        sessions["app_entropy"] = grouped["app"].apply(entropy).to_numpy()
        sessions["activity_date"] = sessions["session_start"].dt.strftime("%Y-%m-%d")
        sessions["time_range"] = (
            sessions["session_start"].dt.strftime("%H:%M:%S")
            + "-"
            + sessions["session_end"].dt.strftime("%H:%M:%S")
        )
        sessions["logs_count"] = sessions["click_count"]
        sessions["success_rate"] = (
            sessions["successful_requests"] / sessions["total_requests"].clip(lower=1)
        )
        sessions["label"] = LABEL_MAPPING["human"]
        sessions["bot_type"] = "human"
        sessions["data_source"] = "talkingdata"
        sessions["total_movements"] = sessions["click_count"]
        sessions["total_response_size"] = np.nan
        sessions["avg_response_size"] = np.nan

        return sessionized, sessions

    def _derive_thresholds(self, background_sessions: pd.DataFrame) -> dict:
        """Build conservative thresholds from unlabeled traffic mix."""
        if background_sessions.empty:
            return {
                "rpm_high": 120.0,
                "interval_low": 0.8,
                "interval_std_low": 0.3,
                "channel_entropy_low": 0.2,
            }

        return {
            "rpm_high": float(background_sessions["requests_per_minute"].quantile(0.95)),
            "interval_low": float(background_sessions["request_interval_mean"].quantile(0.10)),
            "interval_std_low": float(background_sessions["request_interval_std"].quantile(0.10)),
            "channel_entropy_low": float(background_sessions["channel_entropy"].quantile(0.10)),
        }

    def _synthetic_behavior_from_intervals(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Map click interval statistics into movement-like features."""
        engineered = sessions.copy()
        interval_mean = engineered["request_interval_mean"].clip(lower=0.0)
        interval_std = engineered["request_interval_std"].clip(lower=0.0)
        variability = interval_std / (interval_mean + 1.0)
        interval_entropy = np.log1p(interval_std + 1.0)
        cadence = np.log1p(engineered["requests_per_minute"].clip(lower=0.0))
        click_mass = np.sqrt(engineered["click_count"].clip(lower=1.0))
        human_signal = (
            1.8 * interval_entropy
            + 1.2 * variability
            + 0.8 * engineered["success_rate"]
            - 0.9 * cadence
        )
        human_signal = (human_signal - human_signal.min()) / (human_signal.max() - human_signal.min() + 1e-6)

        engineered["mouse_speed_mean"] = np.clip(
            85 + 55 * human_signal + 8 * cadence - 6 * variability,
            20,
            None,
        )
        engineered["mouse_speed_std"] = np.clip(
            9 + 18 * variability + 7 * human_signal,
            1,
            None,
        )
        engineered["mouse_path_length"] = np.clip(
            engineered["mouse_speed_mean"] * click_mass * (1.2 + 0.6 * human_signal),
            25,
            None,
        )
        engineered["direction_change_count"] = np.clip(
            np.rint((engineered["click_count"] - 1) * (0.7 + 0.8 * human_signal + 0.3 * variability)),
            0,
            None,
        ).astype(int)
        engineered["coordinate_entropy"] = np.clip(
            1.1 + 2.4 * human_signal + 0.6 * interval_entropy,
            0.1,
            8.0,
        )
        engineered["movement_std"] = np.clip(
            0.35 * engineered["mouse_speed_std"] + 0.25 * engineered["coordinate_entropy"],
            0.2,
            None,
        )
        return engineered

    def _filter_humans(
        self,
        sessions: pd.DataFrame,
        thresholds: dict,
    ) -> pd.DataFrame:
        """Keep sessions with a strong human signal and discard suspicious traffic."""
        candidate = sessions.copy()
        suspicious = (
            (candidate["requests_per_minute"] >= thresholds["rpm_high"])
            | (candidate["request_interval_mean"] <= thresholds["interval_low"])
            | (candidate["request_interval_std"] <= thresholds["interval_std_low"])
            | (candidate["channel_entropy"] <= thresholds["channel_entropy_low"])
        )

        human_sessions = candidate[(candidate["install_count"] > 0) & ~suspicious].copy()
        if len(human_sessions) < self.target_human_sessions:
            deficit = self.target_human_sessions - len(human_sessions)
            low_risk = candidate[
                (candidate["install_count"] == 0)
                & ~suspicious
                & (candidate["click_count"] <= candidate["click_count"].quantile(0.60))
            ].copy()
            if not low_risk.empty:
                top_up = low_risk.sort_values(
                    ["requests_per_minute", "request_interval_std", "channel_entropy"],
                    ascending=[True, False, False],
                ).head(deficit)
                human_sessions = pd.concat([human_sessions, top_up], ignore_index=True)

        human_sessions["label"] = LABEL_MAPPING["human"]
        human_sessions["bot_type"] = "human"
        return human_sessions.sort_values("session_id").reset_index(drop=True)

    def _build_sequences(
        self,
        session_events: pd.DataFrame,
        sessions: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if session_events.empty or sessions.empty:
            return (
                np.zeros((0, self.sequence_length, 5), dtype=np.float32),
                np.array([], dtype=np.int64),
                [],
            )

        session_lookup = sessions.set_index("session_id")[["device_entropy", "channel_entropy", "label"]]
        sequences: list[np.ndarray] = []
        labels: list[int] = []
        session_ids: list[str] = []

        for session_id, frame in session_events.groupby("session_id", sort=False):
            if session_id not in session_lookup.index:
                continue
            ordered = frame.sort_values("click_time")
            deltas = ordered["click_time"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)
            click_frequency = 1.0 / np.clip(deltas + 1.0, 1.0, None)
            progress = np.linspace(0.0, 1.0, len(ordered), dtype=np.float32)
            device_entropy = np.full(len(ordered), float(session_lookup.loc[session_id, "device_entropy"]), dtype=np.float32)
            channel_entropy = np.full(len(ordered), float(session_lookup.loc[session_id, "channel_entropy"]), dtype=np.float32)
            seq = np.column_stack(
                [deltas, click_frequency, progress, device_entropy, channel_entropy]
            ).astype(np.float32)
            if len(seq) >= self.sequence_length:
                seq = seq[: self.sequence_length]
            else:
                seq = np.vstack(
                    [seq, np.zeros((self.sequence_length - len(seq), seq.shape[1]), dtype=np.float32)]
                )
            sequences.append(seq)
            labels.append(int(session_lookup.loc[session_id, "label"]))
            session_ids.append(str(session_id))

        if not sequences:
            return (
                np.zeros((0, self.sequence_length, 5), dtype=np.float32),
                np.array([], dtype=np.int64),
                [],
            )

        return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64), session_ids

    def run(self) -> TalkingDataArtifacts:
        train_events, train_report = self._sample_train_events()
        test_background, test_report = self._sample_test_background()

        all_train_events, train_sessions = self._sessionize(train_events)
        _, background_sessions = self._sessionize(
            pd.concat([train_events[train_events["is_attributed"] == 0], test_background], ignore_index=True)
        )
        thresholds = self._derive_thresholds(background_sessions)

        human_sessions = self._filter_humans(train_sessions, thresholds)
        human_sessions = self._synthetic_behavior_from_intervals(human_sessions)
        human_sessions = add_device_network_features(human_sessions, self.rng)
        human_sessions["bot_likelihood_score"] = human_sessions.apply(calculate_bot_likelihood_score, axis=1)
        human_sessions = ensure_feature_schema(human_sessions)

        human_event_ids = set(human_sessions["session_id"])
        human_events = all_train_events[all_train_events["session_id"].isin(human_event_ids)].copy()
        sequences, labels, session_ids = self._build_sequences(human_events, human_sessions)

        report = {
            **train_report,
            **test_report,
            "background_session_count": int(len(background_sessions)),
            "human_session_count": int(len(human_sessions)),
            "thresholds": {key: round(value, 4) for key, value in thresholds.items()},
        }

        return TalkingDataArtifacts(
            events=human_events,
            sessions=human_sessions,
            sequences=sequences,
            labels=labels,
            session_ids=session_ids,
            sampling_report=report,
        )
