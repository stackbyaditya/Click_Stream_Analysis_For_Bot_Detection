"""Complete modelling pipeline for clickstream bot detection."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.lstm_model import SEQUENCE_FEATURES, build_session_sequences, predict_lstm_probabilities, train_lstm_model

DATASET_PATH = ROOT_DIR / "data" / "processed" / "final_clickstream_dataset_model_ready.csv"
OUTPUT_DIR = ROOT_DIR / "model_outputs"
LABEL_MAP = {0: "human", 1: "moderate_bot", 2: "advanced_bot"}
DROP_COLS = ["session_id", "ip_address", "user_agent", "data_source", "bot_type", "label_name"]


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    warnings.filterwarnings("ignore", message="X does not have valid feature names")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print("dataset shape:", df.shape)
    print("class distribution:", df["label"].value_counts().sort_index().to_dict())
    return df


def prepare_modelling_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    working = df.copy()
    working["label_name"] = working["label"].map(LABEL_MAP)
    X = working.drop(columns=["label"] + [column for column in DROP_COLS if column in working.columns], errors="ignore")
    y = working["label"].astype(int)
    session_ids = working["session_id"] if "session_id" in working.columns else pd.Series(np.arange(len(working)))
    return X, y, session_ids


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def compute_weights(y_train: pd.Series) -> tuple[dict[int, float], np.ndarray]:
    classes = np.sort(y_train.unique())
    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, class_weight_values)}
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    return class_weight, sample_weights


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values())
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    confusion_path: Path,
) -> dict:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            output_dict=True,
            zero_division=0,
        ),
    }
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, f"{name} Confusion Matrix", confusion_path)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_model(model, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def plot_feature_importance(feature_names: np.ndarray, importance: np.ndarray, title: str, path: Path) -> list[dict]:
    top_idx = np.argsort(importance)[-20:][::-1]
    top_features = pd.DataFrame({"feature": feature_names[top_idx], "importance": importance[top_idx]})
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, y="feature", x="importance", color="#2a9d8f")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return top_features.to_dict(orient="records")


def plot_roc_curves(curves: dict[str, np.ndarray], y_true: np.ndarray, path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for model_name, probabilities in curves.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probabilities.ravel())
        auc_score = roc_auc_score(y_true_bin, probabilities, multi_class="ovr", average="weighted")
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> None:
    configure_plot_style()
    ensure_output_dir()

    df = load_dataset()
    X, y, session_ids = prepare_modelling_data(df)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    session_train = session_ids.iloc[train_idx].reset_index(drop=True)
    session_test = session_ids.iloc[test_idx].reset_index(drop=True)

    class_weight, sample_weights = compute_weights(y_train)
    preprocessor = build_preprocessor(X_train)

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_prob = rf_pipeline.predict_proba(X_test)
    rf_metrics = evaluate_model("RandomForest", y_test, rf_pred, rf_prob, OUTPUT_DIR / "confusion_matrix_rf.png")
    save_model(rf_pipeline, OUTPUT_DIR / "random_forest.pkl")

    xgb_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=3,
                    random_state=RANDOM_SEED,
                    tree_method="hist",
                ),
            ),
        ]
    )
    xgb_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_prob = xgb_pipeline.predict_proba(X_test)
    xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob, OUTPUT_DIR / "confusion_matrix_xgb.png")
    save_model(xgb_pipeline, OUTPUT_DIR / "xgboost_model.pkl")

    lgbm_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                    objective="multiclass",
                    num_class=3,
                    verbosity=-1,
                ),
            ),
        ]
    )
    lgbm_pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    lgbm_pred = lgbm_pipeline.predict(X_test)
    lgbm_prob = lgbm_pipeline.predict_proba(X_test)
    lgbm_metrics = evaluate_model("LightGBM", y_test, lgbm_pred, lgbm_prob, OUTPUT_DIR / "confusion_matrix_lgbm.png")
    save_model(lgbm_pipeline, OUTPUT_DIR / "lightgbm_model.pkl")

    feature_names_rf = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    rf_importance = rf_pipeline.named_steps["model"].feature_importances_
    rf_top_features = plot_feature_importance(
        feature_names_rf,
        np.asarray(rf_importance, dtype=float),
        "RandomForest Feature Importance",
        OUTPUT_DIR / "feature_importance_rf.png",
    )

    feature_names_xgb = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
    xgb_importance = xgb_pipeline.named_steps["model"].feature_importances_
    xgb_top_features = plot_feature_importance(
        feature_names_xgb,
        np.asarray(xgb_importance, dtype=float),
        "XGBoost Feature Importance",
        OUTPUT_DIR / "feature_importance_xgb.png",
    )

    X_train_seq = build_session_sequences(X_train, session_train, feature_columns=SEQUENCE_FEATURES, sequence_length=10)
    X_test_seq = build_session_sequences(X_test, session_test, feature_columns=SEQUENCE_FEATURES, sequence_length=10)
    lstm_model, _ = train_lstm_model(
        X_train_seq,
        y_train.to_numpy(),
        class_weight,
        OUTPUT_DIR / "lstm_bot_detector.h5",
        epochs=20,
        batch_size=64,
    )
    lstm_prob = predict_lstm_probabilities(lstm_model, X_test_seq)
    lstm_pred = np.argmax(lstm_prob, axis=1)
    lstm_metrics = evaluate_model("LSTM", y_test, lstm_pred, lstm_prob, OUTPUT_DIR / "confusion_matrix_lstm.png")

    ensemble_prob = 0.7 * xgb_prob + 0.3 * lstm_prob
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    ensemble_metrics = evaluate_model("Ensemble", y_test, ensemble_pred, ensemble_prob, OUTPUT_DIR / "confusion_matrix_ensemble.png")

    plot_roc_curves(
        {
            "RandomForest": rf_prob,
            "XGBoost": xgb_prob,
            "LightGBM": lgbm_prob,
            "Ensemble": ensemble_prob,
        },
        y_test.to_numpy(),
        OUTPUT_DIR / "roc_curves.png",
    )

    performance_summary = {
        "RandomForest": rf_metrics,
        "XGBoost": xgb_metrics,
        "LightGBM": lgbm_metrics,
        "LSTM": lstm_metrics,
        "Ensemble": ensemble_metrics,
        "feature_importance": {
            "RandomForest": rf_top_features,
            "XGBoost": xgb_top_features,
        },
    }
    best_model_name = max(
        performance_summary,
        key=lambda model_name: performance_summary[model_name]["f1"] if "f1" in performance_summary[model_name] else -1,
    )
    summary_payload = {
        "accuracy_per_model": {name: metrics["accuracy"] for name, metrics in performance_summary.items() if "accuracy" in metrics},
        "precision_per_model": {name: metrics["precision"] for name, metrics in performance_summary.items() if "precision" in metrics},
        "recall_per_model": {name: metrics["recall"] for name, metrics in performance_summary.items() if "recall" in metrics},
        "f1_per_model": {name: metrics["f1"] for name, metrics in performance_summary.items() if "f1" in metrics},
        "roc_auc_per_model": {name: metrics["roc_auc"] for name, metrics in performance_summary.items() if "roc_auc" in metrics},
        "best_model": best_model_name,
        "full_metrics": performance_summary,
    }
    with (OUTPUT_DIR / "model_performance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print("Training completed")
    print("Best model:", best_model_name)
    print("Accuracy:", summary_payload["accuracy_per_model"][best_model_name])
    print("F1 score:", summary_payload["f1_per_model"][best_model_name])


if __name__ == "__main__":
    main()
