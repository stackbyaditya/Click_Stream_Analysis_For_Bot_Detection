"""Boosting-oriented training pipeline for the balanced clickstream dataset."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data_balanced" / "balanced_clickstream_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "model_outputs_boosting"
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


def prepare_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["label"] + [column for column in DROP_COLS if column in df.columns], errors="ignore")
    y = df["label"].astype(int)
    return X, y


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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values())
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_names: np.ndarray, importance: np.ndarray, title: str, path: Path) -> None:
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


def plot_roc_curves(probabilities: dict[str, np.ndarray], y_true: np.ndarray, path: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(10, 8))
    for model_name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
        auc_score = roc_auc_score(y_true_bin, probs, multi_class="ovr", average="weighted")
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Boosting Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> None:
    configure_plot_style()
    ensure_output_dir()

    df = load_dataset()
    X, y = prepare_inputs(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_SEED,
            objective="multiclass",
            num_class=3,
            n_jobs=1,
            verbosity=-1,
        ),
    }

    summaries = {}
    roc_probabilities = {}
    model_paths = {
        "RandomForest": OUTPUT_DIR / "random_forest_boost.pkl",
        "XGBoost": OUTPUT_DIR / "xgboost_boost.pkl",
        "LightGBM": OUTPUT_DIR / "lightgbm_boost.pkl",
    }
    cm_paths = {
        "RandomForest": OUTPUT_DIR / "confusion_matrix_rf.png",
        "XGBoost": OUTPUT_DIR / "confusion_matrix_xgb.png",
        "LightGBM": OUTPUT_DIR / "confusion_matrix_lgbm.png",
    }

    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        roc_probabilities[model_name] = y_prob

        metrics = evaluate_predictions(y_test.to_numpy(), y_pred, y_prob)
        summaries[model_name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        }

        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), f"{model_name} Confusion Matrix", cm_paths[model_name])
        joblib.dump(pipeline, model_paths[model_name])

        if model_name in {"RandomForest", "XGBoost"}:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importance = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
            output_name = "feature_importance_rf.png" if model_name == "RandomForest" else "feature_importance_xgb.png"
            plot_feature_importance(feature_names, importance, f"{model_name} Feature Importance", OUTPUT_DIR / output_name)

    plot_roc_curves(roc_probabilities, y_test.to_numpy(), OUTPUT_DIR / "roc_curves_boosting.png")

    with (OUTPUT_DIR / "model_performance_boosting.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    print("Model Performance Summary")
    print("-------------------------")
    for model_name, metrics in summaries.items():
        print(model_name)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("")


if __name__ == "__main__":
    main()
