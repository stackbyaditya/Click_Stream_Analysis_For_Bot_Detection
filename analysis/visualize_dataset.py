"""Final visual verification for the modelling-ready clickstream dataset."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "processed" / "final_clickstream_dataset_model_ready.csv"
OUTPUT_DIR = ROOT_DIR / "analysis_outputs_final"

LABEL_MAP = {0: "human", 1: "moderate_bot", 2: "advanced_bot"}
LABEL_ORDER = ["human", "moderate_bot", "advanced_bot"]
LABEL_COLORS = {
    "human": "#2e8b57",
    "moderate_bot": "#f4a261",
    "advanced_bot": "#d62828",
}
METADATA_COLUMNS = {"session_id", "ip_address", "user_agent", "data_source", "bot_type"}
EXCLUDED_CORR_COLUMNS = {"label", "session_id", "ip_address", "user_agent", "data_source", "bot_type", "label_name"}

BEHAVIORAL_FEATURES = [
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "mouse_acceleration_std",
    "movement_curvature",
    "session_idle_ratio",
    "total_movements",
]
TEMPORAL_FEATURES = [
    "click_interval_entropy",
    "click_interval_mean",
    "request_interval_std",
    "session_duration_sec",
    "burstiness",
]
ACTIVITY_FEATURES = [
    "click_count",
    "successful_requests",
    "install_count",
    "clicks_per_minute",
    "requests_per_minute",
]
DEVICE_FEATURES = ["browser", "operating_system", "device_type", "country", "region", "is_proxy"]
PAIRPLOT_FEATURES = [
    "mouse_speed_mean",
    "mouse_path_length",
    "coordinate_entropy",
    "click_interval_entropy",
    "clicks_per_minute",
    "session_duration_sec",
]


def configure_plot_style() -> None:
    """Set publication-style plotting defaults."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["font.family"] = "DejaVu Sans"


def ensure_output_dir() -> None:
    """Create the final analysis output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    """Load the final modelling dataset and print a concise summary."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df["label_name"] = df["label"].map(LABEL_MAP)
    summary = {
        "dataset_shape": list(df.shape),
        "column_names": df.columns.tolist(),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        "missing_values": {column: int(value) for column, value in df.isna().sum().to_dict().items()},
    }

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", ", ".join(df.columns))
    print("Class distribution:", summary["class_distribution"])
    print("Missing values:", sum(summary["missing_values"].values()))
    return df, summary


def _save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close()


def _valid_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def plot_class_distribution(df: pd.DataFrame) -> dict:
    """Create class count and proportion plots."""
    counts = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=LABEL_COLORS,
        dodge=False,
        legend=False,
    )
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Sessions")
    _save_current_figure("class_distribution.png")

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=[LABEL_COLORS[label] for label in counts.index],
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    plt.title("Class Proportions")
    _save_current_figure("class_pie.png")
    return counts.to_dict()


def analyze_behavioral_features(df: pd.DataFrame) -> None:
    """Plot behavioral feature distributions and boxplots."""
    features = _valid_columns(df, BEHAVIORAL_FEATURES)
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    for ax, feature in zip(axes.flat, features):
        for label_name in LABEL_ORDER:
            subset = df.loc[df["label_name"] == label_name, feature].dropna()
            sns.histplot(
                subset,
                bins=30,
                stat="density",
                kde=True,
                element="step",
                fill=False,
                ax=ax,
                color=LABEL_COLORS[label_name],
                label=label_name,
            )
        ax.set_title(feature)
        ax.legend()
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    _save_current_figure("behavioral_distributions.png")

    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    for ax, feature in zip(axes.flat, features):
        sns.boxplot(data=df, x="label_name", y=feature, order=LABEL_ORDER, palette=LABEL_COLORS, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("Label")
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    _save_current_figure("behavioral_boxplots.png")


def analyze_temporal_features(df: pd.DataFrame) -> None:
    """Plot temporal distributions and a duration-frequency scatter."""
    features = _valid_columns(df, TEMPORAL_FEATURES)
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    for ax, feature in zip(axes.flat[: len(features)], features):
        for label_name in LABEL_ORDER:
            sns.histplot(
                df.loc[df["label_name"] == label_name, feature].dropna(),
                bins=30,
                stat="density",
                kde=True,
                element="step",
                fill=False,
                ax=ax,
                color=LABEL_COLORS[label_name],
                label=label_name,
            )
        ax.set_title(f"{feature} by class")
        ax.legend()

    violin_ax = axes.flat[-2]
    violin_feature = "session_duration_sec" if "session_duration_sec" in df.columns else features[0]
    sns.violinplot(data=df, x="label_name", y=violin_feature, order=LABEL_ORDER, palette=LABEL_COLORS, ax=violin_ax, cut=0)
    violin_ax.set_title(f"{violin_feature} violin plot")

    scatter_ax = axes.flat[-1]
    sns.scatterplot(
        data=df.sample(min(len(df), 2500), random_state=RANDOM_SEED),
        x="session_duration_sec",
        y="clicks_per_minute",
        hue="label_name",
        palette=LABEL_COLORS,
        alpha=0.7,
        ax=scatter_ax,
    )
    scatter_ax.set_title("Session Duration vs Clicks per Minute")
    _save_current_figure("temporal_analysis.png")


def analyze_activity_features(df: pd.DataFrame) -> None:
    """Plot activity feature KDEs and boxplots."""
    features = _valid_columns(df, ACTIVITY_FEATURES)
    fig, axes = plt.subplots(len(features), 2, figsize=(16, 4 * len(features)))
    for row_idx, feature in enumerate(features):
        kde_ax = axes[row_idx, 0]
        box_ax = axes[row_idx, 1]
        for label_name in LABEL_ORDER:
            sns.kdeplot(
                df.loc[df["label_name"] == label_name, feature].dropna(),
                ax=kde_ax,
                color=LABEL_COLORS[label_name],
                label=label_name,
                fill=False,
                warn_singular=False,
            )
        kde_ax.set_title(f"{feature} KDE")
        kde_ax.legend()
        sns.boxplot(data=df, x="label_name", y=feature, order=LABEL_ORDER, palette=LABEL_COLORS, ax=box_ax)
        box_ax.set_title(f"{feature} boxplot")
    _save_current_figure("activity_patterns.png")


def analyze_device_features(df: pd.DataFrame) -> None:
    """Plot device and proxy patterns for the final schema."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    browser_order = df["browser"].fillna("unknown").value_counts().head(8).index
    sns.countplot(
        data=df,
        x="browser",
        hue="label_name",
        order=browser_order,
        palette=LABEL_COLORS,
        ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Browser Frequency by Label")

    proxy_table = pd.crosstab(df["label_name"], df["is_proxy"], normalize="index")
    proxy_table.plot(kind="bar", stacked=True, color=["#90be6d", "#d62828"], ax=axes[1])
    axes[1].set_title("Proxy Usage vs Label")
    axes[1].set_ylabel("Proportion")

    stacked = pd.crosstab(df["device_type"], df["label_name"])
    stacked.plot(kind="bar", stacked=True, color=[LABEL_COLORS[label] for label in LABEL_ORDER], ax=axes[2])
    axes[2].set_title("Device Type Composition by Label")
    axes[2].tick_params(axis="x", rotation=20)
    _save_current_figure("device_patterns.png")


def plot_correlation_matrix(df: pd.DataFrame) -> list[dict]:
    """Plot correlation heatmap using numeric modelling features only."""
    corr_df = df.drop(columns=[column for column in EXCLUDED_CORR_COLUMNS if column in df.columns], errors="ignore")
    numeric = corr_df.select_dtypes(include=[np.number]).copy()
    corr = numeric.corr(method="pearson")

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric Modelling Feature Correlation Heatmap")
    _save_current_figure("correlation_heatmap.png")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top_pairs = upper.stack().sort_values(key=np.abs, ascending=False).head(10)
    return [
        {"feature_a": pair[0], "feature_b": pair[1], "correlation": float(value)}
        for pair, value in top_pairs.items()
    ]


def plot_pairplot(df: pd.DataFrame) -> None:
    """Plot pairwise separability for the most useful final features."""
    features = _valid_columns(df, PAIRPLOT_FEATURES)
    sample_base = df[features + ["label_name"]].dropna()
    sample = sample_base.sample(min(len(sample_base), 1500), random_state=RANDOM_SEED)
    pair_grid = sns.pairplot(
        sample,
        vars=features,
        hue="label_name",
        hue_order=LABEL_ORDER,
        palette=LABEL_COLORS,
        corner=True,
        plot_kws={"alpha": 0.55, "s": 22},
        diag_kws={"fill": False},
    )
    pair_grid.fig.suptitle("Feature Separability Pairplot", y=1.02)
    pair_grid.savefig(OUTPUT_DIR / "feature_pairplot.png", bbox_inches="tight")
    plt.close("all")


def analyze_outliers(df: pd.DataFrame) -> dict:
    """Visualize the precomputed anomaly score distribution."""
    if "anomaly_score" not in df.columns:
        return {"outlier_count": 0, "threshold": None}

    threshold = float(df["anomaly_score"].quantile(0.97))
    outlier_count = int((df["anomaly_score"] >= threshold).sum())

    plt.figure(figsize=(10, 6))
    sns.histplot(df["anomaly_score"], bins=40, kde=True, color="#577590")
    plt.axvline(threshold, color="#d62828", linestyle="--", label="97th percentile")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly score")
    plt.legend()
    _save_current_figure("anomaly_scores.png")
    return {"outlier_count": outlier_count, "threshold": threshold}


def plot_feature_importance(df: pd.DataFrame) -> list[dict]:
    """Train a RandomForest preview model on modelling features only."""
    features = df.drop(columns=[column for column in METADATA_COLUMNS.union({"label", "label_name"}) if column in df.columns], errors="ignore")
    target = df["label"]

    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
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

    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(features, target)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance_values = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    top_idx = np.argsort(importance_values)[-20:][::-1]
    top_features = pd.DataFrame(
        {"feature": np.array(feature_names)[top_idx], "importance": importance_values[top_idx]}
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, y="feature", x="importance", color="#4d908e")
    plt.title("Top 20 RandomForest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    _save_current_figure("feature_importance.png")
    return top_features.to_dict(orient="records")


def generate_dataset_report(
    df: pd.DataFrame,
    dataset_summary: dict,
    top_correlations: list[dict],
    feature_importance_ranking: list[dict],
    outlier_info: dict,
) -> dict:
    """Save the final dataset verification report."""
    report = {
        "dataset_shape": dataset_summary["dataset_shape"],
        "class_distribution": dataset_summary["class_distribution"],
        "missing_values": dataset_summary["missing_values"],
        "top_correlations": top_correlations,
        "feature_importance_ranking": feature_importance_ranking[:20],
        "outlier_count": outlier_info["outlier_count"],
    }
    with (OUTPUT_DIR / "final_dataset_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def list_generated_files() -> list[str]:
    """List saved final analysis outputs."""
    return sorted(str(path.relative_to(ROOT_DIR)) for path in OUTPUT_DIR.glob("*"))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    configure_plot_style()
    ensure_output_dir()

    df, dataset_summary = load_dataset()
    plot_class_distribution(df)
    analyze_behavioral_features(df)
    analyze_temporal_features(df)
    analyze_activity_features(df)
    analyze_device_features(df)
    top_correlations = plot_correlation_matrix(df)
    plot_pairplot(df)
    outlier_info = analyze_outliers(df)
    feature_importance_ranking = plot_feature_importance(df)
    generate_dataset_report(df, dataset_summary, top_correlations, feature_importance_ranking, outlier_info)

    print("Visualization outputs saved to analysis_outputs_final/")
    for file_name in list_generated_files():
        print(file_name)


if __name__ == "__main__":
    main()
