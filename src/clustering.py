"""KMeans clustering pipeline for supply chain order risk tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"

CLUSTER_FEATURES: list[str] = [
    "Days for shipment (scheduled)",
    "Order Item Total",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Sales per customer",
    "Benefit per order",
    "Late_delivery_risk",
    "profit_margin",
    "composite_risk_score",
]

LABEL_ENCODERS: dict[str, Any] = {}


def load_artifact(path: Path) -> Any:
    """Load a serialized artifact from disk."""
    return joblib.load(path)


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise an informative error when required columns are missing."""
    missing_columns: list[str] = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def decode_value(column_name: str, value: Any) -> str:
    """Decode encoded categorical values to readable labels when possible."""
    encoder: Any | None = LABEL_ENCODERS.get(column_name)
    if encoder is None:
        return str(value)
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(encoder.classes_):
        return str(encoder.classes_[int(value)])
    return str(value)


def prepare_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the selected clustering features and save the scaler."""
    require_columns(df, CLUSTER_FEATURES)
    print("[2/5] Preparing clustering features...")

    feature_df: pd.DataFrame = df[CLUSTER_FEATURES].copy()
    scaler: StandardScaler = StandardScaler()
    # Standard scaling prevents high-value monetary columns from dominating the cluster geometry.
    scaled_values: np.ndarray = scaler.fit_transform(feature_df)
    scaled_df: pd.DataFrame = pd.DataFrame(scaled_values, columns=CLUSTER_FEATURES, index=df.index)

    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.pkl")
    return scaled_df


def find_optimal_k(scaled_df: pd.DataFrame, max_k: int = 8) -> int:
    """Compute elbow inertias for k=2..max_k, save the elbow chart, and return the default k."""
    print("[3/5] Finding optimal k...")
    inertias: list[float] = []
    k_values: list[int] = list(range(2, max_k + 1))

    for k in k_values:
        # Multiple initialisations reduce sensitivity to unlucky centroid starts.
        model: KMeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_df)
        inertias.append(float(model.inertia_))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method for KMeans")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "elbow_curve.png", dpi=300)
    plt.close()

    return 4


def train_kmeans(scaled_df: pd.DataFrame, n_clusters: int = 4) -> KMeans:
    """Train and save a KMeans clustering model."""
    print("[4/5] Training KMeans...")
    # Four clusters are used so risk tiers can map cleanly from low to critical.
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_df)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    return kmeans


def label_risk_tiers(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Assign human-readable risk tiers by ordering clusters on mean late-delivery rate."""
    require_columns(
        df,
        [
            "Late_delivery_risk",
            "Order Item Total",
            "Days for shipment (scheduled)",
            "composite_risk_score",
        ],
    )
    print("[5/5] Labeling risk tiers...")

    labeled_df: pd.DataFrame = df.copy()
    labeled_df["cluster"] = cluster_labels.astype(int)

    cluster_stats: pd.DataFrame = (
        labeled_df.groupby("cluster")
        .agg(
            mean_late_delivery_risk=("Late_delivery_risk", "mean"),
            mean_order_item_total=("Order Item Total", "mean"),
            mean_scheduled_days=("Days for shipment (scheduled)", "mean"),
            mean_composite_risk_score=("composite_risk_score", "mean"),
            order_count=("Late_delivery_risk", "size"),
        )
        .sort_values("mean_late_delivery_risk")
    )

    ordered_clusters: list[int] = cluster_stats.index.tolist()
    risk_labels: list[str] = ["LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK"]
    cluster_to_tier: dict[int, str] = {
        cluster_id: risk_labels[idx]
        for idx, cluster_id in enumerate(ordered_clusters)
    }
    labeled_df["risk_tier"] = labeled_df["cluster"].map(cluster_to_tier)

    profile_payload: dict[str, Any] = {
        str(cluster_id): {
            "risk_tier": cluster_to_tier[cluster_id],
            "mean_late_delivery_risk": float(cluster_stats.loc[cluster_id, "mean_late_delivery_risk"]),
            "mean_order_item_total": float(cluster_stats.loc[cluster_id, "mean_order_item_total"]),
            "mean_scheduled_days": float(cluster_stats.loc[cluster_id, "mean_scheduled_days"]),
            "mean_composite_risk_score": float(cluster_stats.loc[cluster_id, "mean_composite_risk_score"]),
            "order_count": int(cluster_stats.loc[cluster_id, "order_count"]),
        }
        for cluster_id in ordered_clusters
    }
    (MODELS_DIR / "cluster_profiles.json").write_text(
        json.dumps(profile_payload, indent=2),
        encoding="utf-8",
    )

    return labeled_df


def plot_clusters(df: pd.DataFrame, output_path: Path) -> None:
    """Reduce the clustering feature space to 2D with PCA and save a colored scatter plot."""
    require_columns(df, CLUSTER_FEATURES + ["risk_tier"])
    print("Plotting PCA cluster visualization...")

    pca: PCA = PCA(n_components=2, random_state=42)
    reduced: np.ndarray = pca.fit_transform(df[CLUSTER_FEATURES])
    plot_df: pd.DataFrame = pd.DataFrame(reduced, columns=["PC1", "PC2"], index=df.index)
    plot_df["risk_tier"] = df["risk_tier"].astype(str)

    color_map: dict[str, str] = {
        "LOW RISK": "#2ca02c",
        "MEDIUM RISK": "#ffbf00",
        "HIGH RISK": "#ff7f0e",
        "CRITICAL RISK": "#d62728",
    }

    plt.figure(figsize=(10, 6))
    for tier, tier_df in plot_df.groupby("risk_tier"):
        plt.scatter(
            tier_df["PC1"],
            tier_df["PC2"],
            s=12,
            alpha=0.5,
            label=tier,
            c=color_map.get(tier, "#1f77b4"),
        )

    plt.title("KMeans Risk Clusters (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_cluster_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a tier-based summary of the clustered order population."""
    require_columns(
        df,
        [
            "risk_tier",
            "Late_delivery_risk",
            "Order Item Total",
            "Days for shipment (scheduled)",
            "Order Region",
            "Department Name",
        ],
    )

    total_orders: int = int(len(df))
    tier_order: list[str] = ["LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL RISK"]
    clusters: list[dict[str, Any]] = []

    for tier in tier_order:
        tier_df: pd.DataFrame = df[df["risk_tier"] == tier]
        if tier_df.empty:
            continue

        top_region_value: Any = tier_df["Order Region"].mode().iloc[0]
        top_department_value: Any = tier_df["Department Name"].mode().iloc[0]

        clusters.append(
            {
                "tier": tier,
                "order_count": int(len(tier_df)),
                "pct_of_total": float(len(tier_df) / total_orders),
                "avg_late_risk": float(tier_df["Late_delivery_risk"].mean()),
                "avg_order_value": float(tier_df["Order Item Total"].mean()),
                "avg_scheduled_days": float(tier_df["Days for shipment (scheduled)"].mean()),
                "top_region": decode_value("Order Region", top_region_value),
                "top_department": decode_value("Department Name", top_department_value),
            }
        )

    return {"total_orders": total_orders, "clusters": clusters}


def main() -> None:
    """Run the full KMeans risk clustering pipeline."""
    global LABEL_ENCODERS

    print("[1/5] Loading data...")
    processed_df: pd.DataFrame = load_artifact(MODELS_DIR / "processed_df.pkl")
    label_encoders_path: Path = MODELS_DIR / "label_encoders.pkl"
    if label_encoders_path.exists():
        LABEL_ENCODERS = load_artifact(label_encoders_path)

    scaled_df: pd.DataFrame = prepare_clustering_features(processed_df)
    optimal_k: int = find_optimal_k(scaled_df, max_k=8)
    kmeans: KMeans = train_kmeans(scaled_df, n_clusters=optimal_k)
    labeled_df: pd.DataFrame = label_risk_tiers(processed_df, kmeans.labels_)

    scaled_labeled_df: pd.DataFrame = scaled_df.copy()
    scaled_labeled_df["risk_tier"] = labeled_df["risk_tier"]
    plot_clusters(scaled_labeled_df, MODELS_DIR / "cluster_plot.png")

    summary: dict[str, Any] = get_cluster_summary(labeled_df)
    print("Cluster summary:")
    print(json.dumps(summary, indent=2))

    print("Risk tier distribution:")
    print(labeled_df["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
