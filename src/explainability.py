"""SHAP explainability pipeline for Supply Chain Stress Tester."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap


matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
SHAP_SAMPLE_SIZE: int = 2000


def load_artifact(path: Path) -> Any:
    """Load a serialized artifact from disk."""
    return joblib.load(path)


def configure_model_for_shap(model: Any) -> Any:
    """Reduce parallelism for sklearn tree models so SHAP runs reliably in this environment."""
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def require_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Raise an informative error if expected feature columns are missing."""
    missing_columns: list[str] = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required feature columns: {missing_columns}")


def extract_positive_class_shap(
    explainer: shap.TreeExplainer,
    shap_values: Any,
) -> tuple[np.ndarray, float]:
    """Normalize SHAP outputs so downstream code always receives positive-class values."""
    expected_value: Any = explainer.expected_value

    if isinstance(shap_values, list):
        base_value: float = float(expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value)
        return np.asarray(shap_values[1], dtype=float), base_value

    shap_array: np.ndarray = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 3:
        base_value = float(expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value)
        return shap_array[:, :, 1], base_value

    base_value = float(expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value)
    return shap_array, base_value


def compute_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[shap.TreeExplainer, np.ndarray]:
    """Build a tree explainer and compute SHAP values on the first 2000 test rows."""
    require_columns(X_test, feature_columns)
    print("[2/4] Computing SHAP values (2000 samples)...")

    X_sample: pd.DataFrame = X_test[feature_columns].head(SHAP_SAMPLE_SIZE).copy()
    # TreeExplainer is the efficient exact explainer for tree ensembles such as XGBoost and Random Forest.
    explainer: shap.TreeExplainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )
    raw_shap_values: Any = explainer.shap_values(X_sample, check_additivity=False)
    shap_array: np.ndarray
    shap_array, _ = extract_positive_class_shap(explainer, raw_shap_values)
    return explainer, shap_array


def plot_global_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """Save a global SHAP bar chart for the top 15 features."""
    print("[3/4] Plotting SHAP visualizations...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample[feature_columns],
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the SHAP beeswarm summary plot showing impact direction."""
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def explain_single_order(
    explainer: shap.TreeExplainer,
    order_features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Explain one order using SHAP contributions for the positive late-risk class."""
    require_columns(order_features, feature_columns)
    if len(order_features) != 1:
        raise ValueError("order_features must contain exactly one row")

    row_df: pd.DataFrame = order_features[feature_columns].copy()
    raw_single_shap: Any = explainer.shap_values(row_df, check_additivity=False)
    single_shap: np.ndarray
    base_value: float
    single_shap, base_value = extract_positive_class_shap(explainer, raw_single_shap)
    single_values: np.ndarray = np.asarray(single_shap[0], dtype=float)
    prediction: float = float(explainer.model.predict_proba(row_df)[:, 1][0])

    importance_order: np.ndarray = np.argsort(np.abs(single_values))[::-1]
    top_risk_factors: list[dict[str, Any]] = []
    top_3_reasons: list[str] = []

    for index in importance_order[:5]:
        feature_name: str = feature_columns[index]
        feature_value: float = float(row_df.iloc[0][feature_name])
        shap_value: float = float(single_values[index])
        impact: str = "increases risk" if shap_value >= 0 else "decreases risk"
        top_risk_factors.append(
            {
                "feature": feature_name,
                "shap_value": shap_value,
                "feature_value": feature_value,
                "impact": impact,
            }
        )

    for factor in top_risk_factors[:3]:
        direction_word: str = "High" if factor["feature_value"] >= 0 else "Low"
        top_3_reasons.append(
            f"{direction_word} {factor['feature']} {factor['impact']} by {abs(factor['shap_value']):.2f}"
        )

    return {
        "base_value": base_value,
        "prediction": prediction,
        "top_risk_factors": top_risk_factors,
        "top_3_reasons": top_3_reasons,
    }


def save_shap_summary(
    shap_values: np.ndarray,
    feature_columns: list[str],
    output_path: Path,
) -> dict[str, float]:
    """Save mean absolute SHAP importance by feature and print the top 10 drivers."""
    print("[4/4] Saving SHAP summary...")
    mean_abs_shap: np.ndarray = np.mean(np.abs(shap_values), axis=0)
    importance_map: dict[str, float] = {
        feature_name: float(importance_value)
        for feature_name, importance_value in zip(feature_columns, mean_abs_shap)
    }
    sorted_importance: dict[str, float] = dict(
        sorted(importance_map.items(), key=lambda item: item[1], reverse=True)
    )

    output_path.write_text(json.dumps(sorted_importance, indent=2), encoding="utf-8")

    print("Top 10 SHAP features:")
    for feature_name, importance_value in list(sorted_importance.items())[:10]:
        print(f"{feature_name}: {importance_value:.6f}")

    return sorted_importance


def main() -> None:
    """Run the full SHAP explainability workflow."""
    try:
        print("[1/4] Loading model and data...")
        model: Any = configure_model_for_shap(load_artifact(MODELS_DIR / "xgboost_model.pkl"))
        X_test: pd.DataFrame = load_artifact(MODELS_DIR / "X_test.pkl")
        feature_columns: list[str] = load_artifact(MODELS_DIR / "feature_columns.pkl")

        explainer: shap.TreeExplainer
        shap_values: np.ndarray
        explainer, shap_values = compute_shap_values(model, X_test, feature_columns)
        X_sample: pd.DataFrame = X_test[feature_columns].head(SHAP_SAMPLE_SIZE).copy()

        plot_global_importance(
            shap_values,
            X_sample,
            feature_columns,
            MODELS_DIR / "shap_global_importance.png",
        )
        plot_shap_summary(
            shap_values,
            X_sample,
            MODELS_DIR / "shap_summary.png",
        )

        explanation: dict[str, Any] = explain_single_order(
            explainer,
            X_test[feature_columns].head(1).copy(),
            feature_columns,
        )
        print("Single-order SHAP explanation:")
        print(json.dumps(explanation, indent=2))

        save_shap_summary(
            shap_values,
            feature_columns,
            MODELS_DIR / "shap_feature_importance.json",
        )
    except Exception as exc:
        print(f"SHAP pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
