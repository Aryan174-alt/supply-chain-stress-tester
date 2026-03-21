"""Train and compare late-delivery risk models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
TOTAL_ROWS: int = 180519
POSITIVE_CLASS_WEIGHT: float = 81542 / 98977
RANDOM_STATE: int = 42


def load_artifact(path: Path) -> Any:
    """Load a serialized artifact from disk."""
    return joblib.load(path)


def load_data(models_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the preprocessed train/test splits and feature names."""
    x_train: pd.DataFrame = load_artifact(models_dir / "X_train.pkl")
    x_test: pd.DataFrame = load_artifact(models_dir / "X_test.pkl")
    y_train: pd.Series = load_artifact(models_dir / "y_train.pkl")
    y_test: pd.Series = load_artifact(models_dir / "y_test.pkl")
    feature_columns: list[str] = load_artifact(models_dir / "feature_columns.pkl")
    return x_train, x_test, y_train, y_test, feature_columns


def print_shapes(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_columns: list[str],
) -> None:
    """Print input shapes so the saved preprocessing outputs are easy to verify."""
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"feature_columns count: {len(feature_columns)}")


def build_models() -> dict[str, Any]:
    """Create both candidate classifiers with the requested settings."""
    # XGBoost handles nonlinear interactions well and the class weight corrects the mild label imbalance.
    xgb_model: XGBClassifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=POSITIVE_CLASS_WEIGHT,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    # Random Forest provides a strong tree-based baseline with balanced class weighting.
    rf_model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {"XGBoost": xgb_model, "Random Forest": rf_model}


def train_models(
    models: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Fit each candidate model on the training split."""
    trained_models: dict[str, Any] = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        trained_models[model_name] = model
    return trained_models


def evaluate_model(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute and print the standard binary classification metrics for a model."""
    y_pred: np.ndarray = model.predict(x_test)
    y_score: np.ndarray = model.predict_proba(x_test)[:, 1]

    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred)
    recall: float = recall_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_score)
    matrix: np.ndarray = confusion_matrix(y_test, y_pred)
    report: str = classification_report(y_test, y_pred)
    fpr: np.ndarray
    tpr: np.ndarray
    fpr, tpr, _ = roc_curve(y_test, y_score)

    print(f"\n{model_name} Classification Report:")
    print(report)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1: {f1:.4f}")
    print(f"{model_name} Confusion Matrix:\n{matrix}")

    return {
        "name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc_curves(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save a combined ROC chart so model discrimination can be compared visually."""
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(
            result["fpr"],
            result["tpr"],
            label=f'{result["name"]} (AUC = {result["roc_auc"]:.4f})',
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_columns: list[str],
    output_path: Path,
    model_name: str,
) -> None:
    """Save the top 15 feature importances for the selected model."""
    # Tree-based importances are a simple, model-native way to summarize which inputs drive predictions most.
    importances: np.ndarray = np.asarray(model.feature_importances_, dtype=float)
    importance_frame: pd.DataFrame = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False)
    top_features: pd.DataFrame = importance_frame.head(15).sort_values("importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"], color="#1f77b4")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importances ({model_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_results(best_result: dict[str, Any], output_path: Path, feature_count: int) -> None:
    """Persist the winning model metrics in JSON format."""
    serializable_results: dict[str, Any] = {
        "best_model": best_result["name"],
        "accuracy": float(best_result["accuracy"]),
        "precision": float(best_result["precision"]),
        "recall": float(best_result["recall"]),
        "f1": float(best_result["f1"]),
        "roc_auc": float(best_result["roc_auc"]),
        "total_rows": TOTAL_ROWS,
        "features_used": feature_count,
    }
    output_path.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")


def plot_confusion_matrix_for_best(best_result: dict[str, Any], output_path: Path) -> None:
    """Optionally save the winning confusion matrix as a quick visual artifact."""
    display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix=best_result["confusion_matrix"],
        display_labels=[0, 1],
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.title(f'Confusion Matrix ({best_result["name"]})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """Run the full training, comparison, and persistence pipeline."""
    print("[1/4] Loading preprocessed data...")
    x_train, x_test, y_train, y_test, feature_columns = load_data(MODELS_DIR)
    print_shapes(x_train, x_test, y_train, y_test, feature_columns)

    print("[2/4] Training models...")
    models: dict[str, Any] = build_models()
    trained_models: dict[str, Any] = train_models(models, x_train, y_train)

    print("[3/4] Evaluating and plotting...")
    evaluation_results: list[dict[str, Any]] = []
    for model_name, model in trained_models.items():
        evaluation_results.append(evaluate_model(model_name, model, x_test, y_test))

    plot_roc_curves(evaluation_results, MODELS_DIR / "roc_curve.png")

    best_result: dict[str, Any] = max(evaluation_results, key=lambda result: result["f1"])
    best_model: Any = trained_models[best_result["name"]]
    plot_feature_importance(
        best_model,
        feature_columns,
        MODELS_DIR / "feature_importance.png",
        best_result["name"],
    )
    plot_confusion_matrix_for_best(best_result, MODELS_DIR / "confusion_matrix.png")

    print("[4/4] Saving best model...")
    # The output filename stays fixed so downstream code can load a single canonical model artifact.
    joblib.dump(best_model, MODELS_DIR / "xgboost_model.pkl")
    save_results(best_result, MODELS_DIR / "training_results.json", len(feature_columns))

    print(f"Best model by F1: {best_result['name']} ({best_result['f1']:.4f})")


if __name__ == "__main__":
    main()
