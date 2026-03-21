"""Supply chain disruption simulator for late-delivery risk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"

DISRUPTION_CONFIGS: dict[str, dict[str, dict[str, float]]] = {
    "COVID_PANDEMIC": {
        "mild": {"scheduled_days": 1.3, "quantity": 0.85, "sales": 0.9},
        "moderate": {"scheduled_days": 2.0, "quantity": 0.6, "sales": 0.7},
        "severe": {"scheduled_days": 3.0, "quantity": 0.4, "sales": 0.5},
    },
    "FLOOD_DISASTER": {
        "mild": {"scheduled_days": 1.5, "sales": 0.8, "order_total": 0.8},
        "moderate": {"scheduled_days": 2.5, "sales": 0.55, "order_total": 0.55},
        "severe": {"scheduled_days": 4.0, "sales": 0.3, "order_total": 0.3},
    },
    "PORT_STRIKE": {
        "mild": {"scheduled_days": 1.2, "discount_rate": 1.3, "profit_ratio": 0.85},
        "moderate": {"scheduled_days": 1.8, "discount_rate": 1.5, "profit_ratio": 0.7},
        "severe": {"scheduled_days": 2.5, "discount_rate": 2.0, "profit_ratio": 0.5},
    },
    "SUPPLIER_BANKRUPTCY": {
        "mild": {"order_total": 0.7, "profit": 0.6, "benefit": 0.7},
        "moderate": {"order_total": 0.45, "profit": 0.3, "benefit": 0.4},
        "severe": {"order_total": 0.2, "profit": 0.1, "benefit": 0.15},
    },
    "DEMAND_SURGE": {
        "mild": {"quantity": 1.3, "sales": 1.2, "scheduled_days": 1.15},
        "moderate": {"quantity": 1.8, "sales": 1.6, "scheduled_days": 1.4},
        "severe": {"quantity": 2.5, "sales": 2.0, "scheduled_days": 1.8},
    },
}

COLUMN_MULTIPLIER_MAP: dict[str, str] = {
    "scheduled_days": "Days for shipment (scheduled)",
    "quantity": "Order Item Quantity",
    "sales": "Sales",
    "order_total": "Order Item Total",
    "discount_rate": "Order Item Discount Rate",
    "profit_ratio": "Order Item Profit Ratio",
    "profit": "Order Profit Per Order",
    "benefit": "Benefit per order",
}

LABEL_ENCODERS: dict[str, Any] = {}
TARGET_ENCODINGS: dict[str, Any] = {}


def load_artifact(path: Path) -> Any:
    """Load a serialized project artifact."""
    return joblib.load(path)


def configure_model_for_inference(model: Any) -> Any:
    """Force safe local inference settings for models that expose worker counts."""
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def decode_column(values: pd.Series, column_name: str) -> pd.Series:
    """Decode label-encoded values for readable summaries when encoders are available."""
    encoder: Any | None = LABEL_ENCODERS.get(column_name)
    if encoder is None:
        return values

    def decode_value(value: Any) -> Any:
        if pd.isna(value):
            return value
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(encoder.classes_):
            return str(encoder.classes_[int(value)])
        return value

    return values.map(decode_value)


def resolve_affected_region_values(affected_regions: list[str], series: pd.Series) -> list[Any]:
    """Resolve region filters to the same representation stored in the dataframe."""
    if pd.api.types.is_numeric_dtype(series) and "Order Region" in LABEL_ENCODERS:
        encoder: Any = LABEL_ENCODERS["Order Region"]
        region_map: dict[str, int] = {str(label): int(idx) for idx, label in enumerate(encoder.classes_)}
        return [region_map[region] for region in affected_regions if region in region_map]
    return affected_regions


def recalculate_derived_features(df: pd.DataFrame, base_threshold: float) -> pd.DataFrame:
    """Recompute model features that depend on the disrupted numeric columns."""
    df = df.copy()

    # Discount value must track changes to both discount rate and order total.
    df["discount_impact"] = (
        df["Order Item Discount Rate"].astype(float) * df["Order Item Total"].astype(float)
    )

    # Profit margin is clipped to preserve the same bounded representation used during training.
    df["profit_margin"] = np.where(
        df["Order Item Total"].astype(float) == 0,
        0.0,
        df["Order Profit Per Order"].astype(float) / df["Order Item Total"].astype(float),
    )
    df["profit_margin"] = df["profit_margin"].clip(lower=-1, upper=1)

    # High-value orders should respond to total-value changes using the original portfolio threshold.
    df["high_value_order"] = (df["Order Item Total"].astype(float) > base_threshold).astype(int)

    # Interaction features need to be refreshed because the trained model consumes them directly.
    df["scheduled_days_x_shipping_mode"] = (
        df["Days for shipment (scheduled)"].astype(float) * df["Shipping Mode"].astype(float)
    )
    df["quantity_x_discount"] = (
        df["Order Item Quantity"].astype(float) * df["Order Item Discount Rate"].astype(float)
    )
    df["profit_x_quantity"] = (
        df["Order Item Profit Ratio"].astype(float) * df["Order Item Quantity"].astype(float)
    )
    df["total_x_scheduled"] = (
        df["Order Item Total"].astype(float) * df["Days for shipment (scheduled)"].astype(float)
    )

    # Composite risk score is rebuilt from the precomputed target-encoding features.
    df["composite_risk_score"] = (
        df["region_late_rate"].astype(float) * 0.3
        + df["department_late_rate"].astype(float) * 0.2
        + df["shipping_mode_late_rate"].astype(float) * 0.3
        + df["market_late_rate"].astype(float) * 0.2
    )
    return df


def simulate_disruption(
    df: pd.DataFrame,
    disruption_type: str,
    severity: str,
    affected_regions: list[str] | None = None,
) -> pd.DataFrame:
    """Apply a disruption scenario to a dataframe and return the modified copy."""
    print(f"Simulating disruption: type={disruption_type}, severity={severity}")

    if disruption_type not in DISRUPTION_CONFIGS:
        raise ValueError(f"Unsupported disruption_type: {disruption_type}")
    if severity not in DISRUPTION_CONFIGS[disruption_type]:
        raise ValueError(f"Unsupported severity '{severity}' for disruption '{disruption_type}'")

    disrupted_df: pd.DataFrame = df.copy(deep=True)
    scenario: dict[str, float] = DISRUPTION_CONFIGS[disruption_type][severity]
    base_threshold: float = float(df["Order Item Total"].astype(float).quantile(0.75))

    row_mask: pd.Series = pd.Series(True, index=disrupted_df.index)
    if affected_regions is not None:
        if "Order Region" not in disrupted_df.columns:
            raise KeyError("'Order Region' column is required for region filtering")
        region_values: list[Any] = resolve_affected_region_values(affected_regions, disrupted_df["Order Region"])
        if not region_values:
            raise ValueError(f"No matching regions found for filter: {affected_regions}")
        row_mask = disrupted_df["Order Region"].isin(region_values)
        print(f"Applying disruption to {int(row_mask.sum())} rows in regions: {affected_regions}")
    else:
        print(f"Applying disruption to all {len(disrupted_df)} rows")

    for multiplier_key, multiplier in scenario.items():
        column_name: str = COLUMN_MULTIPLIER_MAP[multiplier_key]
        if column_name not in disrupted_df.columns:
            raise KeyError(f"Required column '{column_name}' not found in dataframe")
        disrupted_df[column_name] = disrupted_df[column_name].astype(float)
        # Each multiplier expresses how the disruption changes an order-time business signal.
        disrupted_df.loc[row_mask, column_name] = (
            disrupted_df.loc[row_mask, column_name].astype(float) * multiplier
        )

    disrupted_df = recalculate_derived_features(disrupted_df, base_threshold)
    return disrupted_df


def calculate_risk_scores(
    original_df: pd.DataFrame,
    disrupted_df: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Score baseline and disrupted orders and return a comparison dataframe."""
    print("Calculating baseline and disrupted risk scores...")

    missing_original: list[str] = [column for column in feature_columns if column not in original_df.columns]
    missing_disrupted: list[str] = [column for column in feature_columns if column not in disrupted_df.columns]
    if missing_original:
        raise KeyError(f"Original dataframe missing features: {missing_original}")
    if missing_disrupted:
        raise KeyError(f"Disrupted dataframe missing features: {missing_disrupted}")

    original_scores: np.ndarray = model.predict_proba(original_df[feature_columns])[:, 1]
    disrupted_scores: np.ndarray = model.predict_proba(disrupted_df[feature_columns])[:, 1]

    results_df: pd.DataFrame = pd.DataFrame(
        {
            "original_risk_score": original_scores.astype(float),
            "disrupted_risk_score": disrupted_scores.astype(float),
            "risk_increase": (disrupted_scores - original_scores).astype(float),
            "revenue_at_risk": (
                original_df["Order Item Total"].astype(float) * disrupted_scores.astype(float)
            ),
            "Order Region": decode_column(original_df["Order Region"], "Order Region"),
            "Department Name": decode_column(original_df["Department Name"], "Department Name"),
            "Order Item Total": original_df["Order Item Total"].astype(float),
        },
        index=original_df.index,
    )

    results_df["status"] = np.select(
        [
            results_df["disrupted_risk_score"] < 0.35,
            results_df["disrupted_risk_score"] < 0.65,
        ],
        ["STABLE", "AT RISK"],
        default="CRITICAL",
    )
    return results_df


def get_disruption_summary(
    results_df: pd.DataFrame,
    disruption_type: str,
    severity: str,
) -> dict[str, Any]:
    """Aggregate disruption outcomes into a compact summary payload."""
    print(f"Summarising disruption results for {disruption_type} / {severity}...")

    total_orders: int = int(len(results_df))
    stable_count: int = int((results_df["status"] == "STABLE").sum())
    at_risk_count: int = int((results_df["status"] == "AT RISK").sum())
    critical_count: int = int((results_df["status"] == "CRITICAL").sum())

    if total_orders == 0:
        raise ValueError("results_df is empty; cannot build summary")

    region_risk: pd.Series = results_df.groupby("Order Region")["risk_increase"].mean()
    department_risk: pd.Series = results_df.groupby("Department Name")["risk_increase"].mean()

    return {
        "disruption_type": disruption_type,
        "severity": severity,
        "total_orders": total_orders,
        "stable_count": stable_count,
        "at_risk_count": at_risk_count,
        "critical_count": critical_count,
        "stable_pct": float(stable_count / total_orders),
        "at_risk_pct": float(at_risk_count / total_orders),
        "critical_pct": float(critical_count / total_orders),
        "total_revenue_at_risk": float(results_df["revenue_at_risk"].sum()),
        "avg_risk_increase": float(results_df["risk_increase"].mean()),
        "most_affected_region": str(region_risk.idxmax()),
        "most_affected_department": str(department_risk.idxmax()),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a summary dictionary in a readable format."""
    print("\n" + "=" * 60)
    print(f"{summary['disruption_type']} ({summary['severity']})")
    print("=" * 60)
    for key, value in summary.items():
        if key in {"disruption_type", "severity"}:
            continue
        print(f"{key}: {value}")


def main() -> None:
    """Load artifacts and run the required simulator test cases."""
    global LABEL_ENCODERS, TARGET_ENCODINGS

    try:
        print("Loading simulator artifacts...")
        processed_df: pd.DataFrame = load_artifact(MODELS_DIR / "processed_df.pkl")
        model: Any = configure_model_for_inference(load_artifact(MODELS_DIR / "xgboost_model.pkl"))
        feature_columns: list[str] = load_artifact(MODELS_DIR / "feature_columns.pkl")
        LABEL_ENCODERS = load_artifact(MODELS_DIR / "label_encoders.pkl")
        TARGET_ENCODINGS = load_artifact(MODELS_DIR / "target_encodings.pkl")
        print(f"Loaded dataframe with shape: {processed_df.shape}")
        print(f"Loaded {len(feature_columns)} feature columns")

        covid_df: pd.DataFrame = simulate_disruption(
            processed_df,
            disruption_type="COVID_PANDEMIC",
            severity="moderate",
        )
        covid_results: pd.DataFrame = calculate_risk_scores(
            processed_df,
            covid_df,
            model,
            feature_columns,
        )
        covid_summary: dict[str, Any] = get_disruption_summary(
            covid_results,
            disruption_type="COVID_PANDEMIC",
            severity="moderate",
        )
        print_summary(covid_summary)

        flood_df: pd.DataFrame = simulate_disruption(
            processed_df,
            disruption_type="FLOOD_DISASTER",
            severity="severe",
            affected_regions=["Western Europe", "Central America"],
        )
        flood_results: pd.DataFrame = calculate_risk_scores(
            processed_df,
            flood_df,
            model,
            feature_columns,
        )
        flood_summary: dict[str, Any] = get_disruption_summary(
            flood_results,
            disruption_type="FLOOD_DISASTER",
            severity="severe",
        )
        print_summary(flood_summary)
    except Exception as exc:
        print(f"Simulator run failed: {exc}")
        raise


if __name__ == "__main__":
    main()
