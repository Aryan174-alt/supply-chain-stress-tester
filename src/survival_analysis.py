"""Survival analysis pipeline for supply chain order failure prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
LABEL_ENCODERS_PATH: Path = MODELS_DIR / "label_encoders.pkl"

DURATION_COLUMN: str = "Days for shipment (scheduled)"
EVENT_COLUMN: str = "Late_delivery_risk"
COX_COVARIATES: list[str] = [
    "Order Item Total",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Sales per customer",
    "Benefit per order",
    "order_month",
    "order_dayofweek",
    "high_value_order",
]

LABEL_ENCODERS: dict[str, Any] = {}


def load_artifact(path: Path) -> Any:
    """Load a serialized artifact from disk."""
    return joblib.load(path)


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear error when a required dataframe column is missing."""
    missing_columns: list[str] = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def decode_group_value(column_name: str, value: Any) -> str:
    """Decode label-encoded group values when encoders are available."""
    encoder: Any | None = LABEL_ENCODERS.get(column_name)
    if encoder is None:
        return str(value)
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(encoder.classes_):
        return str(encoder.classes_[int(value)])
    return str(value)


def plot_kaplan_meier(df: pd.DataFrame, group_col: str, output_path: Path) -> None:
    """Plot Kaplan-Meier survival curves for on-time delivery grouped by one categorical field."""
    require_columns(df, [DURATION_COLUMN, EVENT_COLUMN, group_col])
    print(f"Plotting Kaplan-Meier curves for {group_col}...")

    plt.figure(figsize=(10, 6))
    kmf: KaplanMeierFitter = KaplanMeierFitter()

    # The event is late delivery, so 1 - event is not needed; Kaplan-Meier directly estimates on-time survival.
    for group_value in sorted(df[group_col].dropna().unique().tolist()):
        group_mask: pd.Series = df[group_col] == group_value
        group_df: pd.DataFrame = df.loc[group_mask, [DURATION_COLUMN, EVENT_COLUMN]]
        if group_df.empty:
            continue
        kmf.fit(
            durations=group_df[DURATION_COLUMN].astype(float),
            event_observed=group_df[EVENT_COLUMN].astype(int),
            label=decode_group_value(group_col, group_value),
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title(f"Kaplan-Meier Survival Curves by {group_col}")
    plt.xlabel("Scheduled Days")
    plt.ylabel("Probability of On-Time Delivery")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_cox_model(df: pd.DataFrame) -> CoxPHFitter:
    """Train a Cox proportional hazards model on the requested covariates."""
    require_columns(df, [DURATION_COLUMN, EVENT_COLUMN] + COX_COVARIATES)
    print("Training Cox proportional hazards model...")

    # The Cox model uses only order-time covariates so the hazard remains leakage-safe.
    model_df: pd.DataFrame = df[[DURATION_COLUMN, EVENT_COLUMN] + COX_COVARIATES].copy()
    model_df[DURATION_COLUMN] = model_df[DURATION_COLUMN].clip(lower=1).astype(float)
    model_df[EVENT_COLUMN] = model_df[EVENT_COLUMN].astype(int)

    # A small penalizer improves numerical stability when financial features are correlated.
    cox_model: CoxPHFitter = CoxPHFitter(penalizer=0.01)
    cox_model.fit(model_df, duration_col=DURATION_COLUMN, event_col=EVENT_COLUMN)

    print(cox_model.summary.to_string())

    summary_df: pd.DataFrame = cox_model.summary.copy()
    summary_df["hazard_ratio"] = np.exp(summary_df["coef"])
    top_risk_factors: pd.DataFrame = summary_df.sort_values("hazard_ratio", ascending=False).head(5)
    print("\nTop 5 risk factors by hazard ratio:")
    print(top_risk_factors[["hazard_ratio", "p"]].to_string())

    joblib.dump(cox_model, MODELS_DIR / "cox_model.pkl")

    hazard_ratios: dict[str, float] = {
        covariate: float(hazard_ratio)
        for covariate, hazard_ratio in summary_df["hazard_ratio"].to_dict().items()
    }
    (MODELS_DIR / "hazard_ratios.json").write_text(
        json.dumps(hazard_ratios, indent=2),
        encoding="utf-8",
    )

    return cox_model


def build_order_frame(order_features: dict[str, Any]) -> pd.DataFrame:
    """Convert a single-order feature dict into a Cox-model input frame."""
    required_fields: list[str] = [DURATION_COLUMN] + COX_COVARIATES
    missing_fields: list[str] = [field for field in required_fields if field not in order_features]
    if missing_fields:
        raise KeyError(f"Missing order_features keys: {missing_fields}")

    order_frame: pd.DataFrame = pd.DataFrame([{field: order_features[field] for field in required_fields}])
    order_frame[DURATION_COLUMN] = order_frame[DURATION_COLUMN].astype(float).clip(lower=1)
    return order_frame


def first_numeric(value: Any) -> float:
    """Extract a float from scalar, Series, or ndarray-like lifelines outputs."""
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    if isinstance(value, pd.DataFrame):
        return float(value.iloc[0, 0])
    if isinstance(value, np.ndarray):
        return float(value.ravel()[0])
    return float(value)


def predict_survival_days(
    cox_model: CoxPHFitter,
    order_features: dict[str, Any],
) -> dict[str, Any]:
    """Predict survival metrics for a single order."""
    order_frame: pd.DataFrame = build_order_frame(order_features)
    scheduled_days: int = int(order_frame.iloc[0][DURATION_COLUMN])

    # Survival probability is the model's estimate of remaining on-time beyond each day threshold.
    survival_fn: pd.DataFrame = cox_model.predict_survival_function(order_frame, times=[3, 5, 7])
    median_survival_days_value: float = first_numeric(cox_model.predict_median(order_frame))
    partial_hazard: float = first_numeric(cox_model.predict_partial_hazard(order_frame))
    survival_at_schedule: float = float(
        cox_model.predict_survival_function(order_frame, times=[scheduled_days]).iloc[0, 0]
    )
    late_probability_day5: float = 1.0 - float(survival_fn.loc[5].iloc[0])

    interpretation: str
    if survival_at_schedule < 0.4:
        interpretation = f"High risk: {late_probability_day5:.0%} chance of late delivery by day 5"
    elif survival_at_schedule < 0.7:
        interpretation = f"Moderate risk: {late_probability_day5:.0%} chance of late delivery by day 5"
    else:
        interpretation = f"Low risk: {survival_at_schedule:.0%} chance of on-time delivery"

    return {
        "median_survival_days": median_survival_days_value,
        "risk_score": partial_hazard,
        "survival_probability_day3": float(survival_fn.loc[3].iloc[0]),
        "survival_probability_day5": float(survival_fn.loc[5].iloc[0]),
        "survival_probability_day7": float(survival_fn.loc[7].iloc[0]),
        "interpretation": interpretation,
    }


def score_orders_survival(df: pd.DataFrame, cox_model: CoxPHFitter) -> pd.DataFrame:
    """Score a batch of orders with survival risk and operational tiers."""
    require_columns(df, [DURATION_COLUMN] + COX_COVARIATES)
    print(f"Scoring {len(df)} orders with survival model...")

    feature_df: pd.DataFrame = df[[DURATION_COLUMN] + COX_COVARIATES].copy()
    feature_df[DURATION_COLUMN] = feature_df[DURATION_COLUMN].astype(float).clip(lower=1)

    partial_hazards: pd.Series = cox_model.predict_partial_hazard(feature_df)
    median_survival_days: pd.Series = cox_model.predict_median(feature_df)

    # Score each order at its own scheduled duration so tiers reflect its planned fulfillment window.
    survival_probabilities: list[float] = []
    for idx, row in feature_df.iterrows():
        duration_value: int = int(row[DURATION_COLUMN])
        row_frame: pd.DataFrame = row.to_frame().T
        survival_probabilities.append(
            float(cox_model.predict_survival_function(row_frame, times=[duration_value]).iloc[0, 0])
        )

    scored_df: pd.DataFrame = df.copy()
    scored_df["survival_risk_score"] = (1.0 - pd.Series(survival_probabilities, index=feature_df.index)).astype(float)
    scored_df["expected_delay_days"] = (
        median_survival_days.astype(float) - feature_df[DURATION_COLUMN].astype(float)
    )
    scored_df["survival_tier"] = np.select(
        [
            pd.Series(survival_probabilities, index=feature_df.index) > 0.7,
            pd.Series(survival_probabilities, index=feature_df.index) >= 0.4,
        ],
        ["SAFE", "WATCH"],
        default="DANGER",
    )
    scored_df["cox_partial_hazard"] = partial_hazards.astype(float)
    return scored_df


def main() -> None:
    """Run the full survival-analysis workflow requested for the project."""
    global LABEL_ENCODERS

    try:
        print("[1/4] Loading data...")
        processed_df: pd.DataFrame = load_artifact(MODELS_DIR / "processed_df.pkl")
        if LABEL_ENCODERS_PATH.exists():
            LABEL_ENCODERS = load_artifact(LABEL_ENCODERS_PATH)

        print("[2/4] Plotting Kaplan-Meier curves...")
        plot_kaplan_meier(processed_df, "Shipping Mode", MODELS_DIR / "km_shipping_mode.png")
        plot_kaplan_meier(processed_df, "Market", MODELS_DIR / "km_market.png")

        print("[3/4] Training Cox model...")
        cox_model: CoxPHFitter = train_cox_model(processed_df)

        sample_order: dict[str, Any] = processed_df.iloc[0][[DURATION_COLUMN] + COX_COVARIATES].to_dict()
        prediction: dict[str, Any] = predict_survival_days(cox_model, sample_order)
        print("\nSample order survival prediction:")
        print(json.dumps(prediction, indent=2))

        print("[4/4] Scoring orders...")
        scored_df: pd.DataFrame = score_orders_survival(processed_df.head(1000).copy(), cox_model)
        print("Survival tier distribution (first 1000 orders):")
        print(scored_df["survival_tier"].value_counts().to_string())
    except Exception as exc:
        print(f"Survival pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
