"""Leakage-safe feature engineering pipeline for Supply Chain Stress Tester."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COLUMN: str = "Late_delivery_risk"
RANDOM_STATE: int = 42

BASE_FEATURE_COLUMNS: List[str] = [
    "Days for shipment (scheduled)",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Benefit per order",
    "Sales per customer",
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Department Name",
    "Order Region",
    "Type",
    "order_month",
    "order_dayofweek",
    "high_value_order",
    "discount_impact",
    "profit_margin",
]

TARGET_ENCODING_COLUMNS: Dict[str, str] = {
    "Order Region": "region_late_rate",
    "Department Name": "department_late_rate",
    "Shipping Mode": "shipping_mode_late_rate",
    "Market": "market_late_rate",
    "Customer Segment": "customer_segment_late_rate",
}

INTERACTION_FEATURE_COLUMNS: List[str] = [
    "scheduled_days_x_shipping_mode",
    "quantity_x_discount",
    "profit_x_quantity",
    "total_x_scheduled",
]

RISK_SCORE_COLUMN: str = "composite_risk_score"

SAFE_FEATURE_COLUMNS: List[str] = (
    BASE_FEATURE_COLUMNS
    + list(TARGET_ENCODING_COLUMNS.values())
    + INTERACTION_FEATURE_COLUMNS
    + [RISK_SCORE_COLUMN]
)

LEAKAGE_COLUMNS: List[str] = [
    "Days for shipping (real)",
    "delay_days",
    "delay_ratio",
    "is_late",
    "Delivery Status",
    "revenue_at_risk",
]

CATEGORICAL_COLUMNS: List[str] = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Department Name",
    "Order Region",
    "Order Status",
    "Type",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    """[1/5] Load the source CSV and apply basic cleaning."""
    print("[1/5] Loading and cleaning...")
    df: pd.DataFrame = pd.read_csv(path, encoding="latin-1", low_memory=False)

    drop_cols: List[str] = [
        "Customer Email",
        "Customer Password",
        "Customer Fname",
        "Customer Lname",
        "Customer Street",
        "Product Description",
        "Product Image",
    ]
    for column in drop_cols:
        if column in df.columns:
            df = df.drop(columns=column)

    for date_col in ["order date (DateOrders)", "shipping date (DateOrders)"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols: List[str] = df.select_dtypes(include=[object, "category"]).columns.tolist()

    for column in numeric_cols:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_cols:
        if df[column].isna().any():
            mode: str = df[column].mode(dropna=True)[0] if not df[column].mode(dropna=True).empty else "missing"
            df[column] = df[column].fillna(mode)

    return df


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """[2/5] Create features that are fully known at order time."""
    print("[2/5] Engineering base features...")
    df = df.copy()

    denom: pd.Series = df.get("Order Item Total", pd.Series(np.ones(len(df)), index=df.index))
    df["profit_margin"] = np.where(
        denom.astype(float) == 0,
        0.0,
        df.get("Order Profit Per Order", 0.0).astype(float) / denom.astype(float),
    )
    df["profit_margin"] = df["profit_margin"].clip(lower=-1, upper=1)

    if "order date (DateOrders)" in df.columns:
        df["order_month"] = df["order date (DateOrders)"].dt.month.fillna(0).astype(int)
        df["order_dayofweek"] = df["order date (DateOrders)"].dt.dayofweek.fillna(0).astype(int)
    else:
        df["order_month"] = 0
        df["order_dayofweek"] = 0

    if "Order Item Total" in df.columns:
        threshold: float = float(df["Order Item Total"].quantile(0.75))
        df["high_value_order"] = (df["Order Item Total"].astype(float) > threshold).astype(int)
    else:
        df["high_value_order"] = 0

    df["discount_impact"] = (
        df.get("Order Item Discount Rate", 0.0).astype(float)
        * df.get("Order Item Total", 0.0).astype(float)
    )

    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any post-order outcome fields that leak the target."""
    removable_columns: List[str] = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    if removable_columns:
        print(f"Removing leakage columns: {removable_columns}")
        df = df.drop(columns=removable_columns)
    return df


def split_dataframe(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the raw engineered dataframe before any target-based encoding is computed."""
    print("[3/5] Splitting before target encoding...")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    y: pd.Series = df[target_col].astype(int)
    print("Class distribution (full dataset):")
    print(y.value_counts(dropna=False).to_string())

    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Class distribution (train):")
    print(y_train.value_counts().to_string())
    print("Class distribution (test):")
    print(y_test.value_counts().to_string())
    return train_df.copy(), test_df.copy(), y_train.copy(), y_test.copy()


def build_target_encoding_maps(
    train_df: pd.DataFrame,
    target_col: str,
) -> tuple[dict[str, dict[str, float]], float]:
    """Build leakage-safe target encoding maps using training rows only."""
    global_late_rate: float = float(train_df[target_col].astype(float).mean())
    encoding_maps: dict[str, dict[str, float]] = {}

    for source_column, encoded_column in TARGET_ENCODING_COLUMNS.items():
        grouped: pd.Series = train_df.groupby(source_column)[target_col].mean()
        encoding_maps[encoded_column] = {str(key): float(value) for key, value in grouped.items()}

    payload: dict[str, Any] = {
        "global_late_rate": global_late_rate,
        "encodings": encoding_maps,
    }
    joblib.dump(payload, MODELS_DIR / "target_encodings.pkl")
    return encoding_maps, global_late_rate


def apply_target_encoding(
    df: pd.DataFrame,
    encoding_maps: dict[str, dict[str, float]],
    global_late_rate: float,
) -> pd.DataFrame:
    """Apply training-derived target statistics to any dataframe split."""
    df = df.copy()
    for source_column, encoded_column in TARGET_ENCODING_COLUMNS.items():
        mapping: dict[str, float] = encoding_maps[encoded_column]
        df[encoded_column] = (
            df[source_column]
            .astype(str)
            .map(mapping)
            .fillna(global_late_rate)
            .astype(float)
        )
    return df


def fit_label_encoders(train_df: pd.DataFrame, categorical_cols: List[str]) -> dict[str, LabelEncoder]:
    """Fit label encoders on the training split so categorical mappings stay train-only."""
    encoders: dict[str, LabelEncoder] = {}
    for column in categorical_cols:
        if column not in train_df.columns:
            continue
        encoder: LabelEncoder = LabelEncoder()
        encoder.fit(train_df[column].astype(str).fillna("missing"))
        encoders[column] = encoder
    joblib.dump(encoders, MODELS_DIR / "label_encoders.pkl")
    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Apply train-fitted label encoders and map unseen categories to -1."""
    df = df.copy()
    for column, encoder in encoders.items():
        classes: set[str] = set(encoder.classes_.tolist())
        values: pd.Series = df[column].astype(str).fillna("missing")
        df[column] = values.map(lambda value: int(encoder.transform([value])[0]) if value in classes else -1)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create nonlinear interaction features from safe base predictors."""
    df = df.copy()

    # Scheduled lead time interacts with chosen shipping mode and helps separate risky route/service combinations.
    df["scheduled_days_x_shipping_mode"] = (
        df["Days for shipment (scheduled)"].astype(float) * df["Shipping Mode"].astype(float)
    )
    # Quantity with discount captures bulk discounted orders that may strain fulfillment capacity.
    df["quantity_x_discount"] = (
        df["Order Item Quantity"].astype(float) * df["Order Item Discount Rate"].astype(float)
    )
    # Profitability per unit can proxy operational priority and order handling complexity.
    df["profit_x_quantity"] = (
        df["Order Item Profit Ratio"].astype(float) * df["Order Item Quantity"].astype(float)
    )
    # Expensive orders with longer planned lead times often belong to slower, riskier logistics lanes.
    df["total_x_scheduled"] = (
        df["Order Item Total"].astype(float) * df["Days for shipment (scheduled)"].astype(float)
    )

    return df


def add_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Blend the strongest target-encoding signals into one dense feature."""
    df = df.copy()
    df[RISK_SCORE_COLUMN] = (
        df["region_late_rate"].astype(float) * 0.3
        + df["department_late_rate"].astype(float) * 0.2
        + df["shipping_mode_late_rate"].astype(float) * 0.3
        + df["market_late_rate"].astype(float) * 0.2
    )
    return df


def define_feature_set() -> List[str]:
    """[4/5] Persist the final safe feature list."""
    print("[4/5] Defining feature set...")
    joblib.dump(SAFE_FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")
    return SAFE_FEATURE_COLUMNS.copy()


def save_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> None:
    """[5/5] Save model-ready train/test artifacts to models/."""
    print("[5/5] Saving model artifacts...")
    x_train: pd.DataFrame = train_df.reindex(columns=feature_cols)
    x_test: pd.DataFrame = test_df.reindex(columns=feature_cols)
    y_train: pd.Series = train_df[target_col].astype(int)
    y_test: pd.Series = test_df[target_col].astype(int)
    processed_df: pd.DataFrame = pd.concat([train_df, test_df], axis=0).sort_index()

    joblib.dump(x_train, MODELS_DIR / "X_train.pkl")
    joblib.dump(x_test, MODELS_DIR / "X_test.pkl")
    joblib.dump(y_train, MODELS_DIR / "y_train.pkl")
    joblib.dump(y_test, MODELS_DIR / "y_test.pkl")
    joblib.dump(processed_df, MODELS_DIR / "processed_df.pkl")


def main() -> None:
    """Run the full leakage-safe feature engineering pipeline end-to-end."""
    df: pd.DataFrame = load_and_clean(DATA_PATH)
    df = engineer_base_features(df)
    df = remove_leakage_columns(df)

    train_df, test_df, _, _ = split_dataframe(df, TARGET_COLUMN)

    encoding_maps, global_late_rate = build_target_encoding_maps(train_df, TARGET_COLUMN)
    train_df = apply_target_encoding(train_df, encoding_maps, global_late_rate)
    test_df = apply_target_encoding(test_df, encoding_maps, global_late_rate)

    encoders: dict[str, LabelEncoder] = fit_label_encoders(train_df, CATEGORICAL_COLUMNS)
    train_df = apply_label_encoders(train_df, encoders)
    test_df = apply_label_encoders(test_df, encoders)

    train_df = add_interaction_features(train_df)
    test_df = add_interaction_features(test_df)
    train_df = add_composite_risk_score(train_df)
    test_df = add_composite_risk_score(test_df)

    feature_cols: List[str] = define_feature_set()
    present_features: List[str] = [feature for feature in feature_cols if feature in train_df.columns]
    missing_features: set[str] = set(feature_cols) - set(present_features)
    if missing_features:
        print(f"Warning: missing requested features: {sorted(missing_features)}")

    save_artifacts(train_df, test_df, present_features, TARGET_COLUMN)

    print(f"Final feature count: {len(present_features)}")
    print("Final class balance:")
    print(df[TARGET_COLUMN].value_counts().to_string())
    print("Target encoding features created:")
    print(", ".join(TARGET_ENCODING_COLUMNS.values()))


if __name__ == "__main__":
    main()
