"""
Feature engineering for ML models (delay prediction, etc.) in the Delivery Performance Dashboard.
"""
import pandas as pd

# Columns used as inputs for delay prediction (must exist in fact_orders or be derived).
DELAY_CATEGORICAL_COLUMNS = ["customer_state", "product_category_mode", "payment_type_mode"]
DELAY_NUMERIC_COLUMNS = ["freight_value_sum", "payment_value_sum", "month", "day_of_week"]
DELAY_TARGET_COLUMN = "delay_days"


def _ensure_delay_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; derive month and day_of_week from order_purchase_timestamp."""
    out = df.copy()
    if "order_purchase_timestamp" in out.columns:
        ts = pd.to_datetime(out["order_purchase_timestamp"], errors="coerce")
        out["month"] = ts.dt.month.fillna(1).astype(int)
        out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
    else:
        out["month"] = 1
        out["day_of_week"] = 0
    for col in ["freight_value_sum", "payment_value_sum"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in DELAY_CATEGORICAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")
    return out


def build_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature DataFrame for delay prediction from fact_orders.
    Adds month and day_of_week, fills missing numeric/categorical columns.
    Returns a DataFrame with only the columns needed for the delay model.
    """
    out = _ensure_delay_input_columns(df)
    return out[DELAY_CATEGORICAL_COLUMNS + DELAY_NUMERIC_COLUMNS].copy()


def get_delay_training_data(df: pd.DataFrame):
    """
    From fact_orders, return (X, y) for delay model training.
    X is the feature DataFrame (same columns as build_delay_features), y is delay_days.
    Only includes delivered orders with non-null delay_days.
    """
    delivered = df.loc[df["order_status"] == "delivered"].copy()
    if "delay_days" not in delivered.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    valid = delivered["delay_days"].notna()
    subset = delivered.loc[valid]
    if subset.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = build_delay_features(subset)
    y = subset["delay_days"].astype(float)
    return X, y
