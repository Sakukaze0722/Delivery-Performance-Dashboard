"""
Metrics calculation for the Delivery Performance Dashboard.
"""
from datetime import datetime

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[datetime | None, datetime | None] | None = None,
    states: list[str] | None = None,
    categories: list[str] | None = None,
    payment_types: list[str] | None = None,
    delivered_only: bool = False,
) -> pd.DataFrame:
    """
    Apply filters to the fact orders dataframe.
    Handles empty dataframe and NaNs; empty filter lists mean no filter applied.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    # Parse order_purchase_timestamp for date filtering
    if "order_purchase_timestamp" in result.columns:
        result["order_purchase_timestamp"] = pd.to_datetime(
            result["order_purchase_timestamp"], errors="coerce"
        )

    # Date range filter
    if date_range is not None and len(date_range) >= 2:
        start_dt, end_dt = date_range[0], date_range[1]
        if start_dt is not None:
            result = result[
                result["order_purchase_timestamp"].fillna(pd.Timestamp.min) >= start_dt
            ]
        if end_dt is not None:
            result = result[
                result["order_purchase_timestamp"].fillna(pd.Timestamp.max) <= end_dt
            ]

    # State filter (excludes rows with NaN state if not in list)
    if states is not None and len(states) > 0:
        result = result[result["customer_state"].isin(states)]

    # Category filter
    if categories is not None and len(categories) > 0:
        result = result[result["product_category_mode"].isin(categories)]

    # Payment type filter
    if payment_types is not None and len(payment_types) > 0:
        result = result[result["payment_type_mode"].isin(payment_types)]

    # Delivered only
    if delivered_only:
        result = result[result["order_status"] == "delivered"]

    return result.reset_index(drop=True)


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute KPIs from the filtered fact orders dataframe.
    Returns dict with keys: total_orders, delivered_orders, on_time_count, on_time_rate,
    avg_delay_days, total_payment_value, total_freight_value.
    Handles empty dataframe and NaNs with safe defaults.
    """
    defaults = {
        "total_orders": 0,
        "delivered_orders": 0,
        "on_time_count": 0,
        "on_time_rate": 0.0,
        "avg_delay_days": 0.0,
        "total_payment_value": 0.0,
        "total_freight_value": 0.0,
    }

    if df is None or df.empty:
        return defaults.copy()

    total_orders = len(df)

    delivered = df[df["order_status"] == "delivered"]
    delivered_orders = len(delivered)

    on_time_count = 0
    on_time_rate = 0.0
    avg_delay_days = 0.0
    if delivered_orders > 0:
        on_time_valid = delivered["on_time"].dropna()
        on_time_count = int((on_time_valid == True).sum())
        on_time_rate = on_time_count / delivered_orders
        delay_valid = delivered["delay_days"].dropna()
        avg_delay_days = float(delay_valid.mean()) if len(delay_valid) > 0 else 0.0

    total_payment_value = float(
        df["payment_value_sum"].fillna(0).sum()
    ) if "payment_value_sum" in df.columns else 0.0
    total_freight_value = float(
        df["freight_value_sum"].fillna(0).sum()
    ) if "freight_value_sum" in df.columns else 0.0

    return {
        "total_orders": total_orders,
        "delivered_orders": delivered_orders,
        "on_time_count": on_time_count,
        "on_time_rate": on_time_rate,
        "avg_delay_days": avg_delay_days,
        "total_payment_value": total_payment_value,
        "total_freight_value": total_freight_value,
    }


def group_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fact orders by geography for map visualization.
    Groups by customer_state and (mean_lat, mean_lng) at zip level.
    Returns order count and delivery metrics per region.
    Handles empty dataframe and NaNs by dropping rows with missing geo.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "customer_state",
                "mean_lat",
                "mean_lng",
                "order_count",
                "delivered_count",
                "on_time_count",
                "on_time_rate",
                "avg_delay_days",
            ]
        )

    # Drop rows with missing geo for map aggregation
    geo_cols = ["customer_state", "mean_lat", "mean_lng"]
    for col in geo_cols:
        if col not in df.columns:
            return pd.DataFrame(columns=geo_cols + ["order_count"])

    geo_clean = df.dropna(subset=["mean_lat", "mean_lng"]).copy()
    if geo_clean.empty:
        return pd.DataFrame(
            columns=[
                "customer_state",
                "mean_lat",
                "mean_lng",
                "order_count",
                "delivered_count",
                "on_time_count",
                "on_time_rate",
                "avg_delay_days",
            ]
        )

    # Aggregate by state and centroid (mean lat/lng per state for simplicity)
    # For point map: can use zip-level; for choropleth use state-level
    agg = (
        geo_clean.groupby("customer_state")
        .agg(
            mean_lat=("mean_lat", "mean"),
            mean_lng=("mean_lng", "mean"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )

    delivered = geo_clean[geo_clean["order_status"] == "delivered"]
    if not delivered.empty:
        delivered_agg = (
            delivered.groupby("customer_state")
            .agg(
                delivered_count=("order_id", "nunique"),
                on_time_count=("on_time", lambda x: x.fillna(False).sum()),
                avg_delay_days=("delay_days", lambda x: x.dropna().mean()),
            )
            .reset_index()
        )
        agg = agg.merge(delivered_agg, on="customer_state", how="left")
    else:
        agg["delivered_count"] = 0
        agg["on_time_count"] = 0
        agg["avg_delay_days"] = 0.0

    agg["on_time_rate"] = (
        agg["on_time_count"] / agg["delivered_count"].replace(0, pd.NA)
    ).fillna(0.0)
    agg["delivered_count"] = agg["delivered_count"].fillna(0).astype(int)
    agg["on_time_count"] = agg["on_time_count"].fillna(0).astype(int)
    agg["avg_delay_days"] = agg["avg_delay_days"].fillna(0.0)

    return agg
