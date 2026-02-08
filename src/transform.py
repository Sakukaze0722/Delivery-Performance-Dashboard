"""
Data transformation utilities for the Delivery Performance Dashboard.
"""
import pandas as pd
from pathlib import Path

from .data_load import load_required_from_raw


def build_fact_orders(raw_dir: str) -> pd.DataFrame:
    """
    Build a flat fact table for the delivery performance dashboard.

    - Loads required CSVs from raw_dir
    - Creates geo lookup by zip_code_prefix -> mean lat/lng
    - Joins orders + customers + geo
    - Aggregates payments per order (mode payment_type)
    - Aggregates order_items per order (sum freight_value) and derive category
      via products + translation (mode)
    - Computes delay_days and on_time for delivered orders

    Returns a flat fact table with columns needed for dashboard.
    """
    data = load_required_from_raw(raw_dir)
    orders = data["orders"]
    customers = data["customers"]
    order_items = data["order_items"]
    order_payments = data["order_payments"]
    products = data["products"]
    translation = data["product_category_translation"]
    geolocation = data["geolocation"]

    # Create geo lookup: zip_code_prefix -> mean lat/lng
    geo_lookup = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg(
            mean_lat=("geolocation_lat", "mean"),
            mean_lng=("geolocation_lng", "mean"),
        )
        .reset_index()
    )
    geo_lookup = geo_lookup.rename(
        columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"}
    )

    # Join orders + customers + geo
    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(
        geo_lookup,
        on="customer_zip_code_prefix",
        how="left",
    )

    # Parse datetime columns for delay computation
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"], errors="coerce"
    )
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"], errors="coerce"
    )

    # Aggregate payments per order (mode payment_type)
    payment_mode = (
        order_payments.groupby("order_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        .reset_index()
    )
    payment_mode = payment_mode.rename(columns={"payment_type": "payment_type_mode"})
    payment_total = (
        order_payments.groupby("order_id")["payment_value"].sum().reset_index()
    )
    payment_total = payment_total.rename(columns={"payment_value": "payment_value_sum"})
    df = df.merge(payment_mode, on="order_id", how="left")
    df = df.merge(payment_total, on="order_id", how="left")

    # Aggregate order_items per order: sum freight_value, derive category (mode)
    items_with_product = order_items.merge(products, on="product_id", how="left")
    items_with_product = items_with_product.merge(
        translation,
        on="product_category_name",
        how="left",
    )
    items_with_product["product_category_name_english"] = items_with_product[
        "product_category_name_english"
    ].fillna(items_with_product["product_category_name"])

    freight_agg = (
        items_with_product.groupby("order_id")["freight_value"].sum().reset_index()
    )
    freight_agg = freight_agg.rename(columns={"freight_value": "freight_value_sum"})

    def _mode_or_first(series):
        mode_vals = series.mode()
        return mode_vals.iloc[0] if len(mode_vals) > 0 else None

    category_agg = (
        items_with_product.groupby("order_id")["product_category_name_english"]
        .agg(_mode_or_first)
        .reset_index()
    )
    category_agg = category_agg.rename(
        columns={"product_category_name_english": "product_category_mode"}
    )

    df = df.merge(freight_agg, on="order_id", how="left")
    df = df.merge(category_agg, on="order_id", how="left")

    # Compute delay_days and on_time for delivered orders
    delivered_mask = df["order_status"] == "delivered"
    df["delay_days"] = None
    df["on_time"] = None
    df.loc[delivered_mask, "delay_days"] = (
        df.loc[delivered_mask, "order_delivered_customer_date"]
        - df.loc[delivered_mask, "order_estimated_delivery_date"]
    ).dt.days
    df.loc[delivered_mask, "on_time"] = df.loc[delivered_mask, "delay_days"] <= 0

    return df


def get_fact_orders(
    raw_dir: str,
    processed_dir: str | Path,
    use_cache_file: bool = True,
) -> pd.DataFrame:
    """
    Load fact_orders from processed dir if exists, else build from raw and save.
    Call this from app.py with @st.cache_data for in-memory caching.
    """
    processed_path = Path(processed_dir)
    parquet_path = processed_path / "fact_orders.parquet"

    if use_cache_file and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    df = build_fact_orders(raw_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return df
