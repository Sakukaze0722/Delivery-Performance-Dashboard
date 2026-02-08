"""
Configuration settings for the Delivery Performance Dashboard.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Required CSV filenames for Olist dataset (minimum set)
REQUIRED_CSVS = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
    "olist_geolocation_dataset.csv",
]
