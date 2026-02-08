"""
Data loading utilities for the Delivery Performance Dashboard.
"""
import kagglehub
import pandas as pd
from pathlib import Path

def get_dataset_path() -> Path:
    """
    Download the Brazilian ecommerce dataset from Kaggle and return its path.
    Uses cached version if already downloaded.
    """
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    return Path(path)


def load_csv_from_raw(raw_dir: str, filename: str) -> pd.DataFrame:
    """Load a single CSV file from the raw data directory."""
    path = Path(raw_dir) / filename
    return pd.read_csv(path)


def load_required_from_raw(raw_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all required CSV files from data/raw.
    Assumes Kaggle Olist CSV files exist in the given directory.
    """
    return {
        "orders": load_csv_from_raw(raw_dir, "olist_orders_dataset.csv"),
        "customers": load_csv_from_raw(raw_dir, "olist_customers_dataset.csv"),
        "order_items": load_csv_from_raw(raw_dir, "olist_order_items_dataset.csv"),
        "order_payments": load_csv_from_raw(raw_dir, "olist_order_payments_dataset.csv"),
        "products": load_csv_from_raw(raw_dir, "olist_products_dataset.csv"),
        "product_category_translation": load_csv_from_raw(
            raw_dir, "product_category_name_translation.csv"
        ),
        "geolocation": load_csv_from_raw(raw_dir, "olist_geolocation_dataset.csv"),
    }


def load_orders() -> pd.DataFrame:
    """Load the orders dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_orders_dataset.csv")


def load_order_items() -> pd.DataFrame:
    """Load the order items dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_order_items_dataset.csv")


def load_customers() -> pd.DataFrame:
    """Load the customers dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_customers_dataset.csv")


def load_sellers() -> pd.DataFrame:
    """Load the sellers dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_sellers_dataset.csv")


def load_products() -> pd.DataFrame:
    """Load the products dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_products_dataset.csv")


def load_order_payments() -> pd.DataFrame:
    """Load the order payments dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_order_payments_dataset.csv")


def load_order_reviews() -> pd.DataFrame:
    """Load the order reviews dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_order_reviews_dataset.csv")


def load_geolocation() -> pd.DataFrame:
    """Load the geolocation dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "olist_geolocation_dataset.csv")


def load_product_category_translation() -> pd.DataFrame:
    """Load the product category name translation dataset (from Kaggle cache)."""
    path = get_dataset_path()
    return pd.read_csv(path / "product_category_name_translation.csv")


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Load all datasets from the Brazilian ecommerce collection (from Kaggle cache).
    Returns a dictionary mapping dataset names to DataFrames.
    """
    return {
        "orders": load_orders(),
        "order_items": load_order_items(),
        "customers": load_customers(),
        "sellers": load_sellers(),
        "products": load_products(),
        "order_payments": load_order_payments(),
        "order_reviews": load_order_reviews(),
        "geolocation": load_geolocation(),
        "product_category_translation": load_product_category_translation(),
    }
