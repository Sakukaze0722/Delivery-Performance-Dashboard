"""
Train ML models (delay prediction, etc.) from fact_orders and save to data/models/.
Run from project root: python scripts/train_models.py
"""
import sys
from pathlib import Path

# Ensure project root is on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.models import train_delay_model
from src.transform import get_fact_orders


def main():
    print("Loading fact_orders...")
    df = get_fact_orders(
        raw_dir=str(RAW_DATA_DIR),
        processed_dir=str(PROCESSED_DATA_DIR),
        use_cache_file=True,
    )
    if df.empty:
        print("Error: fact_orders is empty. Check data/raw or data/processed.")
        sys.exit(1)
    print("Training delay prediction model...")
    train_delay_model(df)
    print(f"Done. Model saved to {MODELS_DIR / 'delay_model.joblib'}")


if __name__ == "__main__":
    main()
