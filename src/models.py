"""
Model training, loading, and inference for the Delivery Performance Dashboard (delay prediction, etc.).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from src.config import MODELS_DIR
from src.features import (
    DELAY_CATEGORICAL_COLUMNS,
    DELAY_NUMERIC_COLUMNS,
    build_delay_features,
)

DELAY_MODEL_FILENAME = "delay_model.joblib"


def _make_delay_preprocessor():
    """Build ColumnTransformer for delay model: OneHotEncoder for categorical, passthrough for numeric."""
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                DELAY_CATEGORICAL_COLUMNS,
            ),
            ("num", "passthrough", DELAY_NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def train_delay_model(df: pd.DataFrame):
    """
    Train delay prediction model from fact_orders. Uses only delivered orders with delay_days.
    Saves pipeline (preprocessor + RandomForestRegressor) to MODELS_DIR.
    Returns the fitted pipeline and feature names after transform (for importance).
    """
    from src.features import get_delay_training_data

    X, y = get_delay_training_data(df)
    if X.empty or y.empty:
        raise ValueError("No training data: need delivered orders with delay_days.")

    preprocessor = _make_delay_preprocessor()
    regressor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / DELAY_MODEL_FILENAME
    joblib.dump(pipeline, path)

    return pipeline


def load_delay_model():
    """Load delay model from MODELS_DIR. Returns None if file does not exist."""
    path = MODELS_DIR / DELAY_MODEL_FILENAME
    if not path.exists():
        return None
    return joblib.load(path)


def predict_delay(model, df: pd.DataFrame) -> pd.Series:
    """Predict delay_days for each row. df must have columns required by build_delay_features."""
    X = build_delay_features(df)
    return pd.Series(model.predict(X), index=df.index)


def get_delay_feature_importance(model) -> pd.DataFrame:
    """
    Return feature importance for the delay model.
    model must be the fitted pipeline (preprocessor + RandomForestRegressor).
    """
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        DELAY_CATEGORICAL_COLUMNS
    )
    num_names = list(DELAY_NUMERIC_COLUMNS)
    names = list(cat_names) + list(num_names)
    imp = regressor.feature_importances_
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
        "importance", ascending=False
    )
