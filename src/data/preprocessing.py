"""
Feature preprocessing utilities.

Provides a reusable sklearn Pipeline (StandardScaler) and helper functions
for splitting data and describing feature distributions.

The preprocessing pipeline is intentionally kept separate from the classifier
so it can be composed in different ways (standalone, GridSearchCV, Kubeflow step).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.data.ingestion import FEATURE_NAMES, TARGET_NAME
from src.logging_config import get_logger

log = get_logger(__name__)


# ── Pipeline factory ──────────────────────────────────────────────────────────


def build_preprocessing_pipeline() -> Pipeline:
    """Return an unfitted preprocessing Pipeline (StandardScaler only)."""
    return Pipeline([("scaler", StandardScaler())])


# ── Splitting ─────────────────────────────────────────────────────────────────


def split(
    df: pd.DataFrame,
    test_size: float = settings.test_size,
    random_state: int = settings.random_state,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a validated DataFrame into train/test arrays.

    Returns:
        X_train, X_test, y_train, y_test — numpy arrays ready for sklearn.
    """
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    log.info(
        "data_split",
        n_train=len(X_train),
        n_test=len(X_test),
        test_size=test_size,
    )
    return X_train, X_test, y_train, y_test


def features_to_array(row: dict) -> np.ndarray:
    """Convert a single prediction request dict into a (1, 4) numpy array."""
    return np.array([[row[f] for f in FEATURE_NAMES]])


# ── Descriptive statistics ────────────────────────────────────────────────────


def describe_features(df: pd.DataFrame) -> dict[str, dict]:
    """Return per-feature descriptive statistics."""
    stats: dict[str, dict] = {}
    for col in FEATURE_NAMES:
        s = df[col]
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "q75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
        }
    return stats


def class_distribution(df: pd.DataFrame) -> dict[int, int]:
    """Return {class_id: count} distribution from a DataFrame with TARGET_NAME column."""
    if TARGET_NAME not in df.columns:
        return {}
    return df[TARGET_NAME].value_counts().sort_index().to_dict()
