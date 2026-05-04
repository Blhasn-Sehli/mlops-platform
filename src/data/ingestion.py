"""
Data ingestion layer.

Loads raw data from a CSV file or directly from the sklearn Iris dataset (dev/CI).
Validates schema, types, missing values, and value ranges before returning a clean DataFrame.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from src.logging_config import get_logger

log = get_logger(__name__)

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAME = "species"

# Reasonable domain bounds for Iris features (cm)
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "sepal_length": (0.0, 50.0),
    "sepal_width": (0.0, 50.0),
    "petal_length": (0.0, 50.0),
    "petal_width": (0.0, 50.0),
}


# ── Public API ────────────────────────────────────────────────────────────────


def load_from_sklearn() -> pd.DataFrame:
    """Load the Iris dataset from scikit-learn. Used in dev, CI, and Kubeflow."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    df[TARGET_NAME] = iris.target
    log.info("data_loaded", source="sklearn", n_rows=len(df))
    return df


def load_from_csv(path: str | Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    The file must have a header row with columns matching FEATURE_NAMES.
    TARGET_NAME column is optional (not required for inference).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    log.info("data_loaded", source=str(path), n_rows=len(df), columns=list(df.columns))
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """
    Run schema and quality checks on a DataFrame.
    Raises ValueError with a descriptive message on the first failure.
    """
    _check_not_empty(df)
    _check_required_columns(df)
    _check_numeric_types(df)
    _check_no_nulls(df)
    _check_value_bounds(df)
    log.info("data_validated", n_rows=len(df), status="ok")


# ── Validation helpers ────────────────────────────────────────────────────────


def _check_not_empty(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("Dataset is empty")


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _check_numeric_types(df: pd.DataFrame) -> None:
    non_numeric = [c for c in FEATURE_NAMES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Columns must be numeric: {non_numeric}")


def _check_no_nulls(df: pd.DataFrame) -> None:
    null_counts = df[FEATURE_NAMES].isnull().sum()
    bad = null_counts[null_counts > 0].to_dict()
    if bad:
        raise ValueError(f"Null values found: {bad}")


def _check_value_bounds(df: pd.DataFrame) -> None:
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out) > 0:
            raise ValueError(
                f"Column '{col}' has {len(out)} values outside [{lo}, {hi}]: "
                f"{df[col].describe().to_dict()}"
            )
