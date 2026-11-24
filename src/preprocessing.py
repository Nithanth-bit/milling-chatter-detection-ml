"""Leakage-free preprocessing, dataset validation, and scikit-learn pipeline factories."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction import MODEL_FEATURE_COLUMNS

COLUMN_ALIASES = {
    "Omega (rpm)": "Omega",
    "Depth of Cut (m)": "Depth",
    "Depth of Cut": "Depth",
    "Critical Depth (m)": "Critical Depth",
    "Dominant_Freq_Hz": "Dominant Frequency",
    "Peak_to_Peak": "Peak-to-Peak",
    "Max_FFT": "Max FFT",
}


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and derive Peak-to-Peak if only Max/Min Acc are present."""
    clean = df.copy()
    rename_map = {k: v for k, v in COLUMN_ALIASES.items() if k in clean.columns and v not in clean.columns}
    if rename_map:
        clean = clean.rename(columns=rename_map)

    if "Peak-to-Peak" not in clean.columns and {"Max Acc", "Min Acc"}.issubset(clean.columns):
        clean["Peak-to-Peak"] = clean["Max Acc"] - clean["Min Acc"]

    return clean


def validate_feature_dataframe(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate uploaded or loaded feature table and return (cleaned_df, missing_columns)."""
    req = required_features or MODEL_FEATURE_COLUMNS
    clean = standardize_dataframe_columns(df)
    missing = [col for col in req if col not in clean.columns]
    if missing:
        return clean, missing

    for col in req:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if clean[req].isnull().any().any():
        bad_rows = int(clean[req].isnull().any(axis=1).sum())
        raise ValueError(
            f"Feature data contains NaN or non-numeric values in {bad_rows} row(s). "
            f"Check columns: {req}."
        )
    if np.isinf(clean[req].to_numpy(dtype=float)).any():
        raise ValueError(f"Feature data contains infinite values in columns: {req}.")

    return clean, []


def split_features_and_metadata(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Strictly separate vibration ML inputs (X) from physics target (y) and dataset groups."""
    req = feature_columns or MODEL_FEATURE_COLUMNS
    clean, missing = validate_feature_dataframe(df, req)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if "Chatter" not in clean.columns:
        raise ValueError("Missing ground-truth target column 'Chatter'.")
    if "Dataset" not in clean.columns:
        raise ValueError("Missing grouping column 'Dataset' for LODO cross-validation.")

    X = clean[req].copy()
    y = clean["Chatter"].astype(int).copy()
    groups = clean["Dataset"].astype(int).copy()
    return X, y, groups


def build_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Return leakage-safe scikit-learn Pipelines for the 3 journal algorithms."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
