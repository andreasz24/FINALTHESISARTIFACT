"""Data loading, preprocessing, and the company-aware train/test split.

Preprocessing mirrors the thesis pipeline exactly:
  1. Collapse notched ratings (e.g. BB+ / BB-  ->  BB).
  2. Winsorise every feature at the 1st / 99th percentile.
  3. Encode the broad ratings as consecutive integers (D=0 ... AAA=9).
  4. Split so that no company (``GROUP_COL``) appears in both train and test.
"""
from __future__ import annotations

from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config


def load_raw(csv_path=None) -> pd.DataFrame:
    """Load the raw corporate-credit-rating CSV."""
    path = csv_path or config.RAW_CSV
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ratings, winsorise features, and add the integer target column.

    Returns a copy; the input frame is not modified.
    """
    df = df.copy()

    # 1. Collapse notched ratings to broad classes.
    df["Rating"] = (
        df["Rating"].str.replace("+", "", regex=False).str.replace("-", "", regex=False)
    )

    # 2. Winsorise outliers at the 1st / 99th percentile.
    for col in config.FEATURE_COLS:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # 3. Encode broad ratings as consecutive integers starting from 0.
    rating_order = [r for r in config.RATING_LABELS if r in df["Rating"].unique()]
    rating_map = {r: i for i, r in enumerate(rating_order)}
    df["Rating_Num"] = df["Rating"].map(rating_map)
    return df


def get_xy_groups(df: pd.DataFrame):
    """Return the feature matrix X, target y, and the company group vector."""
    X = df[config.FEATURE_COLS]
    y = df["Rating_Num"]
    groups = df[config.GROUP_COL]
    return X, y, groups


def make_split(X, y, groups, test_size=config.TEST_SIZE,
               random_state=config.RANDOM_STATE):
    """Single company-aware train/test split (no company in both sides)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
    )


def prepare_and_save_splits(csv_path=None, save_path=None):
    """Full pipeline: load -> preprocess -> split -> persist to disk.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    df = preprocess(load_raw(csv_path))
    X, y, groups = get_xy_groups(df)
    X_train, X_test, y_train, y_test = make_split(X, y, groups)

    path = save_path or config.SPLITS_PATH
    joblib.dump((X_train, X_test, y_train, y_test), path)
    return X_train, X_test, y_train, y_test


def load_splits(path=None) -> Tuple:
    """Load the saved ``(X_train, X_test, y_train, y_test)`` tuple."""
    return joblib.load(path or config.SPLITS_PATH)
