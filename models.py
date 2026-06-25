"""Model factories and the class-padding helper used by every experiment.

Keeping the constructors in one place guarantees XGBoost and TabPFN-3 use
identical hyperparameters across the baseline, stability, and learning-curve
experiments.
"""
from __future__ import annotations

import pandas as pd

from . import config


def pad_missing_classes(X_train, y_train):
    """Add one all-zero dummy row for any class absent from ``y_train``.

    Some subsamples (used in the learning curve) drop rare rating classes
    entirely. XGBoost needs to see every label to keep a consistent 10-class
    output space, so we inject a single dummy row per missing class.
    """
    present = set(y_train.unique())
    missing = [c for c in config.ALL_CLASSES if c not in present]
    if not missing:
        return X_train, y_train

    dummy_X = pd.DataFrame(0.0, index=range(len(missing)), columns=X_train.columns)
    dummy_y = pd.Series(missing)
    return (
        pd.concat([X_train, dummy_X], ignore_index=True),
        pd.concat([y_train, dummy_y], ignore_index=True),
    )


def make_xgb():
    """XGBoost baseline classifier with the thesis hyperparameters."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def make_tabpfn(n_estimators: int = 1, device: str | None = None):
    """TabPFN-3 classifier. Ensures the license token is set before construction."""
    from tabpfn import TabPFNClassifier
    config.ensure_tabpfn_token()
    return TabPFNClassifier(
        n_estimators=n_estimators,
        device=device or config.get_device(),
        ignore_pretraining_limits=True,  # our ~6.2k rows exceed the soft limit
        random_state=config.RANDOM_STATE,
    )


def make_autogluon_train_frame(X, y, label: str = "Rating_Num"):
    """AutoGluon wants the label inside the dataframe; build that frame."""
    frame = X.copy()
    frame[label] = y.values
    return frame
