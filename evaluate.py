"""Evaluation utilities: metrics, per-class report, and shared figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro and weighted F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def present_classes_labels(y_test):
    """Classes actually present in the test set, with their text labels.

    Class D (=0) has no test rows, so we pin the label set to what is present
    to keep every model's report and confusion matrix aligned.
    """
    present = sorted(pd.Series(y_test).unique())
    labels = [config.RATING_LABELS[i] for i in present]
    return present, labels


def print_classification_report(y_true, y_pred, header: str = ""):
    """Print a per-class precision/recall/F1 table."""
    present, labels = present_classes_labels(y_true)
    if header:
        print(header)
    print(classification_report(
        y_true, y_pred, labels=present, target_names=labels, zero_division=0,
    ))


def plot_confusion_matrix(y_true, y_pred, title: str, cmap: str = "Blues",
                          save_path=None, show: bool = True):
    """Side-by-side count and row-normalised confusion matrices."""
    present, labels = present_classes_labels(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} — Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} — Confusion Matrix (row-normalised)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_feature_importance(importance: pd.Series, title: str,
                            color: str = "steelblue", top_n: int = 15,
                            kind: str = "bar", save_path=None, show: bool = True):
    """Bar chart of the top-N feature importances."""
    imp = importance.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 5))
    if kind == "barh":
        imp.sort_values().plot(kind="barh", color=color, edgecolor="white")
        plt.xlabel("Importance score")
    else:
        imp.plot(kind="bar", color=color, edgecolor="white")
        plt.ylabel("Importance score")
        plt.xlabel("Feature")
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
