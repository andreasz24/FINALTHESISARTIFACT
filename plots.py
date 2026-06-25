"""Experiment-specific plots (sweep curves, grid heatmaps, stability, learning curve).

Generic confusion-matrix and feature-importance plots live in ``evaluate``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config


def plot_estimator_sweep(df, save_path=None, show: bool = True):
    """Left: F1/accuracy vs n_estimators (log-x). Right: inference time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = df["n_estimators"]
    axes[0].plot(n, df["f1_macro"], marker="o", color="darkorange", lw=2, label="F1 Macro")
    axes[0].plot(n, df["f1_weighted"], marker="s", color="steelblue", lw=2, label="F1 Weighted")
    axes[0].plot(n, df["accuracy"], marker="^", color="seagreen", lw=2, label="Accuracy")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(list(n)); axes[0].set_xticklabels([str(v) for v in n])
    axes[0].set_xlabel("n_estimators (log scale)"); axes[0].set_ylabel("Score")
    axes[0].set_title("TabPFN-3 — Performance vs n_estimators")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(v) for v in n], df["inference_time_s"],
                color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("n_estimators"); axes[1].set_ylabel("Inference time (s)")
    axes[1].set_title("TabPFN-3 — Inference Time vs n_estimators")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_marginal_gain(df, save_path=None, show: bool = True):
    """Incremental F1-macro improvement per estimator step."""
    gain = df["f1_macro"].diff().fillna(0)
    plt.figure(figsize=(8, 4))
    plt.bar([str(v) for v in df["n_estimators"]], gain,
            color="steelblue", edgecolor="white", alpha=0.85)
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("n_estimators"); plt.ylabel("ΔF1 Macro (vs previous)")
    plt.title("TabPFN-3 — Marginal Gain in F1 Macro per Estimator Step")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_grid_heatmaps(df, save_dir=None, show: bool = True):
    """Heatmaps for CV F1, test F1, and training time over the 2x2 grid."""
    save_dir = save_dir or config.FIGURES_DIR
    specs = [("cv_f1_macro", "CV F1 Macro", "Greens", ".4f"),
             ("test_f1_macro", "Test F1 Macro", "Greens", ".4f"),
             ("train_time_s", "Training time (s)", "Oranges", ".0f")]
    for metric, label, cmap, fmt in specs:
        pivot = df.pivot(index="num_bag_folds", columns="num_stack_levels", values=metric)
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                    cbar_kws={"label": label})
        plt.title(f"AutoGluon Grid Search — {label}")
        plt.xlabel("num_stack_levels"); plt.ylabel("num_bag_folds")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/ag_grid_{metric}.png", dpi=150)
        if show:
            plt.show()


def plot_stability_box(res, save_path=None, show: bool = True):
    """Box plot of F1-macro across all stability splits."""
    order = [m for m in ["XGBoost", "TabPFN-3", "AutoGluon"] if m in res["model"].unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [res.loc[res["model"] == m, "f1_macro"].values for m in order],
        labels=order, showmeans=True, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(config.MODEL_COLORS[m]); patch.set_alpha(0.4)
    ax.set_ylabel("F1 macro")
    ax.set_title(f"F1 macro distribution across {res['split'].nunique()} group-splits")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


def plot_learning_curve(lc_df, save_path=None, show: bool = True):
    """F1-macro vs training-set size with mean ± std bands per model."""
    agg = lc_df.groupby("frac").agg(
        n_rows=("n_rows", "mean"),
        xgb_mean=("xgb_f1m", "mean"), xgb_std=("xgb_f1m", "std"),
        tab_mean=("tab_f1m", "mean"), tab_std=("tab_f1m", "std"),
    ).reset_index()

    has_ag = "ag_f1m" in lc_df.columns and lc_df["ag_f1m"].notna().any()
    if has_ag:
        ag_agg = lc_df.groupby("frac").agg(
            ag_mean=("ag_f1m", "mean"), ag_std=("ag_f1m", "std")).reset_index()
        agg = agg.merge(ag_agg, on="frac")

    x = agg["n_rows"].values
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [("tab", "TabPFN-3", "darkorange", "o"),
              ("xgb", "XGBoost", "steelblue", "s")]
    if has_ag:
        series.append(("ag", "AutoGluon", "seagreen", "^"))
    for key, label, color, marker in series:
        ax.plot(x, agg[f"{key}_mean"], marker=marker, color=color, lw=2, label=label)
        ax.fill_between(x, agg[f"{key}_mean"] - agg[f"{key}_std"],
                        agg[f"{key}_mean"] + agg[f"{key}_std"], color=color, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Training set size (rows, log scale)"); ax.set_ylabel("F1 macro")
    ax.set_title("Learning curve — F1 macro vs training size (mean ± std, 5 seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return agg
