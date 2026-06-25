"""Experiment routines that produce every result in the thesis.

Each function is self-contained, returns plain data structures (dicts /
DataFrames), and optionally saves to ``config.RESULTS_DIR``. The notebooks call
these and handle display.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

from . import config, evaluate
from .models import (
    make_autogluon_train_frame,
    make_tabpfn,
    make_xgb,
    pad_missing_classes,
)


# ── XGBoost baseline ────────────────────────────────────────────────────────
def run_xgboost_baseline(X_train, X_test, y_train, y_test, save: bool = True) -> dict:
    """Train the XGBoost baseline and return a results dict (acc, F1s, timings)."""
    model = make_xgb()
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t1

    metrics = evaluate.compute_metrics(y_test, y_pred)
    results = {
        "model": "XGBoost", **metrics,
        "train_time_s": train_time, "inference_time_s": inference_time,
        "y_pred": y_pred,
        "feature_importance": pd.Series(model.feature_importances_, index=X_train.columns),
    }
    if save:
        joblib.dump(
            {k: v for k, v in results.items() if k != "feature_importance"},
            config.RESULTS_DIR / "xgb_results.pkl",
        )
    return model, results


# ── TabPFN-3 n_estimators sweep ─────────────────────────────────────────────
def tabpfn_n_estimators_sweep(X_train, X_test, y_train, y_test,
                              n_values=(1, 2, 4, 8, 16), device=None):
    """Sweep TabPFN-3 ``n_estimators`` and return a results DataFrame.

    Each estimator samples a different random feature subset; the final
    prediction averages their probability distributions. With only 16 features
    gains are expected to plateau quickly.
    """
    device = device or config.get_device()
    rows = []
    for n_est in n_values:
        clf = make_tabpfn(n_estimators=n_est, device=device)
        t0 = time.time(); clf.fit(X_train, y_train); fit_time = time.time() - t0
        t1 = time.time(); y_pred = clf.predict(X_test); inf_time = time.time() - t1

        m = evaluate.compute_metrics(y_test, y_pred)
        rows.append({
            "n_estimators": n_est,
            "accuracy": round(m["accuracy"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "f1_weighted": round(m["f1_weighted"], 4),
            "fit_time_s": round(fit_time, 2),
            "inference_time_s": round(inf_time, 2),
            "total_time_s": round(fit_time + inf_time, 2),
        })
        print(f"n_estimators={n_est:2d}  F1_macro={m['f1_macro']:.4f}  "
              f"acc={m['accuracy']:.4f}  infer={inf_time:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "tabpfn3_estimators_results.csv", index=False)
    return df


def tabpfn_fit_best(X_train, X_test, y_train, y_test, best_n: int, device=None,
                    save: bool = True):
    """Refit TabPFN-3 at the chosen ``n_estimators`` for the final evaluation."""
    device = device or config.get_device()
    clf = make_tabpfn(n_estimators=best_n, device=device)
    t0 = time.time(); clf.fit(X_train, y_train); fit_time = time.time() - t0
    t1 = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    inf_time = time.time() - t1

    metrics = evaluate.compute_metrics(y_test, y_pred)
    results = {
        "model": "TabPFN-3", **metrics,
        "fit_time_s": fit_time, "inference_time_s": inf_time,
        "n_estimators": best_n, "y_pred": y_pred, "y_prob": y_prob,
    }
    if save:
        joblib.dump(results, config.RESULTS_DIR / "tabpfn3_results.pkl")
    return clf, results


# ── AutoGluon 2x2 grid search ───────────────────────────────────────────────
def autogluon_grid_search(train_data, test_data, y_test,
                          folds=(3, 8), stacks=(1, 2), time_limit: int = 300):
    """Grid over ``num_bag_folds`` x ``num_stack_levels``.

    Selection uses AutoGluon's internal CV score (``score_val``) so the held-out
    test set never drives model choice; the test metrics are reported alongside
    for reference only.
    """
    from autogluon.tabular import TabularPredictor

    rows = []
    for num_folds in folds:
        for num_stacks in stacks:
            name = f"folds{num_folds}_stacks{num_stacks}"
            path = config.RESULTS_DIR / "ag_grid" / name
            t0 = time.time()
            predictor = TabularPredictor(
                label="Rating_Num", problem_type="multiclass",
                eval_metric="f1_macro", path=str(path), verbosity=0,
            ).fit(
                train_data=train_data, time_limit=time_limit,
                presets="medium_quality",
                num_bag_folds=num_folds, num_stack_levels=num_stacks,
            )
            train_time = time.time() - t0

            best_val = predictor.leaderboard(silent=True)["score_val"].max()
            y_pred = predictor.predict(test_data.drop(columns=["Rating_Num"])).astype(int)
            m = evaluate.compute_metrics(y_test, y_pred)
            rows.append({
                "num_bag_folds": num_folds, "num_stack_levels": num_stacks,
                "cv_f1_macro": round(best_val, 4),
                "test_accuracy": round(m["accuracy"], 4),
                "test_f1_macro": round(m["f1_macro"], 4),
                "test_f1_weighted": round(m["f1_weighted"], 4),
                "train_time_s": round(train_time, 1),
            })
            print(f"folds={num_folds} stacks={num_stacks}  "
                  f"CV_F1={best_val:.4f}  test_F1={m['f1_macro']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "autogluon_grid_results.csv", index=False)
    return df


def autogluon_fit_best(train_data, test_data, y_test,
                       best_folds: int = 8, best_stacks: int = 2,
                       time_limit: int = 900, save: bool = True):
    """Refit the CV-selected best AutoGluon config and produce final outputs."""
    import random
    from autogluon.tabular import TabularPredictor

    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    t0 = time.time()
    predictor = TabularPredictor(
        label="Rating_Num", problem_type="multiclass", eval_metric="f1_macro",
        path=str(config.RESULTS_DIR / "autogluon_best"), verbosity=1,
    ).fit(
        train_data=train_data, time_limit=time_limit, presets="medium_quality",
        num_bag_folds=best_folds, num_stack_levels=best_stacks,
    )
    train_time = time.time() - t0

    y_pred = predictor.predict(test_data.drop(columns=["Rating_Num"])).astype(int)
    metrics = evaluate.compute_metrics(y_test, y_pred)
    results = {
        "model": "AutoGluon", **metrics, "train_time_s": train_time,
        "num_bag_folds": best_folds, "num_stack_levels": best_stacks,
        "y_pred": y_pred,
    }
    if save:
        joblib.dump(results, config.RESULTS_DIR / "autogluon_results.pkl")
    return predictor, results


# ── Stability analysis ──────────────────────────────────────────────────────
def nadeau_bengio(diffs, n_test_frac: float = config.TEST_SIZE):
    """Corrected resampled t-test for overlapping train/test splits.

    Returns ``(mean_diff, std_error, t_stat, p_value)``.
    """
    diffs = np.asarray(diffs)
    N = len(diffs)
    mean_d = diffs.mean()
    var_d = diffs.var(ddof=1)
    r = n_test_frac / (1 - n_test_frac)          # 0.25 for an 80/20 split
    se = np.sqrt((1.0 / N + r) * var_d)
    t = mean_d / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=N - 1)
    return mean_d, se, t, p


def run_stability_analysis(X, y, groups, n_splits: int = 20, run_ag: bool = True,
                           ag_time: int = 120, test_frac: float = config.TEST_SIZE,
                           device=None, save: bool = True):
    """Repeat the experiment over ``n_splits`` group-splits for honest error bars."""
    device = device or config.get_device()
    records = []
    for s in range(n_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=s)
        tr, te = next(gss.split(X, y, groups))
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        xgb = make_xgb(); xgb.fit(Xtr, ytr)
        records.append({"split": s, "model": "XGBoost",
                        **evaluate.compute_metrics(yte, xgb.predict(Xte))})

        tab = make_tabpfn(device=device); tab.fit(Xtr, ytr)
        records.append({"split": s, "model": "TabPFN-3",
                        **evaluate.compute_metrics(yte, tab.predict(Xte))})

        if run_ag:
            from autogluon.tabular import TabularPredictor
            trdf = make_autogluon_train_frame(Xtr, ytr)
            ag = TabularPredictor(
                label="Rating_Num", problem_type="multiclass",
                eval_metric="f1_macro", verbosity=0,
            ).fit(trdf, time_limit=ag_time, presets="medium_quality",
                  num_bag_folds=3, num_stack_levels=2)
            yp = ag.predict(Xte).astype(int)
            records.append({"split": s, "model": "AutoGluon",
                            **evaluate.compute_metrics(yte, yp)})
        print(f"Split {s + 1}/{n_splits} done")

    res = pd.DataFrame(records)
    if save:
        res.to_csv(config.RESULTS_DIR / "stability_runs.csv", index=False)
    return res


def summarize_stability(res: pd.DataFrame,
                        pairs=(("TabPFN-3", "XGBoost"),
                               ("TabPFN-3", "AutoGluon"),
                               ("AutoGluon", "XGBoost"))):
    """Print mean±std per model and the corrected paired F1-macro comparisons."""
    summary = res.groupby("model")[["accuracy", "f1_macro", "f1_weighted"]].agg(["mean", "std"])
    print("── Mean ± std across splits ──")
    print(summary.round(4), "\n")

    piv = res.pivot(index="split", columns="model", values="f1_macro")
    print("── Paired F1-macro comparison (Nadeau–Bengio corrected) ──")
    for a, b in pairs:
        if a not in piv.columns or b not in piv.columns:
            continue
        d = (piv[a] - piv[b]).dropna()
        md, se, t_val, p_val = nadeau_bengio(d)
        print(f"  {a} − {b}:  mean Δ = {md:+.4f}  |  win rate = {(d > 0).mean():.0%}  "
              f"|  corrected p = {p_val:.3f}")
    return summary


# ── Learning curve ──────────────────────────────────────────────────────────
def run_learning_curve(X, y, groups, fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                       seeds=(0, 1, 2, 3, 4), run_ag: bool = True,
                       ag_time: int = 120, device=None, save: bool = True):
    """Fixed seed-42 test set; subsample training companies at each fraction."""
    device = device or config.get_device()

    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr_full, y_tr_full = X.iloc[train_idx], y.iloc[train_idx]
    groups_tr = groups.iloc[train_idx]
    X_test_fix, y_test_fix = X.iloc[test_idx], y.iloc[test_idx]
    companies = groups_tr.unique()

    records = []
    for frac in fractions:
        for seed in seeds:
            rng = np.random.RandomState(seed)
            n_comp = max(1, int(len(companies) * frac))
            chosen = rng.choice(companies, size=n_comp, replace=False)
            mask = groups_tr.isin(chosen)
            Xtr, ytr = X_tr_full[mask], y_tr_full[mask]
            row = {"frac": frac, "n_rows": len(Xtr), "seed": seed}

            xgb = make_xgb()
            Xp, yp_pad = pad_missing_classes(Xtr, ytr)
            xgb.fit(Xp, yp_pad)
            m = evaluate.compute_metrics(y_test_fix, xgb.predict(X_test_fix))
            row.update(xgb_acc=m["accuracy"], xgb_f1m=m["f1_macro"], xgb_f1w=m["f1_weighted"])

            tab = make_tabpfn(device=device); tab.fit(Xtr, ytr)
            m = evaluate.compute_metrics(y_test_fix, tab.predict(X_test_fix))
            row.update(tab_acc=m["accuracy"], tab_f1m=m["f1_macro"], tab_f1w=m["f1_weighted"])

            if run_ag:
                from autogluon.tabular import TabularPredictor
                trdf = make_autogluon_train_frame(Xtr, ytr)
                ag = TabularPredictor(
                    label="Rating_Num", problem_type="multiclass",
                    eval_metric="f1_macro", verbosity=0,
                ).fit(trdf, time_limit=ag_time, presets="medium_quality",
                      num_bag_folds=3, num_stack_levels=2)
                m = evaluate.compute_metrics(y_test_fix, ag.predict(X_test_fix.copy()).astype(int))
                row.update(ag_acc=m["accuracy"], ag_f1m=m["f1_macro"], ag_f1w=m["f1_weighted"])

            records.append(row)
            msg = (f"frac={frac:.1f} seed={seed} n={len(Xtr):4d}  "
                   f"xgb={row['xgb_f1m']:.3f} tab={row['tab_f1m']:.3f}")
            print(msg + (f" ag={row['ag_f1m']:.3f}" if run_ag else ""))

    lc_df = pd.DataFrame(records)
    if save:
        lc_df.to_csv(config.RESULTS_DIR / "learning_curve_results.csv", index=False)
    return lc_df
