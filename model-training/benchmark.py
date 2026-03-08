#!/usr/bin/env python3
"""
Benchmark models fairly: RF, XGBoost, Logistic Regression.

Uses 5-fold cross-validation. No changes to synthetic data.
Reports accuracy, AUC, Brier score (calibration). Picks best model
only if it meaningfully beats the baseline.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

# Reuse preprocessor from train
from train import FEATURE_COLUMNS, TARGET_COLUMN, _Preprocessor, load_data

RANDOM_STATE = 42
CV_FOLDS = 5

DATA_PATH = Path(__file__).parent.parent / "synthetic-data-generator" / "synthetic_logs.json"


def build_rf_default() -> Pipeline:
    return Pipeline([
        ("preprocess", _Preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def build_rf_deeper() -> Pipeline:
    return Pipeline([
        ("preprocess", _Preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=400, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def build_logistic() -> Pipeline:
    return Pipeline([
        ("preprocess", _Preprocessor()),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def try_xgboost():
    """Try XGBoost if available."""
    try:
        from xgboost import XGBClassifier

        def build_xgb() -> Pipeline:
            return Pipeline([
                ("preprocess", _Preprocessor()),
                ("classifier", XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=RANDOM_STATE,
                )),
            ])
        return build_xgb
    except ImportError:
        return None


def run_benchmark():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data not found: {DATA_PATH}\n"
            "Run: cd synthetic-data-generator && python generate.py --count 10000"
        )

    X_raw, y = load_data(DATA_PATH)

    models = {
        "RandomForest (current)": build_rf_default(),
        "RandomForest (deeper)": build_rf_deeper(),
        "LogisticRegression": build_logistic(),
    }

    xgb_builder = try_xgboost()
    if xgb_builder:
        models["XGBoost"] = xgb_builder()

    results = []

    for name, pipe in models.items():
        # cross_validate gives us scores per fold
        scorers = ["accuracy", "roc_auc"]
        cv_scores = cross_validate(pipe, X_raw, y, cv=CV_FOLDS, scoring=scorers, n_jobs=1)

        # For Brier we need predictions
        y_prob = cross_val_predict(pipe, X_raw, y, cv=CV_FOLDS, method="predict_proba", n_jobs=1)[:, 1]
        brier = brier_score_loss(y, y_prob)

        acc_mean = cv_scores["test_accuracy"].mean()
        acc_std = cv_scores["test_accuracy"].std()
        auc_mean = cv_scores["test_roc_auc"].mean()
        auc_std = cv_scores["test_roc_auc"].std()

        results.append({
            "model": name,
            "accuracy_mean": float(acc_mean),
            "accuracy_std": float(acc_std),
            "auc_mean": float(auc_mean),
            "auc_std": float(auc_std),
            "brier": float(brier),
        })

        print(f"\n{name}")
        print(f"  Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"  AUC-ROC: {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"  Brier (lower=better): {brier:.3f}")

    # Best by AUC (primary metric for imbalanced-ish binary)
    best = max(results, key=lambda r: r["auc_mean"])
    print(f"\nBest by AUC: {best['model']} (AUC {best['auc_mean']:.3f})")

    # Honest note: is the improvement meaningful?
    rf_current = next(r for r in results if r["model"] == "RandomForest (current)")
    auc_gain = best["auc_mean"] - rf_current["auc_mean"]
    if auc_gain > 0.02:  # 2% AUC gain is meaningful
        print(f"  -> {best['model']} beats RF by {auc_gain:.3f} AUC. Consider switching.")
    else:
        print(f"  -> Improvement over RF ({auc_gain:+.3f}) is marginal. RF remains acceptable.")

    return results


if __name__ == "__main__":
    run_benchmark()
