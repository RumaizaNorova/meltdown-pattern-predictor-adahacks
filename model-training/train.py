#!/usr/bin/env python3
"""
Train meltdown prediction model on synthetic logs.

Uses Logistic Regression (benchmark winner on synthetic data). Run
benchmark.py to compare models. Saves a predictor pipeline for inference.

Usage:
    python train.py [--data path] [--output path]
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "sleepHours",
    "noiseLevel",
    "sugarAfter6",
    "screenAfter7",
    "routineChange",
    "mealAfter7",
]
TARGET_COLUMN = "meltdownOccurred"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> Tuple[List[dict], np.ndarray]:
    """Load logs from JSON, return list of feature dicts and target array."""
    with open(path) as f:
        logs = json.load(f)

    X_raw: List[dict] = []
    y_list: List[int] = []

    for log in logs:
        try:
            row = {k: log[k] for k in FEATURE_COLUMNS}
            if any(row.get(k) is None for k in FEATURE_COLUMNS):
                continue
            target = log.get(TARGET_COLUMN)
            if target is None:
                continue
            X_raw.append(row)
            y_list.append(1 if target == "Yes" else 0)
        except (KeyError, TypeError):
            continue

    return X_raw, np.array(y_list, dtype=np.int64)


def build_pipeline() -> Pipeline:
    """Build preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", _Preprocessor()),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class _Preprocessor(BaseEstimator, TransformerMixin):
    """
    Convert raw feature values to numeric matrix for Random Forest.
    Encodes binary Yes/No as 0/1 and one-hot encodes noise level.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        records = X if isinstance(X, list) else list(X)

        for row in records:
            sleep = float(row["sleepHours"])
            sugar = 1 if row["sugarAfter6"] == "Yes" else 0
            screen = 1 if row["screenAfter7"] == "Yes" else 0
            routine = 1 if row["routineChange"] == "Yes" else 0
            meal = 1 if row["mealAfter7"] == "Yes" else 0

            noise = row["noiseLevel"]
            noise_low = 1 if noise == "Low" else 0
            noise_medium = 1 if noise == "Medium" else 0
            noise_high = 1 if noise == "High" else 0

            rows.append([sleep, sugar, screen, routine, meal, noise_low, noise_medium, noise_high])

        return np.array(rows, dtype=np.float64)

    def get_feature_names_out(self):
        return [
            "sleepHours",
            "sugarAfter6",
            "screenAfter7",
            "routineChange",
            "mealAfter7",
            "noise_Low",
            "noise_Medium",
            "noise_High",
        ]


def main():
    parser = argparse.ArgumentParser(description="Train meltdown prediction model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "synthetic-data-generator" / "synthetic_logs.json",
        help="Path to synthetic logs JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "model.pkl",
        help="Output path for trained pipeline",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(__file__).parent / "metrics.json",
        help="Output path for evaluation metrics",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Data not found: {args.data}\n"
            "Run the synthetic data generator first:\n"
            "  cd synthetic-data-generator && python generate.py --count 10000"
        )

    X_raw, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "auc_roc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Feature importance (scaled coefficients for Logistic Regression)
    clf = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scale"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coef_scaled = clf.coef_[0] * scaler.scale_
    importances = dict(zip(feature_names, map(float, np.abs(coef_scaled))))
    metrics["feature_importance"] = importances

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pipeline, f)

    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {args.metrics}")
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"AUC-ROC: {auc:.3f}")
    print("\nFeature importance:")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    main()
