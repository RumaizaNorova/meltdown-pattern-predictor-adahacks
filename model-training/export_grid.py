#!/usr/bin/env python3
"""
Export prediction grid for all input combinations.
Used by the frontend for client-side lookup (no API needed).
Includes model-based risk contributors: factors that are "on" (bad) ranked by
feature importance.
"""

import json
import pickle
import sys
from pathlib import Path

# Pickle references train._Preprocessor; add to __main__ so deserialization works
import train

sys.modules["__main__"]._Preprocessor = train._Preprocessor

MODEL_PATH = Path(__file__).parent / "model.pkl"
OUTPUT_PATH = Path(__file__).parent.parent / "predict_grid.json"

SLEEP_VALUES = [round(x * 0.5, 1) for x in range(8, 25)]  # 4.0 to 12.0
NOISE_VALUES = ["Low", "Medium", "High"]
BINARY_VALUES = ["Yes", "No"]

# Features that increase risk when "on", with display labels
# Ordered by typical importance (routine, noise high, etc.)
RISK_FACTOR_SPEC = [
    ("routineChange", "Yes", "Routine change"),
    ("noiseLevel", "High", "High sensory environment"),
    ("noiseLevel", "Medium", "Medium sensory environment"),
    ("screenAfter7", "Yes", "Late screen exposure"),
    ("mealAfter7", "Yes", "Late meal"),
    ("sugarAfter6", "Yes", "Late sugar intake"),
    ("sleepHours", lambda x: float(x) < 7, "Low sleep (<7h)"),
]


def get_contributors(record, importance_order):
    """Return risk factors that are present, ordered by feature importance."""
    contrib = []
    for spec in importance_order:
        if len(spec) == 3:
            key, val, label = spec
            if callable(val):
                if val(record[key]):
                    contrib.append(label)
            elif record.get(key) == val:
                contrib.append(label)
    return contrib


def main():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    clf = pipeline.named_steps["classifier"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = list(preprocess.get_feature_names_out())
    importances = clf.feature_importances_

    # Build importance order: map our RISK_FACTOR_SPEC to feature indices
    # Preprocessor order: sleepHours, sugarAfter6, screenAfter7, routineChange, mealAfter7,
    #                     noise_Low, noise_Medium, noise_High, sleep_below_7, sleep_x_noise_high
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    # Order by importance (routineChange, noise_High, etc. have high importance)
    spec_with_imp = []
    for spec in RISK_FACTOR_SPEC:
        label = spec[2]
        if "routine" in label.lower():
            idx = name_to_idx.get("routineChange", 0)
        elif "High sensory" in label:
            idx = name_to_idx.get("noise_High", 0)
        elif "Medium sensory" in label:
            idx = name_to_idx.get("noise_Medium", 0)
        elif "screen" in label.lower():
            idx = name_to_idx.get("screenAfter7", 0)
        elif "meal" in label.lower():
            idx = name_to_idx.get("mealAfter7", 0)
        elif "sugar" in label.lower():
            idx = name_to_idx.get("sugarAfter6", 0)
        elif "sleep" in label.lower():
            idx = max(name_to_idx.get("sleep_below_7", 0), name_to_idx.get("sleepHours", 0))
        else:
            idx = 0
        spec_with_imp.append((importances[idx], spec))
    spec_with_imp.sort(key=lambda x: -x[0])
    importance_order = [s for _, s in spec_with_imp]

    grid = {}

    for sleep in SLEEP_VALUES:
        for noise in NOISE_VALUES:
            for sugar in BINARY_VALUES:
                for screen in BINARY_VALUES:
                    for routine in BINARY_VALUES:
                        for meal in BINARY_VALUES:
                            records = [{
                                "sleepHours": sleep,
                                "noiseLevel": noise,
                                "sugarAfter6": sugar,
                                "screenAfter7": screen,
                                "routineChange": routine,
                                "mealAfter7": meal,
                            }]
                            record = records[0]
                            prob = float(pipeline.predict_proba(records)[0, 1])
                            pct = round(prob * 100)

                            if pct < 25:
                                tier = "Low"
                            elif pct < 50:
                                tier = "Medium"
                            else:
                                tier = "High"

                            contributors = get_contributors(record, importance_order)

                            key = f"{sleep}|{noise}|{sugar}|{screen}|{routine}|{meal}"
                            grid[key] = {
                                "probability": pct,
                                "tier": tier,
                                "contributors": contributors,
                            }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(grid, f)

    print(f"Exported {len(grid)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
