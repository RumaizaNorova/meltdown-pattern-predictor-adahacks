# Meltdown Prediction Model Training

This folder trains a machine learning model to predict meltdown likelihood from daily factors. The model is trained on synthetic data produced by the `synthetic-data-generator` module.

## Model Choice: Logistic Regression

We use Logistic Regression because it won a fair benchmark (see `benchmark.py`).

**Benchmark results (5-fold CV on 10k synthetic logs):**

| Model | Accuracy | AUC-ROC | Brier |
|-------|----------|---------|-------|
| Logistic Regression | 64.2% | 0.676 | 0.226 |
| Random Forest | 63.2% | 0.653 | 0.230 |
| Random Forest (deeper) | 62.7% | 0.646 | 0.233 |

Logistic Regression has the best AUC and calibration. This makes sense: the synthetic data is generated with a logistic-style formula, so the model matches the data structure.

**Caveat:** On real caregiver data, the true relationship may be non-linear or include interactions we did not model. If you later get real logs and retrain, run `benchmark.py` again to compare models on that data. Logistic Regression may or may not remain best.

## Pipeline

The saved artifact is a sklearn Pipeline with two steps:

1. **Preprocessing.** Converts raw inputs (sleep hours, noise level, Yes/No fields) into a numeric matrix. Binary fields become 0/1. Noise level is one-hot encoded (Low, Medium, High).

2. **Logistic Regression.** Max iter 1000, class weight balanced.

The pipeline is saved with pickle so the inference API can load it and call `predict_proba` with the same input format.

## Benchmarking

Run `python benchmark.py` to compare models with 5-fold cross-validation. Uses the same data and reports accuracy, AUC-ROC, and Brier score. Install `xgboost` to include it in the comparison.

## Prerequisites

1. Generate synthetic data:

   ```bash
   cd ../synthetic-data-generator
   python generate.py --count 10000 --output synthetic_logs.json
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

From the `model-training` directory:

```bash
python train.py
```

Or with explicit paths:

```bash
python train.py --data ../synthetic-data-generator/synthetic_logs.json --output model.pkl --metrics metrics.json
```

## Output

- **model.pkl** – Trained pipeline (preprocessor + classifier). Load with `pickle.load()`.
- **metrics.json** – Accuracy, AUC-ROC, classification report, confusion matrix, feature importance.

## Feature Order

The preprocessor expects inputs in this order for `predict_proba`:

- sleepHours (float)
- noiseLevel (Low, Medium, or High)
- sugarAfter6 (Yes or No)
- screenAfter7 (Yes or No)
- routineChange (Yes or No)
- mealAfter7 (Yes or No)

Output: probability of meltdown (0 to 1).
