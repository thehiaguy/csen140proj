"""
Phase 4 — Property Value Regression (Supervised)
Run: python scripts/04_regression.py
Outputs:
  data/processed/model_predictions.parquet
  outputs/figures/04_feature_importance_rf.png
  outputs/figures/04_feature_importance_xgb.png
  outputs/figures/04_predictions_xgb.png
  outputs/models/random_forest.joblib
  outputs/models/xgboost.joblib
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    MERGED_DATA, CLUSTER_LABELS, MODEL_PREDICTIONS,
    FIGURES_DIR, MODELS_DIR, ensure_dirs
)
from src.regression import (
    MODEL_FEATURES, TARGET,
    prepare_model_data, train_models, evaluate_models,
    plot_feature_importance, plot_predictions, save_models
)

ensure_dirs()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading merged data…")
df = pd.read_parquet(MERGED_DATA)

print("Loading cluster labels…")
cluster_labels = pd.read_csv(CLUSTER_LABELS)

# ── Build feature matrix ───────────────────────────────────────────────────────
print("\nPreparing model data…")
df_model, X, y = prepare_model_data(df, cluster_labels)
print(f"Model dataset: {len(X):,} rows  |  Features: {list(X.columns)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
print()
models = train_models(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = evaluate_models(models, X_test, y_test)

# ── Feature importance ────────────────────────────────────────────────────────
print("\nPlotting feature importances…")
plot_feature_importance(
    results['Random Forest']['model'], MODEL_FEATURES,
    title='Random Forest — Feature Importance',
    save_path=FIGURES_DIR / '04_feature_importance_rf.png'
)
plot_feature_importance(
    results['XGBoost']['model'], MODEL_FEATURES,
    title='XGBoost — Feature Importance',
    save_path=FIGURES_DIR / '04_feature_importance_xgb.png'
)

# ── Predicted vs actual (XGBoost) ─────────────────────────────────────────────
plot_predictions(
    y_test, results['XGBoost']['preds'],
    title='XGBoost — Predicted vs Actual Price',
    save_path=FIGURES_DIR / '04_predictions_xgb.png'
)

# ── Save predictions ──────────────────────────────────────────────────────────
best_model = results['XGBoost']['model']
df_model = df_model.copy()
df_model['predicted_price'] = best_model.predict(X)
print(f"\nSaving predictions → {MODEL_PREDICTIONS}")
df_model.to_parquet(MODEL_PREDICTIONS, index=False)

# ── Save models ───────────────────────────────────────────────────────────────
save_models(results, MODELS_DIR)

print("\nRegression complete.")
