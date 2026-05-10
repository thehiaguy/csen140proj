import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

MODEL_FEATURES = [
    'bed', 'bath', 'house_size', 'acre_lot',
    'bed_bath_ratio', 'total_rooms',
    'zhvi_current', 'growth_1yr', 'growth_5yr',
    'cluster',
]
TARGET = 'price'

# Cap training rows for Random Forest to keep wall-clock time reasonable
RF_SAMPLE_SIZE = 500_000


def prepare_model_data(df_merged, cluster_labels):
    cluster_labels = cluster_labels.copy()
    cluster_labels['zip_code'] = cluster_labels['zip_code'].astype(str).str.zfill(5)
    df = df_merged.merge(
        cluster_labels[['zip_code', 'cluster', 'cluster_label']],
        on='zip_code', how='left'
    )
    df = df.dropna(subset=MODEL_FEATURES + [TARGET])
    X  = df[MODEL_FEATURES].astype(float)
    y  = df[TARGET].astype(float)
    return df, X, y


def train_models(X_train, y_train):
    # Random Forest — subsample if large
    if len(X_train) > RF_SAMPLE_SIZE:
        idx      = np.random.default_rng(42).choice(len(X_train), RF_SAMPLE_SIZE, replace=False)
        X_rf     = X_train.iloc[idx]
        y_rf     = y_train.iloc[idx]
    else:
        X_rf, y_rf = X_train, y_train

    print("Training Random Forest…")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15,
        min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    rf.fit(X_rf, y_rf)

    print("Training XGBoost…")
    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42,
        eval_metric='mae', verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    return {'Random Forest': rf, 'XGBoost': xgb}


def evaluate_models(models, X_test, y_test):
    results = {}
    print(f"\n{'Model':<22} {'MAE':>14}  {'R²':>8}")
    print('-' * 48)
    for name, model in models.items():
        preds  = model.predict(X_test)
        mae    = mean_absolute_error(y_test, preds)
        r2     = r2_score(y_test, preds)
        print(f"{name:<22} ${mae:>13,.0f}  {r2:>8.4f}")
        results[name] = {'model': model, 'preds': preds, 'mae': mae, 'r2': r2}
    return results


def plot_feature_importance(model, feature_names, title, save_path=None):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    imp.plot.barh(figsize=(8, 5))
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(y_test, preds, title, save_path=None):
    sample = min(5000, len(y_test))
    idx    = np.random.default_rng(42).choice(len(y_test), sample, replace=False)
    plt.figure(figsize=(7, 7))
    plt.scatter(
        np.array(y_test)[idx], preds[idx],
        alpha=0.3, s=10, color='steelblue'
    )
    lim = max(np.array(y_test).max(), preds.max())
    plt.plot([0, lim], [0, lim], 'r--', linewidth=1)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_models(models, models_dir):
    for name, result in models.items():
        path = models_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(result['model'], path)
        print(f"Saved → {path}")
