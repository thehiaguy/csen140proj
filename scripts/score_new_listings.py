"""
Score new property listings and return undervalued buy recommendations.

Usage:
    python scripts/score_new_listings.py                         # uses data/new_listings.csv
    python scripts/score_new_listings.py my_listings.csv
    python scripts/score_new_listings.py my_listings.csv --threshold 0.15
    python scripts/score_new_listings.py my_listings.csv --top 20

Required CSV columns:
    price, bed, bath, house_size, zip_code

Optional CSV columns (used for display only):
    acre_lot, address, city, state

Outputs:
    Prints ranked buy recommendations to the terminal.
    Saves full results to data/processed/scored_listings.csv
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import engineer_features
from src.utils import PROCESSED, MODELS_DIR


REQUIRED_COLS   = ['price', 'bed', 'bath', 'house_size', 'zip_code']
DISPLAY_COLS    = ['address', 'city', 'state']
FEATURE_NAMES   = ['bed', 'bath', 'house_size', 'acre_lot',
                   'bed_bath_ratio', 'total_rooms',
                   'zhvi_current', 'growth_1yr', 'growth_5yr', 'cluster']


def load_artifacts():
    model_path = MODELS_DIR / 'xgboost.joblib'
    if not model_path.exists():
        sys.exit(f'ERROR: Trained model not found at {model_path}\n'
                 'Run scripts/04_regression.py first.')

    cluster_path = PROCESSED / 'cluster_labels.csv'
    zhvi_path    = PROCESSED / 'zhvi_features.csv'
    for p in (cluster_path, zhvi_path):
        if not p.exists():
            sys.exit(f'ERROR: {p} not found.\nRun scripts/02_preprocessing.py and '
                     'scripts/03_clustering.py first.')

    model    = joblib.load(model_path)
    clusters = pd.read_csv(cluster_path, dtype={'zip_code': str})
    zhvi     = pd.read_csv(zhvi_path,    dtype={'zip_code': str})
    return model, clusters, zhvi


def validate_input(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f'ERROR: Input CSV is missing required columns: {missing}\n'
                 f'Required: {REQUIRED_COLS}')
    if 'acre_lot' not in df.columns:
        df['acre_lot'] = 0.0
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    return df


def score(df, model, clusters, zhvi, threshold):
    df = engineer_features(df.copy())

    # Attach ZHVI neighborhood trend features
    df = df.merge(
        zhvi[['zip_code', 'zhvi_current', 'growth_1yr', 'growth_5yr']],
        on='zip_code', how='left'
    )

    # Attach cluster / growth-stage label
    df = df.merge(
        clusters[['zip_code', 'cluster', 'cluster_label']],
        on='zip_code', how='left'
    )
    df['cluster'] = df['cluster'].fillna(-1).astype(int)
    df['cluster_label'] = df['cluster_label'].fillna('Unknown')

    # Fill ZHVI nulls with dataset medians so unknown zips still get a prediction
    for col in ['zhvi_current', 'growth_1yr', 'growth_5yr']:
        if df[col].isna().any():
            median = zhvi[col].median()
            df[col] = df[col].fillna(median)

    X = df[FEATURE_NAMES].astype(float)
    df['predicted_price'] = model.predict(X)
    df['value_gap']       = df['predicted_price'] - df['price']
    df['value_gap_pct']   = df['value_gap'] / df['price']
    df['undervalued']     = df['value_gap_pct'] >= threshold
    return df


def print_results(df, threshold, top):
    buys = (
        df[df['undervalued']]
        .sort_values('value_gap_pct', ascending=False)
        .head(top)
        .reset_index(drop=True)
    )

    total      = len(df)
    n_buys     = df['undervalued'].sum()
    pct_buys   = n_buys / total * 100 if total else 0

    print(f'\n{"="*65}')
    print(f'  Scored {total:,} listings  |  threshold: {threshold*100:.0f}% gap')
    print(f'  Undervalued: {n_buys:,} ({pct_buys:.1f}%)  |  showing top {len(buys)}')
    print(f'{"="*65}')

    if buys.empty:
        print('  No undervalued properties found at this threshold.')
        return

    for i, row in buys.iterrows():
        label   = row.get('cluster_label', 'Unknown')
        city    = f"  {row['city']}, {row['state']}" if 'city' in row and pd.notna(row.get('city')) else ''
        addr    = f"  {row['address']}" if 'address' in df.columns and pd.notna(row.get('address')) else ''
        zip_    = row['zip_code']

        print(f"\n  #{i+1}  ZIP {zip_}{addr}{city}  [{label}]")
        print(f"       Listed:    ${row['price']:>12,.0f}")
        print(f"       Predicted: ${row['predicted_price']:>12,.0f}")
        print(f"       Gap:       ${row['value_gap']:>12,.0f}  ({row['value_gap_pct']*100:+.1f}%)  ← BUY")
        print(f"       {row['bed']:.0f} bed / {row['bath']:.0f} bath / {row['house_size']:,.0f} sqft")

    print(f'\n{"="*65}')


def main():
    parser = argparse.ArgumentParser(description='Score new listings for undervalued properties.')
    parser.add_argument('input', nargs='?', default='data/new_listings.csv',
                        help='Path to input CSV (default: data/new_listings.csv)')
    parser.add_argument('--threshold', type=float, default=0.20,
                        help='Min value gap to flag as undervalued (default: 0.20 = 20%%)')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top recommendations to display (default: 10)')
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f'ERROR: Input file not found: {csv_path}\n'
                 'Create a CSV with columns: price, bed, bath, house_size, zip_code\n'
                 'See data/sample_listings.csv for an example.')

    print(f'Loading listings from {csv_path}…')
    df = pd.read_csv(csv_path, dtype={'zip_code': str})
    df = validate_input(df)
    print(f'  {len(df):,} listings loaded.')

    print('Loading trained model and pipeline artifacts…')
    model, clusters, zhvi = load_artifacts()

    print('Scoring listings…')
    results = score(df, model, clusters, zhvi, args.threshold)

    print_results(results, args.threshold, args.top)

    out_path = PROCESSED / 'scored_listings.csv'
    display = [c for c in DISPLAY_COLS if c in results.columns]
    save_cols = ['zip_code'] + display + [
        'price', 'predicted_price', 'value_gap', 'value_gap_pct',
        'undervalued', 'cluster_label', 'bed', 'bath', 'house_size', 'acre_lot',
    ]
    results[[c for c in save_cols if c in results.columns]].to_csv(out_path, index=False)
    print(f'Full results saved → {out_path}')


if __name__ == '__main__':
    main()
