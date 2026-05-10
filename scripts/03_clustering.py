"""
Phase 3 — Neighborhood Clustering (Unsupervised)
Run: python scripts/03_clustering.py
Outputs:
  data/processed/neighborhood_features.csv
  data/processed/cluster_labels.csv
  outputs/figures/03_elbow_silhouette.png
  outputs/figures/03_clusters.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from src.utils import (
    MERGED_DATA, NEIGHBORHOOD_FEATS, CLUSTER_LABELS,
    FIGURES_DIR, ensure_dirs
)
from src.clustering import (
    build_neighborhood_features, find_optimal_k, fit_clusters, plot_clusters
)

ensure_dirs()

# ── Load merged data ──────────────────────────────────────────────────────────
print("Loading merged data…")
df = pd.read_parquet(MERGED_DATA)
print(f"Rows: {len(df):,}  |  Zip codes: {df['zip_code'].nunique():,}")

# ── Build zip-code-level feature matrix ───────────────────────────────────────
print("\nBuilding neighborhood features…")
hood = build_neighborhood_features(df)
print(f"Neighborhoods (zip codes with ZHVI): {len(hood):,}")
print(hood.describe().round(2))

print(f"Saving → {NEIGHBORHOOD_FEATS}")
hood.to_csv(NEIGHBORHOOD_FEATS, index=False)

# ── Find optimal k ────────────────────────────────────────────────────────────
print("\nFinding optimal number of clusters…")
best_k, X_scaled = find_optimal_k(
    hood,
    save_path=FIGURES_DIR / '03_elbow_silhouette.png'
)

# Use k=4 for the four growth stages; override if silhouette strongly prefers another
K = 4
print(f"\nUsing k={K} (four interpretable growth stages)")

# ── Fit final clusters ────────────────────────────────────────────────────────
print(f"\nFitting K-Means with k={K}…")
hood, km = fit_clusters(hood, X_scaled, k=K)

print(f"\nSaving → {CLUSTER_LABELS}")
hood[['zip_code', 'cluster', 'cluster_label']].to_csv(CLUSTER_LABELS, index=False)

# ── Visualize ─────────────────────────────────────────────────────────────────
print("\nPlotting clusters…")
plot_clusters(hood, save_path=FIGURES_DIR / '03_clusters.png')

print("\nCluster summary:")
from src.clustering import CLUSTER_FEATURES
print(hood.groupby('cluster_label')[CLUSTER_FEATURES].mean().round(2).to_string())

print("\nClustering complete.")
