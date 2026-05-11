import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Features fed to K-Means (all log-transformed or ratio-based to reduce skew)
CLUSTER_FEATURES = [
    'log_median_price',
    'median_price_sqft',
    'price_to_zhvi',
    'growth_1yr',
    'growth_5yr',
]

# Quadrant labels: assigned by (price level) × (growth trajectory) after clustering
STAGE_NAMES = ['Declining', 'Stable', 'Emerging', 'Gentrifying']


def build_neighborhood_features(df_merged):
    hood = df_merged.groupby('zip_code').agg(
        median_price      = ('price',           'median'),
        median_price_sqft = ('price_per_sqft',  'median'),
        listing_count     = ('price',            'count'),
        median_house_size = ('house_size',       'median'),
    ).reset_index()

    zhvi_zip = (
        df_merged[['zip_code', 'zhvi_current', 'growth_1yr', 'growth_5yr']]
        .drop_duplicates('zip_code')
    )
    hood = hood.merge(zhvi_zip, on='zip_code', how='inner')
    hood = hood.dropna(subset=['median_price', 'median_price_sqft',
                                'zhvi_current', 'growth_1yr', 'growth_5yr'])

    # Derived features
    hood['log_median_price'] = np.log1p(hood['median_price'])
    hood['price_to_zhvi']    = hood['median_price'] / hood['zhvi_current']

    # Clip extreme price_to_zhvi ratios (outliers skew the clustering)
    p99 = hood['price_to_zhvi'].quantile(0.99)
    hood['price_to_zhvi'] = hood['price_to_zhvi'].clip(upper=p99)

    return hood.reset_index(drop=True)


def find_optimal_k(hood, k_range=range(2, 9), save_path=None):
    X_scaled = StandardScaler().fit_transform(
        hood[CLUSTER_FEATURES].astype('float64')
    )

    inertias, sil_scores = [], []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(
            silhouette_score(X_scaled, labels, sample_size=min(5000, len(hood)))
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set(title='Elbow Curve', xlabel='Number of Clusters (k)', ylabel='Inertia')
    axes[1].plot(list(k_range), sil_scores, 'rs-')
    axes[1].set(title='Silhouette Score', xlabel='Number of Clusters (k)', ylabel='Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"\nBest k by silhouette: {best_k}  (score={max(sil_scores):.3f})")
    return best_k, X_scaled


def fit_clusters(hood, X_scaled, k=4):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    hood = hood.copy()
    hood['cluster'] = km.fit_predict(X_scaled)

    # Quadrant-based label assignment using price level and 5yr growth
    cluster_means = hood.groupby('cluster').agg(
        price_mean  = ('median_price', 'mean'),
        growth_mean = ('growth_5yr',   'mean'),
    )
    price_median  = cluster_means['price_mean'].median()
    growth_median = cluster_means['growth_mean'].median()

    label_map = {}
    for cid, row in cluster_means.iterrows():
        high_price  = row['price_mean']  > price_median
        high_growth = row['growth_mean'] > growth_median
        if     high_price and     high_growth: label = 'Emerging'
        elif   high_price and not high_growth: label = 'Stable'
        elif not high_price and     high_growth: label = 'Gentrifying'
        else:                                   label = 'Declining'
        label_map[cid] = label

    hood['cluster_label'] = hood['cluster'].map(label_map)

    final_score = silhouette_score(
        X_scaled, hood['cluster'], sample_size=min(5000, len(hood))
    )
    print(f"\nFinal Silhouette Score (k={k}): {final_score:.3f}\n")
    print(hood.groupby('cluster_label')[CLUSTER_FEATURES + ['median_price', 'growth_5yr']].mean().round(3))
    print("\nZip codes per cluster:")
    print(hood['cluster_label'].value_counts().to_string())

    return hood, km


def plot_clusters(hood, save_path=None):
    # Use axis limits (not data clipping) so outliers fall off the edge
    # without creating artificial pile-ups at the boundary
    x_lo = hood['growth_1yr'].quantile(0.005) * 100
    x_hi = hood['growth_1yr'].quantile(0.995) * 100
    y_hi = hood['median_price_sqft'].quantile(0.995)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, grp in hood.groupby('cluster_label'):
        axes[0].scatter(
            grp['growth_1yr'] * 100,
            grp['median_price_sqft'],
            alpha=0.4, s=12, label=label
        )
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(0, y_hi)
    axes[0].set(
        title='Neighborhood Clusters',
        xlabel='1-Year Price Growth (%)',
        ylabel='Median Price per Sq Ft ($)'
    )
    axes[0].legend()

    counts = hood['cluster_label'].value_counts()
    axes[1].bar(counts.index, counts.values, color='steelblue')
    axes[1].set(title='Zip Codes per Growth Stage', xlabel='Stage', ylabel='Count')
    for i, (stage, count) in enumerate(counts.items()):
        axes[1].text(i, count + 50, str(count), ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
