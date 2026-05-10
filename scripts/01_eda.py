"""
Phase 1 — Exploratory Data Analysis
Run: python scripts/01_eda.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import REALTOR_CSV, ZHVI_CSV, FIGURES_DIR, ensure_dirs

ensure_dirs()

# ── Load a 200k sample for fast exploration ──────────────────────────────────
print("Loading realtor data sample…")
df = pd.read_csv(REALTOR_CSV, low_memory=False).sample(200_000, random_state=42)

print(f"\nShape: {df.shape}")
print("\nColumn dtypes & non-null counts:")
print(df.info())

print("\nMissing values (%):")
print((df.isnull().mean() * 100).round(2).sort_values(ascending=False))

print("\nNumerical summary:")
print(df.describe().T.round(2))

print("\nStatus value counts:")
print(df['status'].value_counts())

print("\nTop 10 states by listing count:")
print(df['state'].value_counts().head(10))

# ── Distributions ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distributions (200k sample)', fontsize=14)

for_sale = df[
    (df['status'] == 'for_sale') &
    (df['price'].between(50_000, 2_000_000)) &
    (df['house_size'].between(200, 8_000))
]

axes[0, 0].hist(for_sale['price'] / 1_000,        bins=60, color='steelblue', edgecolor='white')
axes[0, 0].set(title='Listing Price', xlabel='Price ($K)', ylabel='Count')

axes[0, 1].hist(for_sale['house_size'],            bins=60, color='seagreen',  edgecolor='white')
axes[0, 1].set(title='House Size', xlabel='Sq Ft', ylabel='Count')

axes[0, 2].hist(for_sale['price'] / for_sale['house_size'], bins=60, color='tomato', edgecolor='white')
axes[0, 2].set(title='Price per Sq Ft', xlabel='$/Sq Ft', ylabel='Count')

axes[1, 0].hist(for_sale['bed'].dropna(),          bins=15, color='mediumpurple', edgecolor='white')
axes[1, 0].set(title='Bedrooms', xlabel='Count', ylabel='Count')

axes[1, 1].hist(for_sale['bath'].dropna(),         bins=15, color='darkorange',   edgecolor='white')
axes[1, 1].set(title='Bathrooms', xlabel='Count', ylabel='Count')

axes[1, 2].hist(for_sale['acre_lot'].clip(0, 5),   bins=40, color='goldenrod',    edgecolor='white')
axes[1, 2].set(title='Lot Size (clipped at 5 ac)', xlabel='Acres', ylabel='Count')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/01_distributions.png")

# ── Missing value heatmap ────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
miss = df.isnull().mean().sort_values(ascending=False)
sns.barplot(x=miss.index, y=miss.values * 100, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Missing (%)')
plt.title('Missing Values by Column')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_missing_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/01_missing_values.png")

# ── Top states by median price ───────────────────────────────────────────────
top_states = (
    for_sale.groupby('state')['price']
    .median()
    .sort_values(ascending=False)
    .head(15)
)
plt.figure(figsize=(10, 5))
top_states.plot.bar(color='steelblue')
plt.ylabel('Median Listing Price ($)')
plt.title('Top 15 States by Median Listing Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_price_by_state.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/01_price_by_state.png")

# ── ZHVI quick look ──────────────────────────────────────────────────────────
print("\nLoading ZHVI data…")
zhvi = pd.read_csv(ZHVI_CSV, dtype={'RegionName': str})
date_cols = [c for c in zhvi.columns if c[:4].isdigit()]
print(f"ZHVI shape: {zhvi.shape}  |  Date range: {date_cols[0]} → {date_cols[-1]}")
print(f"Unique zip codes: {zhvi['RegionName'].nunique()}")

# National median home value over time (sample 500 zips)
sample_zhvi = zhvi[date_cols].sample(500, random_state=42).median()
plt.figure(figsize=(12, 4))
sample_zhvi.plot(color='steelblue')
plt.title('National Median ZHVI Over Time (500-zip sample)')
plt.xlabel('Date')
plt.ylabel('Home Value Index ($)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_zhvi_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → outputs/figures/01_zhvi_trend.png")

print("\nEDA complete.")
