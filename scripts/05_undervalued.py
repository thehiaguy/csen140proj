"""
Phase 5 — Undervalued Property Detection
Run: python scripts/05_undervalued.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import MODEL_PREDICTIONS, PROCESSED, FIGURES_DIR, ensure_dirs

ensure_dirs()

UNDERVALUED_THRESHOLD = 0.20

# ── Load predictions ──────────────────────────────────────────────────────────
print("Loading model predictions…")
df = pd.read_parquet(MODEL_PREDICTIONS)
print(f"Rows: {len(df):,}")

# ── Compute value gap ─────────────────────────────────────────────────────────
df['value_gap']     = df['predicted_price'] - df['price']
df['value_gap_pct'] = df['value_gap'] / df['price']
df['undervalued']   = df['value_gap_pct'] >= UNDERVALUED_THRESHOLD

n_undervalued = df['undervalued'].sum()
pct           = n_undervalued / len(df) * 100
print(f"\nUndervalued properties (gap ≥ {UNDERVALUED_THRESHOLD*100:.0f}%): "
      f"{n_undervalued:,}  ({pct:.1f}% of listings)")

print("\nValue gap summary:")
print(df['value_gap_pct'].describe().apply(lambda x: f"{x:.3f}"))

# ── Distribution of value gap ─────────────────────────────────────────────────
# Use data-driven clip bounds instead of hardcoded limits
lo = df['value_gap_pct'].quantile(0.02)
hi = df['value_gap_pct'].quantile(0.98)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['value_gap_pct'].clip(lo, hi), bins=80,
             color='steelblue', edgecolor='white')
axes[0].axvline(UNDERVALUED_THRESHOLD, color='red', linestyle='--',
                label=f'≥{UNDERVALUED_THRESHOLD*100:.0f}% threshold')
axes[0].set(
    title=f'Distribution of Value Gap (clipped to [{lo:.2f}, {hi:.2f}])',
    xlabel='(Predicted - Listed) / Listed',
    ylabel='Count'
)
axes[0].legend()

price_99 = df['price'].quantile(0.99)
sample_idx = df.sample(10_000, random_state=42).index
axes[1].scatter(
    df.loc[sample_idx, 'price'].clip(upper=price_99),
    df.loc[sample_idx, 'value_gap_pct'].clip(lo, hi),
    alpha=0.2, s=5, color='steelblue'
)
axes[1].axhline(UNDERVALUED_THRESHOLD, color='red', linestyle='--',
                label=f'≥{UNDERVALUED_THRESHOLD*100:.0f}% threshold')
axes[1].set(
    title='Value Gap vs Listing Price (10k sample)',
    xlabel='Listing Price ($)',
    ylabel='Value Gap (%)'
)
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_value_gap_distribution.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved → outputs/figures/05_value_gap_distribution.png")

# ── Undervalued properties by growth stage ────────────────────────────────────
if 'cluster_label' in df.columns:
    cross = pd.crosstab(
        df['cluster_label'], df['undervalued'], normalize='index'
    ) * 100
    cross.columns = ['Fairly Valued (%)', 'Undervalued (%)']
    print("\nUndervalued rate by growth stage:")
    print(cross.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_vals = cross['Undervalued (%)'].sort_values()
    bars = ax.barh(sorted_vals.index, sorted_vals.values, color='steelblue')
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(
        xlabel='Undervalued Listings (%)',
        title='Undervalued Rate by Neighborhood Growth Stage',
        xlim=(0, sorted_vals.max() * 1.15)
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_undervalued_by_cluster.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved → outputs/figures/05_undervalued_by_cluster.png")

# ── Undervalued by state ──────────────────────────────────────────────────────
if 'state' in df.columns:
    state_col = df['state'].astype(str)
    state_stats = (
        df.assign(state_str=state_col)
        .groupby('state_str')
        .agg(
            undervalued_rate = ('undervalued', 'mean'),
            avg_gap          = ('value_gap',   'mean'),
            count            = ('price',       'count'),
        )
        .query('count >= 50')
        .sort_values('undervalued_rate', ascending=False)
    )
    print("\nTop 10 states by undervalued rate:")
    top10 = state_stats.head(10).copy()
    top10['undervalued_rate'] = (top10['undervalued_rate'] * 100).round(1)
    print(top10[['undervalued_rate', 'avg_gap', 'count']].to_string())

    top15 = state_stats.head(15)
    sorted_states = (top15['undervalued_rate'] * 100).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(sorted_states.index, sorted_states.values, color='seagreen')
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(
        xlabel='Undervalued Listings (%)',
        title='Top 15 States by Undervalued Listing Rate',
        xlim=(0, sorted_states.max() * 1.15)
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_undervalued_by_state.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved → outputs/figures/05_undervalued_by_state.png")

# ── Save flagged properties ───────────────────────────────────────────────────
out_path = PROCESSED / 'undervalued_properties.csv'
cols = ['zip_code', 'state', 'city', 'price', 'predicted_price',
        'value_gap', 'value_gap_pct', 'undervalued',
        'bed', 'bath', 'house_size', 'cluster_label']
save_cols = [c for c in cols if c in df.columns]

df[df['undervalued']][save_cols].sort_values(
    'value_gap_pct', ascending=False
).to_csv(out_path, index=False)
print(f"\nSaved {n_undervalued:,} undervalued properties → {out_path}")

print("\nUndervalued analysis complete.")
