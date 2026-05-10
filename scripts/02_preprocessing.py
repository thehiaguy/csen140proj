"""
Phase 2 — Preprocessing & Feature Engineering
Run: python scripts/02_preprocessing.py
Outputs:
  data/processed/cleaned_realtor.parquet
  data/processed/zhvi_features.csv
  data/processed/merged_data.parquet
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.utils import (
    CLEANED_REALTOR, ZHVI_FEATURES, MERGED_DATA, ensure_dirs
)
from src.preprocessing import (
    load_and_clean_realtor, engineer_features, process_zhvi, merge_datasets
)

ensure_dirs()

# ── Realtor data ─────────────────────────────────────────────────────────────
print("Loading and cleaning realtor data (full 2.2M rows)…")
df = load_and_clean_realtor()
print(f"After cleaning: {len(df):,} rows")

print("Engineering features…")
df = engineer_features(df)

print(f"Saving → {CLEANED_REALTOR}")
df.to_parquet(CLEANED_REALTOR, index=False)

# ── ZHVI data ─────────────────────────────────────────────────────────────────
print("\nProcessing ZHVI data…")
zhvi = process_zhvi()
print(f"ZHVI zip codes: {len(zhvi):,}")
print(zhvi.describe().round(4))

print(f"Saving → {ZHVI_FEATURES}")
zhvi.to_csv(ZHVI_FEATURES, index=False)

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging datasets on zip_code…")
merged = merge_datasets(df, zhvi)

matched = merged['zhvi_current'].notna().sum()
print(f"Properties matched to ZHVI: {matched:,} / {len(merged):,} ({matched/len(merged)*100:.1f}%)")
print(f"\nFinal merged shape: {merged.shape}")

print(f"Saving → {MERGED_DATA}")
merged.to_parquet(MERGED_DATA, index=False)

print("\nPreprocessing complete.")
