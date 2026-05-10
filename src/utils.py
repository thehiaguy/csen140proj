from pathlib import Path

ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / 'data'
PROCESSED    = DATA_DIR / 'processed'
FIGURES_DIR  = ROOT / 'outputs' / 'figures'
MODELS_DIR   = ROOT / 'outputs' / 'models'

REALTOR_CSV  = DATA_DIR / 'realtor-data.zip.csv'
ZHVI_CSV     = DATA_DIR / 'zhvi_zip.csv'

CLEANED_REALTOR    = PROCESSED / 'cleaned_realtor.parquet'
ZHVI_FEATURES      = PROCESSED / 'zhvi_features.csv'
MERGED_DATA        = PROCESSED / 'merged_data.parquet'
NEIGHBORHOOD_FEATS = PROCESSED / 'neighborhood_features.csv'
CLUSTER_LABELS     = PROCESSED / 'cluster_labels.csv'
MODEL_PREDICTIONS  = PROCESSED / 'model_predictions.parquet'


def ensure_dirs():
    for d in [PROCESSED, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
