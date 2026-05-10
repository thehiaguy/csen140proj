# CSEN140 Project — Real Estate Gentrification & Undervalued Property Analysis

## Setup

### 1. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Download the raw data files

Two files are required. Both are gitignored due to size — you must download them manually and place them in the `data/` folder.

**File 1 — Kaggle USA Real Estate Dataset**
1. Go to https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
2. Download `realtor-data.zip.csv`
3. Place it at `data/realtor-data.zip.csv`

**File 2 — Zillow Home Value Index (ZHVI) by Zip Code**
1. Go to https://www.zillow.com/research/data/
2. Under **Home Values**, find **ZHVI All Homes (SFR, Condo/Co-op) Time Series, Smoothed, Seasonally Adjusted**
3. Set the geography dropdown to **Zip Code**
4. Download the CSV
5. Place it at `data/zhvi_zip.csv`

Your `data/` folder should look like this before running anything:
```
data/
├── realtor-data.zip.csv    ← from Kaggle
├── zhvi_zip.csv            ← from Zillow
└── processed/              ← generated automatically by the pipeline
```

---

## Running the Pipeline

Run the scripts in order from the project root:

```bash
python3 scripts/01_eda.py           # Exploratory data analysis
python3 scripts/02_preprocessing.py # Clean, engineer features, merge datasets
python3 scripts/03_clustering.py    # K-Means neighborhood clustering
python3 scripts/04_regression.py    # Random Forest + XGBoost price prediction
python3 scripts/05_undervalued.py   # Undervalued property detection
```

Each script saves its outputs to `outputs/figures/` (plots) and `data/processed/` (intermediate data). Scripts must be run in order — each one depends on the outputs of the previous.

---

## Project Structure

```
project/
├── data/
│   ├── processed/          # Generated parquet/csv files
│   └── ...                 # Raw data goes here (not tracked by git)
├── outputs/
│   ├── figures/            # All generated plots
│   └── models/             # Trained models (XGBoost checked in)
├── scripts/                # Run these in order (01 → 05)
├── src/                    # Shared modules (preprocessing, clustering, regression)
└── requirements.txt
```
