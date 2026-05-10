import numpy as np
import pandas as pd

from .utils import REALTOR_CSV, ZHVI_CSV


def load_and_clean_realtor(path=REALTOR_CSV):
    dtypes = {
        'status':     'category',
        'price':      'float32',
        'bed':        'float32',
        'bath':       'float32',
        'acre_lot':   'float32',
        'house_size': 'float32',
        'state':      'category',
    }
    df = pd.read_csv(path, dtype=dtypes, low_memory=False)

    df = df[df['status'] == 'for_sale'].copy()
    df = df.dropna(subset=['price', 'house_size', 'zip_code'])

    df = df[
        (df['price']      > 10_000)  & (df['price']      < 10_000_000) &
        (df['house_size'] > 100)     & (df['house_size'] < 20_000)     &
        (df['bed']        >= 1)      & (df['bed']        <= 15)        &
        (df['bath']       >= 1)      & (df['bath']       <= 15)
    ]

    # Normalize zip_code to a clean 5-digit string
    df['zip_code'] = (
        df['zip_code'].astype(str)
        .str.extract(r'(\d{5})')[0]
    )
    df = df.dropna(subset=['zip_code'])

    return df.reset_index(drop=True)


def engineer_features(df):
    df = df.copy()
    df['price_per_sqft'] = df['price'] / df['house_size']
    df['price_per_acre'] = df['price'] / df['acre_lot'].replace(0, np.nan)
    df['bed_bath_ratio'] = df['bed']  / df['bath']
    df['total_rooms']    = df['bed']  + df['bath']
    return df


def process_zhvi(path=ZHVI_CSV):
    zhvi = pd.read_csv(path, dtype={'RegionName': str})

    date_cols = [c for c in zhvi.columns if c[:4].isdigit()]

    zhvi['zip_code']     = zhvi['RegionName'].str.zfill(5)
    zhvi['zhvi_current'] = zhvi[date_cols[-1]]
    zhvi['zhvi_1yr_ago'] = zhvi[date_cols[-13]]
    zhvi['zhvi_5yr_ago'] = zhvi[date_cols[-61]] if len(date_cols) >= 61 else np.nan

    zhvi['growth_1yr'] = (
        (zhvi['zhvi_current'] - zhvi['zhvi_1yr_ago']) / zhvi['zhvi_1yr_ago']
    )
    zhvi['growth_5yr'] = (
        (zhvi['zhvi_current'] - zhvi['zhvi_5yr_ago']) / zhvi['zhvi_5yr_ago']
    )

    keep = ['zip_code', 'zhvi_current', 'zhvi_1yr_ago', 'growth_1yr', 'growth_5yr']
    return zhvi[keep].dropna(subset=['zhvi_current']).reset_index(drop=True)


def merge_datasets(df_realtor, df_zhvi):
    return df_realtor.merge(df_zhvi, on='zip_code', how='left')
