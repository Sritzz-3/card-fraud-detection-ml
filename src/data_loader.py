from pathlib import Path
import pandas as pd
from .data_generator import generate_transactions

RAW = Path('data/raw')
KEEP = ['trans_date_trans_time','cc_num','merchant','category','amt','gender','city_pop','job','dob',
        'lat','long','merch_lat','merch_long','is_fraud']

def _read(path):
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in KEEP if c in header]
    df = pd.read_csv(path, usecols=usecols, parse_dates=['trans_date_trans_time', 'dob'])
    df['merchant'] = df['merchant'].astype(str).str.removeprefix('fraud_')
    return df

def load_data(sample=False, seed=42):
    """Return training, later test data and truthful source label."""
    if sample:
        path = Path('data/sample/transactions_sample.csv')
        if path.exists():
            return _read(path), None, 'synthetic_sample'
        return generate_transactions(5000, seed), None, 'synthetic_sample'
    train_path, test_path = RAW / 'fraudTrain.csv', RAW / 'fraudTest.csv'
    if train_path.exists() and test_path.exists():
        return _read(train_path), _read(test_path), 'kaggle_sparkov_public_simulation'
    fallback_path = RAW / 'transactions.csv'
    if fallback_path.exists():
        return _read(fallback_path), None, 'synthetic_fallback'
    return generate_transactions(1_000_000, seed), None, 'synthetic_fallback'
