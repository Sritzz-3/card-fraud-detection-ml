from pathlib import Path
import pandas as pd
from .data_generator import generate_transactions

def load_data(sample=False, seed=42):
    if sample:
        p=Path('data/sample/transactions_sample.csv')
        if p.exists(): return pd.read_csv(p, parse_dates=['trans_date_trans_time','dob']), 'synthetic'
        return generate_transactions(5000,seed), 'synthetic'
    p=Path('data/raw/transactions.csv')
    if p.exists(): return pd.read_csv(p,parse_dates=['trans_date_trans_time','dob']), 'synthetic'
    return generate_transactions(1_000_000,seed), 'synthetic'
