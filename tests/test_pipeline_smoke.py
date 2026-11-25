from pathlib import Path
from src.data_generator import generate_transactions
from src.features import engineer_features

def test_five_thousand_row_sample_feature_smoke():
    sample=generate_transactions(5000,seed=42)
    X,y,transactions=engineer_features(sample)
    assert len(X)==len(y)==len(transactions)==5000
    assert 0 < y.mean() < .02
    assert 'cc_num' not in X and 'trans_date_trans_time' not in X
    assert Path('data/sample').exists()
