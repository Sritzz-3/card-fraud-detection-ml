from src.data_loader import load_data
from src.features import engineer_features

def test_five_thousand_row_sample_feature_smoke():
    sample,_,source=load_data(sample=True,seed=42)
    X,y,transactions=engineer_features(sample)
    assert len(X)==len(y)==len(transactions)==5000
    assert 0 < y.mean() < .02
    assert 'cc_num' not in X and 'trans_date_trans_time' not in X
    assert source=='synthetic_sample'
