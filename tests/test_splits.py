import pandas as pd
from src.splits import time_split,split_train_validation

def test_time_ordered_split():
    d=pd.DataFrame({'trans_date_trans_time':pd.date_range('2024-01-01',periods=10)})
    tr,va,te=time_split(d)
    assert tr.trans_date_trans_time.max()<va.trans_date_trans_time.min()
    assert va.trans_date_trans_time.max()<te.trans_date_trans_time.min()

def test_fraudtest_must_follow_validation():
    train=pd.DataFrame({'trans_date_trans_time':pd.date_range('2020-01-01',periods=10)})
    test=pd.DataFrame({'trans_date_trans_time':pd.date_range('2020-01-11',periods=3)})
    fit,val,held=split_train_validation(train,test)
    assert fit.trans_date_trans_time.max()<val.trans_date_trans_time.min()
    assert val.trans_date_trans_time.max()<held.trans_date_trans_time.min()
