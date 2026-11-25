import pandas as pd
from src.splits import time_split

def test_time_ordered_split():
    d=pd.DataFrame({'trans_date_trans_time':pd.date_range('2024-01-01',periods=10)})
    tr,va,te=time_split(d)
    assert tr.trans_date_trans_time.max()<va.trans_date_trans_time.min()
    assert va.trans_date_trans_time.max()<te.trans_date_trans_time.min()
