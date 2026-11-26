import pandas as pd
from src.features import engineer_features,haversine_km

def fixture():
    return pd.DataFrame({'trans_date_trans_time':pd.to_datetime(['2024-01-01 00:00','2024-01-01 00:30','2024-01-01 02:00']),'cc_num':[1]*3,'merchant':['a','b','a'],'category':['grocery_pos','shopping_net','grocery_pos'],'amt':[10.,20.,15.],'gender':['F']*3,'city_pop':[100]*3,'job':['teacher']*3,'dob':pd.to_datetime(['1990-01-01']*3),'lat':[0.]*3,'long':[0.]*3,'merch_lat':[0.,0.,1.],'merch_long':[0.]*3,'is_fraud':[0,1,0]})

def test_haversine_zero_and_one_degree():
    assert haversine_km(0,0,0,0)==0
    assert 111 < haversine_km(0,0,1,0) < 112

def test_prior_rolling_count_and_causality():
    X,y,_=engineer_features(fixture())
    assert X.tx_count_1h.tolist()==[0.,1.,0.]
    changed=fixture(); changed.loc[2,'amt']=9000
    X2,_,_=engineer_features(changed)
    assert X.iloc[:2].equals(X2.iloc[:2])
    changed.loc[2,'category']='future_category'; changed.loc[2,'job']='future_job'
    X3,_,_=engineer_features(changed)
    assert X.iloc[:2].equals(X3.iloc[:2])

def test_removing_later_rows_does_not_change_earlier_features():
    full=fixture(); full['trans_date_trans_time']=pd.to_datetime(['2024-01-01 00:00','2024-01-01 00:30','2024-01-01 02:00'])
    Xfull,_,_=engineer_features(full)
    prefix=full.iloc[:2].copy(); Xprefix,_,_=engineer_features(prefix)
    cols=Xprefix.columns
    assert Xfull.iloc[:2][cols].reset_index(drop=True).equals(Xprefix.reset_index(drop=True))
