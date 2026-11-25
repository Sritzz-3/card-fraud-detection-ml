import numpy as np
import pandas as pd
from .data_generator import CATEGORIES, JOBS

def haversine_km(lat1, lon1, lat2, lon2):
    lat1,lon1,lat2,lon2=np.radians([lat1,lon1,lat2,lon2])
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def engineer_features(df):
    d=df.copy(); d['trans_date_trans_time']=pd.to_datetime(d.trans_date_trans_time)
    d=d.sort_values(['cc_num','trans_date_trans_time'],kind='stable').reset_index(drop=True)
    t=d.trans_date_trans_time; g=d.groupby('cc_num',sort=False)
    hour=t.dt.hour; d['hour']=hour; d['day_of_week']=t.dt.dayofweek; d['is_weekend']=(t.dt.dayofweek>=5).astype('int8'); d['is_night']=((hour<6)|(hour>=22)).astype('int8')
    d['hour_sin']=np.sin(2*np.pi*hour/24); d['hour_cos']=np.cos(2*np.pi*hour/24)
    d['log_amt']=np.log1p(d.amt.clip(lower=0)); d['history_mean']=g.amt.transform(lambda x:x.shift().expanding().mean()).fillna(0)
    d['history_std']=g.amt.transform(lambda x:x.shift().expanding().std()).fillna(1).replace(0,1)
    d['amount_z']=((d.amt-d.history_mean)/d.history_std).clip(-20,20); d['amount_to_history_max']=d.amt/(g.amt.transform(lambda x:x.shift().cummax()).fillna(d.amt).clip(lower=1))
    # All rolling windows are shifted so current transaction is excluded.
    # Per-card time-window histories using indexed rolling on prior observations.
    for label,window in [('1h','1h'),('6h','6h'),('24h','24h'),('7d','7d')]:
        vals=np.zeros(len(d),dtype=float)
        for _,idx in d.groupby('cc_num',sort=False).groups.items():
            s=d.loc[idx].set_index('trans_date_trans_time').amt
            vals[np.asarray(list(idx))]=s.rolling(window,closed='left').count().to_numpy()
        d[f'tx_count_{label}']=vals
    for label,window in [('1h','1h'),('24h','24h')]:
        sums=np.zeros(len(d),dtype=float)
        for _,idx in d.groupby('cc_num',sort=False).groups.items():
            s=d.loc[idx].set_index('trans_date_trans_time').amt
            sums[np.asarray(list(idx))]=s.rolling(window,closed='left').sum().fillna(0).to_numpy()
        d[f'tx_amount_{label}']=sums
    d['seconds_since_prev']=g.trans_date_trans_time.diff().dt.total_seconds().fillna(1e9).clip(upper=1e9)
    d['home_merchant_km']=haversine_km(d.lat,d.long,d.merch_lat,d.merch_long)
    prev_lat=g.merch_lat.shift(); prev_lon=g.merch_long.shift(); d['prev_merchant_km']=haversine_km(prev_lat.fillna(d.merch_lat),prev_lon.fillna(d.merch_long),d.merch_lat,d.merch_long)
    d['travel_speed_kmh']=(d.prev_merchant_km/(d.seconds_since_prev/3600).clip(lower=1/3600)).clip(upper=1500)
    d['merchant_new']=(d.groupby(['cc_num','merchant'],sort=False).cumcount()==0).astype('int8')
    d['category_new']=(d.groupby(['cc_num','category'],sort=False).cumcount()==0).astype('int8')
    d['category_share']=d.groupby(['cc_num','category'],sort=False).cumcount()/(g.cumcount()+1).clip(lower=1)
    d['age']=((t-dob_series(d)).dt.days/365.25).clip(18,100)
    d['city_pop_log']=np.log1p(d.city_pop)
    d['gender_code']=d.gender.map({'F':0,'M':1}).fillna(-1)
    categories=CATEGORIES; jobs=JOBS
    d['category_frequency']=d.category.astype(str).map({v:i/max(1,len(categories)-1) for i,v in enumerate(categories)})
    d['job_frequency']=d.job.astype(str).map({v:i/max(1,len(jobs)-1) for i,v in enumerate(jobs)})
    # Drop identifying/high cardinality raw fields. Numeric features only.
    excluded={'trans_date_trans_time','cc_num','merchant','category','gender','job','dob','is_fraud','lat','long','merch_lat','merch_long'}
    X=d[[c for c in d.columns if c not in excluded]].replace([np.inf,-np.inf],0).fillna(0)
    order=np.argsort(d.trans_date_trans_time.to_numpy(),kind='stable')
    return X.iloc[order].reset_index(drop=True),d.is_fraud.iloc[order].reset_index(drop=True),d.iloc[order].reset_index(drop=True)

def dob_series(d):
    return pd.to_datetime(d.dob)
