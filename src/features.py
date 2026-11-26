"""Past-only transaction feature construction over a complete card timeline."""
import numpy as np
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def engineer_features(df):
    d = df.copy()
    d['trans_date_trans_time'] = pd.to_datetime(d.trans_date_trans_time)
    d = d.sort_values(['cc_num', 'trans_date_trans_time'], kind='stable').reset_index(drop=True)
    group = d.groupby('cc_num', sort=False)
    t, hour = d.trans_date_trans_time, d.trans_date_trans_time.dt.hour
    d['hour'], d['day_of_week'] = hour, t.dt.dayofweek
    d['is_weekend'] = (t.dt.dayofweek >= 5).astype('int8')
    d['is_night'] = ((hour < 6) | (hour >= 22)).astype('int8')
    d['hour_sin'], d['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    d['log_amt'] = np.log1p(d.amt.clip(lower=0))
    prior_amount = group.amt.shift()
    d['history_mean'] = prior_amount.groupby(d.cc_num, sort=False).expanding().mean().reset_index(level=0, drop=True).reindex(d.index).fillna(0)
    d['history_std'] = prior_amount.groupby(d.cc_num, sort=False).expanding().std().reset_index(level=0, drop=True).reindex(d.index).fillna(1).replace(0, 1)
    prior_max = prior_amount.groupby(d.cc_num, sort=False).cummax()
    d['amount_z'] = ((d.amt-d.history_mean)/d.history_std).clip(-20, 20)
    d['amount_to_history_max'] = d.amt / prior_max.fillna(d.amt).clip(lower=1)
    # Grouped time-based rolling windows are left-closed: the current row is excluded.
    timed = d.set_index('trans_date_trans_time')
    tg = timed.groupby('cc_num', sort=False)
    for label, window in [('1h','1h'), ('6h','6h'), ('24h','24h'), ('7d','7d')]:
        d[f'tx_count_{label}'] = tg.rolling(window, closed='left').amt.count().to_numpy()
    for label, window in [('1h','1h'), ('24h','24h')]:
        d[f'tx_amount_{label}'] = tg.rolling(window, closed='left').amt.sum().fillna(0).to_numpy()
    d[[f'tx_count_{x}' for x in ('1h','6h','24h','7d')]] = d[[f'tx_count_{x}' for x in ('1h','6h','24h','7d')]].fillna(0)
    d['seconds_since_prev'] = group.trans_date_trans_time.diff().dt.total_seconds().fillna(1e9).clip(upper=1e9)
    d['home_merchant_km'] = haversine_km(d.lat, d.long, d.merch_lat, d.merch_long)
    prev_lat, prev_lon = group.merch_lat.shift(), group.merch_long.shift()
    d['prev_merchant_km'] = haversine_km(prev_lat.fillna(d.merch_lat), prev_lon.fillna(d.merch_long), d.merch_lat, d.merch_long)
    d['travel_speed_kmh'] = (d.prev_merchant_km/(d.seconds_since_prev/3600).clip(lower=1/3600)).clip(upper=1500)
    d['merchant_new'] = (d.groupby(['cc_num','merchant'], sort=False).cumcount() == 0).astype('int8')
    d['category_new'] = (d.groupby(['cc_num','category'], sort=False).cumcount() == 0).astype('int8')
    d['category_share'] = d.groupby(['cc_num','category'], sort=False).cumcount()/(group.cumcount()+1).clip(lower=1)
    d['age'] = ((t-pd.to_datetime(d.dob)).dt.days/365.25).clip(18, 100)
    d['city_pop_log'] = np.log1p(d.city_pop)
    d['gender_code'] = d.gender.map({'F':0, 'M':1}).fillna(-1)
    # Causal, target-free category/job frequency encodings on the combined timeline.
    chrono = d[['category','job','trans_date_trans_time']].sort_values('trans_date_trans_time', kind='stable')
    denom = np.arange(len(chrono), dtype=float) + 1
    cat_freq = chrono.groupby('category', sort=False).cumcount().to_numpy()/denom
    job_freq = chrono.groupby('job', sort=False).cumcount().to_numpy()/denom
    d['category_frequency'] = 0.0; d['job_frequency'] = 0.0
    d.loc[chrono.index, 'category_frequency'] = cat_freq
    d.loc[chrono.index, 'job_frequency'] = job_freq
    excluded = {'trans_date_trans_time','cc_num','merchant','category','gender','job','dob','is_fraud',
                'lat','long','merch_lat','merch_long','first','last','street','city','state','zip','trans_num','unix_time','Unnamed: 0'}
    X = d[[c for c in d.columns if c not in excluded]].select_dtypes(include=[np.number]).replace([np.inf,-np.inf], 0).fillna(0)
    order = np.argsort(d.trans_date_trans_time.to_numpy(), kind='stable')
    return X.iloc[order].reset_index(drop=True), d.is_fraud.astype(int).iloc[order].reset_index(drop=True), d.iloc[order].reset_index(drop=True)
