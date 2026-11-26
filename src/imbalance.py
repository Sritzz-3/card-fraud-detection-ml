import numpy as np
import pandas as pd

def stratified_time_sample(X, y, times, max_rows, seed=42, bins=20):
    """Sample across chronological quantile bins and label strata."""
    if len(X) <= max_rows:
        return X, np.asarray(y)
    labels = np.asarray(y)
    t = pd.to_datetime(times).reset_index(drop=True)
    bin_id = pd.qcut(np.arange(len(t)), q=min(bins, len(t)), labels=False, duplicates='drop')
    strata = pd.DataFrame({'bin':bin_id, 'label':labels})
    counts = strata.groupby(['bin','label'], observed=True).size()
    quota = np.floor(counts/counts.sum()*max_rows).astype(int)
    rem = max_rows-int(quota.sum())
    if rem:
        quota.iloc[np.argsort(-(counts/counts.sum()*max_rows-quota).to_numpy())[:rem]] += 1
    rng = np.random.default_rng(seed)
    chosen=[]
    for key, ids in strata.groupby(['bin','label'], sort=False).groups.items():
        take=min(int(quota.loc[key]),len(ids))
        if take: chosen.extend(rng.choice(np.asarray(list(ids)),take,replace=False))
    chosen=np.sort(np.asarray(chosen,dtype=int))
    return X.iloc[chosen], labels[chosen]

def undersample(X, y, seed=42, max_ratio=5):
    y=np.asarray(y); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos): return X,y
    rng=np.random.default_rng(seed); keep=rng.choice(neg,min(len(neg),len(pos)*max_ratio),replace=False)
    ids=np.sort(np.r_[pos,keep])
    return X.iloc[ids],y[ids]
