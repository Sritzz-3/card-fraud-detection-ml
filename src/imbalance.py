import numpy as np
from sklearn.utils import resample

def undersample(X,y,seed=42,max_ratio=5):
    y=np.asarray(y); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos): return X,y
    rng=np.random.default_rng(seed); keep=rng.choice(neg,min(len(neg),len(pos)*max_ratio),replace=False)
    ids=np.r_[pos,keep]; rng.shuffle(ids)
    return X.iloc[ids],y[ids]
