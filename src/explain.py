import pandas as pd
from sklearn.inspection import permutation_importance

def importance(model,X,y,seed=42):
    r=permutation_importance(model,X,y,scoring='average_precision',n_repeats=3,random_state=seed,n_jobs=-1)
    return pd.DataFrame({'feature':X.columns,'importance':r.importances_mean}).sort_values('importance',ascending=False)
