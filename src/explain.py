import pandas as pd
from sklearn.inspection import permutation_importance

def importance(model,X,y,seed=42):
    r=permutation_importance(model,X,y,scoring='average_precision',n_repeats=3,random_state=seed,n_jobs=-1)
    out=pd.DataFrame({'feature':X.columns,'importance':r.importances_mean})
    if hasattr(model,'get_booster'):
        gain=model.get_booster().get_score(importance_type='gain')
        total=sum(gain.values()) or 1.0
        out['xgb_gain_importance']=out.feature.map({k:v/total for k,v in gain.items()}).fillna(0.0)
    return out.sort_values('importance',ascending=False)
