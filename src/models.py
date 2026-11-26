from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ['Logistic Regression','Random Forest','Neural Network','XGBoost']

def parameter_candidates(seed=42):
    """Small deterministic randomized-search candidate sets."""
    return {
      'Logistic Regression':[{'C':.1},{'C':1.0}],
      'Random Forest':[{'max_depth':8,'min_samples_leaf':3},{'max_depth':14,'min_samples_leaf':2}],
      'Neural Network':[{'hidden_layer_sizes':(32,16),'alpha':.001},{'hidden_layer_sizes':(64,32),'alpha':.01}],
      'XGBoost':[{'max_depth':4,'learning_rate':.08},{'max_depth':6,'learning_rate':.05}]
    }

def build_model(name, seed=42, scale_pos_weight=1.0, strategy='none', params=None):
    params=dict(params or {})
    weighted = strategy == 'class_weight'
    if name == 'Logistic Regression':
        estimator=LogisticRegression(max_iter=250, C=params.get('C',.5), class_weight='balanced' if weighted else None, random_state=seed)
        return make_pipeline(StandardScaler(),estimator)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=90,max_depth=params.get('max_depth',10),min_samples_leaf=params.get('min_samples_leaf',3),class_weight='balanced_subsample' if weighted else None,n_jobs=-1,random_state=seed)
    if name == 'Neural Network':
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=params.get('hidden_layer_sizes',(48,24)),alpha=params.get('alpha',.001),early_stopping=True,max_iter=35,random_state=seed))
    if name == 'XGBoost':
        return XGBClassifier(n_estimators=150,max_depth=params.get('max_depth',5),learning_rate=params.get('learning_rate',.07),subsample=.85,colsample_bytree=.85,scale_pos_weight=scale_pos_weight if weighted else 1.0,eval_metric='aucpr',n_jobs=-1,random_state=seed)
    raise ValueError(name)

def model_specs(seed=42, scale_pos_weight=1, strategy='class_weight'):
    return {name:build_model(name,seed,scale_pos_weight,strategy) for name in MODEL_NAMES}
