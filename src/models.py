from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

def model_specs(seed=42, scale_pos_weight=1):
    return {
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',random_state=seed)),
      'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=-1,random_state=seed),
      'Neural Network':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(48,24),early_stopping=True,max_iter=35,random_state=seed)),
      'XGBoost':XGBClassifier(n_estimators=160,max_depth=5,learning_rate=.08,subsample=.85,colsample_bytree=.85,scale_pos_weight=scale_pos_weight,eval_metric='aucpr',n_jobs=-1,random_state=seed)
    }
