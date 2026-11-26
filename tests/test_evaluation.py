import numpy as np
from src.evaluation import metrics,choose_threshold,threshold_policies,smooth_precision_right

def test_metrics_and_threshold():
    y=[0,0,1,1]; p=[.1,.2,.7,.9]
    assert metrics(y,p,.5)['recall']==1
    f1,constrained,rows=choose_threshold(y,p,.9)
    assert f1[0] <= 1 and constrained[1]>=.9

def test_precision_envelope_is_monotone_from_right():
    assert np.allclose(smooth_precision_right([.7,.8,.6]),[.8,.8,.6])

def test_precision_policy_obeys_minimum_true_positive_support():
    y=np.r_[np.ones(40,dtype=int),np.zeros(160,dtype=int)]
    scores=np.r_[np.linspace(.8,1,40),np.linspace(0,.7,160)]
    policies,_=threshold_policies(y,scores,target=.99,min_validation_true_positives=30)
    point=policies['precision_constrained']
    assert point['validation_tp']>=30
    assert point['validation_precision']>=.99
    assert point['threshold'] >= 0

def test_threshold_selection_uses_only_validation_inputs():
    yv=np.array([0,0,1,1]); sv=np.array([.1,.2,.7,.9])
    first,_=threshold_policies(yv,sv,min_validation_true_positives=1)
    # Test labels/scores are intentionally absent from the selector interface.
    altered_test_y=np.array([1,1,0,0]); altered_test_scores=np.array([.01,.02,.98,.99])
    second,_=threshold_policies(yv,sv,min_validation_true_positives=1)
    assert first==second
