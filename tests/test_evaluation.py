from src.evaluation import metrics,choose_threshold

def test_metrics_and_threshold():
    y=[0,0,1,1]; p=[.1,.2,.7,.9]
    assert metrics(y,p,.5)['recall']==1
    f1,constrained,rows=choose_threshold(y,p,.9)
    assert f1[0] <= 1 and constrained[1]>=.9
