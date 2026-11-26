import numpy as np
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

def metrics(y,p,t):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=p>=t
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
            'f1':f1_score(y,pred,zero_division=0),'pr_auc':average_precision_score(y,p) if y.sum() else 0.0,
            'roc_auc':roc_auc_score(y,p) if len(np.unique(y))>1 else 0.0,
            'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1])}

def smooth_precision_right(precision):
    """Running maximum from right: threshold's precision envelope over stricter cutoffs."""
    precision=np.asarray(precision,dtype=float)
    return np.maximum.accumulate(precision[::-1])[::-1]

def _candidate_table(y,scores,n_grid=3001):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    th=np.unique(np.r_[0.0,np.quantile(scores,np.linspace(0,1,min(n_grid,len(scores)))),1.0])
    order=np.argsort(scores,kind='stable'); sorted_scores=scores[order]; sorted_y=y[order]
    positive_suffix=np.r_[np.cumsum(sorted_y[::-1])[::-1],0]
    ix=np.searchsorted(sorted_scores,th,side='left')
    tp=positive_suffix[ix]
    fp=(len(scores)-ix)-tp
    fn=int(y.sum())-tp
    precision=tp/np.maximum(1,tp+fp); recall=tp/max(1,int(y.sum()))
    f1=2*precision*recall/np.maximum(1e-15,precision+recall)
    return th,precision,recall,f1,tp,fp

def choose_threshold(y,p,target=.90):
    th,pr,rec,f1,tp,fp=_candidate_table(y,p)
    i=int(np.argmax(f1))
    eligible=np.flatnonzero(pr>=target)
    j=eligible[np.argmax(rec[eligible])] if len(eligible) else i
    rows=list(zip(th,pr,rec,f1))
    return (th[i],pr[i],rec[i],f1[i]),(th[j],pr[j],rec[j],f1[j]),rows

def threshold_policies(y, scores, target=.90, min_validation_true_positives=30, capacities=(.0025,.005,.01)):
    th,pr,rec,f1,tp,fp=_candidate_table(y,scores)
    smooth=smooth_precision_right(pr)
    fi=int(np.argmax(f1))
    support=tp>=min_validation_true_positives
    # Require both the monotone envelope and the empirical precision at the
    # selected point; the true-positive support guard prevents tiny lucky tails.
    ok=np.flatnonzero((smooth>=target)&(pr>=target)&support)
    if len(ok):
        ci=ok[np.argmax(rec[ok])]; policy_note='target_precision_and_minimum_support'
    else:
        supported=np.flatnonzero(support)
        if len(supported):
            top=pr[supported].max(); ties=supported[np.isclose(pr[supported],top)]; ci=ties[np.argmax(rec[ties])]
            policy_note='relaxed_precision_to_highest_smoothed_precision_with_support'
        else:
            ci=fi; policy_note='support_unavailable_fell_back_to_f1'
    result={
      'f1_max':{'threshold':float(th[fi]),'validation_precision':float(pr[fi]),'validation_recall':float(rec[fi]),'validation_tp':int(tp[fi]),'note':'maximize_validation_f1'},
      'precision_constrained':{'threshold':float(th[ci]),'validation_precision':float(pr[ci]),'validation_recall':float(rec[ci]),'validation_tp':int(tp[ci]),'smoothed_precision':float(smooth[ci]),'note':policy_note}}
    scores=np.asarray(scores,dtype=float)
    for q in capacities:
        threshold=float(np.quantile(scores,1-q,method='higher'))
        result[f'review_capacity_{q:g}']={'threshold':threshold,'validation_precision':float(precision_score(y,scores>=threshold,zero_division=0)),
            'validation_recall':float(recall_score(y,scores>=threshold,zero_division=0)),'validation_tp':int(((scores>=threshold)&(np.asarray(y)==1)).sum()),'note':f'flag validation top {q:.2%} by score'}
    smooth_table=[{'threshold':float(a),'precision':float(b),'smoothed_precision':float(c),'recall':float(d),'tp':int(e),'fp':int(f)} for a,b,c,d,e,f in zip(th,pr,smooth,rec,tp,fp)]
    return result,smooth_table
