"""Calibration helpers and validation/test score diagnostic artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_curve

def fit_validation_calibrator(y_validation, scores_validation):
    """Fit on first validation half; choose isotonic vs Platt by second-half log loss."""
    y=np.asarray(y_validation,dtype=int); s=np.asarray(scores_validation,dtype=float)
    cut=max(1,len(y)//2); y_fit,s_fit=y[:cut],s[:cut]; y_select,s_select=y[cut:],s[cut:]
    if len(np.unique(y_fit))<2 or len(y_select)==0:
        class Identity:
            def predict(self, values): return np.asarray(values,dtype=float)
        return Identity(),'identity',float('nan')
    iso=IsotonicRegression(out_of_bounds='clip').fit(s_fit,y_fit)
    platt=LogisticRegression(C=1e6,max_iter=1000).fit(s_fit.reshape(-1,1),y_fit)
    candidates=[('isotonic',iso,iso.predict(s_select)),('platt',platt,platt.predict_proba(s_select.reshape(-1,1))[:,1])]
    if len(np.unique(y_select))<2:
        # No valid log-loss comparison is possible; use Platt's lower-variance mapping.
        name,model,values=candidates[1]
        return model,name,float('nan')
    losses=[log_loss(y_select,np.clip(values,1e-8,1-1e-8),labels=[0,1]) for _,_,values in candidates]
    i=int(np.argmin(losses))
    return candidates[i][1],candidates[i][0],float(losses[i])

def apply_calibrator(model, name, scores):
    s=np.asarray(scores,dtype=float)
    if name=='platt': out=model.predict_proba(s.reshape(-1,1))[:,1]
    else: out=model.predict(s)
    return np.clip(np.asarray(out,dtype=float),0,1)

def write_threshold_diagnostics(path, figure_path, model_scores, labels_validation, labels_test,
                                policies, final_model):
    """Save threshold support/score quantiles and validation/test curve/calibration views."""
    yv=np.asarray(labels_validation,dtype=int); yt=np.asarray(labels_test,dtype=int)
    rows=[]
    for model,pack in model_scores.items():
        for calibration in ('uncalibrated','calibrated'):
            sv=np.asarray(pack[f'{calibration}_validation']); st=np.asarray(pack[f'{calibration}_test'])
            for policy,p in policies[model][calibration].items():
                th=float(p['threshold'])
                rows.append({'model':model,'scores':calibration,'policy':policy,'threshold':th,
                  'validation_positives':int(((sv>=th)&(yv==1)).sum()),'validation_flagged':int((sv>=th).sum()),
                  'test_positives':int(((st>=th)&(yt==1)).sum()),'test_flagged':int((st>=th).sum())})
            for period,ys,ss in [('validation',yv,sv),('test',yt,st)]:
                for label in (0,1):
                    vals=ss[ys==label]
                    for q,val in zip([.1,.25,.5,.75,.9],np.quantile(vals,[.1,.25,.5,.75,.9]) if len(vals) else [np.nan]*5):
                        rows.append({'model':model,'scores':calibration,'policy':f'{period}_class_{label}_q{q:g}',
                          'threshold':np.nan,'validation_positives':np.nan,'validation_flagged':np.nan,
                          'test_positives':np.nan,'test_flagged':np.nan,'quantile':q,'score_quantile':val})
    Path(path).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False)
    Path(figure_path).parent.mkdir(parents=True,exist_ok=True)
    # Distribution view for selected final model.
    pack=model_scores[final_model]; fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,period,ys,ss in [(axes[0,0],'Validation',yv,pack['calibrated_validation']),
                            (axes[0,1],'Test',yt,pack['calibrated_test'])]:
        ax.hist(np.asarray(ss)[ys==0],bins=50,density=True,alpha=.65,label='Legitimate')
        ax.hist(np.asarray(ss)[ys==1],bins=50,density=True,alpha=.65,label='Fraud')
        ax.set_title(f'{final_model}: {period} calibrated scores'); ax.set_xlabel('Fraud probability'); ax.set_ylabel('Density'); ax.legend()
    for ax,period,ys,uncal,cal in [(axes[1,0],'Validation',yv,pack['uncalibrated_validation'],pack['calibrated_validation']),
                                  (axes[1,1],'Test',yt,pack['uncalibrated_test'],pack['calibrated_test'])]:
        for scores,label in [(uncal,'Uncalibrated'),(cal,'Calibrated')]:
            frac,mean=calibration_curve(ys,scores,n_bins=10,strategy='quantile')
            ax.plot(mean,frac,marker='o',label=label)
        ax.plot([0,1],[0,1],ls='--',color='gray'); ax.set_title(f'{period} reliability'); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed fraud rate'); ax.legend()
    fig.tight_layout(); fig.savefig(figure_path,dpi=150); plt.close(fig)
    relpath=str(Path(figure_path).with_name('reliability_curves.png'))
    fig,axes=plt.subplots(2,2,figsize=(11,9)); axes=axes.ravel()
    for ax,model in zip(axes,model_scores):
        p=model_scores[model]
        for period,ys,sc in [('Validation',yv,p['calibrated_validation']),('Test',yt,p['calibrated_test'])]:
            frac,mean=calibration_curve(ys,sc,n_bins=10,strategy='quantile')
            ax.plot(mean,frac,marker='o',label=period)
        ax.plot([0,1],[0,1],ls='--',color='gray'); ax.set_title(model); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed fraud rate'); ax.legend()
    fig.tight_layout(); fig.savefig(relpath,dpi=150); plt.close(fig)

def save_pr_overlay(path, model_scores, y_validation, y_test):
    names=list(model_scores); fig,axes=plt.subplots(2,2,figsize=(11,9)); axes=axes.ravel()
    for ax,name in zip(axes,names):
        pack=model_scores[name]
        for label,y,s in [('Validation',y_validation,pack['calibrated_validation']),('Test',y_test,pack['calibrated_test'])]:
            p,r,_=precision_recall_curve(y,s); ax.plot(r,p,label=label)
        ax.set_title(name); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
