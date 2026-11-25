import argparse, time
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve,roc_curve
from .data_loader import load_data
from .features import engineer_features
from .models import model_specs
from .imbalance import undersample
from .evaluation import choose_threshold,metrics
from .explain import importance
from .visualization import setup
from .config import load_config
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample',action='store_true'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); start=time.time()
    cfg=load_config()
    for p in ['outputs/tables','outputs/figures','outputs/models','data/sample']: Path(p).mkdir(parents=True,exist_ok=True)
    df,source=load_data(args.sample,args.seed)
    if args.sample and len(df)>5000: df=df.sample(5000,random_state=args.seed)
    if not args.sample and len(df)>1_000_000: df=df.iloc[:1_000_000]
    # persist the CI fixture from the same seeded generator
    if not Path('data/sample/transactions_sample.csv').exists():
        from .data_generator import generate_transactions
        generate_transactions(5000,args.seed).to_csv('data/sample/transactions_sample.csv',index=False)
    X,y,ordered=engineer_features(df)
    n=len(ordered); i=int(.6*n); j=int(.8*n); Xtr,Xv,Xte=X.iloc[:i],X.iloc[i:j],X.iloc[j:]; ytr,yv,yte=y.iloc[:i],y.iloc[i:j],y.iloc[j:]
    # Small time-ordered tuning comparison: class-weighted estimators and random undersampling.
    rows=[]; predictions={}; fitted={}; tune=[]; validation_scores={}
    for name,model in model_specs(args.seed,max(1,(ytr==0).sum()/max(1,(ytr==1).sum()))).items():
        for strategy in ['class_weight','undersampling','none']:
            if strategy=='undersampling': Xa,ya=undersample(Xtr,ytr,args.seed)
            else: Xa,ya=Xtr,ytr
            candidate=model
            if strategy=='none' and name=='Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                candidate=RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=3,n_jobs=-1,random_state=args.seed)
            candidate.fit(Xa,ya); pv=candidate.predict_proba(Xv)[:,1]
            from sklearn.metrics import average_precision_score
            score=average_precision_score(yv,pv) if yv.sum() else 0
            tune.append({'model':name,'strategy':strategy,'validation_pr_auc':score})
            if strategy=='class_weight': fitted[name]=candidate; validation_scores[name]=score
        model=fitted[name]; pv=model.predict_proba(Xv)[:,1]; pt=model.predict_proba(Xte)[:,1]
        f1p,recp,thresholds=choose_threshold(yv,pv,cfg['precision_target'])
        threshold=recp[0] if recp[1]>=.90 else f1p[0]
        pred=pt>=threshold; m=metrics(yte,pt,threshold); rows.append({'model':name,'threshold':threshold,'policy':'recall_at_precision_0.90' if recp[1]>=.90 else 'maximize_f1',**m})
        predictions[name]=(pt,threshold); joblib.dump(model,f'outputs/models/{name.lower().replace(" ","_")}.joblib')
    comp=pd.DataFrame(rows); comp.to_csv('outputs/tables/model_comparison.csv',index=False); pd.DataFrame(tune).to_csv('outputs/tables/imbalance_comparison.csv',index=False)
    bestname=max(validation_scores,key=validation_scores.get); best=comp.loc[comp.model==bestname].iloc[0]; pt,th=predictions[bestname]
    # Threshold table and two hundred deterministic bootstrap intervals for the final model.
    _,_,sweep=choose_threshold(yv,fitted[bestname].predict_proba(Xv)[:,1]); pd.DataFrame(sweep,columns=['threshold','precision','recall','f1']).to_csv('outputs/tables/thresholds.csv',index=False)
    rng=np.random.default_rng(args.seed); yarr=np.asarray(yte); intervals=[]
    for _ in range(200):
        ix=rng.integers(0,len(yarr),len(yarr)); mm=metrics(yarr[ix],pt[ix],th); intervals.append([mm['precision'],mm['recall']])
    ci=np.percentile(intervals,[2.5,97.5],axis=0)
    pd.DataFrame([{'metric':'precision','lower_95':ci[0,0],'upper_95':ci[1,0]},{'metric':'recall','lower_95':ci[0,1],'upper_95':ci[1,1]}]).to_csv('outputs/tables/bootstrap_ci.csv',index=False)
    expl=importance(fitted[bestname],Xv.sample(min(2000,len(Xv)),random_state=args.seed),yv.loc[Xv.sample(min(2000,len(Xv)),random_state=args.seed).index]); expl.to_csv('outputs/tables/feature_importance.csv',index=False)
    # Slice metrics
    slices=ordered.iloc[i+j*0:] if False else ordered.iloc[i+j-j:] # test frame starts at i+j? use ordered offset below
    test=ordered.iloc[j:].copy(); test['prediction']=pt>=th; test['score']=pt; test['amount_bucket']=pd.qcut(test.amt,4,duplicates='drop')
    slice_rows=[]
    for col in ['amount_bucket','category']:
        for key,g in test.groupby(col,observed=True):
            mm=metrics(g.is_fraud,g.score,th); slice_rows.append({'slice':col,'value':str(key),**mm})
    pd.DataFrame(slice_rows).to_csv('outputs/tables/slice_analysis.csv',index=False)
    # Figures
    setup()
    plt.figure(); plt.bar(['legitimate','fraud'],[(y==0).sum(),(y==1).sum()]); plt.title('Class imbalance'); plt.ylabel('Transactions'); plt.tight_layout(); plt.savefig('outputs/figures/class_imbalance.png',dpi=150); plt.close()
    plt.figure()
    for name,(score,t) in predictions.items():
        from sklearn.metrics import precision_recall_curve
        pr,rc,_=precision_recall_curve(yte,score); plt.plot(rc,pr,label=name)
    plt.axhline(float(yte.mean()),ls='--',color='gray',label='No-skill baseline'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Test precision-recall curves'); plt.legend(); plt.tight_layout(); plt.savefig('outputs/figures/precision_recall.png',dpi=150); plt.close()
    plt.figure()
    for name,(score,t) in predictions.items():
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(yte,score); plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],ls='--',color='gray'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('Test ROC curves'); plt.legend(); plt.tight_layout(); plt.savefig('outputs/figures/roc_curves.png',dpi=150); plt.close()
    # Illustrative cost curve under a fixed review fee plus missed-fraud amount.
    amount=np.asarray(ordered.amt.iloc[j:]); actual=np.asarray(yte)
    cost_rows=[]
    for threshold in np.unique(np.r_[0,np.quantile(pt,np.linspace(0,1,101)),1]):
        alert=pt>=threshold; cost=float(amount[(actual==1)&(~alert)].sum()+cfg['review_cost']*alert.sum()); cost_rows.append({'threshold':threshold,'total_cost':cost})
    pd.DataFrame(cost_rows).to_csv('outputs/tables/cost_curve.csv',index=False)
    plt.figure(); plt.plot([x['threshold'] for x in cost_rows],[x['total_cost'] for x in cost_rows]); plt.xlabel('Threshold'); plt.ylabel('Estimated cost'); plt.title('Cost versus threshold'); plt.tight_layout(); plt.savefig('outputs/figures/cost_curve.png',dpi=150); plt.close()
    threshold_frame=pd.read_csv('outputs/tables/thresholds.csv'); plt.figure(figsize=(7,4))
    for col in ['precision','recall','f1']: plt.plot(threshold_frame.threshold,threshold_frame[col],label=col)
    plt.xlabel('Validation threshold'); plt.ylabel('Score'); plt.title('Validation threshold sweep'); plt.legend(); plt.tight_layout(); plt.savefig('outputs/figures/threshold_sweep.png',dpi=150); plt.close()
    strategy=pd.DataFrame(tune).groupby('strategy').validation_pr_auc.mean(); plt.figure(); strategy.plot.bar(); plt.ylabel('Mean validation PR-AUC'); plt.title('Imbalance strategy comparison'); plt.tight_layout(); plt.savefig('outputs/figures/imbalance_comparison.png',dpi=150); plt.close()
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig,ax=plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(yte,pt>=th),display_labels=['legitimate','fraud']).plot(ax=ax,colorbar=False); ax.set_title(f'{bestname}: test confusion matrix'); fig.tight_layout(); fig.savefig('outputs/figures/confusion_matrix.png',dpi=150); plt.close(fig)
    expl.head(20).sort_values('importance').plot.barh(x='feature',y='importance',legend=False,figsize=(8,6)); plt.title('Permutation importance'); plt.tight_layout(); plt.savefig('outputs/figures/feature_importance.png',dpi=150); plt.close()
    pd.DataFrame([{'data_source':source,'rows':len(df),'fraud_rate':float(y.mean()),'train_rows':len(Xtr),'validation_rows':len(Xv),'test_rows':len(Xte),'best_model':bestname,'chosen_threshold':th,'test_precision':best.precision,'test_recall':best.recall,'test_f1':best.f1,'test_pr_auc':best.pr_auc,'runtime_seconds':time.time()-start}]).to_csv('outputs/tables/run_summary.csv',index=False)
    print(pd.read_csv('outputs/tables/run_summary.csv').to_string(index=False))
if __name__=='__main__': main()
